"""
PINNFormer 3-D — Transformer-based Physics-Informed Neural Network
with configurable climate physics modules.

Solves a coupled PDE system on a (tidally locked) exoplanet with
selectable physics terms:

  **Atmosphere energy balance:**
    κ_atm ∇²T + S(θ,φ)·(1 − α_eff) − (1 − G)·σ T⁴ + F_oht + Q_tidal + v̄·∇T = 0

  **Ocean mixed-layer (OHT):**
    κ_oht ∇²T_ocean − γ(T_ocean − T_atm) = 0

  **Cloud fraction (diagnostic):**
    f_cloud constrained to sigmoid(T_atm) in loss

  **Ice fraction (diagnostic):**
    f_ice constrained to sigmoid(T_freeze − T_atm) in loss

Physics modules (enable/disable at training time)
--------------------------------------------------
- ``basic``      : κ ∇²T + S − σT⁴ = 0 (bare heat equation)
- ``greenhouse`` : adds optical depth G ∈ [0, 1) so atmosphere traps IR
- ``oht``        : couples ocean mixed-layer PDE (Hu & Yang 2014)
- ``clouds``     : temperature-dependent cloud albedo (Yang et al. 2013)
- ``tidal``      : internal tidal heating Q_tidal (Driscoll & Barnes 2015)
- ``ice_albedo`` : ice-albedo positive feedback below 273 K
- ``advection``  : mean zonal wind v̄·∂T/∂φ hotspot shift (Showman & Polvani 2011)
- ``full``       : all of the above

References
----------
- Hu & Yang (2014): OHT on tidally locked planets
- Yang et al. (2013): cloud feedback stabilizing inner HZ
- Driscoll & Barnes (2015): tidal heating in HZ planets
- Showman & Polvani (2011): superrotating jets on hot Jupiters / tidally locked
- Pierrehumbert (2011): substellar cloud decks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# ── Physical constants ───────────────────────────────────────────────────────

SIGMA = 5.670374419e-8
KAPPA_ATM = 0.025
KAPPA_OHT = 0.8
GAMMA_COUPLE = 15.0
S_MAX = 900.0

ALPHA_BASE = 0.30
DELTA_ALPHA_CLOUD = 0.25
T_CLOUD_THRESH = 280.0
T_CLOUD_SCALE = 15.0

ALPHA_ICE = 0.62
ALPHA_GROUND = 0.12
T_FREEZE = 273.15
T_ICE_SCALE = 8.0

GREENHOUSE_G = 0.4

Q_TIDAL_DEFAULT = 0.5
V_ZONAL_DEFAULT = 5.0


# ── Physics configuration ────────────────────────────────────────────────────


@dataclass
class PINNPhysicsConfig:
    """Toggle-able physics modules for PINNFormer training."""

    enable_greenhouse: bool = False
    enable_oht: bool = False
    enable_clouds: bool = False
    enable_tidal: bool = False
    enable_ice_albedo: bool = False
    enable_advection: bool = False

    greenhouse_G: float = GREENHOUSE_G
    kappa_oht: float = KAPPA_OHT
    gamma_couple: float = GAMMA_COUPLE
    q_tidal: float = Q_TIDAL_DEFAULT
    v_zonal: float = V_ZONAL_DEFAULT

    lambda_pde: float = 1.0
    lambda_oht: float = 0.5
    lambda_cloud: float = 0.3
    lambda_ice: float = 0.3
    lambda_advection: float = 0.2

    @classmethod
    def from_mode(cls, mode: str) -> "PINNPhysicsConfig":
        presets = {
            "basic":      dict(),
            "greenhouse": dict(enable_greenhouse=True),
            "oht":        dict(enable_oht=True),
            "clouds":     dict(enable_clouds=True),
            "tidal":      dict(enable_tidal=True),
            "ice_albedo": dict(enable_ice_albedo=True),
            "advection":  dict(enable_advection=True),
            "oht_clouds": dict(enable_oht=True, enable_clouds=True),
            "full":       dict(
                enable_greenhouse=True, enable_oht=True,
                enable_clouds=True, enable_tidal=True,
                enable_ice_albedo=True, enable_advection=True,
            ),
        }
        if mode not in presets:
            raise ValueError(
                f"Unknown PINN mode '{mode}'. "
                f"Available: {', '.join(sorted(presets))}"
            )
        return cls(**presets[mode])

    def summary(self) -> str:
        flags = []
        if self.enable_greenhouse: flags.append("greenhouse")
        if self.enable_oht:        flags.append("OHT")
        if self.enable_clouds:     flags.append("clouds")
        if self.enable_tidal:      flags.append("tidal_heating")
        if self.enable_ice_albedo: flags.append("ice_albedo")
        if self.enable_advection:  flags.append("advection")
        return ", ".join(flags) if flags else "basic (heat equation only)"

    @property
    def n_output_fields(self) -> int:
        n = 1  # T_atm always
        if self.enable_oht:        n += 1  # T_ocean
        if self.enable_clouds:     n += 1  # f_cloud
        if self.enable_ice_albedo: n += 1  # f_ice
        return n

    @property
    def field_names(self) -> list:
        names = ["T_atm"]
        if self.enable_oht:        names.append("T_ocean")
        if self.enable_clouds:     names.append("f_cloud")
        if self.enable_ice_albedo: names.append("f_ice")
        return names


# ── Model architecture ───────────────────────────────────────────────────────

if _HAS_TORCH:

    class WaveletPositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_freq: int = 10):
            super().__init__()
            freqs = torch.linspace(1, max_freq, d_model // 2)
            self.register_buffer("freqs", freqs)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            pos = torch.arange(x.size(1), device=x.device).float().unsqueeze(-1)
            enc = torch.cat(
                [torch.sin(pos * self.freqs), torch.cos(pos * self.freqs)],
                dim=-1,
            )
            return x + enc.unsqueeze(0)

    class PINNFormer3D(nn.Module):
        """Multi-field PINNFormer with configurable output heads.

        Input:  (θ, φ, z)
        Output: (T_atm, [T_ocean], [f_cloud], [f_ice]) — 1 to 4 fields
        """

        def __init__(
            self,
            d_model: int = 128,
            nhead: int = 4,
            num_layers: int = 4,
            dim_ff: int = 256,
            n_outputs: int = 4,
        ):
            super().__init__()
            self.n_outputs = n_outputs
            self.input_proj = nn.Linear(3, d_model)
            self.pos_enc = WaveletPositionalEncoding(d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ff,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )
            self.heads = nn.ModuleList([
                nn.Linear(d_model, 1) for _ in range(n_outputs)
            ])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Returns shape (N, n_outputs)."""
            h = self.input_proj(x.unsqueeze(1))
            h = self.pos_enc(h)
            h = self.transformer(h)
            h = h.squeeze(1)
            return torch.cat([head(h) for head in self.heads], dim=-1)

    # ── Loss computation ─────────────────────────────────────────────────

    def _laplacian(
        field: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        grad_f = torch.autograd.grad(
            field.sum(), x, create_graph=True
        )[0]
        lap = torch.zeros(x.shape[0], device=x.device)
        for i in range(3):
            g2 = torch.autograd.grad(
                grad_f[:, i].sum(), x, create_graph=True
            )[0]
            lap = lap + g2[:, i]
        return lap

    def pinn_loss_3d(
        model: PINNFormer3D,
        x_colloc: torch.Tensor,
        x_bc: torch.Tensor,
        bc_targets: Dict[str, torch.Tensor],
        cfg: PINNPhysicsConfig,
    ) -> Dict[str, torch.Tensor]:
        """Compute the full physics-informed loss with per-component breakdown.

        Returns a dict of named loss components plus ``"total"``.
        """
        x_colloc = x_colloc.requires_grad_(True)
        out = model(x_colloc)
        losses: Dict[str, torch.Tensor] = {}

        # Parse output fields
        idx = 0
        T_atm = out[:, idx]; idx += 1
        T_ocean = out[:, idx] if cfg.enable_oht else None
        if cfg.enable_oht: idx += 1
        f_cloud = out[:, idx] if cfg.enable_clouds else None
        if cfg.enable_clouds: idx += 1
        f_ice = out[:, idx] if cfg.enable_ice_albedo else None
        if cfg.enable_ice_albedo: idx += 1

        theta, phi = x_colloc[:, 0], x_colloc[:, 1]
        S = S_MAX * torch.clamp(torch.cos(theta) * torch.cos(phi), min=0)

        # ── Effective albedo ──
        alpha_eff = torch.full_like(T_atm, ALPHA_BASE)
        if cfg.enable_clouds and f_cloud is not None:
            alpha_eff = alpha_eff + DELTA_ALPHA_CLOUD * torch.sigmoid(f_cloud)
        if cfg.enable_ice_albedo and f_ice is not None:
            alpha_eff = alpha_eff + (ALPHA_ICE - ALPHA_BASE) * torch.sigmoid(f_ice)
        alpha_eff = torch.clamp(alpha_eff, 0.0, 0.95)

        # ── Atmosphere PDE ──
        lap_atm = _laplacian(T_atm, x_colloc)
        emission_factor = (1.0 - cfg.greenhouse_G) if cfg.enable_greenhouse else 1.0
        res_atm = KAPPA_ATM * lap_atm + S * (1.0 - alpha_eff) - emission_factor * SIGMA * T_atm**4

        if cfg.enable_oht and T_ocean is not None:
            res_atm = res_atm + cfg.gamma_couple * (T_ocean - T_atm)

        if cfg.enable_tidal:
            res_atm = res_atm + cfg.q_tidal

        if cfg.enable_advection:
            grad_T = torch.autograd.grad(
                T_atm.sum(), x_colloc, create_graph=True
            )[0]
            dT_dphi = grad_T[:, 1]
            res_atm = res_atm - cfg.v_zonal * dT_dphi

        losses["L_atm"] = torch.mean(res_atm**2)

        # ── Ocean PDE ──
        if cfg.enable_oht and T_ocean is not None:
            lap_ocean = _laplacian(T_ocean, x_colloc)
            res_ocean = cfg.kappa_oht * lap_ocean - cfg.gamma_couple * (T_ocean - T_atm)
            losses["L_oht"] = cfg.lambda_oht * torch.mean(res_ocean**2)

        # ── Cloud constraint ──
        if cfg.enable_clouds and f_cloud is not None:
            target_cloud = torch.sigmoid(
                (T_atm.detach() - T_CLOUD_THRESH) / T_CLOUD_SCALE
            )
            losses["L_cloud"] = cfg.lambda_cloud * torch.mean(
                (torch.sigmoid(f_cloud) - target_cloud)**2
            )

        # ── Ice constraint ──
        if cfg.enable_ice_albedo and f_ice is not None:
            target_ice = torch.sigmoid(
                (T_FREEZE - T_atm.detach()) / T_ICE_SCALE
            )
            losses["L_ice"] = cfg.lambda_ice * torch.mean(
                (torch.sigmoid(f_ice) - target_ice)**2
            )

        # ── Boundary conditions ──
        out_bc = model(x_bc)
        bc_idx = 0
        L_bc = torch.mean((out_bc[:, bc_idx] - bc_targets["T_atm"])**2)
        bc_idx += 1
        if cfg.enable_oht and "T_ocean" in bc_targets:
            L_bc = L_bc + torch.mean((out_bc[:, bc_idx] - bc_targets["T_ocean"])**2)
            bc_idx += 1
        if cfg.enable_clouds:
            bc_idx += 1
        if cfg.enable_ice_albedo:
            bc_idx += 1
        losses["L_bc"] = L_bc

        total = losses["L_bc"] + cfg.lambda_pde * losses["L_atm"]
        for k, v in losses.items():
            if k not in ("L_bc", "L_atm"):
                total = total + v
        losses["total"] = total
        return losses

    # ── Training ─────────────────────────────────────────────────────────

    def train_pinnformer(
        cfg: Optional[PINNPhysicsConfig] = None,
        n_colloc: int = 8192,
        epochs: int = 10_000,
        lr: float = 1e-3,
        T_sub: float = 320.0,
        T_night: float = 80.0,
        T_ocean_sub: float = 295.0,
        T_ocean_night: float = 270.0,
        device: str = "cuda",
        log_every: int = 500,
    ) -> PINNFormer3D:
        """Train the PINNFormer3D with selectable physics modules.

        Set ``cfg`` to control which PDE terms are active. If None,
        defaults to ``PINNPhysicsConfig()`` (basic heat equation).
        """
        if cfg is None:
            cfg = PINNPhysicsConfig()

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        print(f"  Physics: {cfg.summary()}")
        print(f"  Output fields ({cfg.n_output_fields}): {', '.join(cfg.field_names)}")

        model = PINNFormer3D(n_outputs=cfg.n_output_fields).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=epochs
        )

        x_c = torch.rand(n_colloc, 3, device=device)
        x_c[:, 0] = x_c[:, 0] * np.pi - np.pi / 2
        x_c[:, 1] = x_c[:, 1] * 2 * np.pi

        n_bc = 512
        x_bc_sub = torch.zeros(n_bc // 2, 3, device=device)
        x_bc_sub[:, 2] = torch.rand(n_bc // 2, device=device)
        x_bc_night = torch.zeros(n_bc // 2, 3, device=device)
        x_bc_night[:, 1] = np.pi
        x_bc_night[:, 2] = torch.rand(n_bc // 2, device=device)
        x_bc = torch.cat([x_bc_sub, x_bc_night])

        bc_targets = {
            "T_atm": torch.cat([
                torch.full((n_bc // 2,), T_sub, device=device),
                torch.full((n_bc // 2,), T_night, device=device),
            ])
        }
        if cfg.enable_oht:
            bc_targets["T_ocean"] = torch.cat([
                torch.full((n_bc // 2,), T_ocean_sub, device=device),
                torch.full((n_bc // 2,), T_ocean_night, device=device),
            ])

        header = f"{'Epoch':>8s} | {'Total':>10s} | {'L_atm':>10s} | {'L_bc':>10s}"
        if cfg.enable_oht:        header += f" | {'L_oht':>10s}"
        if cfg.enable_clouds:     header += f" | {'L_cloud':>10s}"
        if cfg.enable_ice_albedo: header += f" | {'L_ice':>10s}"
        print(f"\n  {header}")
        print(f"  {'-' * len(header)}")

        for epoch in range(1, epochs + 1):
            optimiser.zero_grad()
            losses = pinn_loss_3d(model, x_c, x_bc, bc_targets, cfg)
            losses["total"].backward()
            optimiser.step()
            scheduler.step()

            if epoch % log_every == 0 or epoch == 1:
                row = (
                    f"  {epoch:>8d} | {losses['total'].item():>10.4f} | "
                    f"{losses['L_atm'].item():>10.4f} | {losses['L_bc'].item():>10.4f}"
                )
                if "L_oht" in losses:   row += f" | {losses['L_oht'].item():>10.4f}"
                if "L_cloud" in losses: row += f" | {losses['L_cloud'].item():>10.4f}"
                if "L_ice" in losses:   row += f" | {losses['L_ice'].item():>10.4f}"
                print(row)

        return model

    # ── Persistence ──────────────────────────────────────────────────────

    def save_pinnformer(
        model: PINNFormer3D,
        path: str = "models/pinn3d_weights.pt",
        cfg: Optional[PINNPhysicsConfig] = None,
    ) -> None:
        payload = {"state_dict": model.state_dict(), "n_outputs": model.n_outputs}
        if cfg is not None:
            payload["physics_config"] = cfg.__dict__
        torch.save(payload, path)

    def load_pinnformer(
        path: str = "models/pinn3d_weights.pt", device: str = "cpu"
    ) -> PINNFormer3D:
        payload = torch.load(path, map_location=device)
        if isinstance(payload, dict) and "state_dict" in payload:
            n_out = payload.get("n_outputs", 4)
            model = PINNFormer3D(n_outputs=n_out)
            model.load_state_dict(payload["state_dict"])
        else:
            model = PINNFormer3D(n_outputs=2)
            model.load_state_dict(payload)
        model.eval()
        return model

    # ── Sampling helpers ─────────────────────────────────────────────────

    def _sample_grid(
        model: PINNFormer3D,
        n_lat: int = 32,
        n_lon: int = 64,
        device: str = "cpu",
    ) -> torch.Tensor:
        model.eval()
        lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
        lon = np.linspace(-np.pi, np.pi, n_lon)
        LAT, LON = np.meshgrid(lat, lon, indexing="ij")
        coords = np.stack([LAT.ravel(), LON.ravel(), np.zeros(n_lat * n_lon)], axis=-1)
        x = torch.tensor(coords, dtype=torch.float32, device=device)
        with torch.no_grad():
            return model(x)

    def sample_surface_map(
        model: PINNFormer3D,
        T_eq: float = 280.0,
        tidally_locked: bool = True,
        n_lat: int = 32,
        n_lon: int = 64,
        device: str = "cpu",
    ) -> np.ndarray:
        out = _sample_grid(model, n_lat, n_lon, device)
        T_map = out[:, 0].cpu().numpy().reshape(n_lat, n_lon)
        return np.clip(T_map, 30.0, 3000.0)

    def sample_ocean_map(
        model: PINNFormer3D,
        n_lat: int = 32,
        n_lon: int = 64,
        device: str = "cpu",
    ) -> Optional[np.ndarray]:
        if model.n_outputs < 2:
            return None
        out = _sample_grid(model, n_lat, n_lon, device)
        T_map = out[:, 1].cpu().numpy().reshape(n_lat, n_lon)
        return np.clip(T_map, 200.0, 400.0)

    def sample_cloud_map(
        model: PINNFormer3D,
        n_lat: int = 32,
        n_lon: int = 64,
        device: str = "cpu",
    ) -> Optional[np.ndarray]:
        if model.n_outputs < 3:
            return None
        out = _sample_grid(model, n_lat, n_lon, device)
        f_cloud = torch.sigmoid(out[:, 2]).cpu().numpy().reshape(n_lat, n_lon)
        return f_cloud

    def sample_ice_map(
        model: PINNFormer3D,
        n_lat: int = 32,
        n_lon: int = 64,
        device: str = "cpu",
    ) -> Optional[np.ndarray]:
        if model.n_outputs < 4:
            return None
        out = _sample_grid(model, n_lat, n_lon, device)
        f_ice = torch.sigmoid(out[:, 3]).cpu().numpy().reshape(n_lat, n_lon)
        return f_ice

    # Legacy alias
    def sample_cloud_albedo_map(*a, **kw):
        return sample_cloud_map(*a, **kw)


# ── Fallback stubs ───────────────────────────────────────────────────────────

if not _HAS_TORCH:

    class PINNPhysicsConfig:  # type: ignore[no-redef]
        pass

    class PINNFormer3D:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            raise ImportError("PyTorch is required for PINNFormer3D")

    def train_pinnformer(*a, **kw):
        raise ImportError("PyTorch required")

    def save_pinnformer(*a, **kw):
        raise ImportError("PyTorch required")

    def load_pinnformer(*a, **kw):
        raise ImportError("PyTorch required")

    def sample_surface_map(*a, **kw):
        raise ImportError("PyTorch required")

    def sample_ocean_map(*a, **kw):
        raise ImportError("PyTorch required")

    def sample_cloud_map(*a, **kw):
        raise ImportError("PyTorch required")

    def sample_ice_map(*a, **kw):
        raise ImportError("PyTorch required")

    def sample_cloud_albedo_map(*a, **kw):
        raise ImportError("PyTorch required")
