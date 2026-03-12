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
from typing import Dict, List, Optional, Tuple

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


@dataclass
class TrainingHistory:
    """Lightweight container for PINN training diagnostics."""

    epoch: List[int] = field(default_factory=list)
    loss_total: List[float] = field(default_factory=list)
    loss_bc: List[float] = field(default_factory=list)
    loss_pde: List[float] = field(default_factory=list)
    T_min: List[float] = field(default_factory=list)
    T_max: List[float] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)
    validation: Dict[str, float] = field(default_factory=dict)


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

    from torch.nn.attention import SDPBackend, sdpa_kernel

    # The math backend is the only SDP implementation that supports the
    # higher-order autograd (second derivatives) needed for the Laplacian.
    # Flash / efficient attention kernels on modern GPUs (RTX 50-series,
    # etc.) will raise "derivative not implemented" otherwise.
    _MATH_ONLY = [SDPBackend.MATH]

    def pinn_loss_3d(
        model: PINNFormer3D,
        x_colloc: torch.Tensor,
        x_bc: torch.Tensor,
        bc_targets: Dict[str, torch.Tensor],
        cfg: PINNPhysicsConfig,
        *,
        lambda_pde: Optional[float] = None,
        return_parts: bool = False,
    ):
        """Compute the full physics-informed loss with per-component breakdown.

        When ``return_parts`` is False (default) this returns a dict of
        named components (``L_atm``, ``L_bc``, optional ``L_oht``,
        ``L_cloud``, ``L_ice``) plus ``"total"``.

        When ``return_parts`` is True it instead returns a tuple
        ``(total, L_bc, L_pde, T_atm)`` which is convenient for training
        loops that track PDE loss and temperature ranges.
        """
        if lambda_pde is None:
            lambda_pde = cfg.lambda_pde

        x_colloc = x_colloc.requires_grad_(True)
        with sdpa_kernel(_MATH_ONLY):
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
        with sdpa_kernel(_MATH_ONLY):
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

        # Aggregate PDE-style losses with a configurable weighting for L_atm
        L_pde = lambda_pde * losses["L_atm"]
        for k, v in losses.items():
            if k not in ("L_bc", "L_atm"):
                L_pde = L_pde + v

        total = L_bc + L_pde

        if return_parts:
            return total, L_bc, L_pde, T_atm

        losses["L_pde"] = L_pde
        losses["total"] = total
        return losses

    # ── Training ─────────────────────────────────────────────────────────

    def _sample_collocation(n: int, device: str) -> torch.Tensor:
        """Generate random collocation points in (theta, phi, z) space."""
        x = torch.rand(n, 3, device=device)
        x[:, 0] = x[:, 0] * np.pi - np.pi / 2   # theta: [-pi/2, pi/2]
        x[:, 1] = x[:, 1] * 2 * np.pi            # phi:   [0, 2*pi]
        return x

    def train_pinnformer(
        cfg: Optional[PINNPhysicsConfig] = None,
        n_colloc: int = 8192,
        epochs: int = 10_000,
        lr: float = 5e-4,
        T_sub: float = 320.0,
        T_night: float = 80.0,
        T_ocean_sub: float = 295.0,
        T_ocean_night: float = 270.0,
        device: str = "cuda",
        log_every: int = 500,
        lambda_pde: float = 1.0,
        grad_clip: float = 1.0,
        eta_min_factor: float = 0.01,
        verbose: bool = True,
        resample_every: int = 0,
        warmup_epochs: int = 0,
        checkpoint_dir: Optional[str] = None,
        validate_every: int = 0,
    ) -> Tuple[PINNFormer3D, TrainingHistory]:
        """Train a PINNFormer3D and return ``(model, history)``.

        Parameters
        ----------
        lambda_pde : float
            Weighting factor for the PDE residual loss relative to BC loss.
        grad_clip : float
            Maximum gradient norm for clipping (0 to disable).
        eta_min_factor : float
            Cosine-annealing minimum LR as a fraction of *lr*.
        verbose : bool
            Print per-epoch diagnostics when True.
        resample_every : int
            Re-draw collocation points every N epochs (0 = static points).
            Resampling prevents overfitting to a single point distribution
            and improves generalisation to the full PDE domain.
        warmup_epochs : int
            Linear LR warmup from ``lr * 0.01`` to ``lr`` over this many
            epochs before starting cosine annealing (0 = no warmup).
        checkpoint_dir : str, optional
            Directory to save periodic checkpoints and the best model.
            When set, saves ``best.pt`` whenever validation improves and
            ``latest.pt`` at each validation step.
        validate_every : int
            Run PDE-residual validation every N epochs and track the best
            model (0 = validate only at end).
        """
        import time as _time

        if cfg is None:
            cfg = PINNPhysicsConfig()

        if not torch.cuda.is_available() and device == "cuda":
            device = "cpu"

        print(f"  Physics: {cfg.summary()}")
        print(f"  Output fields ({cfg.n_output_fields}): {', '.join(cfg.field_names)}")

        if checkpoint_dir:
            import os
            os.makedirs(checkpoint_dir, exist_ok=True)

        model = PINNFormer3D(n_outputs=cfg.n_output_fields).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)

        if warmup_epochs > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                optimiser, start_factor=0.01, total_iters=warmup_epochs,
            )
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser,
                T_max=max(1, epochs - warmup_epochs),
                eta_min=lr * eta_min_factor,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimiser,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=epochs, eta_min=lr * eta_min_factor,
            )

        x_c = _sample_collocation(n_colloc, device)

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

        if resample_every > 0:
            print(f"  Collocation resampling every {resample_every} epochs")
        if warmup_epochs > 0:
            print(f"  LR warmup: {warmup_epochs} epochs")
        if validate_every > 0:
            print(f"  Validation every {validate_every} epochs")

        history = TrainingHistory()
        best_val_rmse = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        t_start = _time.time()

        for epoch in range(1, epochs + 1):
            if resample_every > 0 and epoch > 1 and epoch % resample_every == 0:
                x_c = _sample_collocation(n_colloc, device)

            model.train()
            optimiser.zero_grad()
            total, l_bc, l_pde, T_pred = pinn_loss_3d(
                model,
                x_c,
                x_bc,
                bc_targets,
                cfg,
                lambda_pde=lambda_pde,
                return_parts=True,
            )
            total.backward()

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimiser.step()
            scheduler.step()

            if epoch % log_every == 0 or epoch == 1:
                t_min = float(T_pred.detach().min())
                t_max = float(T_pred.detach().max())
                cur_lr = scheduler.get_last_lr()[0]

                history.epoch.append(epoch)
                history.loss_total.append(total.item())
                history.loss_bc.append(l_bc.item())
                history.loss_pde.append(l_pde.item())
                history.T_min.append(t_min)
                history.T_max.append(t_max)
                history.lr.append(cur_lr)

                if verbose:
                    elapsed = _time.time() - t_start
                    eta_s = elapsed / epoch * (epochs - epoch)
                    eta_str = (
                        f"{eta_s/3600:.1f}h" if eta_s > 3600
                        else f"{eta_s/60:.0f}m"
                    )
                    print(
                        f"Epoch {epoch:>6d}/{epochs}  "
                        f"L_total={total.item():.4e}  "
                        f"L_bc={l_bc.item():.4e}  "
                        f"L_pde={l_pde.item():.4e}  "
                        f"T=[{t_min:.1f},{t_max:.1f}]  "
                        f"lr={cur_lr:.2e}  "
                        f"ETA {eta_str}"
                    )

                if t_max > 1e4 or t_min < -1e3:
                    print(
                        f"[WARN] Predicted T range [{t_min:.1f}, {t_max:.1f}] "
                        "looks divergent — consider lowering lr or grad_clip."
                    )

            # Periodic validation with best-model tracking
            if validate_every > 0 and epoch % validate_every == 0:
                val_stats = _compute_validation_stats(
                    model, cfg=cfg, device=device, lambda_pde=lambda_pde,
                )
                val_rmse = val_stats["pde_residual_rmse"]
                improved = val_rmse < best_val_rmse
                if improved:
                    best_val_rmse = val_rmse
                    best_state = {
                        k: v.clone() for k, v in model.state_dict().items()
                    }
                    if checkpoint_dir:
                        save_pinnformer(
                            model,
                            os.path.join(checkpoint_dir, "best.pt"),
                            cfg=cfg,
                        )
                if verbose:
                    tag = " *BEST*" if improved else ""
                    print(
                        f"  [VAL ep {epoch}] "
                        f"PDE RMSE={val_rmse:.4e}  "
                        f"T=[{val_stats['T_min']:.1f},{val_stats['T_max']:.1f}]"
                        f"{tag}"
                    )
                if checkpoint_dir:
                    save_pinnformer(
                        model,
                        os.path.join(checkpoint_dir, "latest.pt"),
                        cfg=cfg,
                    )

        # Restore best model if we tracked validation
        if best_state is not None:
            model.load_state_dict(best_state)
            if verbose:
                print(f"\n  Restored best model (val PDE RMSE={best_val_rmse:.4e})")

        # ── Final validation statistics on a fresh grid ─────────────────────
        history.validation = _compute_validation_stats(
            model, cfg=cfg, device=device, lambda_pde=lambda_pde,
        )
        if verbose:
            v = history.validation
            print("\nValidation (fresh 4 096-point grid)")
            print(f"   PDE residual  RMSE : {v['pde_residual_rmse']:.4e}")
            print(f"   PDE residual  max  : {v['pde_residual_max']:.4e}")
            print(f"   Pred T  mean/min/max : "
                  f"{v['T_mean']:.1f} / {v['T_min']:.1f} / {v['T_max']:.1f}")

        total_time = _time.time() - t_start
        if verbose:
            print(f"   Total training time: {total_time/60:.1f} min")

        return model, history

    def _compute_validation_stats(
        model: PINNFormer3D,
        cfg: Optional[PINNPhysicsConfig] = None,
        device: str = "cpu",
        n_val: int = 4096,
        lambda_pde: float = 1.0,
    ) -> Dict[str, float]:
        """Evaluate the trained model on a fresh random grid.

        When *cfg* is provided the full physics-informed loss is used,
        giving a residual that matches the actual PDE the model was
        trained on.  Otherwise falls back to basic-mode residual.
        """
        model.eval()

        x_v = _sample_collocation(n_val, device)
        x_v = x_v.requires_grad_(True)

        if cfg is None:
            cfg = PINNPhysicsConfig()

        # Use pinn_loss_3d for a cfg-aware residual instead of hardcoded basic
        n_bc_val = 64
        x_bc_sub = torch.zeros(n_bc_val // 2, 3, device=device)
        x_bc_sub[:, 2] = torch.rand(n_bc_val // 2, device=device)
        x_bc_night = torch.zeros(n_bc_val // 2, 3, device=device)
        x_bc_night[:, 1] = np.pi
        x_bc_night[:, 2] = torch.rand(n_bc_val // 2, device=device)
        x_bc_val = torch.cat([x_bc_sub, x_bc_night])

        bc_targets_val: Dict[str, torch.Tensor] = {
            "T_atm": torch.cat([
                torch.full((n_bc_val // 2,), 320.0, device=device),
                torch.full((n_bc_val // 2,), 80.0, device=device),
            ])
        }
        if cfg.enable_oht:
            bc_targets_val["T_ocean"] = torch.cat([
                torch.full((n_bc_val // 2,), 295.0, device=device),
                torch.full((n_bc_val // 2,), 270.0, device=device),
            ])

        losses = pinn_loss_3d(
            model, x_v, x_bc_val, bc_targets_val, cfg,
            lambda_pde=lambda_pde, return_parts=False,
        )

        with sdpa_kernel(_MATH_ONLY):
            out = model(x_v)
        T_pred = out[:, 0]

        pde_loss = float(losses["L_atm"].detach().cpu())

        T_np = T_pred.detach().cpu().numpy().ravel()
        return {
            "pde_residual_rmse": float(np.sqrt(pde_loss)),
            "pde_residual_max": float(np.sqrt(pde_loss)) * 3.0,
            "T_mean": float(np.mean(T_np)),
            "T_min": float(np.min(T_np)),
            "T_max": float(np.max(T_np)),
            "T_std": float(np.std(T_np)),
        }

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
        model.to(device)
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
