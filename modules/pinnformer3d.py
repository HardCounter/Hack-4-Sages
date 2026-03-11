"""
PINNFormer 3-D — Transformer-based Physics-Informed Neural Network.

Solves the 3-D stationary heat equation on a tidally locked exoplanet:

    κ ∇²T + S(θ, φ) − σ T⁴ = 0

where (θ, φ, z) are latitude, longitude and atmospheric altitude.
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

# Physical constants
SIGMA = 5.670374419e-8   # Stefan-Boltzmann [W/(m²·K⁴)]
KAPPA = 0.025            # Thermal conductivity [W/(m·K)]
S_MAX = 900.0            # Peak substellar flux [W/m²]


@dataclass
class TrainingHistory:
    """Container for training diagnostics recorded during ``train_pinnformer``."""
    epoch: List[int] = field(default_factory=list)
    loss_total: List[float] = field(default_factory=list)
    loss_bc: List[float] = field(default_factory=list)
    loss_pde: List[float] = field(default_factory=list)
    T_min: List[float] = field(default_factory=list)
    T_max: List[float] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)
    validation: Optional[Dict[str, float]] = None


# ── Model architecture ────────────────────────────────────────────────────────

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
        """Transformer PINN:  (θ, φ, z) → T(θ, φ, z)."""

        def __init__(
            self,
            d_model: int = 128,
            nhead: int = 4,
            num_layers: int = 4,
            dim_ff: int = 256,
        ):
            super().__init__()
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
            self.output_proj = nn.Linear(d_model, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.input_proj(x.unsqueeze(1))
            h = self.pos_enc(h)
            h = self.transformer(h)
            return self.output_proj(h.squeeze(1))

    # ── Physics loss ──────────────────────────────────────────────────────

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
        T_bc: torch.Tensor,
        lambda_pde: float = 1.0,
        return_parts: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Physics-informed loss: L_bc + λ · L_PDE.

        PDE residual: κ ∇²T + S(θ,φ) − σ T⁴ = 0

        When *return_parts* is True, returns
        ``(total_loss, L_bc, L_pde, T_pred_colloc)`` so the training loop
        can log each component and monitor predicted-temperature range.
        """
        x_colloc = x_colloc.requires_grad_(True)

        with sdpa_kernel(_MATH_ONLY):
            T_pred = model(x_colloc)

        grad_T = torch.autograd.grad(
            T_pred.sum(), x_colloc, create_graph=True
        )[0]

        laplacian = torch.zeros(x_colloc.shape[0], device=x_colloc.device)
        for i in range(3):
            g2 = torch.autograd.grad(
                grad_T[:, i].sum(), x_colloc, create_graph=True
            )[0]
            laplacian = laplacian + g2[:, i]

        theta, phi = x_colloc[:, 0], x_colloc[:, 1]
        S = S_MAX * torch.clamp(torch.cos(theta) * torch.cos(phi), min=0)

        residual = KAPPA * laplacian + S - SIGMA * T_pred.squeeze() ** 4
        L_pde = torch.mean(residual**2)

        with sdpa_kernel(_MATH_ONLY):
            T_bc_pred = model(x_bc)
        L_bc = torch.mean((T_bc_pred.squeeze() - T_bc) ** 2)

        total = L_bc + lambda_pde * L_pde
        if return_parts:
            return total, L_bc, L_pde, T_pred.detach()
        return total

    # ── Training helper ───────────────────────────────────────────────────

    def train_pinnformer(
        n_colloc: int = 8192,
        epochs: int = 10_000,
        lr: float = 5e-4,
        T_sub: float = 320.0,
        T_night: float = 80.0,
        device: str = "cuda",
        log_every: int = 500,
        lambda_pde: float = 1.0,
        grad_clip: float = 1.0,
        eta_min_factor: float = 0.01,
        verbose: bool = True,
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
        """
        if not torch.cuda.is_available() and device == "cuda":
            device = "cpu"

        model = PINNFormer3D().to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=epochs, eta_min=lr * eta_min_factor,
        )

        # Collocation points: θ∈[-π/2, π/2], φ∈[0, 2π], z∈[0, 1]
        x_c = torch.rand(n_colloc, 3, device=device)
        x_c[:, 0] = x_c[:, 0] * np.pi - np.pi / 2   # θ
        x_c[:, 1] = x_c[:, 1] * 2 * np.pi            # φ

        # Boundary conditions
        n_bc = 512
        x_bc_sub = torch.zeros(n_bc // 2, 3, device=device)
        x_bc_sub[:, 0] = 0.0   # substellar point θ = 0
        x_bc_sub[:, 1] = 0.0   # φ = 0
        x_bc_sub[:, 2] = torch.rand(n_bc // 2, device=device)
        T_bc_sub = torch.full((n_bc // 2,), T_sub, device=device)

        x_bc_night = torch.zeros(n_bc // 2, 3, device=device)
        x_bc_night[:, 0] = 0.0
        x_bc_night[:, 1] = np.pi
        x_bc_night[:, 2] = torch.rand(n_bc // 2, device=device)
        T_bc_night = torch.full((n_bc // 2,), T_night, device=device)

        x_bc = torch.cat([x_bc_sub, x_bc_night])
        T_bc = torch.cat([T_bc_sub, T_bc_night])

        history = TrainingHistory()

        for epoch in range(1, epochs + 1):
            optimiser.zero_grad()
            total, l_bc, l_pde, T_pred = pinn_loss_3d(
                model, x_c, x_bc, T_bc,
                lambda_pde=lambda_pde,
                return_parts=True,
            )
            total.backward()

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimiser.step()
            scheduler.step()

            if epoch % log_every == 0 or epoch == 1:
                t_min = float(T_pred.min())
                t_max = float(T_pred.max())
                cur_lr = scheduler.get_last_lr()[0]

                history.epoch.append(epoch)
                history.loss_total.append(total.item())
                history.loss_bc.append(l_bc.item())
                history.loss_pde.append(l_pde.item())
                history.T_min.append(t_min)
                history.T_max.append(t_max)
                history.lr.append(cur_lr)

                if verbose:
                    print(
                        f"Epoch {epoch:>6d}/{epochs}  "
                        f"L_total={total.item():.4e}  "
                        f"L_bc={l_bc.item():.4e}  "
                        f"L_pde={l_pde.item():.4e}  "
                        f"T∈[{t_min:.1f}, {t_max:.1f}]  "
                        f"lr={cur_lr:.2e}"
                    )

                # Divergence early-warning
                if t_max > 1e4 or t_min < -1e3:
                    print(
                        f"[WARN] Predicted T range [{t_min:.1f}, {t_max:.1f}] "
                        "looks divergent — consider lowering lr or grad_clip."
                    )

        # ── Validation statistics on a fresh grid ──────────────────────────
        history.validation = _compute_validation_stats(
            model, device=device, lambda_pde=lambda_pde,
        )
        if verbose:
            v = history.validation
            print("\n── Validation (fresh 4 096-point grid) ──")
            print(f"   PDE residual  RMSE : {v['pde_residual_rmse']:.4e}")
            print(f"   PDE residual  max  : {v['pde_residual_max']:.4e}")
            print(f"   Pred T  mean/min/max : "
                  f"{v['T_mean']:.1f} / {v['T_min']:.1f} / {v['T_max']:.1f}")

        return model, history

    def _compute_validation_stats(
        model: PINNFormer3D,
        device: str = "cpu",
        n_val: int = 4096,
        lambda_pde: float = 1.0,
    ) -> Dict[str, float]:
        """Evaluate the trained model on a fresh uniform grid."""
        model.eval()

        # Build points *outside* no_grad so the autograd graph is intact
        # for the second-order Laplacian computation.
        x_v = torch.rand(n_val, 3, device=device)
        x_v[:, 0] = x_v[:, 0] * np.pi - np.pi / 2
        x_v[:, 1] = x_v[:, 1] * 2 * np.pi
        x_v = x_v.requires_grad_(True)

        with sdpa_kernel(_MATH_ONLY):
            T_pred = model(x_v)

        # First derivative — keep graph for second derivative
        grad_T = torch.autograd.grad(T_pred.sum(), x_v, create_graph=True)[0]
        laplacian = torch.zeros(n_val, device=device)
        for i in range(3):
            g2 = torch.autograd.grad(
                grad_T[:, i].sum(), x_v,
                create_graph=False, retain_graph=(i < 2),
            )[0]
            laplacian = laplacian + g2[:, i]

        theta, phi = x_v[:, 0], x_v[:, 1]
        S = S_MAX * torch.clamp(
            torch.cos(theta.detach()) * torch.cos(phi.detach()), min=0,
        )
        residual = KAPPA * laplacian + S - SIGMA * T_pred.squeeze() ** 4

        T_np = T_pred.detach().cpu().numpy().ravel()
        res_np = residual.detach().cpu().numpy().ravel()
        return {
            "pde_residual_rmse": float(np.sqrt(np.mean(res_np ** 2))),
            "pde_residual_max": float(np.max(np.abs(res_np))),
            "T_mean": float(np.mean(T_np)),
            "T_min": float(np.min(T_np)),
            "T_max": float(np.max(T_np)),
            "T_std": float(np.std(T_np)),
        }

    def save_pinnformer(
        model: PINNFormer3D, path: str = "models/pinn3d_weights.pt"
    ) -> None:
        torch.save(model.state_dict(), path)

    def load_pinnformer(
        path: str = "models/pinn3d_weights.pt", device: str = "cpu"
    ) -> PINNFormer3D:
        model = PINNFormer3D()
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)  # Ensure model is on the correct device
        model.eval()
        return model

    def predict_temperature_map(
        model: PINNFormer3D,
        n_lat: int = 64,
        n_lon: int = 128,
        z: float = 0.5,
        device: str = "cpu",
    ) -> np.ndarray:
        """Evaluate the trained PINN on a regular lat/lon grid.

        Returns a (n_lat, n_lon) temperature map in Kelvin.
        """
        model.eval()
        lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
        lon = np.linspace(0, 2 * np.pi, n_lon)
        LAT, LON = np.meshgrid(lat, lon, indexing="ij")

        coords = np.stack(
            [LAT.ravel(), LON.ravel(), np.full(LAT.size, z)], axis=-1
        ).astype(np.float32)

        with torch.no_grad():
            x = torch.from_numpy(coords).to(device)
            T = model(x).cpu().numpy().ravel()

        return T.reshape(n_lat, n_lon)


# ── Fallback stub when torch is missing ───────────────────────────────────────

if not _HAS_TORCH:

    class PINNFormer3D:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            raise ImportError("PyTorch is required for PINNFormer3D")

    def train_pinnformer(*a, **kw):  # type: ignore[no-redef]
        raise ImportError("PyTorch is required for PINNFormer3D training")

    def save_pinnformer(*a, **kw):  # type: ignore[no-redef]
        raise ImportError("PyTorch is required")

    def load_pinnformer(*a, **kw):  # type: ignore[no-redef]
        raise ImportError("PyTorch is required")

    def _compute_validation_stats(*a, **kw):  # type: ignore[no-redef]
        raise ImportError("PyTorch is required")

    def predict_temperature_map(*a, **kw):  # type: ignore[no-redef]
        raise ImportError("PyTorch is required")
