"""
PINNFormer 3-D — Transformer-based Physics-Informed Neural Network.

Solves the 3-D stationary heat equation on a tidally locked exoplanet:

    κ ∇²T + S(θ, φ) − σ T⁴ = 0

where (θ, φ, z) are latitude, longitude and atmospheric altitude.
"""

from typing import Tuple

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

    def pinn_loss_3d(
        model: PINNFormer3D,
        x_colloc: torch.Tensor,
        x_bc: torch.Tensor,
        T_bc: torch.Tensor,
        lambda_pde: float = 1.0,
    ) -> torch.Tensor:
        """Physics-informed loss: L_bc + λ · L_PDE.

        PDE residual: κ ∇²T + S(θ,φ) − σ T⁴ = 0
        """
        x_colloc = x_colloc.requires_grad_(True)
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

        T_bc_pred = model(x_bc)
        L_bc = torch.mean((T_bc_pred.squeeze() - T_bc) ** 2)

        return L_bc + lambda_pde * L_pde

    # ── Training helper ───────────────────────────────────────────────────

    def train_pinnformer(
        n_colloc: int = 8192,
        epochs: int = 10_000,
        lr: float = 1e-3,
        T_sub: float = 320.0,
        T_night: float = 80.0,
        device: str = "cuda",
        log_every: int = 500,
    ) -> PINNFormer3D:
        """Train a PINNFormer3D and return the model."""
        if not torch.cuda.is_available() and device == "cuda":
            device = "cpu"

        model = PINNFormer3D().to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=epochs
        )

        # Collocation points: θ∈[-π/2, π/2], φ∈[0, 2π], z∈[0, 1]
        x_c = torch.rand(n_colloc, 3, device=device)
        x_c[:, 0] = x_c[:, 0] * np.pi - np.pi / 2   # θ
        x_c[:, 1] = x_c[:, 1] * 2 * np.pi            # φ
        # z already in [0, 1]

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

        for epoch in range(1, epochs + 1):
            optimiser.zero_grad()
            loss = pinn_loss_3d(model, x_c, x_bc, T_bc)
            loss.backward()
            optimiser.step()
            scheduler.step()

            if epoch % log_every == 0 or epoch == 1:
                print(f"Epoch {epoch:>6d}/{epochs}  loss={loss.item():.6f}")

        return model

    def save_pinnformer(
        model: PINNFormer3D, path: str = "models/pinn3d_weights.pt"
    ) -> None:
        torch.save(model.state_dict(), path)

    def load_pinnformer(
        path: str = "models/pinn3d_weights.pt", device: str = "cpu"
    ) -> PINNFormer3D:
        model = PINNFormer3D()
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model


# ── Fallback stub when torch is missing ───────────────────────────────────────

if not _HAS_TORCH:

    class PINNFormer3D:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            raise ImportError("PyTorch is required for PINNFormer3D")

    def train_pinnformer(*a, **kw):
        raise ImportError("PyTorch is required for PINNFormer3D training")

    def save_pinnformer(*a, **kw):
        raise ImportError("PyTorch is required")

    def load_pinnformer(*a, **kw):
        raise ImportError("PyTorch is required")
