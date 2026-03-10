"""
Lightweight tests for the PINNFormer3D module.

These are smoke / sanity tests only:
* ensure the model can do a forward pass on a small batch,
* ensure pinn_loss_3d returns a finite scalar,
* ensure a very short training run executes without error,
* ensure sample_surface_map returns a finite 2-D array.

Tests are skipped automatically if PyTorch is not available.
"""

import numpy as np
import pytest


try:
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover - env without torch
    _HAS_TORCH = False


pytestmark = pytest.mark.skipif(
    not _HAS_TORCH, reason="PyTorch not installed; skipping PINNFormer3D tests"
)


from modules.pinnformer3d import (  # type: ignore  # noqa: E402
    PINNFormer3D,
    pinn_loss_3d,
    sample_surface_map,
)


def test_forward_pass_shapes():
    """PINNFormer3D forward pass returns correct shape."""
    model = PINNFormer3D(d_model=32, nhead=4, num_layers=1, dim_ff=64)
    x = torch.zeros(10, 3)  # 10 points in (theta, phi, z)
    y = model(x)
    assert y.shape == (10, 1)


def test_pinn_loss_finite():
    """pinn_loss_3d returns a finite scalar for small toy batch."""
    model = PINNFormer3D(d_model=32, nhead=4, num_layers=1, dim_ff=64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    x_colloc = torch.rand(64, 3, device=device)
    x_bc = torch.zeros(16, 3, device=device)
    T_bc = torch.full((16,), 300.0, device=device)

    loss = pinn_loss_3d(model, x_colloc, x_bc, T_bc)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_short_training_run():
    """Very short manual training loop runs without error."""
    model = PINNFormer3D(d_model=32, nhead=4, num_layers=1, dim_ff=64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    x_colloc = torch.rand(128, 3, device=device)
    x_bc = torch.zeros(32, 3, device=device)
    T_bc = torch.full((32,), 300.0, device=device)

    for _ in range(3):
        optimiser.zero_grad()
        loss = pinn_loss_3d(model, x_colloc, x_bc, T_bc)
        loss.backward()
        optimiser.step()

    assert torch.isfinite(loss)


def test_sample_surface_map_shape_and_finite():
    """sample_surface_map returns a finite 2-D NumPy array."""
    model = PINNFormer3D(d_model=32, nhead=4, num_layers=1, dim_ff=64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    n_lat, n_lon = 16, 32
    temp_map = sample_surface_map(
        model,
        n_lat=n_lat,
        n_lon=n_lon,
        z_level=0.0,
        device=device,
        target_T_eq=300.0,
    )

    assert isinstance(temp_map, np.ndarray)
    assert temp_map.shape == (n_lat, n_lon)
    assert np.isfinite(temp_map).all()

