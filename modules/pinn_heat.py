"""
DeepXDE-based 1-D PINN fallback.

Solves the stationary heat equation along the terminator line
(day-night boundary) of a tidally locked planet:

    κ · T'' + S_abs(x) − σ T⁴ = 0,   x ∈ [0, π]

This is a simpler, CPU-friendly alternative to the full 3-D
PINNFormer.
"""

from typing import Optional, Tuple

import numpy as np

try:
    import deepxde as dde

    _HAS_DDE = True
except ImportError:
    _HAS_DDE = False

SIGMA = 5.670374419e-8
KAPPA = 0.025
S_MAX = 900.0


def _pde(x, T):
    """Residual: κ T'' + S(x) - σ T⁴ = 0"""
    dT_xx = dde.grad.hessian(T, x)
    S = S_MAX * dde.backend.maximum(0, dde.backend.cos(x))
    return KAPPA * dT_xx + S - SIGMA * T**4


def train_1d_pinn(
    T_sub: float = 320.0,
    T_night: float = 80.0,
    epochs: int = 10_000,
) -> Optional[object]:
    """Train a 1-D PINN and return the DeepXDE model."""
    if not _HAS_DDE:
        raise ImportError("deepxde is required for the 1-D PINN")

    geom = dde.geometry.Interval(0, np.pi)
    bc_left = dde.icbc.DirichletBC(
        geom, lambda _: T_sub, lambda x, on: np.isclose(x[0], 0)
    )
    bc_right = dde.icbc.DirichletBC(
        geom, lambda _: T_night, lambda x, on: np.isclose(x[0], np.pi)
    )

    data = dde.data.PDE(
        geom, _pde, [bc_left, bc_right], num_domain=256, num_boundary=2
    )
    net = dde.nn.FNN([1] + [64] * 3 + [1], "tanh", "Glorot normal")

    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)
    model.train(epochs=epochs, display_every=max(1, epochs // 10))
    return model


def predict_terminator_profile(
    model: object,
    n_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate trained 1-D PINN along the terminator.

    Returns (x_positions, T_predictions).
    """
    x = np.linspace(0, np.pi, n_points).reshape(-1, 1)
    T = model.predict(x)
    return x.flatten(), T.flatten()
