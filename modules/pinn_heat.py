"""
DeepXDE-based 1-D PINN fallback.

Solves the stationary heat equation along the terminator line
(day-night boundary) of a tidally locked planet:

    κ · T'' + S_abs(x) − σ T⁴ = 0,   x ∈ [0, π]

**Role in the application:**
This is a lightweight, CPU-friendly alternative to the full 3-D
PINNFormer.  It is *not* exposed in the Streamlit UI by default —
its primary use cases are:

  1. Quick research/prototyping tool to sanity-check 1-D terminator
     temperature profiles without a GPU.
  2. CPU fallback when PyTorch CUDA is unavailable and only a 1-D
     cross-section is needed (e.g. in CI tests or on low-end machines).

If ``deepxde`` is not installed the module exposes safe stub functions
that raise ``ImportError`` with a helpful message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import deepxde as dde

    _HAS_DDE = True
except ImportError:
    _HAS_DDE = False

SIGMA = 5.670374419e-8
KAPPA = 0.025
S_MAX = 900.0


@dataclass
class PINN1DHistory:
    """Lightweight training-history container for the 1-D PINN."""
    final_train_loss: Optional[float] = None
    final_test_loss: Optional[float] = None
    residual_rmse: Optional[float] = None
    T_min: Optional[float] = None
    T_max: Optional[float] = None
    T_mean: Optional[float] = None


def _pde(x, T):
    """Residual: κ T'' + S(x) - σ T⁴ = 0"""
    dT_xx = dde.grad.hessian(T, x)
    S = S_MAX * dde.backend.maximum(0, dde.backend.cos(x))
    return KAPPA * dT_xx + S - SIGMA * T**4


def train_1d_pinn(
    T_sub: float = 320.0,
    T_night: float = 80.0,
    epochs: int = 10_000,
    lr: float = 1e-3,
    num_domain: int = 256,
    hidden_layers: int = 3,
    hidden_size: int = 64,
    verbose: bool = True,
) -> Tuple[object, PINN1DHistory]:
    """Train a 1-D PINN and return ``(model, history)``.

    Parameters
    ----------
    T_sub : float
        Boundary temperature at x = 0 (substellar side, K).
    T_night : float
        Boundary temperature at x = π (nightside, K).
    epochs : int
        Number of Adam optimiser iterations.
    lr : float
        Learning rate for Adam.
    num_domain : int
        Number of collocation points in the interior.
    hidden_layers / hidden_size : int
        Network depth and width.
    verbose : bool
        Print DeepXDE training logs.
    """
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
        geom, _pde, [bc_left, bc_right],
        num_domain=num_domain, num_boundary=2,
    )
    layer_sizes = [1] + [hidden_size] * hidden_layers + [1]
    net = dde.nn.FNN(layer_sizes, "tanh", "Glorot normal")

    model = dde.Model(data, net)
    model.compile("adam", lr=lr)

    display_every = max(1, epochs // 10) if verbose else epochs + 1
    loss_history, _ = model.train(epochs=epochs, display_every=display_every)

    # --- Post-training diagnostics ---
    history = PINN1DHistory()
    if hasattr(loss_history, "loss_train") and len(loss_history.loss_train) > 0:
        history.final_train_loss = float(np.sum(loss_history.loss_train[-1]))
    if hasattr(loss_history, "loss_test") and len(loss_history.loss_test) > 0:
        history.final_test_loss = float(np.sum(loss_history.loss_test[-1]))

    x_eval = np.linspace(0, np.pi, 200).reshape(-1, 1)
    T_eval = model.predict(x_eval).ravel()
    history.T_min = float(T_eval.min())
    history.T_max = float(T_eval.max())
    history.T_mean = float(T_eval.mean())

    # PDE residual at interior points
    try:
        residuals = model.predict(x_eval, operator=_pde).ravel()
        history.residual_rmse = float(np.sqrt(np.mean(residuals ** 2)))
    except Exception:
        pass

    if verbose:
        print("\n── 1-D PINN Post-Training Summary ──")
        print(f"   Final train loss : {history.final_train_loss}")
        print(f"   Final test loss  : {history.final_test_loss}")
        print(f"   Residual RMSE    : {history.residual_rmse}")
        print(f"   T range          : [{history.T_min:.1f}, {history.T_max:.1f}] K")
        print(f"   T mean           : {history.T_mean:.1f} K")

    return model, history


def predict_terminator_profile(
    model: object,
    n_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate trained 1-D PINN along the terminator.

    Returns (x_positions, T_predictions) where x ∈ [0, π].
    """
    x = np.linspace(0, np.pi, n_points).reshape(-1, 1)
    T = model.predict(x)
    return x.flatten(), T.flatten()
