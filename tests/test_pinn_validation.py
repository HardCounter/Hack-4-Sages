"""Lightweight physics-aware validation tests for PINNFormer 3-D."""

import numpy as np
import pytest

pytest.importorskip("torch")

from modules.astro_physics import habitable_surface_fraction
from modules.model_evaluation import summarise_pinn_history
from modules.pinnformer3d import PINNPhysicsConfig, sample_surface_map, train_pinnformer


@pytest.mark.slow
def test_pinn_training_produces_finite_validation_metrics():
    cfg = PINNPhysicsConfig.from_mode("basic")
    model, history = train_pinnformer(
        cfg=cfg,
        n_colloc=1024,
        epochs=50,
        device="cpu",
        log_every=25,
        verbose=False,
    )

    summary = summarise_pinn_history(history)

    assert np.isfinite(summary.pde_residual_rmse)
    assert np.isfinite(summary.pde_residual_max)
    assert np.isfinite(summary.T_mean)
    assert 30.0 <= summary.T_min < summary.T_max <= 3000.0

    tmap = sample_surface_map(model, device="cpu")
    assert tmap.shape == (32, 64)
    assert np.all(np.isfinite(tmap))

    f_hab = habitable_surface_fraction(tmap)
    assert 0.0 <= f_hab <= 1.0

