"""Extended evaluation of the ELM climate surrogate against GCM benchmarks."""

import numpy as np
import pytest

from modules.elm_surrogate import (
    ELMClimateSurrogate,
    generate_analytical_training_data,
)
from modules.gcm_benchmarks import get_gcm_benchmark
from modules.model_evaluation import evaluate_elm_against_gcm


@pytest.mark.slow
def test_elm_matches_gcm_benchmarks_reasonably():
    # Train a small ELM ensemble on analytical data
    X, y = generate_analytical_training_data(
        n_samples=800,
        n_lat=ELMClimateSurrogate.N_LAT,
        n_lon=ELMClimateSurrogate.N_LON,
        regime_weights=None,
        verbose=False,
    )
    model = ELMClimateSurrogate(n_ensemble=4, n_neurons=200, alpha=1e-4)
    model.train(X, y)

    results = evaluate_elm_against_gcm(model)

    # All canonical benchmarks should be present
    for key in ("earth_like", "proxima_b", "hot_rock"):
        bench = get_gcm_benchmark(key)
        assert bench is not None
        assert key in results

    # The earth_like benchmark should be well reproduced:
    earth = results["earth_like"]
    assert earth["pattern_correlation"] > 0.85
    assert earth["rmse_K"] < 20.0

    # Other benchmarks are looser but should still show positive correlation.
    for key in ("proxima_b", "hot_rock"):
        metrics = results[key]
        assert metrics["pattern_correlation"] > 0.6
        assert metrics["rmse_K"] < 60.0

