"""Tests for CTGAN physical post-filters and statistics."""

import numpy as np
import pandas as pd

from modules.data_augmentation import ExoplanetDataAugmenter
from modules.model_evaluation import summarise_ctgan_statistics


def _make_fake_catalog(n: int = 200) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    return pd.DataFrame(
        {
            "radius_earth": rng.uniform(0.5, 4.0, size=n),
            "mass_earth": rng.uniform(0.1, 20.0, size=n),
            "semi_major_axis_au": rng.uniform(0.02, 5.0, size=n),
            "period_days": rng.uniform(0.5, 500.0, size=n),
            "insol_earth": rng.uniform(0.05, 5.0, size=n),
            "t_eq_K": rng.uniform(150.0, 800.0, size=n),
            "star_teff_K": rng.uniform(2600.0, 6500.0, size=n),
            "star_radius_solar": rng.uniform(0.1, 1.5, size=n),
            "star_mass_solar": rng.uniform(0.1, 1.5, size=n),
        }
    )


def test_validate_synthetic_data_filters_unphysical_rows():
    base = _make_fake_catalog(50)
    bad = base.copy()
    bad.loc[0, "radius_earth"] = -1.0
    bad.loc[1, "t_eq_K"] = 9000.0
    bad.loc[2, "insol_earth"] = 1e6

    cleaned = ExoplanetDataAugmenter.validate_synthetic_data(bad)
    assert len(cleaned) < len(bad)
    assert (cleaned["radius_earth"] > 0.3).all()
    assert (cleaned["t_eq_K"] < 4000.0).all()
    assert (cleaned["insol_earth"] < 1e4).all()


def test_summarise_ctgan_statistics_produces_finite_metrics():
    real = _make_fake_catalog(150)
    synth = _make_fake_catalog(150)

    summary = summarise_ctgan_statistics(real, synth)

    assert "real" in summary and "synthetic" in summary and "summary" in summary
    assert summary["summary"]["max_correlation_difference"] >= 0.0
    for stats in (summary["real"], summary["synthetic"]):
        for value in stats.values():
            assert np.isfinite(value)

