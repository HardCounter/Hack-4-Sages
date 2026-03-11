"""
Model evaluation utilities for the Exoplanetary Digital Twin.

This module provides lightweight, physics-aware diagnostics for:

* ELM climate surrogate vs. synthetic GCM benchmarks
* CTGAN synthetic catalog vs. reference catalog statistics
* PINNFormer 3-D training summaries

The functions here are intentionally deterministic and avoid any heavy
training so they can be called from both the training script and the
pytest suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
import pandas as pd

from .elm_surrogate import ELMClimateSurrogate
from .gcm_benchmarks import compare_surrogate_to_gcm, get_gcm_benchmark
from .pinnformer3d import TrainingHistory


ELM_GCM_CASES = ("earth_like", "proxima_b", "hot_rock")


def _build_elm_features_from_gcm(case_key: str) -> Dict[str, float]:
    """
    Construct an ELM feature dictionary from a GCM benchmark definition.

    Uses the standard irradiance scaling
    S / S_earth = (T_eff / 5778 K)^4 (R_star / R_sun)^2 / a^2,
    and assigns mass and tidal locking state with conservative defaults
    when not explicitly given in the benchmark metadata.
    """
    case = get_gcm_benchmark(case_key)
    if case is None:
        raise KeyError(f"Unknown GCM benchmark: {case_key}")

    star_teff = float(case["star_teff"])
    star_radius = float(case["star_radius"])
    semi_major = float(case["semi_major_axis_au"])
    albedo = float(case.get("albedo", 0.3))
    radius_earth = float(case.get("planet_radius_earth", 1.0))
    tidally_locked = bool(case.get("tidally_locked", False))

    insol_earth = (star_teff / 5778.0) ** 4 * star_radius**2 / semi_major**2

    mass_earth = float(case.get("planet_mass_earth", radius_earth**3.0))

    return {
        "radius_earth": radius_earth,
        "mass_earth": mass_earth,
        "semi_major_axis_au": semi_major,
        "star_teff_K": star_teff,
        "star_radius_solar": star_radius,
        "insol_earth": insol_earth,
        "albedo": albedo,
        "tidally_locked": 1 if tidally_locked else 0,
    }


def evaluate_elm_against_gcm(model: ELMClimateSurrogate) -> Dict[str, Dict[str, float]]:
    """
    Compare an ELMClimateSurrogate to all built-in GCM benchmark cases.

    Returns
    -------
    dict
        Mapping case_key -> metrics dict with keys
        ``pattern_correlation``, ``rmse_K``, ``bias_K``,
        ``zonal_mean_rmse_K``.
    """
    results: Dict[str, Dict[str, float]] = {}
    for key in ELM_GCM_CASES:
        bench = get_gcm_benchmark(key)
        if bench is None:
            continue
        params = _build_elm_features_from_gcm(key)
        tmap = model.predict_from_params(params)
        metrics = compare_surrogate_to_gcm(tmap, bench["temperature_map"])
        results[key] = metrics
    return results


CTGAN_NUMERIC_COLS = [
    "radius_earth",
    "mass_earth",
    "semi_major_axis_au",
    "insol_earth",
    "t_eq_K",
    "star_teff_K",
]


def summarise_ctgan_statistics(
    real: pd.DataFrame,
    synthetic_valid: pd.DataFrame,
) -> Dict[str, Mapping[str, float]]:
    """
    Summarise first- and second-order statistics for CTGAN output.

    The goal is not to perform an exhaustive statistical test suite but
    to provide clear, judge-readable diagnostics: means, standard
    deviations, and simple correlation differences between the real
    catalog and the post-filtered synthetic catalog.
    """
    cols = [c for c in CTGAN_NUMERIC_COLS if c in real.columns and c in synthetic_valid.columns]
    if not cols:
        raise ValueError("No overlapping numeric columns between real and synthetic data.")

    def _stats(df: pd.DataFrame) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for col in cols:
            series = df[col].dropna().to_numpy()
            out[f"{col}_mean"] = float(np.mean(series))
            out[f"{col}_std"] = float(np.std(series))
        return out

    real_stats = _stats(real)
    synth_stats = _stats(synthetic_valid)

    real_corr = real[cols].corr().to_numpy()
    synth_corr = synthetic_valid[cols].corr().to_numpy()
    corr_diff = np.abs(real_corr - synth_corr)
    max_corr_diff = float(np.nanmax(corr_diff))

    return {
        "real": real_stats,
        "synthetic": synth_stats,
        "summary": {"max_correlation_difference": max_corr_diff},
    }


@dataclass
class PinnSummary:
    """
    Compact view of PINNFormer 3-D training and validation metrics.

    Primarily a light wrapper around TrainingHistory.validation, but
    exposed as a dataclass to make JSON serialisation and pytest
    assertions straightforward.
    """

    pde_residual_rmse: float
    pde_residual_max: float
    T_mean: float
    T_min: float
    T_max: float
    T_std: float


def summarise_pinn_history(history: TrainingHistory) -> PinnSummary:
    """
    Convert a TrainingHistory.validation dict into a PinnSummary.

    This does not recompute any physics; it simply normalises field
    access and types for downstream use.
    """
    v = history.validation or {}
    return PinnSummary(
        pde_residual_rmse=float(v.get("pde_residual_rmse", np.nan)),
        pde_residual_max=float(v.get("pde_residual_max", np.nan)),
        T_mean=float(v.get("T_mean", np.nan)),
        T_min=float(v.get("T_min", np.nan)),
        T_max=float(v.get("T_max", np.nan)),
        T_std=float(v.get("T_std", np.nan)),
    )

