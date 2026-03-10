"""
Graceful degradation manager.

Wraps each system module with try/fallback logic so that a failure
in any single component (LLM, ELM, 3-D renderer, CTGAN) does not
crash the entire application — instead the system transparently
downgrades to a simpler mode.

Degradation levels
------------------
L1  LLM / agent unavailable   → manual sliders + algebraic calc
L2  ELM generates unphysical  → analytical cos^{1/4} profile
L3  3-D render timeout         → 2-D heatmap
L4  CTGAN fails to converge    → NASA-only data (no augmentation)
"""

import time
from typing import Any, Callable

import numpy as np
import streamlit as st


class GracefulDegradation:

    @staticmethod
    def run_with_fallback(
        primary_fn: Callable,
        fallback_fn: Callable,
        timeout: float = 10.0,
        label: str = "module",
    ) -> Any:
        """Execute *primary_fn*; on failure or timeout switch to *fallback_fn*."""
        try:
            start = time.time()
            result = primary_fn()
            elapsed = time.time() - start
            if elapsed > timeout:
                st.warning(
                    f"\u26a0\ufe0f {label} exceeded timeout "
                    f"({elapsed:.1f}s). Switching to simplified mode."
                )
                return fallback_fn()
            return result
        except Exception as exc:
            st.warning(
                f"\u26a0\ufe0f {label} failure: {exc}. "
                "Switching to simplified mode."
            )
            return fallback_fn()

    @staticmethod
    def validate_temperature_map(temp_map: np.ndarray) -> bool:
        if temp_map is None:
            return False
        if np.any(np.isnan(temp_map)):
            return False
        if np.any(temp_map < 0):
            return False
        if np.any(temp_map > 5000):
            return False
        return True


def run_simulation_pipeline(params: dict) -> dict:
    """Full pipeline with built-in degradation at every stage."""
    gd = GracefulDegradation()

    # L2: ELM → fallback to analytical
    def elm_prediction():
        from modules.elm_surrogate import ELMClimateSurrogate

        model = ELMClimateSurrogate()
        model.load("models/elm_ensemble.pkl")
        return model.predict_from_params(params)

    def analytical_fallback():
        from modules.astro_physics import equilibrium_temperature
        from modules.visualization import generate_eyeball_map

        T_eq = equilibrium_temperature(
            params["star_teff_K"],
            params["star_radius_solar"],
            params["semi_major_axis_au"],
            params.get("albedo", 0.3),
            bool(params.get("tidally_locked", True)),
        )
        return generate_eyeball_map(
            T_eq, tidally_locked=bool(params.get("tidally_locked", 1))
        )

    temp_map = gd.run_with_fallback(
        elm_prediction, analytical_fallback, timeout=5.0, label="ELM Surrogate"
    )

    if not gd.validate_temperature_map(temp_map):
        st.error("\u274c Temperature map unphysical. Falling back to algebraic model.")
        temp_map = analytical_fallback()

    # L3: 3-D → fallback to 2-D
    def render_3d():
        from modules.visualization import create_3d_globe
        return create_3d_globe(
            temp_map, planet_name=params.get("name", "Exoplanet")
        )

    def render_2d():
        from modules.visualization import create_2d_heatmap
        return create_2d_heatmap(
            temp_map, planet_name=params.get("name", "Exoplanet")
        )

    fig = gd.run_with_fallback(
        render_3d, render_2d, timeout=8.0, label="3-D Renderer"
    )

    return {
        "temperature_map": temp_map,
        "figure": fig,
        "T_min": float(temp_map.min()),
        "T_max": float(temp_map.max()),
        "T_mean": float(temp_map.mean()),
    }
