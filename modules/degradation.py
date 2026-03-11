"""
Graceful degradation manager.

Wraps each system module with try/fallback logic so that a failure
in any single component (LLM, ELM, 3-D renderer, CTGAN) does not
crash the entire application — instead the system transparently
downgrades to a simpler mode and displays an explicit banner
indicating the active runtime profile.

Degradation levels
------------------
L0  Dual-LLM mode active      → full orchestrator + astro-agent
L0b Single-LLM mode active    → astro-agent only (lighter VRAM)
L1  LLM / agent unavailable   → deterministic tools only
L2  ELM generates unphysical  → analytical cos^{1/4} profile
L3  3-D render timeout         → 2-D heatmap
L4  CTGAN fails to converge    → NASA-only data (no augmentation)
"""

import logging
import time
from typing import Any, Callable

import numpy as np
import streamlit as st

logger = logging.getLogger(__name__)


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
                logger.warning("%s exceeded timeout (%.1fs)", label, elapsed)
                return fallback_fn()
            return result
        except Exception as exc:
            st.warning(
                f"\u26a0\ufe0f {label} failure: {exc}. "
                "Switching to simplified mode."
            )
            logger.warning("%s failure: %s", label, exc)
            return fallback_fn()

    @staticmethod
    def validate_temperature_map(temp_map: np.ndarray) -> bool:
        if temp_map is None:
            return False
        try:
            arr = np.asarray(temp_map, dtype=np.float64)
        except (ValueError, TypeError):
            return False
        if arr.size == 0:
            return False
        if np.any(np.isnan(arr)):
            return False
        if np.any(np.isinf(arr)):
            return False
        if np.any(arr < 0):
            return False
        if np.any(arr > 5000):
            return False
        return True

    @staticmethod
    def check_ollama_available() -> bool:
        """Return True if Ollama is reachable."""
        try:
            import ollama as _oll
            _oll.list()
            return True
        except Exception:
            return False

    @staticmethod
    def display_mode_banner(mode: str) -> None:
        """Show an informational banner about the current runtime profile."""
        if mode == "deterministic":
            st.info(
                "Running in **Deterministic Tools Only** mode. "
                "All physics, ML, and visualization tools are active. "
                "No LLM narratives or AI interpretation are available.",
                icon="\u2699\ufe0f",
            )
        elif mode == "single_llm":
            st.info(
                "Running in **Single-LLM** mode (astro-agent). "
                "One LLM handles both orchestration and domain expertise.",
                icon="\U0001f916",
            )
        elif mode == "dual_llm":
            pass  # no banner needed for full mode


def run_simulation_pipeline(params: dict) -> dict:
    """Full pipeline with built-in degradation at every stage."""
    gd = GracefulDegradation()

    # L2: ELM → PINNFormer → analytical
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

    def pinn_fallback():
        from modules.astro_physics import equilibrium_temperature
        from modules.pinnformer3d import load_pinnformer, predict_temperature_map
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_pinnformer("models/pinn3d_weights.pt", device=device)
        raw = predict_temperature_map(model, n_lat=64, n_lon=128, z=0.5, device=device)
        T_eq = equilibrium_temperature(
            params["star_teff_K"],
            params["star_radius_solar"],
            params["semi_major_axis_au"],
            params.get("albedo", 0.3),
            bool(params.get("tidally_locked", True)),
        )
        T_sub = T_eq * 1.4
        T_night = max(T_eq * 0.3, 40)
        pmin, pmax = float(raw.min()), float(raw.max())
        span = pmax - pmin
        if span < 1e-3:
            return np.full_like(raw, T_eq)
        return T_night + (raw - pmin) / span * (T_sub - T_night)

    def _pinn_or_analytical():
        return gd.run_with_fallback(
            pinn_fallback, analytical_fallback,
            timeout=10.0, label="PINNFormer 3-D",
        )

    temp_map = gd.run_with_fallback(
        elm_prediction, _pinn_or_analytical, timeout=5.0, label="ELM Surrogate"
    )

    _ICON_NO = "⬡"  # U+2B21 — change here to update this module's icon
    _ICON_WARN = '<span style="color:#ff4b4b">⬢</span>'  # U+2B22 — warning (slider red)
    if not gd.validate_temperature_map(temp_map):
        st.error(f"{_ICON_NO} Temperature map unphysical. Falling back to algebraic model.")
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
