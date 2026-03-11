"""
Quick sanity checks for the ELM climate surrogate.

Run after training models (``python train_models.py``). It:

* Loads ``models/elm_ensemble.pkl``.
* Evaluates a few benchmark planets (Earth-like, Proxima b-like, hot world).
* Prints basic statistics and habitable-surface fractions.
* Saves 2-D temperature-map HTML files into ``diagnostics/`` for inspection.
"""

import os
import sys
from typing import Dict

import numpy as np

# Ensure project root (with local ``modules`` package) is on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from modules.elm_surrogate import ELMClimateSurrogate
from modules.astro_physics import equilibrium_temperature, habitable_surface_fraction
from modules.visualization import create_2d_heatmap


REPO_DIR = ROOT_DIR
MODELS_DIR = os.path.join(REPO_DIR, "models")
DIAG_DIR = os.path.join(REPO_DIR, "diagnostics")
ELM_PATH = os.path.join(MODELS_DIR, "elm_ensemble.pkl")


BENCHMARK_PLANETS: Dict[str, Dict] = {
    "Earth_like": {
        "radius_earth": 1.0,
        "mass_earth": 1.0,
        "semi_major_axis_au": 1.0,
        "star_teff_K": 5778.0,
        "star_radius_solar": 1.0,
        "insol_earth": 1.0,
        "albedo": 0.3,
        "tidally_locked": 0,
    },
    "Proxima_b_like": {
        "radius_earth": 1.1,
        "mass_earth": 1.3,
        "semi_major_axis_au": 0.0485,
        "star_teff_K": 3042.0,
        "star_radius_solar": 0.141,
        "insol_earth": 0.65,
        "albedo": 0.3,
        "tidally_locked": 1,
    },
    "Hot_rock": {
        "radius_earth": 1.2,
        "mass_earth": 2.0,
        "semi_major_axis_au": 0.03,
        "star_teff_K": 6000.0,
        "star_radius_solar": 1.2,
        "insol_earth": 10.0,
        "albedo": 0.2,
        "tidally_locked": 1,
    },
    "Cold_super_earth": {
        "radius_earth": 1.8,
        "mass_earth": 5.0,
        "semi_major_axis_au": 1.5,
        "star_teff_K": 5200.0,
        "star_radius_solar": 0.86,
        "insol_earth": 0.22,
        "albedo": 0.45,
        "tidally_locked": 0,
    },
}


def main() -> None:
    if not os.path.exists(ELM_PATH):
        print(
            f"[ERROR] ELM model not found at {ELM_PATH}.\n"
            "Run `python train_models.py` first."
        )
        return

    os.makedirs(DIAG_DIR, exist_ok=True)

    model = ELMClimateSurrogate().load(ELM_PATH)
    print(f"Loaded ELM surrogate from {ELM_PATH}")

    for name, params in BENCHMARK_PLANETS.items():
        print("\n" + "=" * 72)
        print(f"Benchmark: {name}")
        print("-" * 72)
        for k, v in params.items():
            print(f"  {k:18s}: {v}")

        T_eq = equilibrium_temperature(
            stellar_temp=params["star_teff_K"],
            stellar_radius=params["star_radius_solar"],
            semi_major_axis=params["semi_major_axis_au"],
            albedo=params.get("albedo", 0.3),
            tidally_locked=bool(params.get("tidally_locked", 1)),
        )
        print(f"  Equilibrium T_eq     : {T_eq:.1f} K")

        mean, lower, upper = model.predict_from_params_with_ci(params, alpha=0.1)
        T_min, T_max, T_mean = float(mean.min()), float(mean.max()), float(mean.mean())
        frac_hab = habitable_surface_fraction(mean)

        print(f"  Predicted map T_min  : {T_min:.1f} K")
        print(f"  Predicted map T_max  : {T_max:.1f} K")
        print(f"  Predicted map T_mean : {T_mean:.1f} K")
        print(f"  Habitable surface frac (273–373 K): {frac_hab:.3f}")

        # Very rough sanity check versus T_eq
        if not (0.2 * T_eq <= T_mean <= 2.0 * T_eq):
            print("  [WARN] Mean temperature far from T_eq — check inputs/model.")

        # Save 2-D diagnostic plots
        fig = create_2d_heatmap(mean, planet_name=name)
        out_html = os.path.join(DIAG_DIR, f"elm_{name}.html")
        fig.write_html(out_html)
        print(f"  Saved 2-D temperature map to: {out_html}")

        # Simple check on uncertainty width
        ci_width = float((upper - lower).mean())
        print(f"  Mean 90% CI width     : {ci_width:.2f} K")


if __name__ == "__main__":
    main()

