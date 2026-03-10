"""
Sanity checks for the CTGAN exoplanet data augmenter.

Run after training with `python train_models.py --ctgan`.
This script:

* Loads the combined exoplanet catalog (NASA + Exoplanet.eu + DACE).
* Prepares the tabular dataset and isolates the "habitable" subset.
* Loads (or, if missing, quickly trains) the CTGAN model.
* Generates synthetic habitable planets.
* Prints side-by-side summary statistics for key parameters so you can
  see how well the synthetic distribution matches the real one.
"""

import os
import sys

import numpy as np
import pandas as pd

# Ensure project root (with local ``modules`` package) is on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from modules.data_augmentation import ExoplanetDataAugmenter
from modules.combined_catalog import build_combined_catalog


REPO_DIR = ROOT_DIR
MODELS_DIR = os.path.join(REPO_DIR, "models")
CTGAN_PATH = os.path.join(MODELS_DIR, "ctgan_exoplanets.pkl")


def _summary(df: pd.DataFrame, col: str) -> str:
    series = df[col].dropna()
    if series.empty:
        return "n/a"
    return (
        f"mean={series.mean():.3g}, std={series.std():.3g}, "
        f"min={series.min():.3g}, max={series.max():.3g}, n={len(series)}"
    )


def main() -> None:
    print("Building combined exoplanet catalog (NASA + Exoplanet.eu + DACE)...")
    raw = build_combined_catalog()
    print(f"  Combined catalog size: {len(raw)} unique planets.")

    augmenter = ExoplanetDataAugmenter()
    data = augmenter.prepare_normalised_data(raw)

    real_hab = data[data["habitable"] == 1].copy()
    print(f"Real 'habitable' subset: {len(real_hab)} planets.")

    # Load or (lightly) train CTGAN
    if os.path.exists(CTGAN_PATH):
        print(f"Loading CTGAN model from {CTGAN_PATH}...")
        augmenter.load_model(CTGAN_PATH)
    else:
        print(
            f"[WARN] CTGAN model not found at {CTGAN_PATH}.\n"
            "Training a quick model (fewer epochs) for diagnostics only..."
        )
        augmenter.epochs = 50
        augmenter.train(data)

    print("Sampling synthetic habitable planets...")
    synthetic = augmenter.generate_synthetic_planets(
        n_samples=5000, condition_column="habitable", condition_value=1
    )
    synthetic = ExoplanetDataAugmenter.validate_synthetic_data(synthetic)
    print(f"Synthetic 'habitable' after validation: {len(synthetic)} planets.")

    cols = [
        "radius_earth",
        "mass_earth",
        "semi_major_axis_au",
        "period_days",
        "insol_earth",
        "t_eq_K",
        "star_teff_K",
        "star_radius_solar",
        "star_mass_solar",
    ]

    print("\nParameter comparison: real vs synthetic (habitable subset)")
    print("-" * 80)
    for col in cols:
        real_stats = _summary(real_hab, col)
        synth_stats = _summary(synthetic, col)
        print(f"{col:20s} | real: {real_stats}")
        print(f"{'':20s} | synth: {synth_stats}")
        print("-" * 80)


if __name__ == "__main__":
    main()

