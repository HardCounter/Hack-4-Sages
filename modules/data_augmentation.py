"""
CTGAN data augmentation for exoplanet datasets.

Solves the extreme class-imbalance problem in the NASA Exoplanet
Archive (only ~60 potentially habitable planets out of ~5 700) by
synthesising physically plausible planetary configurations.
"""

import pickle
from typing import List, Optional

import pandas as pd
import numpy as np

try:
    from ctgan import CTGAN

    _HAS_CTGAN = True
except ImportError:
    _HAS_CTGAN = False


class ExoplanetDataAugmenter:
    """Train a CTGAN on NASA data and conditionally sample habitable worlds."""

    def __init__(self, epochs: int = 300, batch_size: int = 500):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model: Optional[object] = None
        self.discrete_columns: List[str] = []

    # ── data prep ─────────────────────────────────────────────────────────

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean, convert units, and label habitability."""
        cols = [
            "pl_radj", "pl_bmassj", "pl_orbsmax", "pl_orbper",
            "pl_insol", "pl_eqt", "st_teff", "st_rad", "st_mass",
        ]
        data = df[cols].copy().dropna()

        # Jupiter → Earth units
        data["pl_radj"] = data["pl_radj"] * 11.209
        data["pl_bmassj"] = data["pl_bmassj"] * 317.83

        # Binary habitability label
        data["habitable"] = (
            data["pl_radj"].between(0.5, 2.5)
            & data["pl_insol"].between(0.2, 2.0)
            & data["st_teff"].between(2500, 7000)
        ).astype(int)

        data.columns = [
            "radius_earth", "mass_earth", "semi_major_axis_au",
            "period_days", "insol_earth", "t_eq_K",
            "star_teff_K", "star_radius_solar", "star_mass_solar",
            "habitable",
        ]

        self.discrete_columns = ["habitable"]
        return data

    # ── training ──────────────────────────────────────────────────────────

    def train(self, data: pd.DataFrame) -> None:
        if not _HAS_CTGAN:
            raise RuntimeError("ctgan package is not installed")

        self.model = CTGAN(
            epochs=self.epochs,
            batch_size=min(self.batch_size, len(data)),
            generator_dim=(256, 256),
            discriminator_dim=(256, 256),
            generator_lr=2e-4,
            discriminator_lr=2e-4,
            discriminator_steps=1,
            verbose=True,
        )
        self.model.fit(data, discrete_columns=self.discrete_columns)

    # ── sampling ──────────────────────────────────────────────────────────

    def generate_synthetic_planets(
        self,
        n_samples: int = 5000,
        condition_column: str = "habitable",
        condition_value: int = 1,
    ) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("CTGAN model has not been trained")
        return self.model.sample(
            n=n_samples,
            condition_column=condition_column,
            condition_value=condition_value,
        )

    # ── post-hoc physics filter ───────────────────────────────────────────

    @staticmethod
    def validate_synthetic_data(synthetic: pd.DataFrame) -> pd.DataFrame:
        mask = (
            (synthetic["radius_earth"] > 0.3)
            & (synthetic["radius_earth"] < 25.0)
            & (synthetic["mass_earth"] > 0.01)
            & (synthetic["mass_earth"] < 5000)
            & (synthetic["semi_major_axis_au"] > 0.001)
            & (synthetic["star_teff_K"] > 2300)
            & (synthetic["star_teff_K"] < 10000)
            & (synthetic["t_eq_K"] > 10)
            & (synthetic["t_eq_K"] < 3000)
            & (synthetic["period_days"] > 0.1)
        )
        return synthetic[mask].copy()

    # ── persistence ───────────────────────────────────────────────────────

    def save_model(self, path: str = "models/ctgan_exoplanets.pkl") -> None:
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, path: str = "models/ctgan_exoplanets.pkl") -> None:
        with open(path, "rb") as f:
            self.model = pickle.load(f)
