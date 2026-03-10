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

    def prepare_normalised_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data when columns are already in the normalised schema.

        Expects columns matching the combined catalog schema:
        radius_earth, mass_earth, semi_major_axis_au, period_days,
        insol_earth, t_eq_K, star_teff_K, star_radius_solar, star_mass_solar.
        """
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
        data = df[cols].copy().dropna()

        # Binary habitability label in the same spirit as prepare_data()
        data["habitable"] = (
            data["radius_earth"].between(0.5, 2.5)
            & data["insol_earth"].between(0.2, 2.0)
            & data["star_teff_K"].between(2500, 7000)
        ).astype(int)

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
        # Hard physical sanity filters to remove clearly unphysical samples.
        s = synthetic.copy()

        # Drop any rows with NaNs in core columns first.
        core_cols = [
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
        s = s.dropna(subset=[c for c in core_cols if c in s.columns])

        mask = (
            (s["radius_earth"] > 0.3)
            & (s["radius_earth"] < 25.0)
            & (s["mass_earth"] > 0.01)
            & (s["mass_earth"] < 5000)
            & (s["semi_major_axis_au"] > 0.001)
            & (s["semi_major_axis_au"] < 50.0)
            & (s["insol_earth"] > 0.0)
            & (s["insol_earth"] < 1e4)
            & (s["t_eq_K"] > 10)
            & (s["t_eq_K"] < 4000)
            & (s["star_teff_K"] > 2300)
            & (s["star_teff_K"] < 10000)
            & (s["star_radius_solar"] > 0.05)
            & (s["star_radius_solar"] < 20.0)
            & (s["star_mass_solar"] > 0.05)
            & (s["star_mass_solar"] < 20.0)
            & (s["period_days"] > 0.1)
        )
        s = s[mask].copy()

        # Clip remaining extremes to the 1st–99th percentile range per column.
        # This keeps the bulk of the distribution while discarding outliers.
        q_low, q_high = 0.01, 0.99
        for col in core_cols:
            if col not in s.columns or s[col].empty:
                continue
            lo, hi = np.quantile(s[col], [q_low, q_high])
            s = s[(s[col] >= lo) & (s[col] <= hi)]

        return s.copy()

    # ── persistence ───────────────────────────────────────────────────────

    def save_model(self, path: str = "models/ctgan_exoplanets.pkl") -> None:
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, path: str = "models/ctgan_exoplanets.pkl") -> None:
        with open(path, "rb") as f:
            self.model = pickle.load(f)
