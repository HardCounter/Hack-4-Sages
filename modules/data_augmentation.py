"""
CTGAN data augmentation for exoplanet datasets.

Solves the extreme class-imbalance problem in the NASA Exoplanet
Archive (only ~60 potentially habitable planets out of ~5 700) by
synthesising physically plausible planetary configurations.
"""

import io
import pickle
from typing import List, Optional

import pandas as pd
import numpy as np


class _DeviceRemappingUnpickler(pickle.Unpickler):
    """Unpickler that remaps CUDA tensor storages to the available device.

    Fixes 'Attempting to deserialize object on a CUDA device but
    torch.cuda.is_available() is False' when loading a model that was
    trained on CUDA/ROCm on a machine that has no GPU or a different backend.
    """

    def __init__(self, f):
        super().__init__(f)
        try:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            self._device = "cpu"

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            dev = self._device

            def _load(b):
                import torch
                return torch.load(io.BytesIO(b), map_location=dev, weights_only=False)

            return _load
        return super().find_class(module, name)

try:
    from ctgan import CTGAN

    _HAS_CTGAN = True
except ImportError:
    _HAS_CTGAN = False


class ExoplanetDataAugmenter:
    """Train a CTGAN on NASA data and conditionally sample habitable worlds."""

    LOG_COLS = [
        "mass_earth", "semi_major_axis_au", "period_days",
        "star_teff_K", "star_radius_solar", "star_mass_solar",
    ]

    def __init__(self, epochs: int = 300, batch_size: int = 500):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model: Optional[object] = None
        self.discrete_columns: List[str] = []
        self._log_transformed: bool = False

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

        train_data = data.copy()

        # Log-transform heavily right-skewed columns so CTGAN's internal
        # VGM (variational Gaussian mixture) normalizer can model them
        # more accurately — critical for mass_earth.
        self._log_transformed = True
        for col in self.LOG_COLS:
            if col in train_data.columns:
                train_data[col] = np.log1p(train_data[col])

        self.model = CTGAN(
            epochs=self.epochs,
            batch_size=min(self.batch_size, len(train_data)),
            generator_dim=(256, 256),
            discriminator_dim=(256, 256),
            generator_lr=2e-4,
            discriminator_lr=2e-4,
            discriminator_steps=5,
            verbose=True,
        )
        self.model.fit(train_data, discrete_columns=self.discrete_columns)

    # ── sampling ──────────────────────────────────────────────────────────

    def generate_synthetic_planets(
        self,
        n_samples: int = 5000,
        condition_column: str = "habitable",
        condition_value: int = 1,
    ) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("CTGAN model has not been trained")
        result = self.model.sample(
            n=n_samples,
            condition_column=condition_column,
            condition_value=condition_value,
        )
        if self._log_transformed:
            for col in self.LOG_COLS:
                if col in result.columns:
                    result[col] = np.expm1(result[col])
        return result

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
        payload = {
            "model": self.model,
            "log_transformed": self._log_transformed,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load_model(self, path: str = "models/ctgan_exoplanets.pkl") -> None:
        with open(path, "rb") as f:
            payload = _DeviceRemappingUnpickler(f).load()
        if isinstance(payload, dict):
            self.model = payload["model"]
            self._log_transformed = payload.get("log_transformed", False)
        else:
            self.model = payload
            self._log_transformed = False
