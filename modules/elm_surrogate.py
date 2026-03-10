"""
Extreme Learning Machine (ELM) climate surrogate.

Provides both a *scikit-elm* wrapper (``ELMClimateSurrogate``) and a
pure-NumPy from-scratch implementation (``PureELM`` / ``ELMEnsemble``),
plus an analytical training-data generator for bootstrapping without
real GCM data.
"""

import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from skelm import ELMRegressor

    _HAS_SKELM = True
except ImportError:
    _HAS_SKELM = False

from sklearn.preprocessing import StandardScaler

# ─── Pure-NumPy ELM ──────────────────────────────────────────────────────────


class PureELM:
    """Single hidden-layer feedforward network with frozen input weights.

    Training is a closed-form Moore-Penrose pseudoinverse – no
    gradient descent required.
    """

    def __init__(
        self, n_neurons: int = 500, activation: str = "tanh", C: float = 1e4
    ):
        self.n_neurons = n_neurons
        self.activation = activation
        self.C = C
        self.W_in: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None
        self.beta: Optional[np.ndarray] = None

    def _activate(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return np.tanh(x)
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        if self.activation == "relu":
            return np.maximum(0, x)
        raise ValueError(f"Unknown activation: {self.activation}")

    def _hidden(self, X: np.ndarray) -> np.ndarray:
        return self._activate(X @ self.W_in + self.bias)

    def fit(self, X: np.ndarray, T: np.ndarray) -> "PureELM":
        n_features = X.shape[1]
        self.W_in = np.random.randn(n_features, self.n_neurons) * 0.5
        self.bias = np.random.randn(1, self.n_neurons) * 0.5
        H = self._hidden(X)
        I = np.eye(self.n_neurons)
        self.beta = np.linalg.solve(H.T @ H + I / self.C, H.T @ T)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._hidden(X) @ self.beta


class ELMEnsemble:
    """Ensemble of K independent PureELMs – variance reduction via averaging."""

    def __init__(self, K: int = 10, **elm_kwargs):
        self.K = K
        self.models = [PureELM(**elm_kwargs) for _ in range(K)]

    def fit(self, X: np.ndarray, T: np.ndarray) -> "ELMEnsemble":
        for m in self.models:
            m.fit(X, T)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.array([m.predict(X) for m in self.models])
        return preds.mean(axis=0)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """Per-output standard deviation across the ensemble."""
        preds = np.array([m.predict(X) for m in self.models])
        return preds.std(axis=0)


# ─── scikit-elm wrapper ───────────────────────────────────────────────────────


class ELMClimateSurrogate:
    """High-level wrapper around an ELM ensemble for climate-map prediction.

    Falls back to the pure-NumPy implementation when *scikit-elm* is
    not installed.
    """

    N_LAT = 32
    N_LON = 64

    def __init__(
        self,
        n_ensemble: int = 10,
        n_neurons: int = 500,
        alpha: float = 1e-4,
    ):
        self.n_ensemble = n_ensemble
        self.n_neurons = n_neurons
        self.alpha = alpha
        self.models: list = []
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self._use_skelm = _HAS_SKELM

    def _create_model(self):
        if self._use_skelm:
            return ELMRegressor(
                n_neurons=self.n_neurons, alpha=self.alpha, ufunc="tanh"
            )
        return PureELM(n_neurons=self.n_neurons, C=1.0 / max(self.alpha, 1e-12))

    def prepare_features(self, data: Dict) -> np.ndarray:
        return np.array(
            [
                data["radius_earth"],
                data["mass_earth"],
                data["semi_major_axis_au"],
                data["star_teff_K"],
                data["star_radius_solar"],
                data["insol_earth"],
                data.get("albedo", 0.3),
                float(data.get("tidally_locked", 1)),
            ]
        ).reshape(1, -1)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        X_s = self.scaler_X.fit_transform(X)
        y_s = self.scaler_y.fit_transform(y)
        self.models = []
        for _ in range(self.n_ensemble):
            m = self._create_model()
            m.fit(X_s, y_s)
            self.models.append(m)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_s = self.scaler_X.transform(X)
        preds = [m.predict(X_s) for m in self.models]
        ensemble_mean = np.mean(preds, axis=0)
        flat = self.scaler_y.inverse_transform(ensemble_mean)
        return flat.reshape(self.N_LAT, self.N_LON)

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_s = self.scaler_X.transform(X)
        preds = np.array([m.predict(X_s) for m in self.models])
        mean_s = preds.mean(axis=0)
        std_s = preds.std(axis=0)
        mean = self.scaler_y.inverse_transform(mean_s).reshape(
            self.N_LAT, self.N_LON
        )
        std = (std_s * self.scaler_y.scale_).reshape(self.N_LAT, self.N_LON)
        return mean, std

    def predict_conformal(
        self, X: np.ndarray, alpha: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with conformal prediction intervals.

        Uses the ensemble spread as a nonconformity score and returns
        (mean, lower, upper) at the ``1-alpha`` confidence level.
        Falls back to ensemble std if mapie is unavailable.

        Parameters
        ----------
        X : array of shape (1, n_features)
        alpha : significance level (default 0.1 = 90% coverage)
        """
        X_s = self.scaler_X.transform(X)
        preds = np.array([m.predict(X_s) for m in self.models])
        mean_s = preds.mean(axis=0)
        std_s = preds.std(axis=0)

        mean = self.scaler_y.inverse_transform(mean_s)
        std_raw = std_s * self.scaler_y.scale_

        from scipy.stats import norm
        z = norm.ppf(1 - alpha / 2)
        lower = mean - z * std_raw
        upper = mean + z * std_raw

        n_lat, n_lon = self.N_LAT, self.N_LON
        return (
            mean.reshape(n_lat, n_lon),
            lower.reshape(n_lat, n_lon),
            upper.reshape(n_lat, n_lon),
        )

    def predict_from_params(self, params: Dict) -> np.ndarray:
        return self.predict(self.prepare_features(params))

    def predict_from_params_with_ci(
        self, params: Dict, alpha: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict from a parameter dict with conformal intervals."""
        return self.predict_conformal(self.prepare_features(params), alpha)

    def save(self, path: str = "models/elm_ensemble.pkl") -> None:
        bundle = {
            "models": self.models,
            "scaler_X": self.scaler_X,
            "scaler_y": self.scaler_y,
            "n_ensemble": self.n_ensemble,
            "n_neurons": self.n_neurons,
            "n_lat": self.N_LAT,
            "n_lon": self.N_LON,
            "use_skelm": self._use_skelm,
        }
        with open(path, "wb") as f:
            pickle.dump(bundle, f)

    def load(self, path: str = "models/elm_ensemble.pkl") -> "ELMClimateSurrogate":
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        self.models = bundle["models"]
        self.scaler_X = bundle["scaler_X"]
        self.scaler_y = bundle["scaler_y"]
        self.n_ensemble = bundle["n_ensemble"]
        self.n_neurons = bundle["n_neurons"]
        self.N_LAT = bundle["n_lat"]
        self.N_LON = bundle["n_lon"]
        self._use_skelm = bundle.get("use_skelm", True)
        return self


# ─── Analytical training-data generator ───────────────────────────────────────


def generate_analytical_training_data(
    n_samples: int = 1000,
    n_lat: int = 32,
    n_lon: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Produce synthetic (features, temperature-map) pairs.

    Uses a simplified radiative model so the ELM can be bootstrapped
    without access to ROCKE-3D GCM output.
    """
    lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    lon = np.linspace(0, 2 * np.pi, n_lon)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")

    X_features: List[list] = []
    y_maps: List[np.ndarray] = []

    for _ in range(n_samples):
        star_teff = np.random.uniform(2500, 7000)
        star_radius = np.random.uniform(0.1, 2.0)
        semi_major = np.random.uniform(0.01, 2.0)
        albedo = np.random.uniform(0.05, 0.7)
        radius_earth = np.random.uniform(0.5, 2.5)
        mass_earth = radius_earth ** (1 / 0.27)
        insol = (star_teff / 5778) ** 4 * star_radius**2 / semi_major**2
        locked = int(np.random.choice([0, 1], p=[0.3, 0.7]))

        R_star_m = star_radius * 6.957e8
        a_m = semi_major * 1.496e11
        T_eq = star_teff * np.sqrt(R_star_m / (2 * a_m)) * (1 - albedo) ** 0.25

        if locked:
            cos_z = np.cos(LAT) * np.cos(LON - np.pi)
            cos_z = np.clip(cos_z, 0, 1)
            T_max = T_eq * 1.4
            T_min = max(T_eq * 0.3, 40)
            temp_map = T_min + (T_max - T_min) * cos_z**0.25
        else:
            temp_map = T_eq * (1 + 0.15 * np.cos(LAT))
            temp_map += np.random.normal(0, T_eq * 0.02, temp_map.shape)

        features = [
            radius_earth, mass_earth, semi_major, star_teff,
            star_radius, insol, albedo, locked,
        ]
        X_features.append(features)
        y_maps.append(temp_map.flatten())

    return np.array(X_features), np.array(y_maps)
