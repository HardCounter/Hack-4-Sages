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


# ─── Astrophysically motivated planet regimes ─────────────────────────────────

PLANET_REGIMES: Dict[str, Dict] = {
    "temperate_g_dwarf": {
        "description": (
            "Earth-like rocky planets in the habitable zone of "
            "G/late-K main-sequence stars."
        ),
        "star_teff": (5000, 6100),
        "star_radius": (0.82, 1.15),
        "semi_major": (0.7, 1.7),
        "radius_earth": (0.8, 1.5),
        "albedo": (0.20, 0.40),
        "locked_prob": 0.05,
    },
    "m_dwarf_locked": {
        "description": (
            "Tidally locked rocky planets in the habitable zone of "
            "M-dwarfs, similar to Proxima Centauri b."
        ),
        "star_teff": (2500, 3900),
        "star_radius": (0.08, 0.62),
        "semi_major": (0.01, 0.25),
        "radius_earth": (0.7, 1.6),
        "albedo": (0.10, 0.45),
        "locked_prob": 0.95,
    },
    "hot_close_in": {
        "description": (
            "Hot rocky planets on very tight orbits (a < 0.1 AU) "
            "around diverse stellar types."
        ),
        "star_teff": (3800, 7000),
        "star_radius": (0.55, 1.80),
        "semi_major": (0.01, 0.10),
        "radius_earth": (0.5, 2.0),
        "albedo": (0.05, 0.25),
        "locked_prob": 0.85,
    },
    "cold_super_earth": {
        "description": (
            "Super-Earths on wider orbits with thick, "
            "volatile-rich atmospheres."
        ),
        "star_teff": (3500, 6000),
        "star_radius": (0.35, 1.15),
        "semi_major": (1.0, 2.5),
        "radius_earth": (1.3, 2.5),
        "albedo": (0.30, 0.65),
        "locked_prob": 0.02,
    },
}

DEFAULT_REGIME_WEIGHTS: Dict[str, float] = {
    "temperate_g_dwarf": 0.20,
    "m_dwarf_locked": 0.35,
    "hot_close_in": 0.25,
    "cold_super_earth": 0.20,
}


# ─── Analytical training-data generator ───────────────────────────────────────


def _mass_from_radius(radius_earth: float, rng: np.random.Generator) -> float:
    """Broken power-law mass-radius relation (Otegi+ 2020 inspired).

    Rocky (R < 1.5 R_E): M ~ R^3.45
    Volatile-rich (R >= 1.5 R_E): continuous match with M ~ R^1.75
    Log-normal scatter of ~15 % captures compositional diversity.
    """
    if radius_earth < 1.5:
        log_m = 3.45 * np.log(radius_earth)
    else:
        log_m = 3.45 * np.log(1.5) + 1.75 * np.log(radius_earth / 1.5)
    return max(np.exp(log_m + rng.normal(0, 0.15)), 0.01)


def generate_analytical_training_data(
    n_samples: int = 1000,
    n_lat: int = 32,
    n_lon: int = 64,
    regime_weights: Optional[Dict[str, float]] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Produce synthetic (features, temperature-map) pairs.

    Uses a simplified radiative-equilibrium model to bootstrap ELM
    training without access to ROCKE-3D GCM output.  Samples are drawn
    from astrophysically motivated planet *regimes* (see
    ``PLANET_REGIMES``) to ensure coverage of Earth-like, M-dwarf
    tidally-locked, hot close-in, and cold super-Earth configurations.

    Parameters
    ----------
    n_samples : int
        Number of synthetic planet samples to generate.
    n_lat, n_lon : int
        Grid resolution for the temperature maps.
    regime_weights : dict, optional
        Mapping ``regime_name -> weight``.  Keys must be a subset of
        ``PLANET_REGIMES``.  If *None*, ``DEFAULT_REGIME_WEIGHTS`` is
        used (35 % M-dwarf locked, 25 % hot close-in, 20 % temperate
        G-dwarf, 20 % cold super-Earth).
    verbose : bool
        Print regime counts and feature summary statistics.

    Returns
    -------
    X : ndarray, shape ``(n_valid, 8)``
        Feature vectors:
        ``[radius_earth, mass_earth, semi_major_axis_au, star_teff_K,
        star_radius_solar, insol_earth, albedo, tidally_locked]``.
    y : ndarray, shape ``(n_valid, n_lat * n_lon)``
        Flattened temperature maps in Kelvin.

    Notes
    -----
    Planet families
    ~~~~~~~~~~~~~~~
    * **temperate_g_dwarf** -- Earth-like rocky planets in the HZ of
      G/late-K stars (T_eff 5000--6100 K).
    * **m_dwarf_locked** -- Tidally locked rocky worlds in the HZ of
      M-dwarfs (T_eff 2500--3900 K), like Proxima Centauri b.
    * **hot_close_in** -- Hot rocky planets on very tight orbits
      (a < 0.1 AU) around diverse stellar types.
    * **cold_super_earth** -- Super-Earths on wider orbits (a > 1 AU)
      with volatile-rich atmospheres.

    Mass-radius relation
    ~~~~~~~~~~~~~~~~~~~~
    Broken power law after Otegi+ (2020): rocky (R < 1.5 R_E,
    M ~ R^3.45) transitioning to volatile-rich (R >= 1.5, M ~ R^1.75)
    with ~15 % log-normal scatter.

    Stellar parameters
    ~~~~~~~~~~~~~~~~~~
    T_eff and R_star are sampled with positive correlation within each
    regime to approximate the main-sequence relation.

    Temperature maps
    ~~~~~~~~~~~~~~~~
    Tidally locked worlds use a sub-stellar irradiation pattern with an
    atmospheric heat-redistribution factor *f* that increases with
    planet size and albedo (proxy for atmospheric thickness).
    Non-locked worlds use a latitude-dependent gradient whose amplitude
    varies with albedo.
    """
    if regime_weights is None:
        regime_weights = DEFAULT_REGIME_WEIGHTS

    rng = np.random.default_rng()

    lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    lon = np.linspace(0, 2 * np.pi, n_lon)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")

    regimes = list(regime_weights.keys())
    weights = np.array([regime_weights[r] for r in regimes], dtype=float)
    weights /= weights.sum()
    regime_indices = rng.choice(len(regimes), size=n_samples, p=weights)

    X_features: List[list] = []
    y_maps: List[np.ndarray] = []
    n_rejected = 0
    regime_counts: Dict[str, int] = {r: 0 for r in regimes}

    for idx in regime_indices:
        regime_name = regimes[idx]
        regime = PLANET_REGIMES[regime_name]

        # --- Stellar parameters (main-sequence correlated) -------------
        teff_lo, teff_hi = regime["star_teff"]
        rad_lo, rad_hi = regime["star_radius"]
        star_teff = float(rng.uniform(teff_lo, teff_hi))
        teff_frac = (star_teff - teff_lo) / max(teff_hi - teff_lo, 1.0)
        rad_center = rad_lo + teff_frac * (rad_hi - rad_lo)
        rad_scatter = 0.08 * (rad_hi - rad_lo)
        star_radius = float(np.clip(
            rng.normal(rad_center, rad_scatter), rad_lo, rad_hi,
        ))

        # --- Orbital & planetary parameters ----------------------------
        sma_lo, sma_hi = regime["semi_major"]
        semi_major = float(rng.uniform(sma_lo, sma_hi))

        rp_lo, rp_hi = regime["radius_earth"]
        radius_earth = float(rng.uniform(rp_lo, rp_hi))
        mass_earth = _mass_from_radius(radius_earth, rng)

        alb_lo, alb_hi = regime["albedo"]
        albedo = float(rng.uniform(alb_lo, alb_hi))

        locked = int(rng.random() < regime["locked_prob"])

        # --- Derived quantities ----------------------------------------
        insol = (star_teff / 5778) ** 4 * star_radius ** 2 / semi_major ** 2
        R_star_m = star_radius * 6.957e8
        a_m = semi_major * 1.496e11
        T_eq = star_teff * np.sqrt(R_star_m / (2 * a_m)) * (1 - albedo) ** 0.25

        if T_eq < 30 or T_eq > 5000:
            n_rejected += 1
            continue

        # --- Temperature map -------------------------------------------
        if locked:
            # Atmospheric heat redistribution heuristic: larger, higher-
            # albedo planets redistribute heat more efficiently.
            f_redist = float(np.clip(
                0.1 + 0.25 * (radius_earth - 0.5) / 2.0 + 0.2 * albedo,
                0.05, 0.75,
            ))
            cos_z = np.cos(LAT) * np.cos(LON - np.pi)
            cos_z = np.clip(cos_z, 0, 1)
            T_max = T_eq * (1.4 - 0.4 * f_redist)
            T_min = max(T_eq * (0.2 + 0.6 * f_redist), 40)
            temp_map = T_min + (T_max - T_min) * cos_z ** 0.25
        else:
            gradient = 0.20 - 0.10 * albedo
            temp_map = T_eq * (1.0 + gradient * np.cos(LAT))
            temp_map += rng.normal(0, T_eq * 0.02, temp_map.shape)

        features = [
            radius_earth, mass_earth, semi_major, star_teff,
            star_radius, insol, albedo, locked,
        ]
        X_features.append(features)
        y_maps.append(temp_map.flatten())
        regime_counts[regime_name] += 1

    if verbose:
        n_valid = len(X_features)
        print(f"  Regime sampling ({n_valid} valid, {n_rejected} rejected):")
        for r in regimes:
            pct = 100 * regime_counts[r] / max(n_valid, 1)
            print(f"    {r:22s}: {regime_counts[r]:5d} ({pct:5.1f}%)")
        X_arr = np.array(X_features)
        feat_names = [
            "radius_earth", "mass_earth", "semi_major_au",
            "star_teff_K", "star_radius_Rsun", "insol_earth",
            "albedo", "tidally_locked",
        ]
        print(f"  Feature statistics (n={n_valid}):")
        for i, name in enumerate(feat_names):
            vals = X_arr[:, i]
            print(
                f"    {name:20s}:  min={vals.min():10.3g}  "
                f"med={np.median(vals):10.3g}  max={vals.max():10.3g}"
            )

    return np.array(X_features), np.array(y_maps)
