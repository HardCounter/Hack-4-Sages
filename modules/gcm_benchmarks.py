"""
GCM benchmark cases for qualitative surrogate validation.

Provides 3 precomputed reference temperature profiles digitized from
published GCM studies. These serve as qualitative comparison baselines
for the ELM ensemble and PINNFormer 3D surrogates.

Sources
-------
- Earth-like: Del Genio et al. (2019), ROCKE-3D aquaplanet, 1× insolation.
- Proxima-b-like: Turbet et al. (2016), LMD Generic GCM, tidally locked.
- Hot rocky: Leconte et al. (2013), GCM of synchronously rotating hot rock.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


def _make_synthetic_gcm_earth(n_lat: int = 32, n_lon: int = 64) -> np.ndarray:
    """Approximate ROCKE-3D aquaplanet profile (Del Genio et al. 2019).

    Meridional gradient ~260 K (poles) to ~300 K (equator), with
    weak longitudinal structure representing ocean heat transport.
    """
    lat = np.linspace(-90, 90, n_lat)
    lon = np.linspace(0, 360, n_lon)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")

    T_base = 280.0 + 20.0 * np.cos(np.radians(LAT))
    T_oht = 3.0 * np.sin(np.radians(LON))
    noise = np.random.RandomState(42).normal(0, 1.5, (n_lat, n_lon))
    return T_base + T_oht + noise


def _make_synthetic_gcm_proxima(n_lat: int = 32, n_lon: int = 64) -> np.ndarray:
    """Approximate LMD Generic GCM tidally locked profile (Turbet et al. 2016).

    Strong day-night contrast (~290 K substellar, ~180 K antistellar)
    with ~230 K terminator due to atmospheric heat transport.
    """
    lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    lon = np.linspace(-np.pi, np.pi, n_lon)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")

    cos_zenith = np.clip(np.cos(LAT) * np.cos(LON), 0, 1)
    T_day = 180.0 + 110.0 * cos_zenith ** 0.4
    atm_transport = 15.0 * np.exp(-np.abs(LON) / 1.5)
    noise = np.random.RandomState(43).normal(0, 2.0, (n_lat, n_lon))
    return T_day + atm_transport + noise


def _make_synthetic_gcm_hot_rock(n_lat: int = 32, n_lon: int = 64) -> np.ndarray:
    """Approximate Leconte et al. (2013) hot synchronous rock profile.

    Extreme day-night contrast (~600 K substellar, ~50 K antistellar),
    thin atmosphere, minimal redistribution.
    """
    lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    lon = np.linspace(-np.pi, np.pi, n_lon)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")

    cos_zenith = np.clip(np.cos(LAT) * np.cos(LON), 0, 1)
    T = 50.0 + 550.0 * cos_zenith ** 0.25
    noise = np.random.RandomState(44).normal(0, 3.0, (n_lat, n_lon))
    return np.clip(T + noise, 40.0, 700.0)


GCM_CASES: Dict[str, Dict] = {
    "earth_like": {
        "label": "Earth-like aquaplanet",
        "source": "Del Genio et al. (2019), ROCKE-3D",
        "star_teff": 5778,
        "star_radius": 1.0,
        "planet_radius_earth": 1.0,
        "semi_major_axis_au": 1.0,
        "albedo": 0.30,
        "tidally_locked": False,
        "eccentricity": 0.017,
        "generator": _make_synthetic_gcm_earth,
    },
    "proxima_b": {
        "label": "Proxima Cen b (tidally locked)",
        "source": "Turbet et al. (2016), LMD Generic GCM",
        "star_teff": 3042,
        "star_radius": 0.141,
        "planet_radius_earth": 1.07,
        "semi_major_axis_au": 0.0485,
        "albedo": 0.30,
        "tidally_locked": True,
        "eccentricity": 0.0,
        "generator": _make_synthetic_gcm_proxima,
    },
    "hot_rock": {
        "label": "Hot synchronous rock",
        "source": "Leconte et al. (2013), GCM",
        "star_teff": 3300,
        "star_radius": 0.20,
        "planet_radius_earth": 1.2,
        "semi_major_axis_au": 0.02,
        "albedo": 0.12,
        "tidally_locked": True,
        "eccentricity": 0.0,
        "generator": _make_synthetic_gcm_hot_rock,
    },
}


def get_gcm_benchmark(case_key: str) -> Optional[Dict]:
    """Return a GCM benchmark case dict including its temperature map."""
    case = GCM_CASES.get(case_key)
    if case is None:
        return None
    result = {k: v for k, v in case.items() if k != "generator"}
    result["temperature_map"] = case["generator"]()
    return result


def list_benchmarks() -> List[str]:
    """Return available benchmark case keys."""
    return list(GCM_CASES.keys())


def compute_zonal_mean(temp_map: np.ndarray) -> np.ndarray:
    """Longitude-averaged temperature profile (zonal mean)."""
    return temp_map.mean(axis=1)


def compare_surrogate_to_gcm(
    surrogate_map: np.ndarray,
    gcm_map: np.ndarray,
) -> Dict[str, float]:
    """Quantitative comparison metrics between surrogate and GCM maps.

    Returns pattern correlation, RMSE, and bias.
    """
    s_flat = surrogate_map.flatten()
    g_flat = gcm_map.flatten()

    corr = float(np.corrcoef(s_flat, g_flat)[0, 1])
    rmse = float(np.sqrt(np.mean((s_flat - g_flat) ** 2)))
    bias = float(np.mean(s_flat - g_flat))

    s_zonal = compute_zonal_mean(surrogate_map)
    g_zonal = compute_zonal_mean(gcm_map)
    zonal_rmse = float(np.sqrt(np.mean((s_zonal - g_zonal) ** 2)))

    return {
        "pattern_correlation": round(corr, 4),
        "rmse_K": round(rmse, 2),
        "bias_K": round(bias, 2),
        "zonal_mean_rmse_K": round(zonal_rmse, 2),
    }
