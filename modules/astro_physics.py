"""
Astrophysics calculation engine.

Implements equilibrium temperature, stellar flux, ESI, SEPHI,
habitable-zone boundaries (Kopparapu 2013), habitable surface
fraction, and a full analysis pipeline.
"""

from typing import Dict, Tuple

import numpy as np

# ─── Physical constants ───────────────────────────────────────────────────────

STEFAN_BOLTZMANN = 5.670374419e-8   # σ  [W/(m²·K⁴)]
L_SUN = 3.828e26                    # Solar luminosity [W]
R_SUN = 6.957e8                     # Solar radius [m]
AU = 1.496e11                       # Astronomical unit [m]
S_EARTH = 1361.0                    # Solar constant at Earth [W/m²]

# ─── Core functions ───────────────────────────────────────────────────────────


def equilibrium_temperature(
    stellar_temp: float,
    stellar_radius: float,
    semi_major_axis: float,
    albedo: float = 0.3,
    tidally_locked: bool = False,
) -> float:
    """Radiative equilibrium temperature [K].

    T_eq = T_* × √(R_* / (f·a)) × (1 − A_B)^{1/4}

    where *f* = √2 for a tidally locked planet (re-emission from one
    hemisphere) and 2 for a fast rotator.
    """
    R_star_m = stellar_radius * R_SUN
    a_m = semi_major_axis * AU
    redistribution = np.sqrt(2) if tidally_locked else 2.0
    T_eq = (
        stellar_temp
        * np.sqrt(R_star_m / (redistribution * a_m))
        * (1 - albedo) ** 0.25
    )
    return round(float(T_eq), 2)


def stellar_flux(
    stellar_temp: float,
    stellar_radius: float,
    semi_major_axis: float,
) -> Tuple[float, float]:
    """Stellar flux on the planet orbit.

    Returns (S_abs [W/m²], S_norm [S_⊕]).
    """
    R_star_m = stellar_radius * R_SUN
    a_m = semi_major_axis * AU
    L_star = 4 * np.pi * R_star_m**2 * STEFAN_BOLTZMANN * stellar_temp**4
    S = L_star / (4 * np.pi * a_m**2)
    S_norm = S / S_EARTH
    return round(float(S), 2), round(float(S_norm), 4)


# ─── Habitability indices ─────────────────────────────────────────────────────

def compute_esi(
    radius: float,
    density: float,
    escape_vel: float,
    surface_temp: float,
) -> float:
    """Earth Similarity Index (Schulze-Makuch et al. 2011).

    ESI ∈ [0, 1] where 1 = identical to Earth.
    """
    ref = {"radius": 1.0, "density": 5.51, "escape_vel": 11.19, "temperature": 288.0}
    weights = {"radius": 0.57, "density": 1.07, "escape_vel": 0.70, "temperature": 5.58}
    values = {
        "radius": radius,
        "density": density,
        "escape_vel": escape_vel,
        "temperature": surface_temp,
    }
    n = len(values)
    esi = 1.0
    for key in values:
        x, x_ref, w = values[key], ref[key], weights[key]
        if (x + x_ref) == 0:
            continue
        similarity = 1.0 - abs(x - x_ref) / (x + x_ref)
        esi *= similarity ** (w / n)
    return round(float(esi), 4)


def estimate_escape_velocity(mass_earth: float, radius_earth: float) -> float:
    """v_e ≈ 11.19 × √(M/R) [km/s] in Earth units."""
    if radius_earth <= 0:
        return 0.0
    return 11.19 * np.sqrt(mass_earth / radius_earth)


def estimate_density(mass_earth: float, radius_earth: float) -> float:
    """ρ ≈ 5.51 × M/R³ [g/cm³] in Earth units."""
    if radius_earth <= 0:
        return 0.0
    return 5.51 * mass_earth / radius_earth**3


# ─── SEPHI (Rodríguez-Mozos & Moya 2017) ─────────────────────────────────────

def compute_sephi(
    surface_temp: float,
    mass_earth: float,
    radius_earth: float,
) -> Dict[str, object]:
    """Surface Exoplanetary Planetary Habitability Index.

    Returns a dict with boolean criteria and an overall SEPHI score.
    """
    v_esc = estimate_escape_velocity(mass_earth, radius_earth)

    # Thermal criterion: liquid water range
    thermal_ok = 273.0 <= surface_temp <= 373.0

    # Atmospheric criterion: escape velocity must exceed thermal
    # velocity of key atmospheric gases (N2, O2, CO2). Rough
    # threshold ~5 km/s keeps a substantial atmosphere over Gyr.
    atmosphere_ok = v_esc >= 5.0

    # Magnetic criterion (heuristic): planets with mass ≥ 0.5 M⊕
    # and radius < 2.5 R⊕ are likely to sustain a dynamo.
    magnetic_ok = mass_earth >= 0.5 and radius_earth <= 2.5

    criteria_met = sum([thermal_ok, atmosphere_ok, magnetic_ok])
    score = criteria_met / 3.0

    return {
        "thermal_ok": thermal_ok,
        "atmosphere_ok": atmosphere_ok,
        "magnetic_ok": magnetic_ok,
        "criteria_met": criteria_met,
        "sephi_score": round(score, 2),
    }


# ─── Habitable-zone boundaries (Kopparapu et al. 2013) ───────────────────────

_HZ_COEFFS = {
    "recent_venus":   (1.7763, 1.4335e-4, 3.3954e-9, -7.6364e-12, -1.1950e-15),
    "runaway_gh":     (1.0385, 1.2456e-4, 1.4612e-8, -7.6345e-12, -1.7511e-15),
    "max_gh":         (0.3507, 5.9578e-5, 1.6707e-9, -3.0058e-12, -5.1925e-16),
    "early_mars":     (0.3207, 5.4471e-5, 1.5275e-9, -2.1709e-12, -3.8282e-16),
}


def hz_boundaries(
    star_teff: float,
    star_luminosity_solar: float = 1.0,
) -> Dict[str, float]:
    """Habitable zone inner/outer boundaries in AU.

    Uses the parameterisation of Kopparapu et al. (2013).
    *star_luminosity_solar* is L/L☉ (linear, not log).
    """
    T = star_teff - 5780.0
    result: Dict[str, float] = {}
    for label, (s0, a, b, c, d) in _HZ_COEFFS.items():
        S_eff = s0 + a * T + b * T**2 + c * T**3 + d * T**4
        if S_eff > 0:
            dist_au = np.sqrt(star_luminosity_solar / S_eff)
        else:
            dist_au = float("nan")
        result[label] = round(float(dist_au), 4)
    return result


# ─── Habitable surface fraction ───────────────────────────────────────────────

def habitable_surface_fraction(
    temp_map: np.ndarray,
    t_min: float = 273.0,
    t_max: float = 373.0,
) -> float:
    """Fraction of planet surface with temperature in [t_min, t_max].

    Applies cosine-latitude weighting for correct area integration
    on a sphere.
    """
    n_lat, n_lon = temp_map.shape
    lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    weights = np.cos(lat).reshape(-1, 1) * np.ones((1, n_lon))
    mask = (temp_map >= t_min) & (temp_map <= t_max)
    return float(np.sum(mask * weights) / np.sum(weights))


# ─── Full analysis pipeline ──────────────────────────────────────────────────

def compute_full_analysis(
    stellar_temp: float,
    stellar_radius: float,
    planet_radius_jup: float,
    planet_mass_jup: float,
    semi_major_axis: float,
    albedo: float = 0.3,
    tidally_locked: bool = True,
) -> Dict[str, object]:
    """End-to-end analysis from raw NASA parameters to results dict."""
    R_earth = planet_radius_jup * 11.209
    M_earth = planet_mass_jup * 317.83

    T_eq = equilibrium_temperature(
        stellar_temp, stellar_radius, semi_major_axis, albedo, tidally_locked
    )
    S_abs, S_norm = stellar_flux(stellar_temp, stellar_radius, semi_major_axis)

    density = estimate_density(M_earth, R_earth)
    v_escape = estimate_escape_velocity(M_earth, R_earth)
    esi = compute_esi(R_earth, density, v_escape, T_eq)
    sephi = compute_sephi(T_eq, M_earth, R_earth)

    in_hz = 0.2 <= S_norm <= 2.0
    has_liquid_water = 200 <= T_eq <= 380

    return {
        "T_eq_K": T_eq,
        "flux_Wm2": S_abs,
        "flux_earth": S_norm,
        "radius_earth": round(R_earth, 3),
        "mass_earth": round(M_earth, 3),
        "density_gcc": round(density, 3),
        "escape_vel_kms": round(v_escape, 3),
        "ESI": esi,
        "SEPHI": sephi,
        "in_habitable_zone": in_hz,
        "liquid_water_possible": has_liquid_water,
        "tidally_locked": tidally_locked,
    }
