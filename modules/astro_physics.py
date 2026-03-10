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


# ─── Interior-Surface-Atmosphere (ISA) interactions ──────────────────────────

G_EARTH = 9.81   # m/s²
R_EARTH = 6.371e6  # m


def estimate_outgassing_rate(
    mass_earth: float,
    radius_earth: float,
    age_gyr: float = 4.5,
) -> Dict[str, float]:
    """Estimate volcanic outgassing rate relative to Earth.

    Uses a heuristic scaling: outgassing ~ (M/R²)^α × (age/4.5)^(-β),
    reflecting the mantle convection vigor (proportional to surface
    gravity) and radiogenic heat decay over time. Based on simplified
    Kite et al. (2009) parameterisation.
    """
    g_planet = G_EARTH * mass_earth / radius_earth ** 2
    g_ratio = g_planet / G_EARTH
    age_factor = (age_gyr / 4.5) ** (-1.5)
    rate_relative = g_ratio ** 0.75 * age_factor

    co2_flux_earth = 6.0e12  # mol/yr (present-day Earth)
    co2_flux = co2_flux_earth * rate_relative

    return {
        "outgassing_rate_earth": round(rate_relative, 3),
        "co2_flux_mol_yr": round(co2_flux, 2),
        "surface_gravity_ms2": round(g_planet, 2),
        "mantle_convection_vigor": "high" if g_ratio > 1.2 else (
            "moderate" if g_ratio > 0.6 else "low"
        ),
    }


def estimate_isa_interaction(
    mass_earth: float,
    radius_earth: float,
    T_eq: float,
    tidally_locked: bool = True,
    age_gyr: float = 4.5,
) -> Dict[str, object]:
    """Full Interior-Surface-Atmosphere interaction assessment.

    Models the coupling between geological activity, surface
    chemistry, and atmospheric composition — a key criterion for
    realistic habitability assessment (Steinmeyer et al.).
    """
    outgassing = estimate_outgassing_rate(mass_earth, radius_earth, age_gyr)

    has_plate_tectonics = (
        mass_earth >= 0.5 and mass_earth <= 5.0
        and radius_earth <= 2.0
    )

    carbonate_silicate_active = (
        has_plate_tectonics
        and 200 <= T_eq <= 400
        and outgassing["outgassing_rate_earth"] > 0.3
    )

    water_cycling = 273 <= T_eq <= 373
    volatile_retention = estimate_escape_velocity(mass_earth, radius_earth) >= 5.0

    isa_score = sum([
        has_plate_tectonics,
        carbonate_silicate_active,
        water_cycling,
        volatile_retention,
    ]) / 4.0

    return {
        "outgassing": outgassing,
        "plate_tectonics_likely": has_plate_tectonics,
        "carbonate_silicate_cycle": carbonate_silicate_active,
        "water_cycling": water_cycling,
        "volatile_retention": volatile_retention,
        "isa_score": round(isa_score, 2),
        "isa_assessment": (
            "Strong ISA coupling" if isa_score >= 0.75 else
            "Moderate ISA coupling" if isa_score >= 0.5 else
            "Weak ISA coupling"
        ),
    }


# ─── Photochemical false-positive mitigation ─────────────────────────────────

def estimate_uv_flux(
    star_teff: float,
    star_radius: float,
    semi_major_axis: float,
) -> Dict[str, float]:
    """Estimate UV radiation environment at the planet's orbit.

    Hotter stars emit proportionally more UV. This is critical for
    assessing whether biosignature-like gases (O2, O3, CH4) could be
    produced abiotically via photolysis.
    """
    R_star_m = star_radius * R_SUN
    a_m = semi_major_axis * AU

    L_bol = 4 * np.pi * R_star_m**2 * STEFAN_BOLTZMANN * star_teff**4
    t_norm = np.clip((star_teff - 2500) / (10000 - 2500), 0.0, 1.0)
    uv_fraction = 0.005 + 0.20 * t_norm**1.3
    L_uv = L_bol * uv_fraction
    F_uv = L_uv / (4 * np.pi * a_m**2)
    F_uv_earth = S_EARTH * 0.08  # ~8% of solar flux is UV

    return {
        "uv_flux_Wm2": round(float(F_uv), 2),
        "uv_flux_earth": round(float(F_uv / F_uv_earth), 3),
        "uv_hazard": (
            "extreme" if F_uv / F_uv_earth > 5 else
            "high" if F_uv / F_uv_earth > 2 else
            "moderate" if F_uv / F_uv_earth > 0.5 else
            "low"
        ),
    }


def assess_biosignature_false_positives(
    star_teff: float,
    star_radius: float,
    semi_major_axis: float,
    T_eq: float,
    mass_earth: float,
    radius_earth: float,
) -> Dict[str, object]:
    """Assess the risk of photochemical false positives.

    Evaluates whether abiotic processes (UV photolysis, outgassing)
    could mimic biological signatures — a critical step before
    claiming habitability (Petkowski et al.).
    """
    uv = estimate_uv_flux(star_teff, star_radius, semi_major_axis)
    outgassing = estimate_outgassing_rate(mass_earth, radius_earth)

    o2_false_positive_risk = (
        "high" if (uv["uv_flux_earth"] > 3 and T_eq < 250) else
        "moderate" if uv["uv_flux_earth"] > 1.5 else
        "low"
    )

    ch4_false_positive_risk = (
        "high" if outgassing["outgassing_rate_earth"] > 2.0 else
        "moderate" if outgassing["outgassing_rate_earth"] > 0.8 else
        "low"
    )

    o3_abiotic = uv["uv_hazard"] in ("extreme", "high") and T_eq < 280

    risk_flags = []
    if o2_false_positive_risk == "high":
        risk_flags.append("O2 via H2O photolysis (high UV, cold trap)")
    if ch4_false_positive_risk == "high":
        risk_flags.append("CH4 via volcanic outgassing")
    if o3_abiotic:
        risk_flags.append("O3 via abiotic O2 photochemistry")

    overall = (
        "high" if len(risk_flags) >= 2 else
        "moderate" if len(risk_flags) == 1 else
        "low"
    )

    return {
        "uv_environment": uv,
        "o2_false_positive_risk": o2_false_positive_risk,
        "ch4_false_positive_risk": ch4_false_positive_risk,
        "o3_abiotic_likely": o3_abiotic,
        "risk_flags": risk_flags,
        "overall_false_positive_risk": overall,
        "recommendation": (
            "Biosignature claims require careful photochemical modeling"
            if overall != "low" else
            "Low false-positive risk — biosignatures more likely genuine"
        ),
    }


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

    isa = estimate_isa_interaction(M_earth, R_earth, T_eq, tidally_locked)
    false_pos = assess_biosignature_false_positives(
        stellar_temp, stellar_radius, semi_major_axis, T_eq, M_earth, R_earth
    )

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
        "isa_interaction": isa,
        "biosignature_false_positives": false_pos,
    }
