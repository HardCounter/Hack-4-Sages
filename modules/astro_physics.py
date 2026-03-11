"""
Astrophysics calculation engine.

Implements equilibrium temperature, stellar flux, ESI, SEPHI,
habitable-zone boundaries (Kopparapu 2013), habitable surface
fraction, semi-empirical albedo estimation, atmospheric escape
diagnostics, and a full analysis pipeline.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─── Physical constants ───────────────────────────────────────────────────────

STEFAN_BOLTZMANN = 5.670374419e-8   # σ  [W/(m²·K⁴)]
L_SUN = 3.828e26                    # Solar luminosity [W]
R_SUN = 6.957e8                     # Solar radius [m]
AU = 1.496e11                       # Astronomical unit [m]
S_EARTH = 1361.0                    # Solar constant at Earth [W/m²]
G_GRAV = 6.674e-11                  # Gravitational constant [m³/(kg·s²)]
M_EARTH_KG = 5.972e24              # Earth mass [kg]
R_EARTH_M = 6.371e6                # Earth radius [m]
M_PROTON = 1.673e-27               # Proton mass [kg]


# ─── Semi-empirical albedo estimation ─────────────────────────────────────────

_ALBEDO_TABLE: Dict[str, Dict[str, Tuple[float, float]]] = {
    "ocean":       {"thin": (0.06, 0.03), "temperate": (0.30, 0.06), "thick_cloudy": (0.50, 0.08)},
    "desert":      {"thin": (0.30, 0.05), "temperate": (0.35, 0.06), "thick_cloudy": (0.55, 0.08)},
    "ice":         {"thin": (0.60, 0.08), "temperate": (0.55, 0.07), "thick_cloudy": (0.65, 0.08)},
    "mixed_rocky": {"thin": (0.12, 0.04), "temperate": (0.30, 0.05), "thick_cloudy": (0.45, 0.07)},
}


def estimate_albedo(
    surface_type: str = "mixed_rocky",
    atmosphere_type: str = "temperate",
    user_override: Optional[float] = None,
) -> Dict[str, float]:
    """Semi-empirical Bond albedo from surface and atmosphere classes.

    Returns a dict with ``albedo`` (best estimate) and ``albedo_uncertainty``
    (1-sigma spread). Based on aggregate literature values from Kasting
    et al. (1993), Shields et al. (2013), and Kopparapu et al. (2013).

    If *user_override* is provided, it is used directly and the uncertainty
    is set to 0.
    """
    if user_override is not None:
        return {"albedo": float(user_override), "albedo_uncertainty": 0.0}
    surface_entry = _ALBEDO_TABLE.get(surface_type, _ALBEDO_TABLE["mixed_rocky"])
    best, sigma = surface_entry.get(atmosphere_type, surface_entry["temperate"])
    return {"albedo": round(best, 3), "albedo_uncertainty": round(sigma, 3)}


# ─── Redistribution factor ───────────────────────────────────────────────────


def redistribution_factor(
    tidally_locked: bool,
    optical_depth_class: str = "moderate",
) -> float:
    """Continuous redistribution factor f for T_eq computation.

    Instead of the binary sqrt(2) vs 2, uses a literature-motivated
    parameterization spanning tidally locked (hemisphere-only
    reemission) to fast rotator (uniform redistribution), modulated
    by a coarse atmospheric optical-depth proxy.

    Optical depth classes:
      - ``"thin"``:     f_locked ≈ 1.20,  f_fast ≈ 1.85  (weak redistribution)
      - ``"moderate"``: f_locked ≈ sqrt(2), f_fast ≈ 2.0  (standard, Kasting-like)
      - ``"thick"``:    f_locked ≈ 1.60,  f_fast ≈ 2.0   (strong circulation narrows gap)

    Motivation: Leconte et al. (2013), Pierrehumbert (2011) — thick atmospheres
    on tidally locked planets transport heat more efficiently, raising f toward
    the fast-rotator limit.
    """
    _TABLE = {
        "thin":     (1.20, 1.85),
        "moderate": (np.sqrt(2), 2.0),
        "thick":    (1.60, 2.0),
    }
    f_locked, f_fast = _TABLE.get(optical_depth_class, _TABLE["moderate"])
    return float(f_locked if tidally_locked else f_fast)


# ─── Orbit-averaged flux correction ──────────────────────────────────────────


def orbit_averaged_flux_factor(eccentricity: float) -> float:
    """Orbit-averaged correction to ⟨1/r²⟩ for eccentric orbits.

    ⟨F⟩ / F(a) = 1 / √(1 − e²)

    For circular orbits (e=0) this returns 1.0. Standard result from
    celestial mechanics; see Williams & Pollard (2002).
    """
    e = np.clip(eccentricity, 0.0, 0.99)
    return float(1.0 / np.sqrt(1.0 - e**2))


# ─── Core functions ───────────────────────────────────────────────────────────


def equilibrium_temperature(
    stellar_temp: float,
    stellar_radius: float,
    semi_major_axis: float,
    albedo: float = 0.3,
    tidally_locked: bool = False,
    eccentricity: float = 0.0,
    optical_depth_class: str = "moderate",
) -> float:
    """Radiative equilibrium temperature [K].

    T_eq = T_* × √(R_* / (f·a)) × (1 − A_B)^{1/4} × ⟨1/r²⟩^{1/4}

    Uses a continuous redistribution factor *f* (see ``redistribution_factor``)
    and an orbit-averaged flux correction for eccentric orbits
    (see ``orbit_averaged_flux_factor``).
    """
    R_star_m = stellar_radius * R_SUN
    a_m = max(semi_major_axis, 1e-6) * AU
    f = redistribution_factor(tidally_locked, optical_depth_class)
    ecc_corr = orbit_averaged_flux_factor(eccentricity) ** 0.25
    T_eq = (
        stellar_temp
        * np.sqrt(R_star_m / (f * a_m))
        * (1 - albedo) ** 0.25
        * ecc_corr
    )
    return round(float(T_eq), 2)


def stellar_flux(
    stellar_temp: float,
    stellar_radius: float,
    semi_major_axis: float,
    eccentricity: float = 0.0,
) -> Tuple[float, float]:
    """Orbit-averaged stellar flux on the planet.

    Returns (S_abs [W/m²], S_norm [S_⊕]).
    """
    R_star_m = stellar_radius * R_SUN
    a_m = max(semi_major_axis, 1e-6) * AU
    L_star = 4 * np.pi * R_star_m**2 * STEFAN_BOLTZMANN * stellar_temp**4
    S = L_star / (4 * np.pi * a_m**2)
    S *= orbit_averaged_flux_factor(eccentricity)
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
        if (x + x_ref) <= 0:
            continue
        similarity = np.clip(1.0 - abs(x - x_ref) / (x + x_ref), 0.0, 1.0)
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
    if star_teff < 2600 or star_teff > 7200:
        logger.warning(
            "T_eff=%.0fK outside Kopparapu et al. (2013) calibration "
            "range [2600, 7200]K — HZ boundaries are extrapolated.",
            star_teff,
        )
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


def estimate_outgassing_rate(
    mass_earth: float,
    radius_earth: float,
    age_gyr: float = 4.5,
) -> Dict[str, float]:
    """Estimate volcanic outgassing rate relative to Earth.

    Scaling: outgassing ~ (g/g_E)^0.75 × (age/4.5)^(-1.5) × H_radio

    Extends the Kite et al. (2009) parameterisation with a crude
    radiogenic heat budget proxy. Young planets (< 1 Gyr) receive a
    boost reflecting higher concentrations of short-lived radioisotopes
    (⁴⁰K, ²³²Th, ²³⁸U). Massive planets (> 2 M_E) also receive a
    gravity-dependent interior pressure correction per Stamenković
    et al. (2012).
    """
    if radius_earth <= 0:
        return {
            "outgassing_rate_earth": 0.0,
            "co2_flux_mol_yr": 0.0,
            "surface_gravity_ms2": 0.0,
            "mantle_convection_vigor": "low",
            "radiogenic_factor": 1.0,
        }
    age_gyr = max(age_gyr, 0.1)
    g_planet = G_EARTH * mass_earth / radius_earth ** 2
    g_ratio = g_planet / G_EARTH
    age_factor = (age_gyr / 4.5) ** (-1.5)

    radiogenic = 1.0
    if age_gyr < 1.0:
        radiogenic = 1.0 + 0.5 * (1.0 - age_gyr)
    if mass_earth > 2.0:
        radiogenic *= max(0.6, 1.0 - 0.05 * (mass_earth - 2.0))

    rate_relative = g_ratio ** 0.75 * age_factor * radiogenic

    co2_flux_earth = 6.0e12
    co2_flux = co2_flux_earth * rate_relative

    return {
        "outgassing_rate_earth": round(rate_relative, 3),
        "co2_flux_mol_yr": round(co2_flux, 2),
        "surface_gravity_ms2": round(g_planet, 2),
        "mantle_convection_vigor": "high" if g_ratio > 1.2 else (
            "moderate" if g_ratio > 0.6 else "low"
        ),
        "radiogenic_factor": round(radiogenic, 3),
    }


def _tectonic_plausibility(
    mass_earth: float, radius_earth: float, age_gyr: float
) -> str:
    """3-level tectonic plausibility index.

    Based on parameter ranges from Stamenković et al. (2012) and
    Driscoll & Barnes (2015). Returns ``"plausible"``, ``"uncertain"``,
    or ``"unlikely"`` rather than a binary flag.
    """
    if radius_earth > 2.5 or mass_earth < 0.1:
        return "unlikely"
    if mass_earth > 8.0:
        return "unlikely"
    if 0.5 <= mass_earth <= 5.0 and radius_earth <= 2.0 and age_gyr >= 0.5:
        return "plausible"
    return "uncertain"


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
    The plate tectonics assessment uses a 3-level plausibility index
    (Stamenković et al. 2012, Driscoll & Barnes 2015) rather than a
    simple boolean.
    """
    outgassing = estimate_outgassing_rate(mass_earth, radius_earth, age_gyr)
    tectonics = _tectonic_plausibility(mass_earth, radius_earth, age_gyr)

    carbonate_silicate_active = (
        tectonics == "plausible"
        and 200 <= T_eq <= 400
        and outgassing["outgassing_rate_earth"] > 0.3
    )

    water_cycling = 273 <= T_eq <= 373
    volatile_retention = estimate_escape_velocity(mass_earth, radius_earth) >= 5.0

    score_components = [
        1.0 if tectonics == "plausible" else (0.5 if tectonics == "uncertain" else 0.0),
        1.0 if carbonate_silicate_active else 0.0,
        1.0 if water_cycling else 0.0,
        1.0 if volatile_retention else 0.0,
    ]
    isa_score = sum(score_components) / len(score_components)

    return {
        "outgassing": outgassing,
        "plate_tectonics": tectonics,
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

_UV_BOLOMETRIC_TABLE = {
    2800: 0.001,
    3300: 0.005,
    3800: 0.015,
    4500: 0.04,
    5200: 0.06,
    5778: 0.08,
    6500: 0.12,
    7500: 0.18,
    10000: 0.25,
}


def _interpolate_uv_fraction(star_teff: float) -> float:
    """UV/bolometric flux ratio interpolated from tabulated spectral-type data.

    Based on representative UV/bolometric ratios for F-/G-/K-/M-type stars
    compiled from France et al. (2013) and Lammer et al. (2009) spectral
    measurements. Replaces the previous linear UV fraction parameterization.
    """
    temps = sorted(_UV_BOLOMETRIC_TABLE.keys())
    if star_teff <= temps[0]:
        return _UV_BOLOMETRIC_TABLE[temps[0]]
    if star_teff >= temps[-1]:
        return _UV_BOLOMETRIC_TABLE[temps[-1]]
    for i in range(len(temps) - 1):
        if temps[i] <= star_teff <= temps[i + 1]:
            t0, t1 = temps[i], temps[i + 1]
            f0 = _UV_BOLOMETRIC_TABLE[t0]
            f1 = _UV_BOLOMETRIC_TABLE[t1]
            frac = (star_teff - t0) / (t1 - t0)
            return f0 + frac * (f1 - f0)
    return 0.08


def estimate_uv_flux(
    star_teff: float,
    star_radius: float,
    semi_major_axis: float,
) -> Dict[str, float]:
    """Estimate UV radiation environment at the planet's orbit.

    Uses tabulated UV/bolometric ratios for representative spectral types
    (France et al. 2013, Lammer et al. 2009) instead of a single linear
    parametric fit. Critical for assessing abiotic photolysis pathways.
    """
    R_star_m = star_radius * R_SUN
    a_m = max(semi_major_axis, 1e-6) * AU

    L_bol = 4 * np.pi * R_star_m**2 * STEFAN_BOLTZMANN * star_teff**4
    uv_fraction = _interpolate_uv_fraction(star_teff)
    L_uv = L_bol * uv_fraction
    F_uv = L_uv / (4 * np.pi * a_m**2)
    F_uv_earth = S_EARTH * 0.08

    return {
        "uv_flux_Wm2": round(float(F_uv), 2),
        "uv_flux_earth": round(float(F_uv / F_uv_earth), 3),
        "uv_fraction_used": round(uv_fraction, 4),
        "uv_hazard": (
            "extreme" if F_uv / F_uv_earth > 5 else
            "high" if F_uv / F_uv_earth > 2 else
            "moderate" if F_uv / F_uv_earth > 0.5 else
            "low"
        ),
    }


# ─── Atmospheric escape diagnostic ───────────────────────────────────────────


def estimate_atmospheric_escape(
    mass_earth: float,
    radius_earth: float,
    star_teff: float,
    semi_major_axis: float,
    star_radius: float,
    age_gyr: float = 4.5,
) -> Dict[str, object]:
    """Energy-limited atmospheric escape time-scale estimate.

    Implements the energy-limited escape formalism (Watson et al. 1981,
    Erkaev et al. 2007):

        dM/dt = (epsilon * pi * F_XUV * R_p * R_XUV^2) / (G * M_p * K_tide)

    where epsilon is a heating efficiency (~0.15, Lammer et al. 2009),
    F_XUV is the XUV flux, and K_tide is a tidal correction factor
    (set to 1 for simplicity).

    Returns a categorical flag: ``"retained"``, ``"borderline"``,
    or ``"escape_dominated"`` based on the ratio of escape time-scale
    to system age.
    """
    if mass_earth <= 0 or radius_earth <= 0:
        return {"escape_flag": "unknown", "escape_timescale_gyr": None, "mass_loss_rate_kg_s": 0.0}

    uv_data = estimate_uv_flux(star_teff, star_radius, semi_major_axis)
    F_xuv = uv_data["uv_flux_Wm2"] * 0.001

    R_p = radius_earth * R_EARTH_M
    M_p = mass_earth * M_EARTH_KG
    epsilon = 0.15
    R_xuv = R_p * 1.2

    dM_dt = (epsilon * np.pi * F_xuv * R_p * R_xuv**2) / (G_GRAV * M_p)
    dM_dt = max(dM_dt, 1e-20)

    H_envelope_kg = 0.05 * M_p
    tau_s = H_envelope_kg / dM_dt
    tau_gyr = tau_s / (3.156e16)

    if tau_gyr > 10 * age_gyr:
        flag = "retained"
    elif tau_gyr > age_gyr:
        flag = "borderline"
    else:
        flag = "escape_dominated"

    return {
        "escape_flag": flag,
        "escape_timescale_gyr": round(float(tau_gyr), 2) if np.isfinite(tau_gyr) else None,
        "mass_loss_rate_kg_s": round(float(dM_dt), 2),
        "xuv_flux_Wm2": round(float(F_xuv), 2),
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


# ─── Fulton Gap / Radius Valley ───────────────────────────────────────────────

def classify_radius_gap(radius_earth: float) -> Dict[str, object]:
    """Classify planet into radius valley regimes (Fulton et al. 2017).

    The scarcity of planets between 1.5–2.0 R⊕ separates rocky
    super-Earths (atmosphere stripped by photoevaporation) from
    gas-envelope-retaining sub-Neptunes.
    """
    GAP_LOW = 1.5
    GAP_HIGH = 2.0
    GAP_MID = (GAP_LOW + GAP_HIGH) / 2.0

    if radius_earth < GAP_LOW:
        classification = "rocky_super_earth"
        label = "Rocky Super-Earth"
        atmosphere_retention = "likely_lost"
    elif radius_earth <= GAP_HIGH:
        classification = "radius_gap"
        label = "Radius Gap (unstable)"
        atmosphere_retention = "uncertain"
    elif radius_earth <= 3.5:
        classification = "sub_neptune"
        label = "Sub-Neptune"
        atmosphere_retention = "likely_retained"
    else:
        classification = "giant"
        label = "Gas Giant"
        atmosphere_retention = "retained"

    gap_proximity = max(0.0, 1.0 - abs(radius_earth - GAP_MID) / (GAP_MID - GAP_LOW))

    return {
        "classification": classification,
        "label": label,
        "atmosphere_retention": atmosphere_retention,
        "gap_proximity": round(float(gap_proximity), 3),
    }


# ─── Sulfur Chemistry ─────────────────────────────────────────────────────────

def assess_sulfur_chemistry(
    t_eq: float,
    surface_pressure_bar: float,
    atmosphere_type: str = "h2_rich",
) -> Dict[str, object]:
    """Predict sulfur speciation from T_eq, pressure, and atmosphere type.

    Three atmosphere regimes: 'h2_rich', 'o2_rich', 'ch4_co2'.
    Based on Zahnle et al. (2016) sulfur photochemistry constraints.
    """
    h2so4_condensation = t_eq > 400 and surface_pressure_bar > 10
    h2s_condensation = t_eq < 300 and surface_pressure_bar < 1

    if h2so4_condensation:
        dominant_gas = "H2SO4"
        cloud_condensates = ["H2SO4"]
        regime = "oxidising"
    elif t_eq > 300 and surface_pressure_bar > 1:
        dominant_gas = "SO2"
        cloud_condensates = ["SO2"]
        regime = "mixed"
    else:
        dominant_gas = "H2S"
        cloud_condensates = ["H2S"] if h2s_condensation else []
        regime = "reducing"

    mineral_map = {
        "h2_rich":  ["FeS", "FeS2"],
        "o2_rich":  ["CaSO4", "FeSO4"],
        "ch4_co2":  ["FeS", "CaSO4"],
    }
    surface_minerals = mineral_map.get(atmosphere_type, ["FeS"])

    notes_parts = []
    if h2so4_condensation:
        notes_parts.append("Venus-like H2SO4 cloud deck expected.")
    if h2s_condensation:
        notes_parts.append("H2S condensation at surface level.")
    if not notes_parts:
        notes_parts.append("Moderate sulfur activity, SO2 dominated.")

    return {
        "dominant_gas": dominant_gas,
        "cloud_condensates": cloud_condensates,
        "surface_minerals": surface_minerals,
        "regime": regime,
        "h2s_condensation": h2s_condensation,
        "h2so4_condensation": h2so4_condensation,
        "atmosphere_type": atmosphere_type,
        "notes": " ".join(notes_parts),
    }


# ─── Carbon-to-Oxygen Ratio ───────────────────────────────────────────────────

def assess_co_ratio(co_ratio: float) -> Dict[str, object]:
    """Classify planetary composition from the C/O ratio.

    Solar C/O ≈ 0.55. Earth C/O ≈ 0.50.
    Above 0.8 → carbon planet (graphite/carbide surface, no oceans).
    """
    if co_ratio < 0.55:
        classification = "water_world_candidate"
        label = "Water World Candidate"
        ocean_likelihood = "high"
        dominant_surface = "silicates_water"
        atmosphere_bias = "CO2_H2O"
        habitability_modifier = 0.15
    elif co_ratio <= 0.80:
        classification = "transitional"
        label = "Solar-like Composition"
        ocean_likelihood = "moderate"
        dominant_surface = "mixed"
        atmosphere_bias = "mixed"
        habitability_modifier = 0.0
    else:
        classification = "carbon_planet"
        label = "Carbon Planet"
        ocean_likelihood = "low"
        dominant_surface = "graphite_carbides"
        atmosphere_bias = "CH4_dominated"
        habitability_modifier = -0.4

    return {
        "classification": classification,
        "label": label,
        "ocean_likelihood": ocean_likelihood,
        "dominant_surface": dominant_surface,
        "atmosphere_bias": atmosphere_bias,
        "habitability_modifier": round(habitability_modifier, 3),
        "co_ratio": round(co_ratio, 3),
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
    eccentricity: float = 0.0,
) -> Dict[str, object]:
    """End-to-end analysis from raw NASA parameters to results dict."""
    R_earth = planet_radius_jup * 11.209
    M_earth = planet_mass_jup * 317.83

    T_eq = equilibrium_temperature(
        stellar_temp, stellar_radius, semi_major_axis, albedo,
        tidally_locked, eccentricity,
    )
    S_abs, S_norm = stellar_flux(
        stellar_temp, stellar_radius, semi_major_axis, eccentricity,
    )

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
    escape = estimate_atmospheric_escape(
        M_earth, R_earth, stellar_temp, semi_major_axis, stellar_radius,
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
        "eccentricity": eccentricity,
        "isa_interaction": isa,
        "biosignature_false_positives": false_pos,
        "atmospheric_escape": escape,
    }
