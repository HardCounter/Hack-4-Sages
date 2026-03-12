"""
NASA Exoplanet Archive TAP client.

Provides functions to query the NASA Exoplanet Archive via the
Table Access Protocol (TAP) using ADQL (Astronomical Data Query Language).
"""

import io
from typing import Optional

import pandas as pd
import requests

NASA_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# ─── Unit conversion helpers ───────────────────────────────────────────────────

R_JUPITER_TO_EARTH = 11.209
M_JUPITER_TO_EARTH = 317.83
R_SUN_M = 6.957e8       # Solar radius [m]
L_SUN_W = 3.828e26      # Solar luminosity [W]
AU_M = 1.496e11          # Astronomical unit [m]


def jupiter_to_earth_radius(r_jup: float) -> float:
    return r_jup * R_JUPITER_TO_EARTH


def jupiter_to_earth_mass(m_jup: float) -> float:
    return m_jup * M_JUPITER_TO_EARTH


def log_solar_lum_to_watts(log_lum: float) -> float:
    return 10**log_lum * L_SUN_W


def solar_to_meters_radius(r_solar: float) -> float:
    return r_solar * R_SUN_M


def au_to_meters(a_au: float) -> float:
    return a_au * AU_M


# ─── Core query function ──────────────────────────────────────────────────────

def query_nasa_archive(
    adql_query: str, fmt: str = "csv", timeout: int = 30
) -> pd.DataFrame:
    """Execute an ADQL query against the NASA Exoplanet Archive.

    Parameters
    ----------
    adql_query : str
        Query written in ADQL (≈ SQL with geometric extensions).
    fmt : str
        Response format – ``"csv"``, ``"votable"`` or ``"json"``.
    timeout : int
        HTTP timeout in seconds.

    Returns
    -------
    pd.DataFrame
    """
    params = {"query": adql_query, "format": fmt}
    response = requests.get(NASA_TAP_URL, params=params, timeout=timeout)
    response.raise_for_status()

    text = response.text.strip()
    if not text:
        raise ValueError("Empty response from NASA Exoplanet Archive")

    return pd.read_csv(io.StringIO(text))


# ─── Convenience wrappers ─────────────────────────────────────────────────────

def get_planet_data(planet_name: str) -> Optional[pd.Series]:
    """Fetch the full parameter set for a single named planet."""
    query = f"""
    SELECT pl_name, pl_radj, pl_bmassj, pl_orbsmax, pl_orbper,
           pl_insol, pl_eqt, pl_dens,
           st_teff, st_rad, st_lum, st_mass
    FROM pscomppars
    WHERE pl_name = '{planet_name}'
    """
    df = query_nasa_archive(query)
    if df.empty:
        return None
    return df.iloc[0]


def get_habitable_candidates(
    radius_min_earth: float = 0.5,
    radius_max_earth: float = 2.5,
    insol_min: float = 0.1,
    insol_max: float = 10.0,
    teff_min: int = 2500,
    teff_max: int = 7000,
) -> pd.DataFrame:
    """Earth-sized planets inside a broad habitable zone.

    Parameters are in Earth radii; converted to Jupiter radii for the
    ``pl_radj`` column used by the NASA ``pscomppars`` table.
    """
    r_min_jup = radius_min_earth / R_JUPITER_TO_EARTH
    r_max_jup = radius_max_earth / R_JUPITER_TO_EARTH
    query = f"""
    SELECT pl_name, pl_radj, pl_bmassj, pl_orbsmax, pl_orbper,
           pl_insol, pl_eqt, pl_dens,
           st_teff, st_rad, st_lum, st_mass,
           disc_year, discoverymethod
    FROM pscomppars
    WHERE pl_radj BETWEEN {r_min_jup:.6f} AND {r_max_jup:.6f}
      AND pl_insol BETWEEN {insol_min} AND {insol_max}
      AND st_teff BETWEEN {teff_min} AND {teff_max}
      AND pl_radj IS NOT NULL
      AND st_teff IS NOT NULL
    ORDER BY pl_insol ASC
    """
    return query_nasa_archive(query)


def get_all_confirmed_planets() -> pd.DataFrame:
    """Full confirmed-planet catalog (training data for CTGAN)."""
    query = """
    SELECT pl_name, pl_radj, pl_bmassj, pl_orbsmax, pl_orbper,
           pl_insol, pl_eqt, pl_dens, pl_orbeccen,
           st_teff, st_rad, st_lum, st_mass, st_dens,
           sy_dist
    FROM pscomppars
    WHERE pl_radj IS NOT NULL
      AND st_teff IS NOT NULL
      AND pl_orbsmax IS NOT NULL
    """
    return query_nasa_archive(query)
