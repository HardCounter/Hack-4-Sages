"""
Combined exoplanet catalog utilities.

Goal:
* Pull the NASA Exoplanet Archive catalog (pscomppars) in a normalised schema.
* Load European sources (Gaia, Exoplanet.eu, DACE) from CSV files created by
  ``data.py``.
* Normalise their columns to match the NASA-style schema as closely as
  possible.
* Concatenate and de-duplicate into a single dataframe that downstream
  modules can consume.

Normalised schema (column names):
* pl_name              – planet name (string)
* radius_earth         – planetary radius [R_earth]
* mass_earth           – planetary mass [M_earth]
* semi_major_axis_au   – semi-major axis [AU]
* period_days          – orbital period [days]
* insol_earth          – stellar flux at orbit [S_earth]
* t_eq_K               – equilibrium temperature [K]
* star_teff_K          – stellar effective temperature [K]
* star_radius_solar    – stellar radius [R_sun]
* star_mass_solar      – stellar mass [M_sun]
* source               – string flag: 'NASA', 'Gaia', 'ExoplanetEU', 'DACE'
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd

from . import nasa_client


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")


def _normalise_nasa() -> pd.DataFrame:
    """Fetch and normalise NASA pscomppars confirmed-planet catalog."""
    df = nasa_client.get_all_confirmed_planets().copy()

    # Convert units to the shared schema
    df["radius_earth"] = df["pl_radj"] * nasa_client.R_JUPITER_TO_EARTH
    df["mass_earth"] = df["pl_bmassj"] * nasa_client.M_JUPITER_TO_EARTH
    df["semi_major_axis_au"] = df["pl_orbsmax"]
    df["period_days"] = df["pl_orbper"]
    df["insol_earth"] = df["pl_insol"]
    df["t_eq_K"] = df["pl_eqt"]
    df["star_teff_K"] = df["st_teff"]
    df["star_radius_solar"] = df["st_rad"]
    df["star_mass_solar"] = df["st_mass"]

    df_norm = df[
        [
            "pl_name",
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
    ].copy()
    df_norm["source"] = "NASA"
    return df_norm


def _load_csv_if_exists(filename: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _normalise_exoplanet_eu() -> pd.DataFrame:
    """Normalise Exoplanet.eu CSV to the shared schema.

    Expects ``exoplanet_eu_raw.csv`` from the VO TAP service used in
    ``data.py``. Column names in that file are stable and include, e.g.:
    * target_name        – planet name
    * mass               – planet mass [M_jup]
    * radius             – planet radius [R_jup]
    * semi_major_axis    – semi-major axis [AU]
    * period             – orbital period [days]
    * sflux              – stellar flux [S_earth]
    * t_eq               – equilibrium temperature [K]
    * star_teff          – stellar effective temperature [K]
    * star_radius        – stellar radius [R_sun]
    * star_mass          – stellar mass [M_sun]
    """
    df = _load_csv_if_exists("exoplanet_eu_raw.csv")
    if df.empty:
        return df

    out = pd.DataFrame()

    # Planet name: VO table uses 'target_name'
    out["pl_name"] = df.get("target_name", np.nan)

    # Planet properties in Jupiter units → convert to Earth units where needed.
    # If radius/mass are already in Earth units in some future schema, this
    # will slightly rescale them; adjust if you see that in the CSV.
    if "radius" in df.columns:
        out["radius_earth"] = df["radius"] * nasa_client.R_JUPITER_TO_EARTH
    else:
        out["radius_earth"] = np.nan

    if "mass" in df.columns:
        out["mass_earth"] = df["mass"] * nasa_client.M_JUPITER_TO_EARTH
    else:
        out["mass_earth"] = np.nan

    out["semi_major_axis_au"] = df.get("semi_major_axis", np.nan)
    out["period_days"] = df.get("period", np.nan)
    out["insol_earth"] = df.get("sflux", np.nan)
    out["t_eq_K"] = df.get("temp_calculated", df.get("t_eq", np.nan))

    # Host star properties
    out["star_teff_K"] = df.get("star_teff", np.nan)
    out["star_radius_solar"] = df.get("star_radius", np.nan)
    out["star_mass_solar"] = df.get("star_mass", np.nan)

    out["source"] = "ExoplanetEU"
    return out


def _normalise_dace() -> pd.DataFrame:
    """Normalise DACE exoplanet CSV to the shared schema.

    Expects a file ``dace_raw.csv`` created by ``DaceExo.query_database``.
    Column names there are stable and include, for example:
    * planet_name           – planet name
    * planet_mass           – [M_jup]
    * planet_radius         – [R_jup]
    * semi_major_axis       – [AU]
    * period                – [days]
    * insolation_flux_computed / equilibrium_temp_computed
    * stellar_eff_temp      – [K]
    * stellar_radius        – [R_sun]
    * stellar_mass          – [M_sun]
    """
    df = _load_csv_if_exists("dace_raw.csv")
    if df.empty:
        return df

    out = pd.DataFrame()

    # Name is explicitly 'planet_name'
    out["pl_name"] = df.get("planet_name", np.nan)

    # Planet parameters; DACE uses Jupiter units for mass/radius.
    if "planet_radius" in df.columns:
        out["radius_earth"] = df["planet_radius"] * nasa_client.R_JUPITER_TO_EARTH
    elif "planet_radius_Rearth" in df.columns:
        out["radius_earth"] = df["planet_radius_Rearth"]
    else:
        out["radius_earth"] = np.nan

    if "planet_mass" in df.columns:
        out["mass_earth"] = df["planet_mass"] * nasa_client.M_JUPITER_TO_EARTH
    elif "planet_mass_Mearth" in df.columns:
        out["mass_earth"] = df["planet_mass_Mearth"]
    else:
        out["mass_earth"] = np.nan

    out["semi_major_axis_au"] = df.get("semi_major_axis", df.get("a_AU", np.nan))
    out["period_days"] = df.get("period", df.get("P_days", np.nan))
    out["insol_earth"] = df.get("insolation_flux_computed", np.nan)
    out["t_eq_K"] = df.get("equilibrium_temp_computed", df.get("equilibrium_temp", np.nan))

    out["star_teff_K"] = df.get("stellar_eff_temp", np.nan)
    out["star_radius_solar"] = df.get("stellar_radius", df.get("R_star_Rsun", np.nan))
    out["star_mass_solar"] = df.get("stellar_mass", df.get("M_star_Msun", np.nan))

    out["source"] = "DACE"
    return out


def _normalise_gaia() -> pd.DataFrame:
    """Normalise Gaia DR exoplanet transit table to the shared schema.

    Currently Gaia's `vari_planetary_transit` table (as fetched in
    ``data.py``) does not provide enough physical parameters to be useful
    for our climate / habitability models, so we skip it for the combined
    catalog.
    """
    # Keep the CSV for future work, but do not include it in the merged
    # catalog yet.
    return pd.DataFrame()


def build_combined_catalog() -> pd.DataFrame:
    """Return a merged, de-duplicated catalog across all sources.

    De-duplication strategy:
    * Concatenate NASA, Exoplanet.eu, DACE, Gaia (if available).
    * Group by ``pl_name``.
    * Prefer NASA rows when duplicates exist; otherwise keep the first
      non-empty entry.
    """
    frames = [
        _normalise_nasa(),
        _normalise_exoplanet_eu(),
        _normalise_dace(),
        _normalise_gaia(),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)

    # Simple de-duplication by planet name, preferring NASA.
    df_all["source_priority"] = df_all["source"].map(
        {"NASA": 0, "ExoplanetEU": 1, "DACE": 2, "Gaia": 3}
    ).fillna(5)

    df_all.sort_values(
        by=["pl_name", "source_priority"], inplace=True, kind="mergesort"
    )
    df_dedup = df_all.drop_duplicates(subset=["pl_name"], keep="first").drop(
        columns=["source_priority"]
    )
    return df_dedup


__all__ = [
    "build_combined_catalog",
]

