"""
Fetch raw exoplanet catalog data from:
* Gaia DR3 planetary transits (+ host stellar context),
* Exoplanet.eu VO service,
* DACE (Geneva/CHEOPS).

Writes CSVs into the ``data/`` directory:
* data/gaia_raw.csv
* data/exoplanet_eu_raw.csv
* data/dace_raw.csv
"""

import os

import pandas as pd  # noqa: F401  # imported for side effects in astroquery
import pyvo as vo
from astroquery.gaia import Gaia
from dace_query.exoplanet import Exoplanet as DaceExo


def get_gaia_exoplanet_hosts():
    print("Fetching Gaia DR3 exoplanet host stars with stellar parameters...")
    # Use the planetary transit table only as a selector, then join to
    # gaia_source to retrieve stellar context (position, parallax, G mag).
    query = """
    SELECT
        vpt.source_id,
        vpt.transit_reference_time,
        vpt.transit_period,
        vpt.transit_depth,
        vpt.transit_duration,
        gs.ra,
        gs.dec,
        gs.parallax,
        gs.phot_g_mean_mag
    FROM gaiadr3.vari_planetary_transit AS vpt
    JOIN gaiadr3.gaia_source AS gs
      ON vpt.source_id = gs.source_id
    """
    job = Gaia.launch_job(query)
    return job.get_results().to_pandas()


def get_exoplanet_eu_data():
    print("Fetching Exoplanet.eu (Encyclopedia) Data...")
    # Uses the Virtual Observatory (VO) TAP service
    service = vo.dal.TAPService("http://voparis-tap-planeto.obspm.fr/tap")
    query = "SELECT * FROM exoplanet.epn_core"
    results = service.search(query)
    return results.to_table().to_pandas()


def get_dace_data():
    print("Fetching DACE (Geneva/CHEOPS) Data...")
    # DACE aggregates several catalogs including their own refined values
    # In 'public' mode, no API key is required for basic catalog queries
    return DaceExo.query_database(output_format="pandas")


def main() -> None:
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # 1. Get Gaia Data (ESA transiting exoplanet hosts + stellar context)
    df_gaia = get_gaia_exoplanet_hosts()
    df_gaia.to_csv(os.path.join(data_dir, "gaia_raw.csv"), index=False)

    # 2. Get Exoplanet.eu (The main European consolidated catalog)
    df_eu = get_exoplanet_eu_data()
    df_eu.to_csv(os.path.join(data_dir, "exoplanet_eu_raw.csv"), index=False)

    # 3. Get DACE Data (High-precision refined European data)
    df_dace = get_dace_data()
    df_dace.to_csv(os.path.join(data_dir, "dace_raw.csv"), index=False)

    print("\nSuccess! Files saved:")
    print(f"- Gaia: {len(df_gaia)} rows")
    print(f"- Exoplanet.eu: {len(df_eu)} rows")
    print(f"- DACE: {len(df_dace)} rows")


if __name__ == "__main__":
    main()

