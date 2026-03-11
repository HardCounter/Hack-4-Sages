"""
Helper script to build the combined exoplanet catalog
and dump it to a CSV for manual inspection.

Usage (from repo root, inside venv):
    python tools/build_combined_catalog_preview.py

This will create ``data/combined_catalog_preview.csv``.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.combined_catalog import DATA_DIR, build_combined_catalog


def main() -> None:
    df = build_combined_catalog()
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "combined_catalog_preview.csv")

    if df.empty:
        print("Combined catalog is empty – did you run tools/data_fetch.py?")
        return

    df.to_csv(out_path, index=False)
    print(f"Wrote combined catalog with {len(df)} unique planets to:")
    print(f"  {out_path}")


if __name__ == "__main__":
    main()

