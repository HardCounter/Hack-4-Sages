"""
Diagnostics for the PINNFormer3D physics-informed climate model.

Run after training with:

    python train_models.py --pinn

This script:
* Loads ``models/pinn3d_weights.pt`` (if available).
* Samples a surface temperature map from the 3-D PINN for a canonical
  tidally locked configuration.
* Compares basic map statistics against an analytical eyeball model
  with the same equilibrium temperature.
* Saves 2-D HTML maps for visual inspection into ``diagnostics/``.
"""

import os
import sys

import numpy as np

# Ensure project root (with local ``modules`` package) is on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from modules.astro_physics import equilibrium_temperature, habitable_surface_fraction
from modules.pinnformer3d import load_pinnformer, sample_surface_map
from modules.visualization import create_2d_heatmap, generate_eyeball_map


REPO_DIR = ROOT_DIR
MODELS_DIR = os.path.join(REPO_DIR, "models")
DIAG_DIR = os.path.join(REPO_DIR, "diagnostics")
PINN_PATH = os.path.join(MODELS_DIR, "pinn3d_weights.pt")


def main() -> None:
    if not os.path.exists(PINN_PATH):
        print(
            f"[ERROR] PINNFormer weights not found at {PINN_PATH}.\n"
            "Run `python train_models.py --pinn` first."
        )
        return

    os.makedirs(DIAG_DIR, exist_ok=True)

    # Canonical tidally locked test planet: Proxima-b-like
    params = {
        "radius_earth": 1.1,
        "mass_earth": 1.3,
        "semi_major_axis_au": 0.0485,
        "star_teff_K": 3042.0,
        "star_radius_solar": 0.141,
        "insol_earth": 0.65,
        "albedo": 0.3,
        "tidally_locked": 1,
    }

    print("Loading PINNFormer3D model...")
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    model = load_pinnformer(PINN_PATH, device=device)
    print(f"  Loaded model on device={device}")

    print("\nComputing equilibrium temperature and analytical map...")
    T_eq = equilibrium_temperature(
        stellar_temp=params["star_teff_K"],
        stellar_radius=params["star_radius_solar"],
        semi_major_axis=params["semi_major_axis_au"],
        albedo=params.get("albedo", 0.3),
        tidally_locked=True,
    )
    print(f"  Equilibrium T_eq   : {T_eq:.1f} K")

    ana_map = generate_eyeball_map(T_eq, tidally_locked=True, n_lat=32, n_lon=64)
    ana_stats = {
        "T_min": float(ana_map.min()),
        "T_max": float(ana_map.max()),
        "T_mean": float(ana_map.mean()),
        "HSF": habitable_surface_fraction(ana_map),
    }
    print("Analytical eyeball map stats:")
    print(
        f"  T_min={ana_stats['T_min']:.1f} K, "
        f"T_max={ana_stats['T_max']:.1f} K, "
        f"T_mean={ana_stats['T_mean']:.1f} K, "
        f"HSF={ana_stats['HSF']:.3f}"
    )

    print("\nSampling surface map from PINNFormer3D...")
    pinn_map = sample_surface_map(
        model,
        n_lat=32,
        n_lon=64,
        z_level=0.0,
        device=device,
        target_T_eq=T_eq,
    )
    pinn_stats = {
        "T_min": float(pinn_map.min()),
        "T_max": float(pinn_map.max()),
        "T_mean": float(pinn_map.mean()),
        "HSF": habitable_surface_fraction(pinn_map),
    }
    print("PINNFormer3D surface map stats:")
    print(
        f"  T_min={pinn_stats['T_min']:.1f} K, "
        f"T_max={pinn_stats['T_max']:.1f} K, "
        f"T_mean={pinn_stats['T_mean']:.1f} K, "
        f"HSF={pinn_stats['HSF']:.3f}"
    )

    # Rough comparison hints
    print("\nComparison (PINN vs analytical):")
    print(
        f"  ΔT_mean = {pinn_stats['T_mean'] - ana_stats['T_mean']:.1f} K "
        f"(PINN - analytical)"
    )
    print(
        f"  ΔHSF    = {pinn_stats['HSF'] - ana_stats['HSF']:.3f} "
        f"(PINN - analytical)"
    )

    # Save maps for visual inspection
    ana_fig = create_2d_heatmap(ana_map, planet_name="Proxima_b_analytical")
    pinn_fig = create_2d_heatmap(pinn_map, planet_name="Proxima_b_PINNFormer")

    ana_path = os.path.join(DIAG_DIR, "pinn_Proxima_b_analytical.html")
    pinn_path = os.path.join(DIAG_DIR, "pinn_Proxima_b_PINNFormer.html")

    ana_fig.write_html(ana_path)
    pinn_fig.write_html(pinn_path)

    print(f"\nSaved analytical map to: {ana_path}")
    print(f"Saved PINNFormer map to: {pinn_path}")


if __name__ == "__main__":
    main()

