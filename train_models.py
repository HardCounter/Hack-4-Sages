"""
One-shot training script — populates the models/ directory.

Run once after installing dependencies:

    python train_models.py

What it trains
--------------
1. ELM ensemble   (~5 seconds on CPU)  → models/elm_ensemble.pkl
2. CTGAN           (minutes, GPU helps) → models/ctgan_exoplanets.pkl  [optional]
3. PINNFormer 3-D  (variable on GPU)    → models/pinn3d_weights.pt     [optional]

Pass flags to control scope:

    python train_models.py                                  # ELM only
    python train_models.py --pinn                           # ELM + PINN (basic)
    python train_models.py --pinn --pinn-mode oht_clouds    # ELM + PINN with OHT + clouds
    python train_models.py --pinn --pinn-mode full          # ELM + PINN with all physics
    python train_models.py --ctgan --pinn --pinn-mode full  # everything

Available PINN modes: basic, greenhouse, oht, clouds, tidal, ice_albedo,
                      advection, oht_clouds, full
"""

import argparse
import os
import sys
import time

import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def ensure_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)


# ── 1. ELM ensemble ──────────────────────────────────────────────────────────

BATCH_THRESHOLD = 100_000


def train_elm(n_samples: int = 5000, n_ensemble: int = 10, n_neurons: int = 500):
    from modules.elm_surrogate import ELMClimateSurrogate, generate_analytical_training_data

    print(f"\n{'='*60}")
    print("  ELM Ensemble Training")
    print(f"{'='*60}")
    print(f"  Samples : {n_samples}")
    print(f"  Ensemble: {n_ensemble} models x {n_neurons} neurons")

    model = ELMClimateSurrogate(
        n_ensemble=n_ensemble, n_neurons=n_neurons, alpha=1e-4
    )

    if n_samples >= BATCH_THRESHOLD:
        chunk_size = min(50_000, n_samples // 4)
        print(f"  Mode    : batched (chunk_size={chunk_size})")
        t0 = time.time()
        model.train_batched(n_samples=n_samples, chunk_size=chunk_size)
        print(f"  Training completed in {time.time()-t0:.1f}s")
    else:
        t0 = time.time()
        X, y = generate_analytical_training_data(n_samples=n_samples)
        print(f"  Data generated in {time.time()-t0:.1f}s  (X={X.shape}, y={y.shape})")

        t1 = time.time()
        model.train(X, y)
        print(f"  Training completed in {time.time()-t1:.1f}s")

    path = os.path.join(MODELS_DIR, "elm_ensemble.pkl")
    model.save(path)
    print(f"  Saved → {path}")

    # Quick sanity check — predict Proxima Centauri b
    params = {
        "radius_earth": 1.07,
        "mass_earth": 1.27,
        "semi_major_axis_au": 0.0485,
        "star_teff_K": 3042,
        "star_radius_solar": 0.141,
        "insol_earth": 0.65,
        "albedo": 0.3,
        "tidally_locked": 1,
    }
    tmap = model.predict_from_params(params)
    print(f"  Sanity check (Proxima Cen b): "
          f"T_min={tmap.min():.0f} K, T_max={tmap.max():.0f} K, "
          f"shape={tmap.shape}")


# ── 2. CTGAN ──────────────────────────────────────────────────────────────────

def train_ctgan(epochs: int = 1000):
    import os
    import warnings
    import pandas as pd

    from modules.data_augmentation import ExoplanetDataAugmenter
    from modules.combined_catalog import DATA_DIR, build_combined_catalog

    # Optional: auto-fetch EU/DACE catalogs if missing, so the combined
    # catalog can include more than NASA.
    eu_path = os.path.join(DATA_DIR, "exoplanet_eu_raw.csv")
    dace_path = os.path.join(DATA_DIR, "dace_raw.csv")
    if not (os.path.exists(eu_path) and os.path.exists(dace_path)):
        try:
            from tools.data_fetch import main as fetch_data

            print("  European catalogs missing – running tools/data_fetch.py...")
            fetch_data()
        except Exception as exc:  # pragma: no cover - best-effort fetch
            print(f"  [WARN] Could not fetch European catalogs automatically: {exc}")

    print(f"\n{'='*60}")
    print("  CTGAN Data Augmentation Training")
    print(f"{'='*60}")
    print(f"  Epochs: {epochs}")

    print("  Building combined exoplanet catalog (NASA + EU + DACE)...")
    raw = build_combined_catalog()
    print(f"  Combined catalog size: {len(raw)} unique planets")

    # Prepare data in the normalised schema and light class rebalance so the
    # CTGAN sees more examples of habitable worlds during training.
    aug = ExoplanetDataAugmenter(epochs=epochs)
    data = aug.prepare_normalised_data(raw)

    # Noise-augmented upsampling: instead of exact copies, add small
    # Gaussian perturbations to each replicated habitable row so that
    # CTGAN sees more distributional diversity during training.
    hab = data[data["habitable"] == 1]
    non = data[data["habitable"] == 0]
    if len(hab) > 0:
        factor = max(1, 2 * len(non) // max(1, len(hab)))
        continuous = [c for c in hab.columns if c != "habitable"]
        hab_augmented = [hab]
        for _ in range(factor - 1):
            noisy = hab.copy()
            for col in continuous:
                col_std = noisy[col].std()
                if col_std > 0:
                    noise = np.random.normal(0, 0.05 * col_std, size=len(noisy))
                    noisy[col] = np.maximum(noisy[col] + noise, 1e-6)
            hab_augmented.append(noisy)
        data_balanced = pd.concat([non] + hab_augmented, ignore_index=True)
    else:
        data_balanced = data

    # Suppress benign cuBLAS context warnings from PyTorch during GAN training.
    warnings.filterwarnings(
        "ignore",
        message="Attempting to run cuBLAS, but there was no current CUDA context!",
        category=UserWarning,
    )

    t0 = time.time()
    aug.train(data_balanced)
    print(f"  Training completed in {time.time()-t0:.1f}s")

    synth = aug.generate_synthetic_planets(n_samples=500000)
    valid = aug.validate_synthetic_data(synth)
    print(f"  Generated {len(synth)} → {len(valid)} physically valid synthetics")

    path = os.path.join(MODELS_DIR, "ctgan_exoplanets.pkl")
    aug.save_model(path)
    print(f"  Saved → {path}")


# ── 3. PINNFormer 3-D ────────────────────────────────────────────────────────

def train_pinn(epochs: int = 5000, n_colloc: int = 8192, mode: str = "basic"):
    from modules.pinnformer3d import (
        PINNPhysicsConfig,
        train_pinnformer,
        save_pinnformer,
    )

    cfg = PINNPhysicsConfig.from_mode(mode)

    print(f"\n{'='*60}")
    print("  PINNFormer 3-D Training")
    print(f"{'='*60}")
    print(f"  Mode   : {mode}")
    print(f"  Physics: {cfg.summary()}")
    print(f"  Fields : {cfg.n_output_fields} → {', '.join(cfg.field_names)}")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device : {device}")
    if device == "cuda":
        backend = "ROCm/HIP" if getattr(torch.version, "hip", None) else "CUDA"
        print(f"  Backend: {backend}")
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM   : {vram:.1f} GB")
    print(f"  Collocation points: {n_colloc}")
    print(f"  Epochs: {epochs}")

    t0 = time.time()
    model = train_pinnformer(
        cfg=cfg, n_colloc=n_colloc, epochs=epochs, device=device, log_every=500
    )
    elapsed = time.time() - t0
    print(f"  Training completed in {elapsed/60:.1f} min")

    path = os.path.join(MODELS_DIR, "pinn3d_weights.pt")
    save_pinnformer(model, path, cfg=cfg)
    print(f"  Saved → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train all models for the Exoplanetary Digital Twin."
    )
    parser.add_argument(
        "--ctgan", action="store_true",
        help="Also train CTGAN (requires internet for NASA data)"
    )
    parser.add_argument(
        "--pinn", action="store_true",
        help="Also train PINNFormer 3-D (requires PyTorch; GPU strongly recommended)"
    )
    parser.add_argument(
        "--elm-samples", type=int, default=5000,
        help="Number of synthetic training samples for ELM (default: 5000)"
    )
    parser.add_argument(
        "--elm-neurons", type=int, default=500,
        help="Hidden neurons per ELM model (default: 500)"
    )
    parser.add_argument(
        "--elm-models", type=int, default=10,
        help="Number of ELM models in the ensemble (default: 10)"
    )
    parser.add_argument(
        "--ctgan-epochs", type=int, default=600,
        help="CTGAN training epochs (default: 600)"
    )
    parser.add_argument(
        "--pinn-epochs", type=int, default=5000,
        help="PINNFormer training epochs (default: 5000)"
    )
    parser.add_argument(
        "--pinn-mode", type=str, default="basic",
        choices=[
            "basic", "greenhouse", "oht", "clouds", "tidal",
            "ice_albedo", "advection", "oht_clouds", "full",
        ],
        help=(
            "PINNFormer physics mode (default: basic). "
            "Options: basic, greenhouse, oht, clouds, tidal, "
            "ice_albedo, advection, oht_clouds, full"
        ),
    )
    parser.add_argument(
        "--pinn-colloc", type=int, default=8192,
        help="Number of collocation points for PINN training (default: 8192)"
    )
    args = parser.parse_args()

    ensure_dir()

    # ELM always runs — it's fast and essential
    train_elm(
        n_samples=args.elm_samples,
        n_ensemble=args.elm_models,
        n_neurons=args.elm_neurons,
    )

    if args.ctgan:
        train_ctgan(epochs=args.ctgan_epochs)

    if args.pinn:
        train_pinn(
            epochs=args.pinn_epochs,
            n_colloc=args.pinn_colloc,
            mode=args.pinn_mode,
        )

    print(f"\n{'='*60}")
    print("  All done! Contents of models/:")
    print(f"{'='*60}")
    for f in sorted(os.listdir(MODELS_DIR)):
        size = os.path.getsize(os.path.join(MODELS_DIR, f))
        if size > 1_000_000:
            print(f"  {f:40s} {size/1_000_000:.1f} MB")
        else:
            print(f"  {f:40s} {size/1_000:.1f} KB")

    print("\nYou can now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
