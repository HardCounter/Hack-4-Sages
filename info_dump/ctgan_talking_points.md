# CTGAN Talking Points: Class Imbalance, Augmentation & Safeguards

## 1. The Class-Imbalance Problem

The NASA Exoplanet Archive contains ~5 700 confirmed planets, but only ~60 (~1%) satisfy even conservative habitability criteria (radius 0.5-2.5 R_Earth, insolation 0.2-2.0 S_Earth, host star 2 500-7 000 K). Any supervised model trained naively on this catalog will overwhelmingly learn "not habitable" and ignore the minority class entirely. CTGAN conditional sampling lets us synthesise additional habitable-planet configurations to study the parameter space, but the outputs must be treated as *exploratory*, not as predictions of real planets.

## 2. Why Tabular Augmentation Is Hard

Unlike image augmentation (where flips and crops preserve semantics), exoplanet parameters are tightly constrained by physics. A CTGAN learns statistical correlations from the full catalog, but the ~60 habitable planets occupy a narrow, physically coupled subspace. The generator can produce combinations that look statistically plausible but violate fundamental relationships — for example, an Earth-mass planet at 50 AU with high insolation (violates the inverse-square law), or a 0.5 R_Earth planet with 8 M_Earth (exceeds any known mass-radius relation for rocky bodies). Log-transforming heavily skewed features (mass, semi-major axis, period, insolation) before training helps the GAN model the tail distributions more faithfully, but cannot guarantee physical self-consistency.

## 3. Safeguards Against Nonsense

We apply multiple layers of defence:

1. **Two-tier post-hoc filtering.** A broad "physical sanity" filter removes obviously impossible rows (e.g. negative radii, star temperatures below 2 300 K). A stricter "habitable plausibility" filter (Kopparapu et al. 2014-inspired bounds) rejects rows outside the rocky/super-Earth regime: R < 4 R_Earth, M < 10 M_Earth, T_eq 100-1 000 K, etc.
2. **Explicit provenance tagging.** Every synthetic row carries `is_synthetic=True` and `source="CTGAN-v1"` so downstream code can never accidentally mix synthetic data into the real catalog view.
3. **Distribution diagnostics.** Two-sample Kolmogorov-Smirnov tests and overlay histograms (real vs synthetic) for every parameter, plus a side-by-side correlation-matrix comparison, are generated automatically by `diagnostic/diagnose_ctgan.py` and saved as an HTML report.
4. **UI separation.** The Streamlit app displays synthetic data only inside a clearly-labelled "Exploratory" expander with a warning banner, never in the main catalog table.
