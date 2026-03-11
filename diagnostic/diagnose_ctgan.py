"""
Sanity checks for the CTGAN exoplanet data augmenter.

Run after training with `python train_models.py --ctgan`.
This script:

* Loads the combined exoplanet catalog (NASA + Exoplanet.eu + DACE).
* Prepares the tabular dataset and isolates the "habitable" subset.
* Loads (or, if missing, quickly trains) the CTGAN model.
* Generates synthetic habitable planets.
* Prints side-by-side summary statistics for key parameters so you can
  see how well the synthetic distribution matches the real one.
"""

import os
import sys

import numpy as np
import pandas as pd

# Ensure project root (with local ``modules`` package) is on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from modules.data_augmentation import ExoplanetDataAugmenter
from modules.combined_catalog import build_combined_catalog


REPO_DIR = ROOT_DIR
MODELS_DIR = os.path.join(REPO_DIR, "models")
CTGAN_PATH = os.path.join(MODELS_DIR, "ctgan_exoplanets.pkl")
DIAG_DIR = os.path.join(REPO_DIR, "diagnostic")

COLS = [
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


def _build_real_habitable_ranges(real_hab: pd.DataFrame) -> dict:
    """Compute robust 1–99% ranges for each column in COLS on real habitable planets."""
    ranges: dict[str, tuple[float, float]] = {}
    for col in COLS:
        series = real_hab[col].dropna()
        if series.empty:
            continue
        lo, hi = series.quantile([0.01, 0.99])
        ranges[col] = (float(lo), float(hi))
    return ranges


def _posthoc_match_habitable(
    synthetic: pd.DataFrame,
    real_ranges: dict,
) -> pd.DataFrame:
    """Restrict synthetic habitable planets to the real habitable parameter envelope.

    This does *not* change how CTGAN is trained, only how we filter its
    conditional samples (habitable == 1) for downstream use and diagnostics.
    """
    s = synthetic.copy()
    for col, (lo, hi) in real_ranges.items():
        if col not in s.columns:
            continue
        s = s[(s[col] >= lo) & (s[col] <= hi)]
    return s


def _summary(df: pd.DataFrame, col: str) -> str:
    series = df[col].dropna()
    if series.empty:
        return "n/a"
    return (
        f"mean={series.mean():.3g}, std={series.std():.3g}, "
        f"min={series.min():.3g}, max={series.max():.3g}, n={len(series)}"
    )


def _ks_tests(real: pd.DataFrame, synth: pd.DataFrame, n_iter: int = 50) -> pd.DataFrame:
    from scipy.stats import ks_2samp

    rows = []
    for col in COLS:
        r = real[col].dropna().values
        s = synth[col].dropna().values
        if len(r) < 5 or len(s) < 5:
            rows.append({"column": col, "ks_stat": np.nan, "p_value": np.nan})
            continue
        # Run multiple subsampled KS tests and take the median to reduce
        # variance from the random draw (important when n_real is small).
        stats, pvals = [], []
        for _ in range(n_iter):
            s_sub = s
            if len(s) > len(r):
                s_sub = np.random.choice(s, size=len(r), replace=False)
            stat, pval = ks_2samp(r, s_sub)
            stats.append(stat)
            pvals.append(pval)
        rows.append({
            "column": col,
            "ks_stat": round(float(np.median(stats)), 4),
            "p_value": round(float(np.median(pvals)), 4),
        })
    return pd.DataFrame(rows)


def _dcr_memorization_check(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    cols: list[str] = COLS,
) -> dict:
    """Distance to Closest Record (DCR) memorization diagnostic.

    For each synthetic row, compute its L2 distance (in min-max-scaled
    feature space) to the nearest real row.  Compare against the
    leave-one-out nearest-neighbour distances within the real set itself.

    Returns a dict with:
      dcr_synth   – array of per-synthetic-record DCR values
      dcr_real    – array of per-real-record leave-one-out NN distances
      exact_copy_frac – fraction of synthetic records with DCR ≈ 0
      median_ratio    – median(dcr_synth) / median(dcr_real)
      fifth_pct_synth – 5th-percentile DCR for synthetic records
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import MinMaxScaler

    r = real[cols].dropna().values.astype(np.float64)
    s = synthetic[cols].dropna().values.astype(np.float64)

    scaler = MinMaxScaler().fit(r)
    r_scaled = scaler.transform(r)
    s_scaled = scaler.transform(s)

    # Synthetic → Real: 1-NN
    nn_real = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(r_scaled)
    dcr_synth = nn_real.kneighbors(s_scaled)[0].ravel()

    # Real → Real (leave-one-out): 2-NN then take the second neighbour
    nn_loo = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(r_scaled)
    dcr_real = nn_loo.kneighbors(r_scaled)[0][:, 1]

    eps = 1e-8
    exact_copy_frac = float((dcr_synth < eps).sum()) / len(dcr_synth)
    median_synth = float(np.median(dcr_synth))
    median_real = float(np.median(dcr_real))
    median_ratio = median_synth / median_real if median_real > 0 else np.inf
    fifth_pct = float(np.percentile(dcr_synth, 5))

    return {
        "dcr_synth": dcr_synth,
        "dcr_real": dcr_real,
        "exact_copy_frac": exact_copy_frac,
        "median_synth": median_synth,
        "median_real": median_real,
        "median_ratio": median_ratio,
        "fifth_pct_synth": fifth_pct,
    }


def _build_html_report(
    real_hab: pd.DataFrame,
    synthetic: pd.DataFrame,
    ks_df: pd.DataFrame,
    dcr: dict | None = None,
) -> str:
    """Build a standalone HTML report with histograms and correlation matrices."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return "<html><body><p>Plotly not installed — skipping visual report.</p></body></html>"

    sections = []

    # --- Overlay histograms ---
    n = len(COLS)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig_hist = make_subplots(rows=nrows, cols=ncols, subplot_titles=COLS)

    for i, col in enumerate(COLS):
        r, c = divmod(i, ncols)
        real_vals = real_hab[col].dropna().values
        synth_vals = synthetic[col].dropna().values
        fig_hist.add_trace(
            go.Histogram(x=real_vals, name="Real", marker_color="#2171b5",
                         opacity=0.6, showlegend=(i == 0)),
            row=r + 1, col=c + 1,
        )
        fig_hist.add_trace(
            go.Histogram(x=synth_vals, name="Synthetic", marker_color="#d73027",
                         opacity=0.6, showlegend=(i == 0)),
            row=r + 1, col=c + 1,
        )
    fig_hist.update_layout(
        barmode="overlay", height=300 * nrows, width=1000,
        title_text="Real vs Synthetic: Parameter Distributions",
        paper_bgcolor="#111", plot_bgcolor="#222", font=dict(color="white"),
    )
    sections.append(fig_hist.to_html(full_html=False, include_plotlyjs="cdn"))

    # --- KS test table ---
    ks_html = ks_df.to_html(index=False, classes="ks-table", border=0)
    sections.append(f"<h2>KS Tests (Real vs Synthetic)</h2>{ks_html}")

    # --- Correlation matrix comparison ---
    real_corr = real_hab[COLS].corr()
    synth_corr = synthetic[COLS].corr()

    fig_corr = make_subplots(
        rows=1, cols=2, subplot_titles=["Real correlation", "Synthetic correlation"],
    )
    fig_corr.add_trace(
        go.Heatmap(z=real_corr.values, x=COLS, y=COLS,
                    colorscale="RdBu", zmin=-1, zmax=1, showscale=False),
        row=1, col=1,
    )
    fig_corr.add_trace(
        go.Heatmap(z=synth_corr.values, x=COLS, y=COLS,
                    colorscale="RdBu", zmin=-1, zmax=1),
        row=1, col=2,
    )
    fig_corr.update_layout(
        height=500, width=1100,
        title_text="Correlation Matrix Comparison",
        paper_bgcolor="#111", plot_bgcolor="#222", font=dict(color="white"),
    )
    sections.append(fig_corr.to_html(full_html=False, include_plotlyjs=False))

    # --- DCR memorization histogram ---
    if dcr is not None:
        fig_dcr = go.Figure()
        fig_dcr.add_trace(go.Histogram(
            x=dcr["dcr_real"], name="Real↔Real (LOO)",
            marker_color="#2171b5", opacity=0.6, nbinsx=60,
        ))
        fig_dcr.add_trace(go.Histogram(
            x=dcr["dcr_synth"], name="Synth→Real",
            marker_color="#d73027", opacity=0.6, nbinsx=60,
        ))
        fig_dcr.update_layout(
            barmode="overlay", height=400, width=900,
            title_text="DCR Memorization Check",
            xaxis_title="Distance to Closest Record (min-max scaled L2)",
            yaxis_title="Count",
            paper_bgcolor="#111", plot_bgcolor="#222", font=dict(color="white"),
        )
        dcr_summary = (
            f"<p>Median ratio (synth/real): <b>{dcr['median_ratio']:.3f}</b> "
            f"(median synth={dcr['median_synth']:.4f}, "
            f"median real={dcr['median_real']:.4f})</p>"
            f"<p>Exact-copy fraction: <b>{dcr['exact_copy_frac']:.4%}</b></p>"
            f"<p>5th-percentile synth DCR: <b>{dcr['fifth_pct_synth']:.4f}</b></p>"
        )
        sections.append(
            "<h2>DCR Memorization Check</h2>"
            + dcr_summary
            + fig_dcr.to_html(full_html=False, include_plotlyjs=False)
        )

    body = "\n<hr>\n".join(sections)
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>"
        "body{background:#111;color:#eee;font-family:sans-serif;padding:20px}"
        ".ks-table{border-collapse:collapse;margin:10px 0}"
        ".ks-table td,.ks-table th{padding:6px 12px;border:1px solid #555}"
        "</style></head><body>"
        f"<h1>CTGAN Diagnostics Report</h1>{body}"
        "</body></html>"
    )


def main() -> None:
    print("Building combined exoplanet catalog (NASA + Exoplanet.eu + DACE)...")
    raw = build_combined_catalog()
    print(f"  Combined catalog size: {len(raw)} unique planets.")

    augmenter = ExoplanetDataAugmenter()
    data = augmenter.prepare_normalised_data(raw)

    real_hab = data[data["habitable"] == 1].copy()
    print(f"Real 'habitable' subset: {len(real_hab)} planets.")

    # Robust parameter envelope for real habitable planets (1–99% per column).
    real_ranges = _build_real_habitable_ranges(real_hab)

    # Load or (lightly) train CTGAN
    if os.path.exists(CTGAN_PATH):
        print(f"Loading CTGAN model from {CTGAN_PATH}...")
        augmenter.load_model(CTGAN_PATH)
    else:
        print(
            f"[WARN] CTGAN model not found at {CTGAN_PATH}.\n"
            "Training a quick model (fewer epochs) for diagnostics only..."
        )
        augmenter.epochs = 50
        augmenter.train(data)

    print("Sampling synthetic habitable planets...")
    synthetic = augmenter.generate_synthetic_planets(
        n_samples=500000, condition_column="habitable", condition_value=1
    )
    synthetic = ExoplanetDataAugmenter.validate_synthetic_data(synthetic)
    # Post-hoc physics/astrophysical filter: keep only synthetic habitable
    # planets whose parameters lie within the real habitable envelope.
    synthetic = _posthoc_match_habitable(synthetic, real_ranges)
    print(f"Synthetic 'habitable' after validation: {len(synthetic)} planets.")

    print("\nParameter comparison: real vs synthetic (habitable subset)")
    print("-" * 80)
    for col in COLS:
        real_stats = _summary(real_hab, col)
        synth_stats = _summary(synthetic, col)
        print(f"{col:20s} | real: {real_stats}")
        print(f"{'':20s} | synth: {synth_stats}")
        print("-" * 80)

    # --- KS tests ---
    ks_df = _ks_tests(real_hab, synthetic)
    print("\nTwo-sample KS tests:")
    for _, row in ks_df.iterrows():
        flag = "  PASS" if row["p_value"] > 0.05 else "* FAIL"
        print(f"  {row['column']:20s}  D={row['ks_stat']:.4f}  p={row['p_value']:.4f}  {flag}")

    # --- DCR memorization check ---
    print("\nDCR memorization check (min-max scaled L2)...")
    dcr = _dcr_memorization_check(real_hab, synthetic)
    print(f"  Median DCR synth→real : {dcr['median_synth']:.4f}")
    print(f"  Median DCR real↔real  : {dcr['median_real']:.4f}")
    print(f"  Median ratio (S/R)    : {dcr['median_ratio']:.3f}")
    print(f"  5th-pctl synth DCR    : {dcr['fifth_pct_synth']:.4f}")
    print(f"  Exact-copy fraction   : {dcr['exact_copy_frac']:.4%}")
    if dcr["exact_copy_frac"] > 0.01:
        print("  [WARN] >1% exact copies detected — possible memorization.")
    elif dcr["median_ratio"] < 0.5:
        print("  [WARN] Median ratio < 0.5 — synthetic records cluster too "
              "close to training data.")
    else:
        print("  [OK] No significant memorization detected.")

    # --- HTML report ---
    os.makedirs(DIAG_DIR, exist_ok=True)
    html = _build_html_report(real_hab, synthetic, ks_df, dcr=dcr)
    report_path = os.path.join(DIAG_DIR, "ctgan_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nHTML report saved to {report_path}")


if __name__ == "__main__":
    main()

