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


def _build_summary_stats_table(real_hab: pd.DataFrame, synthetic: pd.DataFrame) -> str:
    """Build an HTML table comparing real vs synthetic summary statistics."""
    rows_html = []
    for col in COLS:
        rv = real_hab[col].dropna()
        sv = synthetic[col].dropna()
        if rv.empty:
            continue
        pct_diff = abs(rv.mean() - sv.mean()) / (rv.mean() + 1e-12) * 100
        emoji = "&#9989;" if pct_diff < 20 else ("&#9888;" if pct_diff < 50 else "&#10060;")
        rows_html.append(
            f"<tr>"
            f"<td>{col}</td>"
            f"<td>{rv.mean():.4g} &plusmn; {rv.std():.4g}</td>"
            f"<td>{sv.mean():.4g} &plusmn; {sv.std():.4g}</td>"
            f"<td>{len(rv)}</td><td>{len(sv)}</td>"
            f"<td>{emoji} {pct_diff:.1f}%</td>"
            f"</tr>"
        )
    header = (
        "<tr><th>Parameter</th><th>Real (mean&plusmn;std)</th>"
        "<th>Synthetic (mean&plusmn;std)</th>"
        "<th>n_real</th><th>n_synth</th><th>Mean diff</th></tr>"
    )
    return f'<table class="ks-table">{header}{"".join(rows_html)}</table>'


_TALKING_POINTS_HTML = """
<h2>Key Talking Points — Class Imbalance &amp; Augmentation</h2>
<div style="background:#1a1a2e;padding:16px;border-radius:8px;margin:12px 0">

<h3>1. Why class imbalance is a core problem</h3>
<p>Out of ~5 700 confirmed exoplanets, only ~60 fall into the
habitable zone by standard criteria (radius 0.5–2.5 R⊕, stellar
flux 0.2–2.0 S⊕, T_eff 2500–7000 K). That is roughly a <b>1:95
imbalance ratio</b>. Any supervised model trained on raw counts
will learn to predict "not habitable" almost every time and still
achieve >98% accuracy — a classic accuracy paradox.</p>

<h3>2. Why augmentation is hard for exoplanet data</h3>
<ul>
  <li><b>Multivariate coupling:</b> Planet parameters are not
  independent — radius, mass, insolation, and orbital period are
  linked by Kepler's third law, the mass–radius relation, and
  stellar luminosity. Naïve oversampling (e.g., SMOTE) ignores
  these astrophysical constraints and can produce planets that
  violate conservation laws.</li>
  <li><b>Small minority class:</b> With only ~60 real habitable
  worlds, the CTGAN's generator has very few exemplars to learn
  from, increasing the risk of <i>mode collapse</i> (generating
  near-identical copies of the training set).</li>
  <li><b>Right-skewed features:</b> Parameters like mass_earth
  and period_days span orders of magnitude. Without
  log-transforms, the GAN's internal Gaussian mixtures cannot
  represent the long tails, producing physically absurd outputs.</li>
</ul>

<h3>3. How we safeguard against nonsense</h3>
<ol>
  <li><b>Log-transform before training:</b> We apply log1p to
  right-skewed columns (mass, period, semi-major axis, stellar
  parameters) before CTGAN training and reverse with expm1 after
  sampling.</li>
  <li><b>Hard physics filter:</b> Every synthetic planet passes
  through <code>validate_synthetic_data()</code> which enforces
  absolute physical bounds (e.g., radius 0.3–25 R⊕, T_eq 10–4000 K,
  period > 0.1 d).</li>
  <li><b>Percentile clipping:</b> After physics filtering, we clip
  each column to its 1st–99th percentile range to remove long-tail
  outliers that passed the coarse filter.</li>
  <li><b>Post-hoc habitable envelope:</b> For conditional sampling
  (habitable=1), we further restrict synthetic planets to lie within
  the 1–99% quantile envelope of real habitable planets per feature.</li>
  <li><b>DCR memorization check:</b> We compute Distance to Closest
  Record (synth→real vs real↔real LOO) to verify the generator is
  not simply memorizing training rows.</li>
  <li><b>Clear labeling:</b> All synthetic data is explicitly marked
  as <i>exploratory</i> in the UI — never presented as real
  discoveries.</li>
</ol>
</div>
"""


def _build_html_report(
    real_hab: pd.DataFrame,
    synthetic: pd.DataFrame,
    ks_df: pd.DataFrame,
    dcr: dict | None = None,
) -> str:
    """Build a standalone HTML report with histograms, correlation matrices,
    summary statistics, and augmentation talking points."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return "<html><body><p>Plotly not installed — skipping visual report.</p></body></html>"

    sections = []

    # --- Summary statistics table ---
    sections.append(
        "<h2>Summary Statistics — Real vs Synthetic (habitable subset)</h2>"
        + _build_summary_stats_table(real_hab, synthetic)
    )

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

    # --- Paired violin / box comparison ---
    key_params = ["radius_earth", "mass_earth", "insol_earth", "t_eq_K"]
    fig_violin = make_subplots(
        rows=1, cols=len(key_params),
        subplot_titles=key_params,
    )
    for i, col in enumerate(key_params):
        rv = real_hab[col].dropna().values
        sv = synthetic[col].dropna().values
        fig_violin.add_trace(
            go.Violin(y=rv, name="Real", side="negative",
                      line_color="#2171b5", showlegend=(i == 0)),
            row=1, col=i + 1,
        )
        fig_violin.add_trace(
            go.Violin(y=sv, name="Synthetic", side="positive",
                      line_color="#d73027", showlegend=(i == 0)),
            row=1, col=i + 1,
        )
    fig_violin.update_layout(
        height=400, width=1100,
        title_text="Side-by-Side Violin — Key Habitable Parameters",
        paper_bgcolor="#111", plot_bgcolor="#222", font=dict(color="white"),
        violingap=0, violinmode="overlay",
    )
    sections.append(fig_violin.to_html(full_html=False, include_plotlyjs=False))

    # --- KS test table ---
    ks_html = ks_df.to_html(index=False, classes="ks-table", border=0)
    ks_pass = (ks_df["p_value"] > 0.05).sum()
    ks_total = len(ks_df)
    sections.append(
        f"<h2>KS Tests (Real vs Synthetic)</h2>"
        f"<p><b>{ks_pass}/{ks_total}</b> columns pass at p&gt;0.05</p>"
        f"{ks_html}"
    )

    # --- Correlation matrix comparison ---
    real_corr = real_hab[COLS].corr()
    synth_corr = synthetic[COLS].corr()
    corr_diff = (real_corr - synth_corr).abs()

    fig_corr = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Real correlation", "Synthetic correlation", "|Difference|"],
    )
    fig_corr.add_trace(
        go.Heatmap(z=real_corr.values, x=COLS, y=COLS,
                    colorscale="RdBu", zmin=-1, zmax=1, showscale=False),
        row=1, col=1,
    )
    fig_corr.add_trace(
        go.Heatmap(z=synth_corr.values, x=COLS, y=COLS,
                    colorscale="RdBu", zmin=-1, zmax=1, showscale=False),
        row=1, col=2,
    )
    fig_corr.add_trace(
        go.Heatmap(z=corr_diff.values, x=COLS, y=COLS,
                    colorscale="YlOrRd", zmin=0, zmax=1),
        row=1, col=3,
    )
    fig_corr.update_layout(
        height=500, width=1400,
        title_text="Correlation Matrix Comparison (+ absolute difference)",
        paper_bgcolor="#111", plot_bgcolor="#222", font=dict(color="white"),
    )
    sections.append(fig_corr.to_html(full_html=False, include_plotlyjs=False))

    # --- DCR memorization histogram ---
    if dcr is not None:
        fig_dcr = go.Figure()
        fig_dcr.add_trace(go.Histogram(
            x=dcr["dcr_real"], name="Real\u2194Real (LOO)",
            marker_color="#2171b5", opacity=0.6, nbinsx=60,
        ))
        fig_dcr.add_trace(go.Histogram(
            x=dcr["dcr_synth"], name="Synth\u2192Real",
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

    # --- Talking points ---
    sections.append(_TALKING_POINTS_HTML)

    body = "\n<hr>\n".join(sections)
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>"
        "body{background:#111;color:#eee;font-family:sans-serif;padding:20px}"
        ".ks-table{border-collapse:collapse;margin:10px 0}"
        ".ks-table td,.ks-table th{padding:6px 12px;border:1px solid #555}"
        "h2{color:#64b5f6;margin-top:32px}"
        "h3{color:#90caf9}"
        "code{background:#222;padding:2px 6px;border-radius:3px}"
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

    # --- Talking points (console) ---
    print("\n" + "=" * 80)
    print("TALKING POINTS — Class Imbalance & Augmentation")
    print("=" * 80)
    print(
        "\n1. CLASS IMBALANCE: ~60 habitable planets out of ~5 700 confirmed "
        "(1:95 ratio).\n"
        "   Any classifier trained on raw counts predicts 'not habitable'\n"
        "   almost always and still hits >98% accuracy (accuracy paradox).\n"
    )
    print(
        "2. WHY AUGMENTATION IS HARD:\n"
        "   • Parameters are astrophysically coupled (Kepler's 3rd law,\n"
        "     mass–radius relation, stellar luminosity). Naïve oversampling\n"
        "     (SMOTE) ignores these constraints → unphysical planets.\n"
        "   • Only ~60 minority exemplars → high risk of mode collapse.\n"
        "   • Right-skewed features spanning orders of magnitude require\n"
        "     log-transforms for the GAN's internal Gaussian mixtures.\n"
    )
    print(
        "3. SAFEGUARDS AGAINST NONSENSE:\n"
        "   • Log1p transform before CTGAN, expm1 after sampling.\n"
        "   • Hard physics filter (validate_synthetic_data): absolute bounds.\n"
        "   • 1–99% percentile clipping per column.\n"
        "   • Post-hoc habitable envelope matching real HZ range.\n"
        "   • DCR memorization check (synth→real vs real↔real LOO).\n"
        "   • All synthetic data labeled 'exploratory' in the UI.\n"
    )


if __name__ == "__main__":
    main()

