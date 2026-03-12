"""
Diagnostic comparison: PINNFormer 3-D vs NASA GCM benchmark cases.

Run after training (``python train_models.py --pinn``).  This script:

* Loads (or quickly trains) a PINNFormer3D model.
* Generates PINN temperature maps on a lat/lon grid.
* Loads synthetic GCM reference maps (Turbet 2016 / Leconte 2013).
* Computes pattern correlation, RMSE, bias, zonal-mean RMSE.
* Renders side-by-side heatmaps, difference maps, and zonal-mean profiles.
* Saves an interactive HTML report.

GCM references
--------------
- Proxima-b: Turbet et al. (2016), LMD Generic GCM, tidally locked.
- Hot rock:  Leconte et al. (2013), hot synchronous planet GCM.
- (earth_like omitted — PINNFormer is designed for tidally locked worlds.)
"""

import os
import sys
from typing import Dict, List

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from modules.gcm_benchmarks import (
    compare_surrogate_to_gcm,
    compute_zonal_mean,
    get_gcm_benchmark,
)
from modules.visualization import SCIENCE_COLORSCALE

REPO_DIR = ROOT_DIR
MODELS_DIR = os.path.join(REPO_DIR, "models")
DIAG_DIR = os.path.join(REPO_DIR, "diagnostic")
WEIGHTS_PATH = os.path.join(MODELS_DIR, "pinn3d_weights.pt")

N_LAT = 32
N_LON = 64

CASES = ["proxima_b", "hot_rock"]

CASE_LABELS = {
    "proxima_b": "Proxima Cen b — LMD Generic GCM (Turbet et al. 2016)",
    "hot_rock": "Hot synchronous rock — GCM (Leconte et al. 2013)",
}


# ── HTML builder ─────────────────────────────────────────────────────────────

def _build_case_html(
    case_key: str,
    pinn_map: np.ndarray,
    gcm_map: np.ndarray,
    metrics: Dict[str, float],
) -> str:
    """Build HTML section for one benchmark case."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return f"<h2>{case_key}</h2><p>Plotly not installed.</p>"

    label = CASE_LABELS.get(case_key, case_key)
    lat_deg = np.linspace(-90, 90, N_LAT)
    lon_deg = np.linspace(-180, 180, N_LON)
    vmin = min(float(pinn_map.min()), float(gcm_map.min()))
    vmax = max(float(pinn_map.max()), float(gcm_map.max()))

    parts: List[str] = [f"<h2>{label}</h2>"]

    # Side-by-side temperature maps
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["PINNFormer 3-D", "GCM Reference"],
        horizontal_spacing=0.08,
    )
    for i, (tmap, name) in enumerate(
        [(pinn_map, "PINN"), (gcm_map, "GCM")], start=1
    ):
        fig.add_trace(
            go.Heatmap(
                z=tmap, x=lon_deg, y=lat_deg,
                colorscale="RdYlBu_r", zmin=vmin, zmax=vmax,
                colorbar=dict(title="T [K]") if i == 2 else dict(title=""),
                showscale=(i == 2),
                name=name,
            ),
            row=1, col=i,
        )
        fig.update_xaxes(title_text="Longitude [°]", row=1, col=i)
        fig.update_yaxes(title_text="Latitude [°]", row=1, col=i)
    fig.update_layout(
        height=380, width=1080,
        paper_bgcolor="#111", plot_bgcolor="#222", font=dict(color="white"),
        title_text=f"Temperature Maps — {case_key}",
    )
    parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    # Difference map
    diff = pinn_map - gcm_map
    abs_max = max(abs(float(diff.min())), abs(float(diff.max())), 1.0)
    fig_diff = go.Figure(data=go.Heatmap(
        z=diff, x=lon_deg, y=lat_deg,
        colorscale="RdBu_r", zmid=0, zmin=-abs_max, zmax=abs_max,
        colorbar=dict(title="\u0394T [K]"),
    ))
    fig_diff.update_layout(
        title=f"Difference (PINN \u2212 GCM) — {case_key}",
        xaxis_title="Longitude [°]", yaxis_title="Latitude [°]",
        height=350, width=700,
        paper_bgcolor="#111", plot_bgcolor="#222", font=dict(color="white"),
    )
    parts.append(fig_diff.to_html(full_html=False, include_plotlyjs=False))

    # Zonal-mean profile
    pinn_zonal = compute_zonal_mean(pinn_map)
    gcm_zonal = compute_zonal_mean(gcm_map)
    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(
        x=lat_deg, y=pinn_zonal, mode="lines",
        name="PINNFormer", line=dict(color="#ff7043", width=2),
    ))
    fig_z.add_trace(go.Scatter(
        x=lat_deg, y=gcm_zonal, mode="lines",
        name="GCM", line=dict(color="#42a5f5", width=2, dash="dash"),
    ))
    fig_z.update_layout(
        title=f"Zonal-Mean Temperature — {case_key}",
        xaxis_title="Latitude [°]", yaxis_title="Temperature [K]",
        height=350, width=700,
        paper_bgcolor="#111", plot_bgcolor="#222", font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    parts.append(fig_z.to_html(full_html=False, include_plotlyjs=False))

    # Metrics table
    rows_html = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in metrics.items()
    )
    parts.append(
        '<table class="m-table">'
        "<tr><th>Metric</th><th>Value</th></tr>"
        f"{rows_html}</table>"
    )

    # Verdict
    corr = metrics.get("pattern_correlation", 0)
    rmse = metrics.get("rmse_K", 999)
    bias = metrics.get("bias_K", 0)
    verdicts: List[str] = []
    if corr > 0.85:
        verdicts.append(f"Pattern correlation {corr:.3f} — strong spatial agreement with GCM.")
    elif corr > 0.6:
        verdicts.append(f"Pattern correlation {corr:.3f} — moderate spatial agreement.")
    else:
        verdicts.append(f"Pattern correlation {corr:.3f} — weak agreement; PINN may need more training or physics modules.")
    if rmse < 20:
        verdicts.append(f"RMSE {rmse:.1f} K — excellent quantitative match.")
    elif rmse < 50:
        verdicts.append(f"RMSE {rmse:.1f} K — reasonable; differences likely from missing physics.")
    else:
        verdicts.append(f"RMSE {rmse:.1f} K — significant gap; consider full-physics mode or longer training.")
    if abs(bias) > 10:
        direction = "warmer" if bias > 0 else "cooler"
        verdicts.append(f"Mean bias {bias:+.1f} K — PINN is systematically {direction} than GCM.")

    verdict_items = "".join(f"<li>{v}</li>" for v in verdicts)
    parts.append(f"<ul style='line-height:1.8'>{verdict_items}</ul>")

    return "\n".join(parts)


def _build_summary_table(all_metrics: Dict[str, Dict[str, float]]) -> str:
    """Cross-case summary table."""
    header = (
        "<tr><th>Case</th><th>Pattern Corr.</th><th>RMSE [K]</th>"
        "<th>Bias [K]</th><th>Zonal RMSE [K]</th>"
        "<th>PINN T range</th><th>GCM T range</th></tr>"
    )
    rows = ""
    for key, m in all_metrics.items():
        pinn_range = f"{m.get('pinn_T_min', '?')}–{m.get('pinn_T_max', '?')} K"
        gcm_range = f"{m.get('gcm_T_min', '?')}–{m.get('gcm_T_max', '?')} K"
        rows += (
            f"<tr><td>{key}</td>"
            f"<td>{m['pattern_correlation']:.4f}</td>"
            f"<td>{m['rmse_K']:.1f}</td>"
            f"<td>{m['bias_K']:+.1f}</td>"
            f"<td>{m['zonal_mean_rmse_K']:.1f}</td>"
            f"<td>{pinn_range}</td>"
            f"<td>{gcm_range}</td></tr>"
        )
    return (
        "<h2>Cross-Case Summary</h2>"
        f'<table class="m-table">{header}{rows}</table>'
    )


def build_full_report(
    pinn_maps: Dict[str, np.ndarray],
    gcm_maps: Dict[str, np.ndarray],
    all_metrics: Dict[str, Dict[str, float]],
) -> str:
    sections = [_build_summary_table(all_metrics)]
    for key in CASES:
        if key in pinn_maps:
            sections.append("<hr>")
            sections.append(
                _build_case_html(key, pinn_maps[key], gcm_maps[key], all_metrics[key])
            )

    body = "\n".join(sections)
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>"
        "body{background:#111;color:#eee;font-family:sans-serif;padding:20px;max-width:1200px;margin:auto}"
        ".m-table{border-collapse:collapse;margin:10px 0;width:100%}"
        ".m-table td,.m-table th{padding:8px 14px;border:1px solid #555;text-align:center}"
        ".m-table th{background:#1a237e;color:#e8eaf6}"
        "h1{color:#90caf9} h2{color:#64b5f6;margin-top:32px}"
        "hr{border:1px solid #333;margin:40px 0}"
        "ul{color:#b0bec5}"
        "</style></head><body>"
        "<h1>PINNFormer 3-D vs GCM Benchmarks</h1>"
        "<p style='color:#78909c'>Synthetic GCM reference maps from published 3-D "
        "general circulation models. PINNFormer is a physics-informed neural network "
        "surrogate — not calibrated to GCM output — so differences reflect both "
        "missing physics and the approximation gap.</p>"
        f"{body}</body></html>"
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(DIAG_DIR, exist_ok=True)

    try:
        import torch
        from modules.pinnformer3d import (
            load_pinnformer,
            sample_surface_map,
            train_pinnformer,
            save_pinnformer,
        )
    except ImportError:
        print("[ERROR] PyTorch is required for PINNFormer diagnostics.")
        return

    if os.path.exists(WEIGHTS_PATH):
        print(f"Loading PINNFormer weights from {WEIGHTS_PATH} ...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_pinnformer(WEIGHTS_PATH, device=device)
    else:
        print(
            f"[WARN] Weights not found at {WEIGHTS_PATH}.\n"
            "Training a quick model (2 000 epochs) for diagnostics only..."
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = train_pinnformer(
            epochs=2000, n_colloc=4096, device=device,
            T_sub=320.0, T_night=80.0, log_every=500,
        )
        os.makedirs(MODELS_DIR, exist_ok=True)
        save_pinnformer(model, WEIGHTS_PATH)

    pinn_maps: Dict[str, np.ndarray] = {}
    gcm_maps: Dict[str, np.ndarray] = {}
    all_metrics: Dict[str, Dict[str, float]] = {}

    for case_key in CASES:
        print(f"\n{'─'*60}")
        print(f"  Benchmark: {case_key}")
        print(f"{'─'*60}")

        bench = get_gcm_benchmark(case_key)
        if bench is None:
            print(f"  [SKIP] Unknown case '{case_key}'")
            continue

        gcm_map = bench["temperature_map"]
        print(f"  GCM   : T = {gcm_map.min():.1f} – {gcm_map.max():.1f} K  "
              f"(mean {gcm_map.mean():.1f} K)")

        pinn_map = sample_surface_map(model, n_lat=N_LAT, n_lon=N_LON, device=device)
        print(f"  PINN  : T = {pinn_map.min():.1f} – {pinn_map.max():.1f} K  "
              f"(mean {pinn_map.mean():.1f} K)")

        metrics = compare_surrogate_to_gcm(pinn_map, gcm_map)
        metrics["pinn_T_mean"] = round(float(pinn_map.mean()), 2)
        metrics["pinn_T_min"] = round(float(pinn_map.min()), 2)
        metrics["pinn_T_max"] = round(float(pinn_map.max()), 2)
        metrics["gcm_T_mean"] = round(float(gcm_map.mean()), 2)
        metrics["gcm_T_min"] = round(float(gcm_map.min()), 2)
        metrics["gcm_T_max"] = round(float(gcm_map.max()), 2)

        print(f"  Correlation: {metrics['pattern_correlation']:.4f}")
        print(f"  RMSE       : {metrics['rmse_K']:.2f} K")
        print(f"  Bias       : {metrics['bias_K']:+.2f} K")
        print(f"  Zonal RMSE : {metrics['zonal_mean_rmse_K']:.2f} K")

        pinn_maps[case_key] = pinn_map
        gcm_maps[case_key] = gcm_map
        all_metrics[case_key] = metrics

    html = build_full_report(pinn_maps, gcm_maps, all_metrics)
    report_path = os.path.join(DIAG_DIR, "pinn_vs_gcm_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n{'='*60}")
    print(f"  HTML report saved to {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
