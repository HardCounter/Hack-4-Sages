"""
Diagnostic comparison: PINNFormer 3-D vs analytical eyeball-state map.

Run after training (``python train_models.py --pinn``).  This script:

* Loads (or quickly trains) a PINNFormer3D model.
* Generates a PINN temperature map on a lat/lon grid.
* Generates the analytical eyeball-state map for the same planet.
* Computes RMSE, max-absolute-error, and smoothness metrics.
* Prints comparison verdicts.
* Saves an HTML report with side-by-side and difference maps.
"""

import os
import sys
from typing import Dict

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from modules.visualization import (
    create_2d_heatmap,
    generate_eyeball_map,
    SCIENCE_COLORSCALE,
)

REPO_DIR = ROOT_DIR
MODELS_DIR = os.path.join(REPO_DIR, "models")
DIAG_DIR = os.path.join(REPO_DIR, "diagnostic")
WEIGHTS_PATH = os.path.join(MODELS_DIR, "pinn3d_weights.pt")

N_LAT = 64
N_LON = 128

# Proxima-b-like defaults
T_EQ = 254.0
T_SUB = 320.0
T_NIGHT = 80.0


# ── Metrics ──────────────────────────────────────────────────────────────────

def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _max_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def _smoothness_dtheta(temp_map: np.ndarray, n_lat: int, n_lon: int) -> Dict[str, float]:
    """Approximate ∂T/∂θ variability (finite differences along latitude)."""
    dlat = np.pi / (n_lat - 1)
    dT_dtheta = np.diff(temp_map, axis=0) / dlat
    return {
        "dT_dtheta_mean_abs": float(np.mean(np.abs(dT_dtheta))),
        "dT_dtheta_max_abs": float(np.max(np.abs(dT_dtheta))),
        "dT_dtheta_std": float(np.std(dT_dtheta)),
    }


def _smoothness_dphi(temp_map: np.ndarray, n_lat: int, n_lon: int) -> Dict[str, float]:
    """Approximate ∂T/∂φ variability (finite differences along longitude)."""
    dlon = 2 * np.pi / (n_lon - 1)
    dT_dphi = np.diff(temp_map, axis=1) / dlon
    return {
        "dT_dphi_mean_abs": float(np.mean(np.abs(dT_dphi))),
        "dT_dphi_max_abs": float(np.max(np.abs(dT_dphi))),
        "dT_dphi_std": float(np.std(dT_dphi)),
    }


# ── Verdicts ─────────────────────────────────────────────────────────────────

def _generate_verdicts(
    pinn_map: np.ndarray,
    analytic_map: np.ndarray,
    rmse: float,
    max_err: float,
) -> list[str]:
    """Return human-readable comparison verdicts."""
    verdicts: list[str] = []

    diff = pinn_map - analytic_map

    # Global bias
    mean_diff = float(np.mean(diff))
    if abs(mean_diff) < 2.0:
        verdicts.append(f"Global mean bias is negligible ({mean_diff:+.2f} K).")
    elif mean_diff > 0:
        verdicts.append(f"PINN is globally warmer than analytic by {mean_diff:+.1f} K on average.")
    else:
        verdicts.append(f"PINN is globally cooler than analytic by {mean_diff:+.1f} K on average.")

    # Dayside vs nightside
    mid_lon = analytic_map.shape[1] // 2
    dayside = diff[:, mid_lon // 2 : mid_lon + mid_lon // 2]
    nightside_left = diff[:, : mid_lon // 2]
    nightside_right = diff[:, mid_lon + mid_lon // 2 :]
    nightside = np.concatenate([nightside_left, nightside_right], axis=1)

    day_bias = float(np.mean(dayside))
    night_bias = float(np.mean(nightside))
    if day_bias > 2.0 and night_bias < -2.0:
        verdicts.append(
            f"PINN is hotter on the dayside (+{day_bias:.1f} K) and cooler on "
            f"the nightside ({night_bias:+.1f} K) than analytic."
        )
    elif day_bias > 2.0:
        verdicts.append(f"PINN dayside is warmer than analytic by +{day_bias:.1f} K.")
    elif night_bias < -2.0:
        verdicts.append(f"PINN nightside is cooler than analytic by {night_bias:+.1f} K.")
    else:
        verdicts.append("Dayside/nightside biases are within 2 K of analytic.")

    # RMSE quality
    if rmse < 5.0:
        verdicts.append(f"RMSE = {rmse:.2f} K — excellent agreement.")
    elif rmse < 20.0:
        verdicts.append(f"RMSE = {rmse:.2f} K — good agreement, minor structure differences.")
    elif rmse < 50.0:
        verdicts.append(f"RMSE = {rmse:.2f} K — moderate disagreement; check training convergence.")
    else:
        verdicts.append(f"RMSE = {rmse:.2f} K — large disagreement; model may need more training.")

    # Max error
    if max_err > 100.0:
        verdicts.append(f"Max absolute error = {max_err:.1f} K — localized hotspot or artifact.")

    return verdicts


# ── HTML report ──────────────────────────────────────────────────────────────

def _build_html_report(
    pinn_map: np.ndarray,
    analytic_map: np.ndarray,
    metrics: Dict[str, float],
    verdicts: list[str],
    smooth_pinn: Dict[str, float],
    smooth_analytic: Dict[str, float],
) -> str:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return "<html><body><p>Plotly not installed — skipping report.</p></body></html>"

    sections: list[str] = []

    # ── Side-by-side heatmaps ──
    lon_deg = np.linspace(0, 360, pinn_map.shape[1])
    lat_deg = np.linspace(-90, 90, pinn_map.shape[0])
    vmin = min(float(pinn_map.min()), float(analytic_map.min()))
    vmax = max(float(pinn_map.max()), float(analytic_map.max()))

    fig_pair = make_subplots(
        rows=1, cols=2,
        subplot_titles=["PINNFormer 3-D", "Analytical Eyeball"],
    )
    for i, (tmap, name) in enumerate(
        [(pinn_map, "PINN"), (analytic_map, "Analytic")], start=1
    ):
        fig_pair.add_trace(
            go.Heatmap(
                z=tmap, x=lon_deg, y=lat_deg,
                colorscale="RdYlBu_r", zmin=vmin, zmax=vmax,
                colorbar=dict(title="T [K]") if i == 2 else dict(title=""),
                showscale=(i == 2),
            ),
            row=1, col=i,
        )
    fig_pair.update_layout(
        height=350, width=1050,
        paper_bgcolor="#111", plot_bgcolor="#222", font=dict(color="white"),
        title_text="PINN vs Analytic Temperature Maps",
    )
    sections.append(fig_pair.to_html(full_html=False, include_plotlyjs="cdn"))

    # ── Difference map ──
    diff = pinn_map - analytic_map
    abs_max = max(abs(float(diff.min())), abs(float(diff.max())), 1.0)

    fig_diff = go.Figure(data=go.Heatmap(
        z=diff, x=lon_deg, y=lat_deg,
        colorscale="RdBu_r", zmid=0, zmin=-abs_max, zmax=abs_max,
        colorbar=dict(title="\u0394T [K]"),
    ))
    fig_diff.update_layout(
        title="Difference Map (PINN \u2212 Analytic)",
        xaxis_title="Longitude [\u00b0]",
        yaxis_title="Latitude [\u00b0]",
        height=350, width=700,
        paper_bgcolor="#111", plot_bgcolor="#222", font=dict(color="white"),
    )
    sections.append(fig_diff.to_html(full_html=False, include_plotlyjs=False))

    # ── Metrics table ──
    rows_html = "".join(
        f"<tr><td>{k}</td><td>{v:.4g}</td></tr>" for k, v in metrics.items()
    )
    sections.append(
        "<h2>Quantitative Metrics</h2>"
        f'<table class="m-table"><tr><th>Metric</th><th>Value</th></tr>'
        f"{rows_html}</table>"
    )

    # ── Smoothness comparison ──
    smooth_rows = ""
    for key in smooth_pinn:
        sp = smooth_pinn[key]
        sa = smooth_analytic.get(key, 0.0)
        smooth_rows += f"<tr><td>{key}</td><td>{sp:.4g}</td><td>{sa:.4g}</td></tr>"
    sections.append(
        "<h2>Smoothness Comparison</h2>"
        '<table class="m-table">'
        "<tr><th>Metric</th><th>PINN</th><th>Analytic</th></tr>"
        f"{smooth_rows}</table>"
    )

    # ── Verdicts ──
    verdict_items = "".join(f"<li>{v}</li>" for v in verdicts)
    sections.append(
        "<h2>Comparison Verdicts</h2>"
        f"<ul style='line-height:1.8'>{verdict_items}</ul>"
    )

    body = "\n<hr>\n".join(sections)
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>"
        "body{background:#111;color:#eee;font-family:sans-serif;padding:20px}"
        ".m-table{border-collapse:collapse;margin:10px 0}"
        ".m-table td,.m-table th{padding:6px 14px;border:1px solid #555}"
        "h2{color:#64b5f6;margin-top:32px}"
        "</style></head><body>"
        "<h1>PINNFormer 3-D Diagnostic Report</h1>"
        f"{body}</body></html>"
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(DIAG_DIR, exist_ok=True)

    # --- Load or train the PINN ---
    try:
        import torch
        from modules.pinnformer3d import (
            load_pinnformer,
            predict_temperature_map,
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
        model, history = train_pinnformer(
            epochs=2000, n_colloc=4096, device=device,
            T_sub=T_SUB, T_night=T_NIGHT, log_every=200,
        )
        os.makedirs(MODELS_DIR, exist_ok=True)
        save_pinnformer(model, WEIGHTS_PATH)

    # --- Generate maps ---
    print("Generating PINN temperature map ...")
    pinn_map = predict_temperature_map(
        model, n_lat=N_LAT, n_lon=N_LON, z=0.5, device=device,
    )

    print("Generating analytical eyeball map ...")
    analytic_map = generate_eyeball_map(
        T_eq=T_EQ, tidally_locked=True, n_lat=N_LAT, n_lon=N_LON,
    )

    # --- Metrics ---
    rmse = _rmse(pinn_map, analytic_map)
    max_err = _max_abs_error(pinn_map, analytic_map)

    metrics: Dict[str, float] = {
        "RMSE [K]": rmse,
        "Max absolute error [K]": max_err,
        "PINN T_mean [K]": float(pinn_map.mean()),
        "PINN T_min [K]": float(pinn_map.min()),
        "PINN T_max [K]": float(pinn_map.max()),
        "Analytic T_mean [K]": float(analytic_map.mean()),
        "Analytic T_min [K]": float(analytic_map.min()),
        "Analytic T_max [K]": float(analytic_map.max()),
    }

    smooth_pinn = {
        **_smoothness_dtheta(pinn_map, N_LAT, N_LON),
        **_smoothness_dphi(pinn_map, N_LAT, N_LON),
    }
    smooth_analytic = {
        **_smoothness_dtheta(analytic_map, N_LAT, N_LON),
        **_smoothness_dphi(analytic_map, N_LAT, N_LON),
    }

    verdicts = _generate_verdicts(pinn_map, analytic_map, rmse, max_err)

    # --- Console output ---
    print("\n" + "=" * 72)
    print("  PINNFormer vs Analytical — Quantitative Comparison")
    print("=" * 72)
    for k, v in metrics.items():
        print(f"  {k:30s}: {v:.2f}")
    print()
    print("  Smoothness (PINN):")
    for k, v in smooth_pinn.items():
        print(f"    {k:25s}: {v:.4f}")
    print("  Smoothness (Analytic):")
    for k, v in smooth_analytic.items():
        print(f"    {k:25s}: {v:.4f}")
    print()
    print("  Verdicts:")
    for v in verdicts:
        print(f"    - {v}")

    # --- HTML report ---
    html = _build_html_report(
        pinn_map, analytic_map, metrics, verdicts,
        smooth_pinn, smooth_analytic,
    )
    report_path = os.path.join(DIAG_DIR, "pinnformer_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nHTML report saved to {report_path}")


if __name__ == "__main__":
    main()
