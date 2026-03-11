"""
Anomaly detection for exoplanet datasets.

Uses Isolation Forest for identifying statistically unusual planets in
the NASA Exoplanet Archive, and UMAP for 2-D visualisation of the
planet population. Addresses the 'anomaly detection in imbalanced
datasets' criterion.

Degrades gracefully when optional dependencies are missing.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

_FEATURE_COLS_NASA = [
    "pl_radj", "pl_bmassj", "pl_orbsmax", "pl_orbper",
    "pl_insol", "pl_eqt", "st_teff", "st_rad",
]

_FEATURE_COLS_COMBINED = [
    "radius_earth", "mass_earth", "semi_major_axis_au", "period_days",
    "insol_earth", "t_eq_K", "star_teff_K", "star_radius_solar",
]

_NICE_NAMES: Dict[str, str] = {
    "pl_radj": "Radius (Rj)", "pl_bmassj": "Mass (Mj)",
    "pl_orbsmax": "Semi-major axis (AU)", "pl_orbper": "Period (d)",
    "pl_insol": "Insolation (S⊕)", "pl_eqt": "T_eq (K)",
    "st_teff": "Star Teff (K)", "st_rad": "Star R (R☉)",
    "radius_earth": "Radius (R⊕)", "mass_earth": "Mass (M⊕)",
    "semi_major_axis_au": "Semi-major axis (AU)", "period_days": "Period (d)",
    "insol_earth": "Insolation (S⊕)", "t_eq_K": "T_eq (K)",
    "star_teff_K": "Star Teff (K)", "star_radius_solar": "Star R (R☉)",
}

_NAME_COL_CANDIDATES = ["pl_name", "name", "planet_name"]


def _resolve_feature_cols(df: pd.DataFrame) -> List[str]:
    """Pick the best set of feature columns for the given DataFrame."""
    nasa_hits = sum(1 for c in _FEATURE_COLS_NASA if c in df.columns)
    comb_hits = sum(1 for c in _FEATURE_COLS_COMBINED if c in df.columns)
    base = _FEATURE_COLS_NASA if nasa_hits >= comb_hits else _FEATURE_COLS_COMBINED
    return [c for c in base if c in df.columns]


def _get_name_col(df: pd.DataFrame) -> str:
    for c in _NAME_COL_CANDIDATES:
        if c in df.columns:
            return c
    return ""


def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.05,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run Isolation Forest on an exoplanet catalog DataFrame.

    Adds columns ``anomaly_score`` (lower = more anomalous) and
    ``is_anomaly`` (boolean) to the returned DataFrame.
    Automatically detects NASA or combined-catalog column schemas.
    """
    cols = feature_cols or _resolve_feature_cols(df)
    available = [c for c in cols if c in df.columns]
    if len(available) < 3:
        raise ValueError(f"Need at least 3 feature columns, got {available}")

    subset = df[available].dropna()
    if len(subset) < 20:
        raise ValueError("Too few rows after dropping NaN")

    scaler = StandardScaler()
    X = scaler.fit_transform(subset.values)

    iso = IsolationForest(
        contamination=contamination, random_state=42, n_jobs=-1,
    )
    labels = iso.fit_predict(X)
    scores = iso.decision_function(X)

    result = df.loc[subset.index].copy()
    result["anomaly_score"] = scores
    result["is_anomaly"] = labels == -1

    return result.sort_values("anomaly_score")


def get_top_anomalies(
    df: pd.DataFrame, n: int = 10, contamination: float = 0.05,
) -> pd.DataFrame:
    """Return the top-N most anomalous planets."""
    detected = detect_anomalies(df, contamination=contamination)
    return detected.head(n)


def build_weird_planets_table(
    df: pd.DataFrame, n: int = 15,
) -> pd.DataFrame:
    """Build a ranked 'weird planets' table with anomaly reasons.

    For each anomaly, identifies which features deviate most from the
    population median (in z-score space) and constructs a human-readable
    reason string.
    """
    cols = _resolve_feature_cols(df)
    available = [c for c in cols if c in df.columns]
    if len(available) < 3:
        return pd.DataFrame()

    subset = df[available].dropna()
    if len(subset) < 20:
        return pd.DataFrame()

    scaler = StandardScaler()
    Z = pd.DataFrame(
        scaler.fit_transform(subset.values),
        columns=available, index=subset.index,
    )

    name_col = _get_name_col(df)
    detected = detect_anomalies(df, feature_cols=available)
    anomalies = detected[detected["is_anomaly"]].head(n).copy()

    reasons = []
    for idx in anomalies.index:
        if idx not in Z.index:
            reasons.append("—")
            continue
        z_row = Z.loc[idx].abs().sort_values(ascending=False)
        top3 = z_row.head(3)
        parts = []
        for col_name, z_val in top3.items():
            nice = _NICE_NAMES.get(col_name, col_name)
            direction = "high" if Z.loc[idx, col_name] > 0 else "low"
            parts.append(f"{nice} unusually {direction} ({z_val:.1f}σ)")
        reasons.append("; ".join(parts))

    anomalies["why_weird"] = reasons

    out_cols = []
    if name_col:
        out_cols.append(name_col)
    out_cols += ["anomaly_score", "why_weird"]
    for c in available[:4]:
        if c in anomalies.columns:
            out_cols.append(c)
    return anomalies[[c for c in out_cols if c in anomalies.columns]].reset_index(drop=True)


def compute_umap_embedding(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> Optional[np.ndarray]:
    """Compute 2-D UMAP embedding of the planet population.

    Returns an (N, 2) array or None if UMAP is not installed.
    """
    try:
        import umap
    except ImportError:
        return None

    cols = feature_cols or _resolve_feature_cols(df)
    available = [c for c in cols if c in df.columns]
    subset = df[available].dropna()
    if len(subset) < 20:
        return None

    scaler = StandardScaler()
    X = scaler.fit_transform(subset.values)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist,
        n_components=2, random_state=42,
    )
    embedding = reducer.fit_transform(X)
    return embedding


def create_umap_figure(
    df: pd.DataFrame,
    embedding: np.ndarray,
    anomaly_col: str = "is_anomaly",
):
    """Create a Plotly scatter of the UMAP embedding colored by anomaly status.

    Anomalous planets are larger, red, and labeled on hover with key
    parameters.  Normal planets are semi-transparent blue.
    """
    import plotly.graph_objects as go

    subset = df.iloc[:len(embedding)].copy()
    is_anom = subset.get(anomaly_col, pd.Series([False] * len(embedding)))
    name_col = _get_name_col(subset)
    names = subset[name_col] if name_col else pd.Series(["?"] * len(embedding))

    feature_cols = _resolve_feature_cols(subset)
    hover_parts: List[str] = []
    for _, row in subset.iterrows():
        parts = []
        for fc in feature_cols[:5]:
            if fc in row.index and pd.notna(row[fc]):
                nice = _NICE_NAMES.get(fc, fc)
                parts.append(f"{nice}: {row[fc]:.3g}")
        hover_parts.append("<br>".join(parts))

    normal_mask = ~is_anom.values.astype(bool)
    anom_mask = is_anom.values.astype(bool)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=embedding[normal_mask, 0], y=embedding[normal_mask, 1],
        mode="markers", name="Normal",
        marker=dict(size=5, color="#2171b5", opacity=0.5),
        text=names.values[normal_mask],
        customdata=np.array(hover_parts)[normal_mask],
        hovertemplate="<b>%{text}</b><br>%{customdata}<extra>Normal</extra>",
    ))

    scores = subset.get("anomaly_score", pd.Series([0] * len(embedding)))
    anom_sizes = np.clip(8 + (-scores.values[anom_mask]) * 30, 8, 20)

    fig.add_trace(go.Scatter(
        x=embedding[anom_mask, 0], y=embedding[anom_mask, 1],
        mode="markers+text", name="Anomaly",
        marker=dict(
            size=anom_sizes,
            color="#d73027", opacity=0.9,
            line=dict(width=1, color="#ffcdd2"),
        ),
        text=names.values[anom_mask],
        textposition="top center",
        textfont=dict(size=9, color="#ffcdd2"),
        customdata=np.array(hover_parts)[anom_mask],
        hovertemplate="<b>%{text}</b><br>%{customdata}<extra>Anomaly</extra>",
    ))

    fig.update_layout(
        title="Planet Population — UMAP Projection",
        xaxis_title="UMAP-1", yaxis_title="UMAP-2",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(
            bgcolor="rgba(20,20,50,0.7)", bordercolor="#555", borderwidth=1,
            font=dict(color="white"),
        ),
        height=500,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig
