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

_FEATURE_COLS = [
    "pl_radj", "pl_bmassj", "pl_orbsmax", "pl_orbper",
    "pl_insol", "pl_eqt", "st_teff", "st_rad",
]


def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.05,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run Isolation Forest on an exoplanet catalog DataFrame.

    Adds columns ``anomaly_score`` (lower = more anomalous) and
    ``is_anomaly`` (boolean) to the returned DataFrame.
    """
    cols = feature_cols or _FEATURE_COLS
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

    cols = feature_cols or _FEATURE_COLS
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
    """Create a Plotly scatter of the UMAP embedding colored by anomaly status."""
    import plotly.graph_objects as go

    subset = df.iloc[:len(embedding)].copy()

    colors = [
        "#d73027" if a else "#2171b5"
        for a in subset.get(anomaly_col, [False] * len(embedding))
    ]
    names = subset.get("pl_name", pd.Series(["?"] * len(embedding)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=embedding[:, 0], y=embedding[:, 1],
        mode="markers",
        marker=dict(size=5, color=colors, opacity=0.7),
        text=names.values,
        hovertemplate="<b>%{text}</b><extra></extra>",
    ))
    fig.update_layout(
        title="Planet Population (UMAP)",
        xaxis_title="UMAP-1", yaxis_title="UMAP-2",
        paper_bgcolor="black", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=400,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig
