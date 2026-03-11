"""
Plotly visualisation helpers.

* 3-D interactive globe with temperature texture
* 2-D Mollweide-projection heatmap (fallback)
* Analytical eyeball-state temperature map generator
* Host-star scatter point in 3-D scene
* Habitable-zone overlay contours
"""

from typing import Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ─── Scientific colour scale (ice → habitable → scorched) ────────────────────

SCIENCE_COLORSCALE = [
    [0.00, "#1a0533"],   # ultra cold  < 100 K  – deep violet
    [0.15, "#08306b"],   # solid ice   ~ 150 K  – navy
    [0.30, "#2171b5"],   # frozen      ~ 200 K  – blue
    [0.45, "#4292c6"],   # ice edge    ~ 240 K  – light blue
    [0.50, "#41ab5d"],   # freezing pt ~ 273 K  – green
    [0.60, "#78c679"],   # temperate   ~ 290 K  – light green
    [0.70, "#fee08b"],   # warm        ~ 310 K  – yellow
    [0.80, "#fc8d59"],   # hot         ~ 340 K  – orange
    [0.90, "#d73027"],   # very hot    ~ 380 K  – red
    [1.00, "#67001f"],   # extreme     ~ 500 K+ – dark red
]


# ─── Analytical temperature-map generators ────────────────────────────────────

def generate_eyeball_map(
    T_eq: float,
    tidally_locked: bool = True,
    n_lat: int = 64,
    n_lon: int = 128,
) -> np.ndarray:
    """Analytical surface-temperature map.

    For a tidally locked planet the map has an *eyeball* topology:
    a warm substellar region centred at (0, 0) surrounded by ice.
    LON=0 maps to the +X axis in the 3-D globe, i.e. towards the star.
    For a fast rotator a simple latitudinal gradient is used.
    """
    lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    lon = np.linspace(0, 2 * np.pi, n_lon)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")

    if tidally_locked:
        cos_zenith = np.cos(LAT) * np.cos(LON)
        cos_zenith = np.clip(cos_zenith, 0, 1)
        T_sub = T_eq * 1.4
        T_night = max(T_eq * 0.3, 40)
        temp_map = T_night + (T_sub - T_night) * cos_zenith**0.25
    else:
        temp_map = T_eq * (1 + 0.15 * np.cos(LAT))

    return temp_map


# ─── 3-D globe ────────────────────────────────────────────────────────────────

def create_3d_globe(
    temperature_map: np.ndarray,
    planet_name: str = "Exoplanet",
    colorscale: Optional[list] = None,
    show_star: bool = True,
    star_teff: Optional[float] = None,
) -> go.Figure:
    """Render an interactive 3-D sphere coloured by surface temperature."""
    n_lat, n_lon = temperature_map.shape
    theta = np.linspace(0, 2 * np.pi, n_lon)
    phi = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    THETA, PHI = np.meshgrid(theta, phi)

    r = 1.0
    X = r * np.cos(PHI) * np.cos(THETA)
    Y = r * np.cos(PHI) * np.sin(THETA)
    Z = r * np.sin(PHI)

    cs = colorscale or SCIENCE_COLORSCALE
    T_min, T_max = float(temperature_map.min()), float(temperature_map.max())

    # Roll by 180° so the substellar (warm) face aligns with the star marker at -X
    tmap_display = np.roll(temperature_map, temperature_map.shape[1] // 2, axis=1)

    surface = go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=tmap_display,
        colorscale=cs,
        cmin=T_min, cmax=T_max,
        colorbar=dict(
            title=dict(text="Temperature [K]", font=dict(size=14)),
            ticksuffix=" K",
            len=0.75,
            thickness=20,
            x=1.02,
        ),
        hovertemplate=(
            "<b>%{surfacecolor:.1f} K</b><br>"
            "(%{customdata[0]:.1f}\u00b0, %{customdata[1]:.1f}\u00b0)"
            "<extra></extra>"
        ),
        customdata=np.stack(
            [np.degrees(PHI), np.degrees(THETA)], axis=-1
        ),
        lighting=dict(
            ambient=0.4, diffuse=0.6, specular=0.15,
            roughness=0.8, fresnel=0.1,
        ),
        lightposition=dict(x=-1000, y=0, z=0),
    )

    fig = go.Figure(data=[surface])

    # Optional host-star point
    if show_star:
        star_color = _star_color(star_teff) if star_teff else "#ffe066"
        fig.add_trace(
            go.Scatter3d(
                x=[-3.0], y=[0.0], z=[0.0],
                mode="markers",
                marker=dict(size=8, color=star_color, symbol="diamond"),
                name="Host star",
                hovertemplate=(
                    f"Host star<br>T_eff={star_teff or '?'} K<extra></extra>"
                ),
            )
        )

    # Rotation animation frames
    n_frames = 100
    frames = []
    for k in range(n_frames):
        angle = 2 * np.pi * k / n_frames
        ex = 1.5 * np.cos(angle)
        ey = 1.5 * np.sin(angle)
        frames.append(go.Frame(
            layout=dict(scene_camera=dict(
                eye=dict(x=ex, y=ey, z=0.5),
                up=dict(x=0, y=0, z=1),
            )),
            name=str(k),
        ))
    fig.frames = frames

    fig.update_layout(
        title=dict(
            text=f"\U0001fa90 {planet_name} \u2014 Surface Temperature",
            font=dict(size=18),
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.5, y=0.5, z=0.5),
                up=dict(x=0, y=0, z=1),
            ),
            bgcolor="black",
        ),
        paper_bgcolor="black",
        font=dict(color="white"),
        width=800,
        height=700,
        margin=dict(l=0, r=80, t=60, b=0),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.05, y=0.95,
            buttons=[
                dict(
                    label="\u25b6 Rotate",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=80, redraw=True),
                        fromcurrent=True,
                        mode="immediate",
                    )],
                ),
                dict(
                    label="\u23f8 Stop",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                    )],
                ),
            ],
        )],
    )
    return fig


# ─── 2-D heatmap fallback ────────────────────────────────────────────────────

def create_2d_heatmap(
    temperature_map: np.ndarray,
    planet_name: str = "Exoplanet",
) -> go.Figure:
    fig = px.imshow(
        temperature_map,
        labels=dict(
            x="Longitude [\u00b0]",
            y="Latitude [\u00b0]",
            color="T [K]",
        ),
        x=np.linspace(0, 360, temperature_map.shape[1]),
        y=np.linspace(-90, 90, temperature_map.shape[0]),
        color_continuous_scale="RdYlBu_r",
        origin="lower",
        aspect="auto",
    )
    fig.update_layout(
        title=f"{planet_name} \u2014 Temperature Map (2-D)",
        coloraxis_colorbar=dict(title="T [K]"),
        width=800,
        height=400,
        paper_bgcolor="black",
        font=dict(color="white"),
    )
    return fig


# ─── Habitable-zone diagram ──────────────────────────────────────────────────

def create_hz_diagram(
    hz_boundaries: dict,
    planet_semi_major: float,
    star_teff: float,
) -> go.Figure:
    """Top-down habitable-zone diagram with planet position."""
    zones = [
        ("recent_venus", "runaway_gh", "Too hot", "rgba(215,48,39,0.25)"),
        ("runaway_gh", "max_gh", "Habitable Zone", "rgba(65,171,93,0.30)"),
        ("max_gh", "early_mars", "Extended HZ", "rgba(66,146,198,0.20)"),
    ]

    fig = go.Figure()
    for inner_key, outer_key, label, colour in zones:
        inner = hz_boundaries.get(inner_key, 0)
        outer = hz_boundaries.get(outer_key, 0)
        fig.add_shape(
            type="rect", x0=inner, x1=outer, y0=0, y1=1,
            fillcolor=colour, line_width=0, layer="below",
        )
        fig.add_annotation(
            x=(inner + outer) / 2, y=0.5, text=label,
            showarrow=False, font=dict(size=11, color="white"),
        )

    fig.add_trace(go.Scatter(
        x=[planet_semi_major], y=[0.5],
        mode="markers+text",
        marker=dict(size=14, color="#00d4ff", symbol="circle"),
        text=["Planet"], textposition="top center",
        textfont=dict(color="white"),
        name="Planet",
    ))

    fig.update_layout(
        title=f"Habitable Zone (T_eff = {star_teff} K)",
        xaxis=dict(title="Distance [AU]", color="white"),
        yaxis=dict(visible=False, range=[0, 1]),
        paper_bgcolor="black",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=250,
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=False,
    )
    return fig


# ─── helpers ──────────────────────────────────────────────────────────────────

def _star_color(teff: Optional[float]) -> str:
    """Map stellar temperature to approximate RGB colour."""
    if teff is None:
        return "#ffe066"
    if teff < 3500:
        return "#ff4500"   # M-dwarf  – red-orange
    if teff < 5000:
        return "#ffa500"   # K-type   – orange
    if teff < 6000:
        return "#fff44f"   # G-type   – yellow
    if teff < 7500:
        return "#fffbe6"   # F-type   – white-yellow
    return "#caf0f8"       # A/B type – blue-white
