"""
Autonomous Exoplanetary Digital Twin — main Streamlit application.

Tabs
----
1. Agent AI          - LLM chat with transparent reasoning
2. Manual Mode       - slider-driven simulation with live 3-D globe
3. Planet Catalog    - NASA archive browser + famous-planet gallery
4. Science Dashboard - HZ diagram, atmospheric cross-section, uncertainty
5. System            - self-diagnostics, health, export
"""

import io
import json
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import  streamlit as st

from modules.llm_helpers import sanitize_latex

# ─── Page config (must be first Streamlit call) ──────────────────────────────

st.set_page_config(
    page_title="Exoplanetary Digital Twin",
    page_icon="ET",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"About": "Autonomous Exoplanetary Digital Twin — Hack4Sages 2026"},
)

# ─── Cosmic dark-theme CSS ───────────────────────────────────────────────────

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Orbitron:wght@400;500;600;700&display=swap');
:root{
  --gap-xs:.4rem;--gap-sm:.75rem;--gap-md:1.25rem;--gap-lg:2rem;
  --radius-card:12px;
  --card-bg:rgba(10,14,39,.7);
  --border-subtle:1px solid rgba(66,165,245,.2);
  --accent:#00d4ff;--accent-dim:#1a237e;
  --text-primary:#e0e0e0;--text-muted:#90a4ae;
}
.stApp{background:radial-gradient(ellipse at center,#0a0e27 0%,#000 100%)}
/* ── Hide sidebar completely ── */
[data-testid="stSidebar"]{display:none!important}
section[data-testid="stSidebarCollapsedControl"]{display:none!important}
h1,h2,h3{font-family:'Orbitron',sans-serif!important;color:#e0e0e0!important}
.stMetricValue{color:#00d4ff!important;font-family:'Orbitron',sans-serif!important}
.stMetricLabel{color:#90a4ae!important}
[data-testid="stMetric"]{background:var(--card-bg);border:var(--border-subtle);border-radius:var(--radius-card);padding:var(--gap-sm) var(--gap-md)!important}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#1a237e 0%,#0d47a1 100%);border:1px solid #42a5f5;color:#fff;font-family:'Space Grotesk',sans-serif}
.stButton>button[kind="primary"]:hover{background:linear-gradient(135deg,#283593 0%,#1565c0 100%);border-color:#64b5f6}
/* ── Redesigned tab styling ── */
.stTabs [data-baseweb="tab-list"]{
  gap:12px;
  background:rgba(10,14,39,.6);
  border:1px solid rgba(66,165,245,.2);
  border-radius:12px;
  padding:6px;
  justify-content:center;
}
.stTabs [data-baseweb="tab"]{
  font-family:'Space Grotesk',sans-serif;
  color:#90a4ae;
  background:transparent;
  border:1px solid transparent;
  border-radius:8px;
  padding:8px 24px!important;
  transition:all .2s ease;
  font-weight:500;
  letter-spacing:.03em;
}
.stTabs [data-baseweb="tab"]:hover{
  background:rgba(0,212,255,.06);
  border-color:rgba(0,212,255,.2);
  color:#b0bec5;
}
.stTabs [aria-selected="true"]{
  color:#00d4ff!important;
  background:rgba(0,212,255,.15)!important;
  border:1px solid rgba(0,212,255,.4)!important;
  border-radius:8px!important;
  padding:8px 24px!important;
  font-weight:600;
  box-shadow:0 0 12px rgba(0,212,255,.15);
}
.stTabs [data-baseweb="tab-highlight"]{display:none!important}
.stTabs [data-baseweb="tab-border"]{display:none!important}
div[data-testid="stChatMessage"]{background:rgba(10,14,39,.6);border:1px solid #1a237e;border-radius:12px;max-width:100%;word-wrap:break-word;margin-bottom:var(--gap-sm)}
.tooltip-term{border-bottom:1px dotted #888;cursor:help;color:#b0bec5}
[data-testid="stExpanderDetails"]{background:var(--card-bg);border-radius:0 0 var(--radius-card) var(--radius-card);padding:var(--gap-md)}
div[data-testid="stAlert"]{border-radius:var(--radius-card)!important;margin-bottom:var(--gap-sm)}
[data-testid="stSlider"] label,[data-testid="stSlider"] span{color:var(--text-primary)!important}
hr{border-color:rgba(66,165,245,.15)!important;margin:var(--gap-lg) 0}
[data-testid="stProgressBar"]>div>div{background:var(--accent)!important}
.js-plotly-plot,.plot-container{max-width:100%!important}
/* ── Science tab grid cards ── */
.science-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:1.25rem;margin-top:1rem}
.science-card{
  background:rgba(10,14,39,.7);
  border:1px solid rgba(66,165,245,.2);
  border-radius:12px;
  padding:1.25rem;
  min-height:120px;
}
.science-card-full{grid-column:1/-1}
@media(max-width:768px){
  .science-grid{grid-template-columns:1fr}
}
/* ── About button in header ── */
.header-bar{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:.5rem}
.about-btn{
  background:rgba(0,212,255,.1);border:1px solid rgba(0,212,255,.3);
  color:#00d4ff;border-radius:8px;padding:6px 18px;
  font-family:'Space Grotesk',sans-serif;font-size:.85rem;
  cursor:pointer;transition:all .2s;text-decoration:none;
}
.about-btn:hover{background:rgba(0,212,255,.2);border-color:#00d4ff}
@media(max-width:640px){
  .stButton>button{width:100%!important}
  [data-testid="column"]{min-width:100%!important;flex:unset!important}
  .stTabs [data-baseweb="tab"]{font-size:.75rem;padding:6px 10px!important}
}
/* ── Slider thumb → hexagon ── */
[data-testid="stSlider"] div[role="slider"]{
  clip-path:polygon(25% 0%,75% 0%,100% 50%,75% 100%,25% 100%,0% 50%)!important;
  border-radius:0!important;
  background:#ff4b4b!important;
  width:18px!important;
  height:18px!important;
}
</style>""",
    unsafe_allow_html=True,
)

# ─── Tooltip helper ──────────────────────────────────────────────────────────

_GLOSSARY = {
    "ESI": "Earth Similarity Index — scale 0-1 measuring similarity to Earth (1.0 = identical).",
    "T_eq": "Equilibrium temperature — theoretical surface temperature assuming radiative balance [K].",
    "Albedo": "Bond albedo — fraction of incident stellar radiation reflected by the planet (0-1).",
    "Tidal locking": "Synchronous rotation where one hemisphere permanently faces the star.",
    "HZ": "Habitable Zone — orbital region where liquid water can exist on the surface.",
    "SEPHI": "Surface Exoplanetary Planetary Habitability Index — multi-criteria habitability score.",
    "HSF": "Habitable Surface Fraction — % of the surface with 273-373 K (liquid water).",
    "Fulton Gap": "Scarcity of planets between 1.5\u20132.0 R\u2295 separating rocky super-Earths from sub-Neptunes.",
    "C/O Ratio": "Carbon-to-Oxygen ratio controlling mineralogy; solar \u2248 0.55, C/O \u2265 1 \u2192 carbon planet.",
    "Sulfur Chemistry": "Dominant sulfur species (H2S, SO2, H2SO4) determined by temperature, pressure, and redox state.",
}


def tip(term: str) -> str:
    expl = _GLOSSARY.get(term, "")
    return f'<span class="tooltip-term" title="{expl}">{term}</span>'


# ─── Chart height token ──────────────────────────────────────────────────────

_CHART_H = 320  # px — consistent height for all science charts

# ─── Status icons — change these to swap symbols everywhere ─────────────────
_ICON_YES  = '<span style="color:#26a641">⬢</span>'  # U+2B22  — positive / pass (green)
_ICON_NO   = '<span style="color:#e05252">⬡</span>'  # U+2B21  — negative / fail (red)
_ICON_WARN = '<span style="color:#ff4b4b">⬢</span>'  # U+2B22  — warning (slider red)

# ─── Cached loaders ──────────────────────────────────────────────────────────

@st.cache_resource
def _load_agent(mode: str = "dual_llm"):
    from modules.agent_setup import AgentMode, build_agent
    agent_mode = AgentMode(mode)
    return build_agent(agent_mode)


@st.cache_resource
def _load_elm():
    from modules.elm_surrogate import ELMClimateSurrogate
    m = ELMClimateSurrogate()
    try:
        m.load("models/elm_ensemble.pkl")
    except FileNotFoundError:
        pass
    return m


@st.cache_resource
def _load_pinn():
    """Load trained PINNFormer 3-D weights. Returns (model, device) or (None, device)."""
    try:
        import torch
        from modules.pinnformer3d import load_pinnformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_pinnformer("models/pinn3d_weights.pt", device=device)
        return model, device
    except (FileNotFoundError, ImportError, Exception):
        return None, "cpu"


# ─── Session-state defaults ──────────────────────────────────────────────────

for k, v in {
    "chat_history": [],
    "current_planet_data": None,
    "temperature_map": None,
    "analysis_history": [],
    "selected_planet": None,
    "llm_mode": "dual_llm",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Header with About button ─────────────────────────────────────────────

st.markdown(
    '<div class="header-bar">'
    '<div>'
    '<h1 style="font-family:\'Orbitron\',sans-serif;'
    'font-size:clamp(1.4rem,3vw,2.2rem);letter-spacing:.04em;'
    'color:#e0e0e0;margin:0">Autonomous Exoplanetary Digital Twin</h1>'
    '<p style="color:#90a4ae;font-size:clamp(.85rem,1.5vw,1rem);'
    'margin-top:.25rem;margin-bottom:0">'
    'GCM-inspired climate surrogate explorer for exoplanets.</p>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown('<div style="margin-bottom:1rem"></div>', unsafe_allow_html=True)
# ─── Tabs ────────────────────────────────────────────────────────────────────

tab_agent, tab_manual, tab_catalog, tab_science, tab_system = st.tabs(
    [
        "Agent AI",
        "Manual Mode",
        "Catalog",
        "Science",
        "System",
    ]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Agent AI with transparent reasoning
# ═══════════════════════════════════════════════════════════════════════════════

with tab_agent:
    col_chat, col_reasoning = st.columns([3, 1])

    with col_chat:
        st.subheader("Conversation with AstroAgent")

        if st.session_state["llm_mode"] == "deterministic":
            st.warning(
                "Running in **Deterministic Tools Only** mode — "
                "no LLM agent available. Switch to Single-LLM or "
                "Dual-LLM mode in the System tab to enable the agent."
            )

        audience = st.radio(
            "Explanation depth",
            ["Scientist", "Outreach"],
            horizontal=True,
            index=0,
        )

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(sanitize_latex(msg["content"]))

        if prompt := st.chat_input("Ask about an exoplanet\u2026"):
            audience_hint = {
                "Scientist": " (respond at expert level, cite equations)",
                "Outreach": " (explain simply, use analogies and vivid language)",
            }
            full_prompt = prompt + audience_hint.get(audience, "")

            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Agent reasoning\u2026"):
                    try:
                        from langchain_core.messages import AIMessage, HumanMessage
                        lc_history = []
                        for m in st.session_state.chat_history[:-1]:
                            cls = HumanMessage if m["role"] == "user" else AIMessage
                            lc_history.append(cls(content=m["content"]))

                        agent = _load_agent(st.session_state["llm_mode"])
                        if agent is None:
                            answer = ("Agent unavailable in Deterministic mode. "
                                      "Switch to Single-LLM or Dual-LLM in the System tab.")
                            steps = []
                        else:
                            response = agent.invoke(
                                {"input": full_prompt, "chat_history": lc_history}
                            )
                            answer = response["output"]
                            steps = response.get("intermediate_steps", [])
                    except Exception as exc:
                        answer = f"Agent unavailable: {exc}"
                        steps = []
                    st.markdown(sanitize_latex(answer))

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.session_state["_last_agent_steps"] = steps

            # Proactive suggestions (LLM-generated via Qwen)
            st.markdown("---")
            st.markdown("**Suggested next steps:**")
            try:
                from modules.llm_helpers import generate_smart_suggestions
                convo_summary = "\n".join(
                    f"{m['role']}: {m['content'][:200]}"
                    for m in st.session_state.chat_history[-6:]
                )
                suggestions = generate_smart_suggestions(convo_summary)
            except Exception:
                suggestions = [
                    "Run a climate simulation for this planet",
                    "Compare with Earth",
                    "Find similar planets in the catalog",
                ]
            s_cols = st.columns(3)
            for c, s in zip(s_cols, suggestions):
                with c:
                    if st.button(s, key=f"sug_{hash(s)}"):
                        st.session_state.chat_history.append({"role": "user", "content": s})
                        st.rerun()

    # Transparent reasoning panel (enhancement B2)
    with col_reasoning:
        st.subheader("Reasoning Chain")
        _steps = st.session_state.get("_last_agent_steps", [])
        if _steps:
            for i, (action, obs) in enumerate(_steps):
                is_expert = action.tool == "consult_domain_expert"
                if is_expert:
                    st.markdown(
                        '<div style="background:linear-gradient(135deg,#1a237e,#4a148c);'
                        'padding:12px;border-radius:8px;border:1px solid #7c4dff;'
                        'margin-bottom:8px">'
                        '<strong style="color:#b388ff">'
                        'Expert Opinion</strong></div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(sanitize_latex(obs[:800]))
                else:
                    with st.expander(f"Step {i+1}: {action.tool}", expanded=(i == 0)):
                        st.markdown(f"**Tool:** `{action.tool}`")
                        st.markdown(f"**Input:** `{action.tool_input}`")
                        st.markdown(f"**Output:** {sanitize_latex(obs[:500])}")
        else:
            st.info("Reasoning steps appear here after each agent response.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Manual Mode with live "What If" globe
# ═══════════════════════════════════════════════════════════════════════════════

with tab_manual:
    col_params, col_viz = st.columns([1, 2], gap="large")

    with col_params:
        with st.container(border=True):
            st.markdown("##### Planet Parameters")
            star_teff = st.slider("Stellar temperature [K]", 2500, 7500, 3042, 50,
                help="Host star effective temperature in Kelvin")
            star_radius = st.slider("Stellar radius [R\u2609]", 0.08, 3.0, 0.141, 0.01,
                help="Host star radius in solar radii")
            planet_radius = st.slider("Planet radius [R\u2295]", 0.5, 2.5, 1.07, 0.01,
                help="Planet radius in Earth radii")
            planet_mass = st.slider("Planet mass [M\u2295]", 0.1, 15.0, 1.27, 0.1,
                help="Planet mass in Earth masses")
            semi_major = st.slider(
                "Semi-major axis [AU]", 0.01, 2.0, 0.0485, 0.001, format="%.4f",
                help="Orbital semi-major axis in astronomical units",
            )
            eccentricity = st.slider("Eccentricity", 0.0, 0.9, 0.0, 0.01,
                help="Orbital eccentricity (0 = circular)")
            _SURFACE_LABELS = {
                "Mixed rocky": "mixed_rocky",
                "Ocean world": "ocean",
                "Desert": "desert",
                "Ice": "ice",
            }
            surface_type = _SURFACE_LABELS[st.selectbox("Surface type",
                list(_SURFACE_LABELS.keys()),
                help="Surface class for albedo estimation")]
            _ATMO_ALBEDO_LABELS = {
                "Thin": "thin",
                "Temperate": "temperate",
                "Thick/Cloudy": "thick_cloudy",
            }
            atmo_albedo_type = _ATMO_ALBEDO_LABELS[st.selectbox("Atmosphere (albedo)",
                list(_ATMO_ALBEDO_LABELS.keys()), index=1,
                help="Atmosphere class for albedo estimation")]
            from modules.astro_physics import estimate_albedo
            _albedo_est = estimate_albedo(surface_type, atmo_albedo_type)
            albedo = st.slider("Bond albedo (override)", 0.0, 1.0,
                _albedo_est["albedo"], 0.01,
                help=f"Estimated: {_albedo_est['albedo']:.2f} ± {_albedo_est['albedo_uncertainty']:.2f}")
            locked = st.checkbox("Tidally locked", True)
            co_ratio = st.slider("C/O Ratio", 0.1, 1.5, 0.55, 0.01,
                help="Carbon-to-Oxygen ratio. Solar \u2248 0.55")
            surface_pressure = st.slider("Surface Pressure (bar)", 0.001, 100.0, 1.0, 0.1,
                help="Surface atmospheric pressure in bar")
            _ATM_LABELS = {
                "H₂-rich (Reducing)": "h2_rich",
                "O₂-rich (Oxidising)": "o2_rich",
                "CH₄/CO₂ (Anoxic)":   "ch4_co2",
            }
            atm_type = _ATM_LABELS[st.selectbox("Atmosphere Type",
                list(_ATM_LABELS.keys()),
                help="Dominant atmospheric regime for sulfur chemistry")]

        st.markdown("##### Climate Model")
        _MODEL_LABELS = ["ELM Ensemble", "PINNFormer 3-D", "Analytical"]
        climate_model = st.radio(
            "Model", _MODEL_LABELS, horizontal=True, index=0,
            help=(
                "ELM — fast data-driven surrogate (parameterised per planet). "
                "PINNFormer — physics-informed neural network (solves the heat PDE). "
                "Analytical — algebraic cos^1/4 profile (always available)."
            ),
        )

        btn_col, toggle_col = st.columns([3, 2], gap="small")
        with btn_col:
            run_sim = st.button("Run Simulation", type="primary", use_container_width=True)
        with toggle_col:
            live_mode = st.toggle('Live "What If" mode', value=False)

    # Debounce guard — only re-run in live mode when a parameter actually changed
    _prev = st.session_state.get("_last_params", {})
    _curr = dict(
        star_teff=star_teff, star_radius=star_radius, planet_radius=planet_radius,
        planet_mass=planet_mass, semi_major=semi_major, albedo=albedo, locked=locked,
        eccentricity=eccentricity,
        co_ratio=co_ratio, surface_pressure=surface_pressure, atm_type=atm_type,
        climate_model=climate_model,
    )
    _params_changed = (_curr != _prev)
    should_compute = run_sim or (live_mode and _params_changed)
    if should_compute:
        st.session_state["_last_params"] = _curr

    if should_compute:
        try:
            with st.status("Running simulation pipeline\u2026", expanded=True) as _pipeline:
                from modules.astro_physics import (
                    assess_co_ratio,
                    assess_sulfur_chemistry,
                    classify_radius_gap,
                    compute_esi,
                    compute_sephi,
                    equilibrium_temperature,
                    estimate_density,
                    estimate_escape_velocity,
                    habitable_surface_fraction,
                    hz_boundaries,
                    stellar_flux,
                )
                from modules.degradation import GracefulDegradation
                from modules.visualization import generate_eyeball_map
                from modules.validators import PlanetaryParameters, SimulationOutput

                _pipeline.write("Validating parameters\u2026")
                PlanetaryParameters(
                    name="Custom",
                    radius_earth=planet_radius,
                    mass_earth=planet_mass,
                    semi_major_axis=semi_major,
                    eccentricity=eccentricity,
                    albedo=albedo,
                    tidally_locked=locked,
                )

                _pipeline.write("Computing habitability indices\u2026")
                T_eq = equilibrium_temperature(
                    star_teff, star_radius, semi_major, albedo, locked,
                    eccentricity=eccentricity,
                )
                S_abs, S_norm = stellar_flux(
                    star_teff, star_radius, semi_major, eccentricity=eccentricity,
                )
                density = estimate_density(planet_mass, planet_radius)
                v_esc = estimate_escape_velocity(planet_mass, planet_radius)
                esi = compute_esi(planet_radius, density, v_esc, T_eq)
                sephi = compute_sephi(T_eq, planet_mass, planet_radius)

                _pipeline.write("Classifying radius gap, sulfur chemistry, C/O ratio\u2026")
                rg     = classify_radius_gap(planet_radius)
                sulfur = assess_sulfur_chemistry(T_eq, surface_pressure, atm_type)
                co     = assess_co_ratio(co_ratio)

                SimulationOutput(T_eq_K=T_eq, ESI=esi, flux_earth=S_norm)

                _pipeline.write(f"Generating climate map ({climate_model})\u2026")
                gd = GracefulDegradation()
                _climate_method = "Analytical Fallback"

                _elm_params = {
                    "radius_earth": planet_radius,
                    "mass_earth": planet_mass,
                    "semi_major_axis_au": semi_major,
                    "star_teff_K": star_teff,
                    "star_radius_solar": star_radius,
                    "insol_earth": S_norm,
                    "albedo": albedo,
                    "tidally_locked": int(locked),
                }

                def _elm_predict():
                    elm = _load_elm()
                    if not elm.models:
                        raise FileNotFoundError("ELM not trained")
                    return elm.predict_from_params(_elm_params)

                def _pinn_predict():
                    pinn_model, pinn_device = _load_pinn()
                    if pinn_model is None:
                        raise FileNotFoundError("PINNFormer not trained")
                    from modules.pinnformer3d import predict_temperature_map
                    raw = predict_temperature_map(
                        pinn_model, n_lat=64, n_lon=128, z=0.5,
                        device=pinn_device,
                    )
                    T_sub_target = T_eq * 1.4
                    T_night_target = max(T_eq * 0.3, 40)
                    pinn_min, pinn_max = float(raw.min()), float(raw.max())
                    span = pinn_max - pinn_min
                    if span < 1e-3:
                        return np.full_like(raw, T_eq)
                    return T_night_target + (raw - pinn_min) / span * (
                        T_sub_target - T_night_target
                    )

                def _pinn_predict():
                    from modules.pinnformer3d import (
                        load_pinnformer,
                        sample_surface_map,
                        sample_cloud_map,
                        sample_ice_map,
                        sample_ocean_map,
                    )

                    pinn = load_pinnformer()
                    t_map = sample_surface_map(pinn, T_eq, tidally_locked=locked)
                    n_lat, n_lon = t_map.shape

                    cloud_map = sample_cloud_map(pinn, n_lat=n_lat, n_lon=n_lon)
                    ice_map = sample_ice_map(pinn, n_lat=n_lat, n_lon=n_lon)
                    ocean_map = sample_ocean_map(pinn, n_lat=n_lat, n_lon=n_lon)

                    return {
                        "temperature": t_map,
                        "cloud": cloud_map,
                        "ice": ice_map,
                        "ocean": ocean_map,
                    }

                def _analytical_fallback():
                    return generate_eyeball_map(T_eq, tidally_locked=locked)

                cloud_map = None
                ice_map = None
                ocean_map = None

                if climate_model == "PINNFormer 3-D":
                    if not locked:
                        st.info(
                            "PINNFormer was trained for tidally locked planets. "
                            "Results are rescaled but the spatial pattern assumes tidal locking."
                        )
                    raw_result = gd.run_with_fallback(
                        _pinn_predict,
                        lambda: gd.run_with_fallback(
                            _elm_predict, _analytical_fallback,
                            timeout=5.0, label="ELM Surrogate",
                        ),
                        timeout=10.0,
                        label="PINNFormer 3-D",
                    )
                elif climate_model == "ELM Ensemble":
                    raw_result = gd.run_with_fallback(
                        _elm_predict,
                        lambda: gd.run_with_fallback(
                            _pinn_predict, _analytical_fallback,
                            timeout=10.0, label="PINNFormer 3-D",
                        ),
                        timeout=5.0,
                        label="ELM Surrogate",
                    )
                else:
                    raw_result = _analytical_fallback()

                if isinstance(raw_result, dict):
                    temp_map = raw_result["temperature"]
                    cloud_map = raw_result.get("cloud")
                    ice_map = raw_result.get("ice")
                    ocean_map = raw_result.get("ocean")
                else:
                    temp_map = raw_result

                if isinstance(temp_map, dict):
                    cloud_map = temp_map.get("cloud")
                    ice_map = temp_map.get("ice")
                    ocean_map = temp_map.get("ocean")
                    temp_map = temp_map["temperature"]

                if not gd.validate_temperature_map(temp_map):
                    temp_map = _analytical_fallback()

                try:
                    temp_map = _elm_predict()
                    _climate_method = "ELM Ensemble"
                    if not gd.validate_temperature_map(temp_map):
                        temp_map = _analytical_fallback()
                        _climate_method = "Analytical Fallback (ELM invalid)"
                except Exception:
                    try:
                        pinn_fields = _pinn_predict()
                        temp_map = pinn_fields["temperature"]
                        _climate_method = "PINNFormer 3D"
                        if not gd.validate_temperature_map(temp_map):
                            temp_map = _analytical_fallback()
                            _climate_method = "Analytical Fallback (PINN invalid)"
                            cloud_map = None
                            ice_map = None
                            ocean_map = None
                        else:
                            cloud_map = pinn_fields.get("cloud")
                            ice_map = pinn_fields.get("ice")
                            ocean_map = pinn_fields.get("ocean")
                    except Exception:
                        temp_map = _analytical_fallback()
                        _climate_method = "Analytical Fallback"

                _pipeline.write(f"Climate method: **{_climate_method}**")
                _pipeline.write("Computing surface fraction\u2026")
                hsf = habitable_surface_fraction(temp_map)

                st.session_state.temperature_map = temp_map
                st.session_state.cloud_map = cloud_map
                st.session_state.ice_map = ice_map
                st.session_state.ocean_map = ocean_map
                st.session_state["climate_method"] = _climate_method
                st.session_state.current_planet_data = {
                    "T_eq": T_eq,
                    "T_min": float(temp_map.min()),
                    "T_max": float(temp_map.max()),
                    "T_mean": float(temp_map.mean()),
                    "ESI": esi,
                    "SEPHI": sephi,
                    "HSF": hsf,
                    "flux_earth": S_norm,
                    "star_teff": star_teff,
                    "star_radius": star_radius,
                    "planet_radius": planet_radius,
                    "planet_mass": planet_mass,
                    "semi_major": semi_major,
                    "albedo": albedo,
                    "eccentricity": eccentricity,
                    "tidally_locked": locked,
                    "climate_method": _climate_method,
                    "radius_gap": rg,
                    "sulfur": sulfur,
                    "co": co,
                }
                _pipeline.update(label="Simulation complete", state="complete")
        except Exception as exc:
            st.error(f"{_ICON_NO} Validation failed: {exc}")

    with col_viz:
        st.subheader("Climate Simulation")

        if st.session_state.temperature_map is not None:
            d = st.session_state.current_planet_data

            # ── ESI gauge (left) + 2×2 metric grid (right) ──
            gauge_col, metrics_col = st.columns([1, 1], gap="medium")

            with gauge_col:
                esi_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=d["ESI"],
                        gauge=dict(
                            axis=dict(range=[0, 1]),
                            bar=dict(color="#00d4ff"),
                            steps=[
                                dict(range=[0, 0.4], color="#d73027"),
                                dict(range=[0.4, 0.7], color="#fee08b"),
                                dict(range=[0.7, 1.0], color="#1a9850"),
                            ],
                            threshold=dict(
                                line=dict(color="white", width=3),
                                thickness=0.75,
                                value=0.8,
                            ),
                        ),
                        title=dict(text="Earth Similarity Index", font=dict(size=12)),
                        number=dict(font=dict(size=28)),
                    )
                )
                esi_gauge.update_layout(
                    height=220,
                    margin=dict(l=20, r=20, t=40, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                )
                st.plotly_chart(esi_gauge, use_container_width=True)

            with metrics_col:
                esi_delta = d['ESI'] - 0.8
                esi_delta_str = f"{esi_delta:+.3f} vs 0.8 threshold"
                esi_delta_color = "#26a641" if esi_delta >= 0 else "#e05252"
                st.markdown(
                    f"""
<div style="display:grid;grid-template-columns:1fr 1fr;grid-auto-rows:1fr;gap:.75rem;height:100%;align-content:center">
  <div style="background:rgba(10,14,39,.7);border:1px solid rgba(66,165,245,.2);border-radius:12px;padding:.75rem 1rem;display:flex;flex-direction:column;justify-content:center">
    <div style="color:#90a4ae;font-size:.8rem">Equilibrium Temp.</div>
    <div style="color:#00d4ff;font-family:'Orbitron',sans-serif;font-size:1.5rem;font-weight:600">{d['T_eq']:.0f} K</div>
    <div style="font-size:.75rem;margin-top:.2rem;visibility:hidden">placeholder</div>
  </div>
  <div style="background:rgba(10,14,39,.7);border:1px solid rgba(66,165,245,.2);border-radius:12px;padding:.75rem 1rem;display:flex;flex-direction:column;justify-content:center">
    <div style="color:#90a4ae;font-size:.8rem">Earth Similarity (ESI)</div>
    <div style="color:#00d4ff;font-family:'Orbitron',sans-serif;font-size:1.5rem;font-weight:600">{d['ESI']:.3f}</div>
    <div style="color:{esi_delta_color};font-size:.75rem;margin-top:.2rem">{'↑' if esi_delta >= 0 else '↓'} {esi_delta_str}</div>
  </div>
  <div style="background:rgba(10,14,39,.7);border:1px solid rgba(66,165,245,.2);border-radius:12px;padding:.75rem 1rem;display:flex;flex-direction:column;justify-content:center">
    <div style="color:#90a4ae;font-size:.8rem">Habitable Surface</div>
    <div style="color:#00d4ff;font-family:'Orbitron',sans-serif;font-size:1.5rem;font-weight:600">{d['HSF']:.1%}</div>
    <div style="font-size:.75rem;margin-top:.2rem;visibility:hidden">placeholder</div>
  </div>
  <div style="background:rgba(10,14,39,.7);border:1px solid rgba(66,165,245,.2);border-radius:12px;padding:.75rem 1rem;display:flex;flex-direction:column;justify-content:center">
    <div style="color:#90a4ae;font-size:.8rem">Stellar Flux</div>
    <div style="color:#00d4ff;font-family:'Orbitron',sans-serif;font-size:1.5rem;font-weight:600">{d['flux_earth']:.2f} S⊕</div>
    <div style="font-size:.75rem;margin-top:.2rem;visibility:hidden">placeholder</div>
  </div>
</div>""",
                    unsafe_allow_html=True,
                )

            # SEPHI traffic lights
            sp = d["SEPHI"]
            _ok = _ICON_YES
            _fail = _ICON_NO
            with st.container(border=True):
                st.markdown(
                    f"**SEPHI** &nbsp; "
                    f"{_ok if sp['thermal_ok'] else _fail} Thermal &nbsp; "
                    f"{_ok if sp['atmosphere_ok'] else _fail} Atmosphere &nbsp; "
                    f"{_ok if sp['magnetic_ok'] else _fail} Magnetic &nbsp; "
                    f"(Score: **{sp['sephi_score']:.2f}**)",
                    unsafe_allow_html=True,
                )

            # ISA Interaction & Biosignature False-Positive badges
            try:
                from modules.astro_physics import (
                    assess_biosignature_false_positives,
                    estimate_isa_interaction,
                )
                isa = estimate_isa_interaction(
                    planet_mass, planet_radius, d["T_eq"], locked,
                )
                fp = assess_biosignature_false_positives(
                    star_teff, star_radius, semi_major,
                    d["T_eq"], planet_mass, planet_radius,
                )
                isa_col, fp_col = st.columns(2, gap="medium")
                with isa_col:
                    isa_icon = (
                        _ICON_YES if isa["isa_score"] >= 0.75 else
                        _ICON_WARN if isa["isa_score"] >= 0.5 else
                        _ICON_NO
                    )
                    st.markdown(
                        f"**ISA Coupling** {isa_icon} "
                        f"{isa['isa_assessment']} (score: {isa['isa_score']:.2f})",
                        unsafe_allow_html=True,
                    )
                with fp_col:
                    fp_icon = {
                        "low": _ICON_YES, "moderate": _ICON_WARN, "high": _ICON_NO
                    }.get(fp["overall_false_positive_risk"], "?")
                    st.markdown(
                        f"**False-Positive Risk** {fp_icon} "
                        f"{fp['overall_false_positive_risk'].title()}",
                        unsafe_allow_html=True,
                    )
            except Exception:
                pass

            # ── Radius Gap / Sulfur / C/O classification ──
            if "radius_gap" in d:
                rg_d    = d["radius_gap"]
                sulfur_d = d["sulfur"]
                co_d    = d["co"]
                with st.container(border=True):
                    st.markdown("##### Composition & Atmospheric Classification")
                    rg_col, sulfur_col, co_col = st.columns(3, gap="medium")
                    with rg_col:
                        st.markdown(f"**Radius Class:** {rg_d['label']}")
                        st.caption(f"Atmosphere retention: `{rg_d['atmosphere_retention']}`")
                        if rg_d["classification"] == "radius_gap":
                            st.progress(rg_d["gap_proximity"], text="Gap proximity")
                    with sulfur_col:
                        st.markdown(f"**Sulfur:** `{sulfur_d['dominant_gas']}` | `{sulfur_d['regime']}`")
                        st.caption(f"Surface minerals: {', '.join(sulfur_d['surface_minerals'])}")
                        if sulfur_d["h2so4_condensation"]:
                            st.warning("Venus-like H2SO4 clouds predicted")
                    with co_col:
                        adjusted_esi = d["ESI"] + co_d["habitability_modifier"]
                        st.markdown(f"**C/O:** {co_d['label']}")
                        st.caption(f"ESI adjusted: `{adjusted_esi:.3f}` (modifier: {co_d['habitability_modifier']:+.3f})")
                        if co_d["classification"] == "carbon_planet":
                            st.error("Carbon-rich composition \u2014 liquid water unlikely")

            # ── Raw Data Export ──
            with st.expander("Raw Data & CSV Export"):
                _metrics_df = pd.DataFrame([{
                    "T_eq_K": d["T_eq"],
                    "T_min_K": d["T_min"],
                    "T_max_K": d["T_max"],
                    "T_mean_K": d["T_mean"],
                    "ESI": d["ESI"],
                    "SEPHI_score": d["SEPHI"]["sephi_score"],
                    "HSF": d["HSF"],
                    "flux_earth": d["flux_earth"],
                    "albedo": d["albedo"],
                    "eccentricity": d.get("eccentricity", 0.0),
                    "tidally_locked": d["tidally_locked"],
                }])
                st.dataframe(_metrics_df, use_container_width=True)
                st.download_button(
                    "Download metrics CSV",
                    data=_metrics_df.to_csv(index=False),
                    file_name="simulation_metrics.csv",
                    mime="text/csv",
                )
                _tmap_df = pd.DataFrame(
                    st.session_state.temperature_map,
                    columns=[f"lon_{i}" for i in range(st.session_state.temperature_map.shape[1])],
                )
                st.download_button(
                    "Download temperature map CSV",
                    data=_tmap_df.to_csv(index=False),
                    file_name="temperature_map.csv",
                    mime="text/csv",
                )

            # Climate method badge + Globe / heatmap toggle
            _method = st.session_state.get("climate_method", "Unknown")
            _method_colors = {
                "ELM Ensemble": "#1a9850",
                "PINNFormer 3D": "#6a3d9a",
                "Analytical Fallback": "#d73027",
            }
            _mc = _method_colors.get(_method, "#90a4ae")
            st.markdown(
                f'<div style="display:inline-block;background:{_mc}22;border:1px solid {_mc};'
                f'border-radius:8px;padding:4px 14px;font-size:.85rem;color:{_mc};'
                f'font-family:\'Space Grotesk\',sans-serif;font-weight:600;margin-bottom:.5rem">'
                f'Climate map: {_method}</div>',
                unsafe_allow_html=True,
            )

            # Layer controls for 3-D visualisation
            view_mode = st.radio("View", ["3D Globe", "2D Heatmap"], horizontal=True)

            from modules.visualization import create_2d_heatmap, create_3d_globe

            cloud_map = getattr(st.session_state, "cloud_map", None)

            overlay_options = []
            if cloud_map is not None:
                overlay_options.append("Clouds (PINN)")

            selected_overlays: list[str] = []
            if view_mode == "3D Globe" and overlay_options:
                selected_overlays = st.multiselect(
                    "Overlays",
                    overlay_options,
                    default=overlay_options,  # show all available overlays by default
                    help="Visualise diagnostic layers from the PINNFormer (e.g. cloud fraction).",
                )

            if view_mode == "3D Globe":
                cloud_overlay = cloud_map if "Clouds (PINN)" in selected_overlays else None
                fig = create_3d_globe(
                    st.session_state.temperature_map,
                    "Custom Planet",
                    star_teff=star_teff,
                    cloud_map=cloud_overlay,
                )
            else:
                fig = create_2d_heatmap(
                    st.session_state.temperature_map, "Custom Planet"
                )
            st.plotly_chart(fig, use_container_width=True)

            # ── AI Interpretation & Physics Review (dual-LLM) ─────────
            with st.expander("AI Interpretation", expanded=False):
                try:
                    from modules.llm_helpers import (
                        classify_climate_state,
                        interpret_simulation,
                        review_elm_output,
                    )
                    with st.spinner("Domain expert interpreting results..."):
                        interp = interpret_simulation(d)
                    st.markdown(sanitize_latex(interp))

                    with st.spinner("Classifying climate state..."):
                        tmap = st.session_state.temperature_map
                        cls = classify_climate_state(
                            float(tmap.min()), float(tmap.max()),
                            float(tmap.mean()), locked,
                        )
                    state_climate = cls.get("state", "Unknown")
                    st.info(
                        f"**Climate state:** {state_climate} "
                        f"({cls.get('confidence', '?')} confidence)\n\n"
                        f"{cls.get('reason', '')}"
                    )

                    with st.spinner("Physics plausibility review..."):
                        params_for_review = {
                            "star_teff": star_teff, "star_radius": star_radius,
                            "planet_radius": planet_radius, "planet_mass": planet_mass,
                            "semi_major": semi_major, "albedo": albedo,
                            "tidally_locked": locked,
                        }
                        review = review_elm_output(
                            params_for_review,
                            float(tmap.min()), float(tmap.max()), float(tmap.mean()),
                        )
                    review = sanitize_latex(review)
                    if review.lower().startswith("plausible"):
                        st.success(f"⬢ {review}")
                    elif review.lower().startswith("warning"):
                        st.warning(review)
                    else:
                        st.info(review)
                except Exception as exc:
                    st.caption(f"*AI interpretation unavailable — {exc}*")
        else:
            st.info("Adjust parameters and press **Run Simulation** (or enable live mode).")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Planet Catalog
# ═══════════════════════════════════════════════════════════════════════════════

with tab_catalog:
    st.subheader("Habitable-Zone Candidates — NASA Exoplanet Archive")

    # ── Search: try direct name lookup first, then LLM-generated ADQL ──
    nl_query = st.text_input(
        "Search planets (name or natural language)",
        placeholder="e.g. TRAPPIST-1 e, or rocky planets closer than 10 parsecs",
    )
    if nl_query:
        from modules.llm_helpers import generate_adql_query, generate_planet_name_query
        from modules.nasa_client import query_nasa_archive

        nl_results = None

        # Always try a case-insensitive name search first
        with st.spinner("Searching by name\u2026"):
            try:
                name_adql = generate_planet_name_query(nl_query)
                nl_results = query_nasa_archive(name_adql)
                if nl_results is not None and nl_results.empty:
                    nl_results = None
            except Exception:
                nl_results = None

        # If no name match, fall back to LLM-generated ADQL
        if nl_results is None:
            try:
                with st.spinner("Translating to ADQL\u2026"):
                    adql = generate_adql_query(nl_query)
                if adql:
                    st.code(adql, language="sql")
                    with st.spinner("Executing query\u2026"):
                        nl_results = query_nasa_archive(adql)
                    if nl_results is not None and nl_results.empty:
                        nl_results = None
            except Exception as exc:
                st.error(f"ADQL search error: {exc}")

        if nl_results is not None:
            st.dataframe(nl_results, use_container_width=True)
            st.success(f"Found {len(nl_results)} results")
        else:
            st.warning("No results found.")

    st.markdown("##### Example Systems")
    st.caption("Pre-loaded queries that trigger real NASA TAP lookups and the full computation pipeline.")
    famous = [
        {"name": "TRAPPIST-1 e", "desc": "Rocky, temperate, tidally locked"},
        {"name": "Proxima Cen b", "desc": "Closest habitable candidate"},
        {"name": "K2-18 b",       "desc": "Water vapour detected (JWST)"},
        {"name": "Kepler-442 b",  "desc": "High-ESI super-Earth"},
        {"name": "TOI-700 d",     "desc": "Earth-size in HZ (TESS)"},
        {"name": "LHS 1140 b",    "desc": "Dense rocky world in HZ"},
    ]
    row1, row2 = famous[:3], famous[3:]
    for row in (row1, row2):
        gcols = st.columns(3, gap="medium")
        for col, p in zip(gcols, row):
            with col:
                if st.button(
                    p["name"],
                    use_container_width=True,
                    help=p["desc"],
                ):
                    st.session_state["selected_planet"] = p["name"]

    if st.session_state.get("selected_planet"):
        with st.spinner(f"Fetching {st.session_state['selected_planet']}\u2026"):
            try:
                from modules.nasa_client import get_planet_data
                row = get_planet_data(st.session_state["selected_planet"])
                if row is not None:
                    raw_dict = row.to_dict()
                    # Domain-expert summary instead of raw JSON
                    try:
                        from modules.llm_helpers import summarise_planet_data
                        with st.spinner("Domain expert summarising..."):
                            summary = summarise_planet_data(raw_dict)
                        st.markdown(sanitize_latex(summary))
                    except Exception:
                        pass
                    with st.expander("Raw data"):
                        st.json(raw_dict)
                else:
                    st.warning("Planet not found in archive.")
            except Exception as exc:
                st.error(str(exc))
        st.session_state["selected_planet"] = None

    if st.button("Fetch full catalog (NASA + European sources)"):
        with st.spinner("Querying NASA, Exoplanet.eu, DACE\u2026"):
            try:
                from modules.combined_catalog import build_combined_catalog
                cand = build_combined_catalog()
                st.dataframe(cand, use_container_width=True)
                sources = cand["source"].value_counts().to_dict() if "source" in cand.columns else {}
                src_summary = " \u00b7 ".join(f"{s}: **{n}**" for s, n in sources.items())
                st.success(f"Found {len(cand)} planets ({src_summary})")

                # ── Anomaly detection + UMAP + Weird Planets ──
                try:
                    from modules.anomaly_detection import (
                        build_weird_planets_table,
                        compute_umap_embedding,
                        create_umap_figure,
                        detect_anomalies,
                    )

                    with st.spinner("Running anomaly detection\u2026"):
                        detected = detect_anomalies(cand)
                        n_anom = int(detected["is_anomaly"].sum())

                    st.markdown("---")
                    st.markdown("### \U0001f50d Anomaly Detection")
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    col_stat1.metric("Planets analysed", len(detected))
                    col_stat2.metric("Anomalies found", n_anom)
                    col_stat3.metric(
                        "Contamination rate",
                        f"{n_anom / len(detected):.1%}" if len(detected) else "—",
                    )

                    umap_col, weird_col = st.columns([3, 2])

                    with umap_col:
                        with st.spinner("Computing UMAP embedding\u2026"):
                            emb = compute_umap_embedding(detected)
                            if emb is not None:
                                fig_umap = create_umap_figure(detected, emb)
                                st.plotly_chart(fig_umap, use_container_width=True)
                            else:
                                st.info("UMAP embedding unavailable (install `umap-learn`).")

                    with weird_col:
                        st.markdown("#### \U0001f47d Weirdest Planets")
                        weird = build_weird_planets_table(cand, n=12)
                        if not weird.empty:
                            st.dataframe(
                                weird,
                                use_container_width=True,
                                height=400,
                            )
                        else:
                            st.info("Not enough data for weird-planet ranking.")

                except Exception as exc_anom:
                    st.warning(f"Anomaly detection skipped: {exc_anom}")
            except Exception as exc:
                st.error(f"NASA error: {exc}")

    # ── Real vs Synthetic habitable-planet comparison ─────────
    st.markdown("---")
    st.markdown("### \U0001f9ec Real vs Synthetic Habitable Planets")
    st.caption(
        "Compare the real habitable-zone candidates with CTGAN-generated "
        "synthetic counterparts. Synthetic data is clearly marked as "
        "**exploratory** — use it to understand distributional coverage, "
        "not as ground truth."
    )

    if st.button("\U0001f52c Run comparison"):
        with st.spinner("Loading catalog and CTGAN model\u2026"):
            try:
                from modules.combined_catalog import build_combined_catalog
                from modules.data_augmentation import ExoplanetDataAugmenter

                cat = build_combined_catalog()
                aug = ExoplanetDataAugmenter()
                data = aug.prepare_normalised_data(cat)
                real_hab = data[data["habitable"] == 1].copy()

                import os as _os
                _ctgan_path = _os.path.join("models", "ctgan_exoplanets.pkl")
                if _os.path.exists(_ctgan_path):
                    aug.load_model(_ctgan_path)
                else:
                    st.warning("CTGAN model not found — training a quick model (50 epochs)\u2026")
                    aug.epochs = 50
                    aug.train(data)

                synth = aug.generate_synthetic_planets(
                    n_samples=2000, condition_column="habitable", condition_value=1,
                )
                synth = ExoplanetDataAugmenter.validate_synthetic_data(synth)

                st.success(
                    f"Real habitable: **{len(real_hab)}** planets  \u00b7  "
                    f"Synthetic habitable (after validation): **{len(synth)}**"
                )

                compare_cols = [
                    "radius_earth", "mass_earth", "semi_major_axis_au",
                    "insol_earth", "t_eq_K", "star_teff_K",
                ]
                nice = {
                    "radius_earth": "Radius (R\u2295)", "mass_earth": "Mass (M\u2295)",
                    "semi_major_axis_au": "Semi-major axis (AU)",
                    "insol_earth": "Insolation (S\u2295)", "t_eq_K": "T_eq (K)",
                    "star_teff_K": "Star T_eff (K)",
                }

                from plotly.subplots import make_subplots
                ncols_fig = 3
                nrows_fig = (len(compare_cols) + ncols_fig - 1) // ncols_fig
                fig_cmp = make_subplots(
                    rows=nrows_fig, cols=ncols_fig,
                    subplot_titles=[nice.get(c, c) for c in compare_cols],
                )
                for i, col in enumerate(compare_cols):
                    r, c = divmod(i, ncols_fig)
                    fig_cmp.add_trace(
                        go.Histogram(
                            x=real_hab[col].dropna(), name="Real",
                            marker_color="#2171b5", opacity=0.65,
                            showlegend=(i == 0),
                        ),
                        row=r + 1, col=c + 1,
                    )
                    fig_cmp.add_trace(
                        go.Histogram(
                            x=synth[col].dropna(), name="Synthetic (exploratory)",
                            marker_color="#d73027", opacity=0.55,
                            showlegend=(i == 0),
                        ),
                        row=r + 1, col=c + 1,
                    )
                fig_cmp.update_layout(
                    barmode="overlay", height=300 * nrows_fig,
                    title_text="Distribution Overlay — Real vs Synthetic Habitable",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    legend=dict(
                        bgcolor="rgba(20,20,50,0.7)", bordercolor="#555",
                        borderwidth=1, font=dict(color="white"),
                    ),
                )
                st.plotly_chart(fig_cmp, use_container_width=True)

                # Summary statistics table
                rows_stats = []
                for col in compare_cols:
                    rv = real_hab[col].dropna()
                    sv = synth[col].dropna()
                    rows_stats.append({
                        "Parameter": nice.get(col, col),
                        "Real mean": f"{rv.mean():.3g}",
                        "Real std": f"{rv.std():.3g}",
                        "Synth mean": f"{sv.mean():.3g}",
                        "Synth std": f"{sv.std():.3g}",
                        "Real n": len(rv),
                        "Synth n": len(sv),
                    })
                st.dataframe(pd.DataFrame(rows_stats), use_container_width=True)

                st.info(
                    "\u26a0\ufe0f **Synthetic data caveat:** These samples are generated "
                    "by a CTGAN trained on ~60 real habitable-zone planets. They "
                    "capture broad distributional shape but should not be treated "
                    "as real discoveries. All synthetic rows are post-filtered "
                    "to the 1\u201399th percentile envelope of real parameters."
                )
            except Exception as exc_cmp:
                st.error(f"Comparison failed: {exc_cmp}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Science Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

with tab_science:
    d = st.session_state.current_planet_data
    tmap = st.session_state.temperature_map

    if d is None or tmap is None:
        st.info("Run a simulation first (Manual Mode tab).")
    else:
        # ── Compute HZ boundaries (shared by narrative + diagram) ──
        _star_rad = d.get("star_radius", 1.0)
        try:
            from modules.astro_physics import hz_boundaries as _hz
            lum_solar = (d["star_teff"] / 5778) ** 4 * _star_rad ** 2
            hz = _hz(d["star_teff"], lum_solar)
        except Exception:
            hz = {}

        # ── Scientific Narrative (domain expert) — full-width card ──
        _narrative_text = ""
        try:
            from modules.llm_helpers import narrate_science_panel
            with st.spinner("Domain expert writing scientific narrative..."):
                equator = tmap[tmap.shape[0] // 2, :]
                cs_stats = {
                    "T_min_equator": float(equator.min()),
                    "T_max_equator": float(equator.max()),
                    "T_mean_equator": float(equator.mean()),
                    "gradient_K": float(equator.max() - equator.min()),
                }
                _narrative_text = narrate_science_panel(
                    hz_data=hz,
                    cross_section_stats=cs_stats,
                    uncertainty_note="ELM ensemble std-dev or +/-15 K analytical uncertainty",
                )
        except Exception:
            pass

        # ── Precompute data for all science cards ──
        _isa_data = None
        try:
            from modules.astro_physics import estimate_isa_interaction
            _p_rad = d.get("planet_radius", 1.0)
            _p_mass = d.get("planet_mass", 1.0)
            _locked = d.get("tidally_locked", True)
            _isa_data = estimate_isa_interaction(_p_mass, _p_rad, d["T_eq"], _locked)
        except Exception:
            pass

        _fp_data = None
        try:
            from modules.astro_physics import assess_biosignature_false_positives
            _p_rad = d.get("planet_radius", 1.0)
            _p_mass = d.get("planet_mass", 1.0)
            _fp_data = assess_biosignature_false_positives(
                d["star_teff"], _star_rad, d["semi_major"],
                d["T_eq"], _p_mass, _p_rad,
            )
        except Exception:
            pass

        # ── Grid layout using columns ──
        # Row 1: full-width Scientific Narrative
        if _narrative_text:
            with st.container(border=True):
                st.markdown("### Scientific Narrative")
                st.markdown(sanitize_latex(_narrative_text))

        # Row 2: HZ diagram | False Positives
        row2_left, row2_right = st.columns(2, gap="medium")

        with row2_left:
            with st.container(border=True):
                if _isa_data:
                    isa = _isa_data
                    st.subheader("Interior-Surface-Atmosphere")
                    ic1, ic2 = st.columns(2)
                    ic1.metric("ISA Score", f"{isa['isa_score']:.2f}")
                    ic2.metric("Outgassing", f"{isa['outgassing']['outgassing_rate_earth']:.2f}x Earth")
                    _ok = _ICON_YES
                    _fail = _ICON_NO
                    st.markdown(
                        f"{_ok if isa['plate_tectonics'] == 'plausible' else (_ICON_WARN if isa['plate_tectonics'] == 'uncertain' else _fail)} Plate tectonics ({isa['plate_tectonics']}) &nbsp; "
                        f"{_ok if isa['carbonate_silicate_cycle'] else _fail} C-Si cycle &nbsp; "
                        f"{_ok if isa['water_cycling'] else _fail} Water cycling &nbsp; "
                        f"{_ok if isa['volatile_retention'] else _fail} Volatile retention",
                        unsafe_allow_html=True,
                    )
                else:
                    st.subheader("Interior-Surface-Atmosphere")
                    st.caption("Data unavailable.")

        with row2_right:
            with st.container(border=True):
                if _fp_data:
                    fp = _fp_data
                    st.subheader("Photochemical False Positives")
                    fp1, fp2, fp3 = st.columns(3)
                    fp1.metric("O\u2082 risk", fp["o2_false_positive_risk"].title())
                    fp2.metric("CH\u2084 risk", fp["ch4_false_positive_risk"].title())
                    fp3.metric("UV flux", f"{fp['uv_environment']['uv_flux_earth']:.1f}x Earth")
                    if fp["risk_flags"]:
                        for flag in fp["risk_flags"]:
                            st.warning(flag)
                    st.info(fp["recommendation"])
                else:
                    st.subheader("Photochemical False Positives")
                    st.caption("Data unavailable.")

        # Row 3: ISA Interaction | Terminator Cross-Section
        row3_left, row3_right = st.columns(2, gap="medium")

        with row3_left:
            with st.container(border=True):
                st.subheader("Habitable Zone")
                try:
                    from modules.visualization import create_hz_diagram
                    if hz:
                        fig_hz = create_hz_diagram(hz, d["semi_major"], d["star_teff"])
                        st.plotly_chart(fig_hz, use_container_width=True)
                    else:
                        st.warning("HZ boundaries could not be computed.")
                except Exception as exc:
                    st.warning(f"HZ diagram unavailable: {exc}")

        with row3_right:
            with st.container(border=True):
                st.subheader("Terminator Cross-Section")
                n_lat = tmap.shape[0]
                equator_idx = n_lat // 2
                profile = tmap[equator_idx, :]
                lons = np.linspace(0, 360, len(profile))

                fig_cs = go.Figure()
                fig_cs.add_trace(
                    go.Scatter(x=lons, y=profile, mode="lines", name="ELM / Analytical")
                )
                fig_cs.add_hline(y=273, line_dash="dash", line_color="#41ab5d",
                                 annotation_text="273 K (freezing)")
                fig_cs.add_hline(y=373, line_dash="dash", line_color="#d73027",
                                 annotation_text="373 K (boiling)")
                fig_cs.update_layout(
                    xaxis_title="Longitude [\u00b0]",
                    yaxis_title="Temperature [K]",
                    paper_bgcolor="black",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    height=_CHART_H,
                    margin=dict(l=40, r=20, t=30, b=40),
                )
                st.plotly_chart(fig_cs, use_container_width=True)

        # Row 4: Uncertainty Estimate (full-width)
        with st.container(border=True):
            st.subheader("Uncertainty Estimate")
            _elm_ci_shown = False
            try:
                elm = _load_elm()
                if elm.models:
                    _p_params = {
                        "radius_earth": d.get("planet_radius", 1.0),
                        "mass_earth": d.get("planet_mass", 1.0),
                        "semi_major_axis_au": d.get("semi_major", 0.05),
                        "star_teff_K": d.get("star_teff", 3000),
                        "star_radius_solar": d.get("star_radius", 0.14),
                        "insol_earth": d.get("flux_earth", 1.0),
                        "albedo": d.get("albedo", 0.3),
                        "tidally_locked": int(d.get("tidally_locked", True)),
                    }
                    mean_map, lower_map, upper_map = elm.predict_from_params_with_ci(
                        _p_params, alpha=0.1,
                    )
                    ci_width = float((upper_map - lower_map).mean())
                    st.markdown(
                        "**Conformal prediction** (90% coverage interval from ELM ensemble):"
                    )
                    uc1, uc2, uc3, uc4 = st.columns(4)
                    uc1.metric("T mean", f"{float(mean_map.mean()):.0f} K")
                    uc2.metric("CI width (avg)", f"\u00b1{ci_width/2:.1f} K")
                    uc3.metric("CI lower bound", f"{float(lower_map.mean()):.0f} K")
                    uc4.metric("CI upper bound", f"{float(upper_map.mean()):.0f} K")
                    _elm_ci_shown = True
            except Exception:
                pass
            if not _elm_ci_shown:
                st.markdown(
                    "Analytical model: static uncertainty estimates (ELM not loaded)."
                )
                uncert_cols = st.columns(3)
                uncert_cols[0].metric("T_eq \u00b1", "\u00b115 K (albedo uncertainty)")
                uncert_cols[1].metric("ESI \u00b1", "\u00b10.05 (propagated)")
                uncert_cols[2].metric("HSF \u00b1", "\u00b15 pp")

        # Row 5: GCM Benchmark Comparison (full-width)
        with st.container(border=True):
            st.subheader("GCM Benchmark Comparison")
            st.caption(
                "Qualitative comparison of surrogate output against "
                "precomputed GCM reference profiles. These are approximate "
                "digitizations from published GCM studies, not full GCM reruns."
            )
            try:
                from modules.gcm_benchmarks import (
                    compare_surrogate_to_gcm,
                    compute_zonal_mean,
                    get_gcm_benchmark,
                    list_benchmarks,
                )
                _bench_key = st.selectbox(
                    "Select benchmark case",
                    list_benchmarks(),
                    format_func=lambda k: {
                        "earth_like": "Earth-like aquaplanet (Del Genio 2019)",
                        "proxima_b": "Proxima Cen b tidally locked (Turbet 2016)",
                        "hot_rock": "Hot synchronous rock (Leconte 2013)",
                    }.get(k, k),
                )
                if st.button("Compare with GCM"):
                    bench = get_gcm_benchmark(_bench_key)
                    if bench is not None:
                        gcm_map = bench["temperature_map"]
                        surr_map = tmap
                        if surr_map.shape != gcm_map.shape:
                            from scipy.ndimage import zoom
                            scale = (gcm_map.shape[0] / surr_map.shape[0],
                                     gcm_map.shape[1] / surr_map.shape[1])
                            surr_map = zoom(surr_map, scale, order=1)

                        metrics = compare_surrogate_to_gcm(surr_map, gcm_map)

                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("Pattern correlation", f"{metrics['pattern_correlation']:.3f}")
                        mc2.metric("RMSE", f"{metrics['rmse_K']:.1f} K")
                        mc3.metric("Bias", f"{metrics['bias_K']:+.1f} K")
                        mc4.metric("Zonal RMSE", f"{metrics['zonal_mean_rmse_K']:.1f} K")

                        surr_zonal = compute_zonal_mean(surr_map)
                        gcm_zonal = compute_zonal_mean(gcm_map)
                        lats = np.linspace(-90, 90, len(surr_zonal))

                        fig_zonal = go.Figure()
                        fig_zonal.add_trace(go.Scatter(
                            x=lats, y=surr_zonal, mode="lines",
                            name="Surrogate (ELM/PINNFormer)",
                        ))
                        fig_zonal.add_trace(go.Scatter(
                            x=lats, y=gcm_zonal, mode="lines",
                            name=f"GCM ({bench['source']})",
                            line=dict(dash="dash"),
                        ))
                        fig_zonal.update_layout(
                            xaxis_title="Latitude [deg]",
                            yaxis_title="Zonal mean T [K]",
                            paper_bgcolor="black",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="white"),
                            height=_CHART_H,
                            margin=dict(l=40, r=20, t=30, b=40),
                        )
                        st.plotly_chart(fig_zonal, use_container_width=True)
                        st.caption(f"Source: {bench['source']}")
                    else:
                        st.warning("Benchmark case not found.")
            except Exception as exc:
                st.warning(f"GCM comparison unavailable: {exc}")

        # Row 6: Compare with Earth | Planetary Soundscape
        st.markdown(
            """<style>
            div[data-testid="stHorizontalBlock"]:has(> div > div > div[data-testid="stVerticalBlockBorderWrapper"])
            > div { display:flex; flex-direction:column; }
            div[data-testid="stHorizontalBlock"]:has(> div > div > div[data-testid="stVerticalBlockBorderWrapper"])
            > div > div[data-testid="stVerticalBlock"] { flex:1; }
            div[data-testid="stHorizontalBlock"]:has(> div > div > div[data-testid="stVerticalBlockBorderWrapper"])
            div[data-testid="stVerticalBlockBorderWrapper"] { height:100%; box-sizing:border-box; }
            </style>""",
            unsafe_allow_html=True,
        )
        row5_left, row5_right = st.columns(2, gap="medium")

        with row5_left:
            with st.container(border=True):
                st.subheader("Compare with Earth")
                if st.button("Compare with Earth"):
                    try:
                        from modules.llm_helpers import compare_planets
                        earth_params = {
                            "name": "Earth",
                            "T_eq": 255, "ESI": 1.0, "SEPHI": {"sephi_score": 1.0},
                            "HSF": 0.65, "flux_earth": 1.0,
                            "radius_earth": 1.0, "mass_earth": 1.0,
                            "semi_major_au": 1.0, "star_teff": 5778,
                        }
                        with st.spinner("Domain expert comparing with Earth..."):
                            comp = compare_planets(d, earth_params)
                        st.markdown(sanitize_latex(comp))
                    except Exception:
                        st.caption("*Comparison unavailable (Ollama not running).*")

        with row5_right:
            with st.container(border=True):
                st.subheader("Outreach: Planetary Soundscape")
                st.caption(
                    "Temperature-to-frequency sonification for outreach and "
                    "engagement purposes. This is **not** a scientific diagnostic."
                )
                if st.button("Generate sound"):
                    from scipy.io import wavfile

                    equator = tmap[tmap.shape[0] // 2, :]
                    T_lo, T_hi = equator.min(), equator.max()
                    dur, sr = 5.0, 22050
                    freqs = 200 + (equator - T_lo) / max(T_hi - T_lo, 1) * 1800
                    t_total = np.linspace(0, dur, int(sr * dur))
                    spf = len(t_total) // len(freqs)
                    audio = np.concatenate(
                        [
                            0.3 * np.sin(2 * np.pi * f * np.linspace(0, spf / sr, spf))
                            for f in freqs
                        ]
                    )[: len(t_total)]
                    audio_i16 = (audio * 32767).astype(np.int16)

                    buf = io.BytesIO()
                    wavfile.write(buf, sr, audio_i16)
                    st.audio(buf.getvalue(), format="audio/wav")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — System (diagnostics, export, health)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_system:
    st.subheader("System Health & Diagnostics")

    # ── LLM Mode Selector ──
    with st.container(border=True):
        st.markdown("##### LLM Runtime Mode")
        _MODE_OPTIONS = {
            "Dual-LLM (orchestrator + astro-agent)": "dual_llm",
            "Single-LLM (astro-agent only)": "single_llm",
            "Deterministic (no LLM)": "deterministic",
        }
        _MODE_LABELS = list(_MODE_OPTIONS.keys())
        _current_label = [k for k, v in _MODE_OPTIONS.items()
                          if v == st.session_state["llm_mode"]][0]
        selected_label = st.radio(
            "Select runtime mode",
            _MODE_LABELS,
            index=_MODE_LABELS.index(_current_label),
            horizontal=True,
            help=(
                "Dual-LLM: Qwen 2.5-14B orchestrator + AstroSage-8B expert (~12 GB VRAM). "
                "Single-LLM: AstroSage-8B handles both roles (~5 GB VRAM). "
                "Deterministic: physics + ML tools only, no AI narratives."
            ),
        )
        new_mode = _MODE_OPTIONS[selected_label]
        if new_mode != st.session_state["llm_mode"]:
            st.session_state["llm_mode"] = new_mode
            from modules.llm_helpers import set_llm_mode
            set_llm_mode(new_mode)
            st.rerun()

        _mode_desc = {
            "dual_llm": "Two LLM instances: Qwen 2.5-14B (orchestration) + AstroSage-8B (domain expertise).",
            "single_llm": "One LLM instance: AstroSage-8B handles both tool orchestration and interpretation.",
            "deterministic": "No LLM loaded. All physics, ML, and visualization tools remain functional.",
        }
        st.caption(_mode_desc[st.session_state["llm_mode"]])

    if st.button("Run Self-Diagnostics"):
        checks = {}

        # 1. NASA API
        with st.status("Running diagnostics\u2026", expanded=True) as status:
            st.write("Testing NASA API...")
            try:
                from modules.nasa_client import get_planet_data
                earth_test = get_planet_data("Proxima Cen b")
                checks["NASA API"] = _ICON_YES if earth_test is not None else _ICON_WARN
            except Exception:
                checks["NASA API"] = _ICON_NO

            # 2. T_eq for Earth
            st.write("Computing T_eq for Earth sanity check...")
            try:
                from modules.astro_physics import equilibrium_temperature
                t_earth = equilibrium_temperature(5778, 1.0, 1.0, 0.3, False)
                checks["T_eq(Earth)"] = (
                    f"{_ICON_YES} {t_earth:.1f} K" if 240 < t_earth < 270 else f"{_ICON_WARN} {t_earth:.1f} K"
                )
            except Exception as exc:
                checks["T_eq(Earth)"] = f"{_ICON_NO} {exc}"

            # 3. Pydantic rejection
            st.write("Testing Pydantic guardrail...")
            try:
                from modules.validators import PlanetaryParameters
                PlanetaryParameters(
                    name="bad", radius_earth=100, semi_major_axis=1, albedo=0.3
                )
                checks["Pydantic guard"] = f"{_ICON_NO} should have rejected"
            except Exception:
                checks["Pydantic guard"] = f"{_ICON_YES} rejected invalid input"

            # 4. ELM model
            st.write("Checking ELM model...")
            try:
                from modules.elm_surrogate import ELMClimateSurrogate
                m = ELMClimateSurrogate()
                m.load("models/elm_ensemble.pkl")
                checks["ELM model"] = f"{_ICON_YES} loaded"
            except FileNotFoundError:
                checks["ELM model"] = f"{_ICON_WARN} not trained yet"
            except Exception as exc:
                checks["ELM model"] = f"{_ICON_NO} {exc}"

            # 5. PINNFormer 3-D
            st.write("Checking PINNFormer 3-D...")
            try:
                pinn_m, pinn_d = _load_pinn()
                if pinn_m is not None:
                    n_params = sum(p.numel() for p in pinn_m.parameters())
                    checks["PINNFormer 3-D"] = (
                        f"{_ICON_YES} loaded ({n_params/1e6:.1f}M params, {pinn_d})"
                    )
                else:
                    checks["PINNFormer 3-D"] = f"{_ICON_WARN} not trained yet"
            except Exception as exc:
                checks["PINNFormer 3-D"] = f"{_ICON_NO} {exc}"

            # 6. Ollama
            st.write("Checking Ollama...")
            try:
                import ollama as _oll
                tags = _oll.list()
                names = [m.model for m in tags.models] if hasattr(tags, "models") else []
                checks["Ollama"] = f"{_ICON_YES} {len(names)} models: {', '.join(names[:5])}"
            except Exception as exc:
                checks["Ollama"] = f"{_ICON_NO} {exc}"

            status.update(label="Diagnostics complete", state="complete")

        for k, v in checks.items():
            st.markdown(f"- **{k}:** {v}", unsafe_allow_html=True)

    # ── Export (enhancement A7) ──
    st.subheader("Export")
    if st.session_state.temperature_map is not None:
        from modules.visualization import create_3d_globe

        fig_export = create_3d_globe(st.session_state.temperature_map, "Export")
        html_bytes = fig_export.to_html(include_plotlyjs="cdn").encode()
        with st.container(border=True):
            st.download_button(
                "Download interactive HTML globe",
                data=html_bytes,
                file_name="planet_globe.html",
                mime="text/html",
            )
    else:
        st.info("Run a simulation to enable export.")

    # ── Architecture diagram (targets Req 4, 7) ──
    with st.expander("System Architecture"):
        st.markdown("""
```mermaid
flowchart TB
    subgraph UI ["Streamlit UI"]
        AgentTab["Agent AI"]
        ManualTab["Manual Mode"]
        CatalogTab["Catalog"]
        ScienceTab["Science Dashboard"]
    end

    subgraph LLM ["Dual LLM Layer (Ollama)"]
        Qwen["Qwen 2.5-14B<br/>Orchestrator"]
        Astro["astro-agent<br/>Domain Expert"]
    end

    subgraph ML ["ML Models"]
        ELM["ELM Ensemble<br/>Climate Surrogate"]
        CTGAN["CTGAN<br/>Data Augmentation"]
        PINN["PINNFormer 3D<br/>Physics-Informed NN"]
        IsoForest["Isolation Forest<br/>Anomaly Detection"]
    end

    subgraph Physics ["Physics Engine"]
        Teq["T_eq Calculator"]
        ESI["ESI / SEPHI"]
        ISA["ISA Interactions"]
        FP["False-Positive<br/>Mitigation"]
        HZ["HZ Boundaries"]
    end

    subgraph Data ["Data Layer"]
        NASA["NASA Exoplanet<br/>Archive (TAP)"]
        RAG["RAG Citations<br/>(ChromaDB)"]
        Pydantic["Pydantic<br/>Guardrails"]
    end

    AgentTab --> Qwen
    Qwen --> Astro
    Qwen --> NASA
    Qwen --> Physics
    ManualTab --> ELM
    ManualTab --> PINN
    ManualTab --> Physics
    CatalogTab --> NASA
    CatalogTab --> IsoForest
    ScienceTab --> Physics
    Physics --> Pydantic
    ELM --> Pydantic
    Qwen --> RAG
```
""")

    # ── Docker instructions ──
    with st.expander("Docker deployment"):
        st.code(
            "docker build -t exo-twin .\n"
            "docker run -p 8501:8501 --gpus all exo-twin",
            language="bash",
        )

# ─── About section (outside tabs so scroll always works) ────────────────────

st.markdown('<div id="about-section"></div>', unsafe_allow_html=True)
st.markdown("---")
with st.expander("About", expanded=False):
    st.markdown("""
**Autonomous Exoplanetary Digital Twin** was built for the Hack4Sages 2026 hackathon.

Source code: [github.com/HardCounter/Hack-4-Sages](https://github.com/HardCounter/Hack-4-Sages)

LinkedIn: [HardCounter Team](https://www.linkedin.com/company/hardcounter-team/)

                
| Name | GitHub | LinkedIn |
|------|--------|----------|
| Aleksander | [Ale-Jez](https://github.com/Ale-Jez/) | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/aleksander-jezowski) |
| Denis | [DenisLisovytskiy](https://github.com/DenisLisovytskiy) | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/denis-lisovytskiy-738b13232/) |
| Piotr | [Piotereko](https://github.com/Piotereko) | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/piotr-czechowski4/) |
| Rafał | [lasotar](https://github.com/lasotar) | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rafal-lasota/) |         

""")
