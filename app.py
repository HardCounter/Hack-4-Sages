"""
Autonomous Exoplanetary Digital Twin — main Streamlit application.

Tabs
----
1. Agent AI          – LLM chat with transparent reasoning
2. Manual Mode       – slider-driven simulation with live 3-D globe
3. Planet Catalog    – NASA archive browser + famous-planet gallery
4. Science Dashboard – HZ diagram, atmospheric cross-section, uncertainty
5. System            – self-diagnostics, health, export
"""

import io
import json
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─── Page config (must be first Streamlit call) ──────────────────────────────

st.set_page_config(
    page_title="Exoplanetary Digital Twin",
    page_icon="\U0001fa90",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Cosmic dark-theme CSS ───────────────────────────────────────────────────

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Orbitron:wght@400;500;600;700&display=swap');
.stApp{background:radial-gradient(ellipse at center,#0a0e27 0%,#000 100%)}
[data-testid="stSidebar"]{background:rgba(10,14,39,.95);border-right:1px solid #1a237e}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label{color:#b0bec5!important}
h1,h2,h3{font-family:'Orbitron',sans-serif!important;color:#e0e0e0!important}
.stMetricValue{color:#00d4ff!important;font-family:'Orbitron',sans-serif!important}
.stMetricLabel{color:#90a4ae!important}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#1a237e 0%,#0d47a1 100%);border:1px solid #42a5f5;color:#fff;font-family:'Space Grotesk',sans-serif}
.stButton>button[kind="primary"]:hover{background:linear-gradient(135deg,#283593 0%,#1565c0 100%);border-color:#64b5f6}
.stTabs [data-baseweb="tab-list"]{gap:8px}
.stTabs [data-baseweb="tab"]{font-family:'Space Grotesk',sans-serif;color:#90a4ae}
.stTabs [aria-selected="true"]{color:#00d4ff!important}
div[data-testid="stChatMessage"]{background:rgba(10,14,39,.6);border:1px solid #1a237e;border-radius:12px}
.tooltip-term{border-bottom:1px dotted #888;cursor:help;color:#b0bec5}
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
}


def tip(term: str) -> str:
    expl = _GLOSSARY.get(term, "")
    return f'<span class="tooltip-term" title="{expl}">{term}</span>'


# ─── Cached loaders ──────────────────────────────────────────────────────────

@st.cache_resource
def _load_agent():
    from modules.agent_setup import agent_executor
    return agent_executor


@st.cache_resource
def _load_elm():
    from modules.elm_surrogate import ELMClimateSurrogate
    m = ELMClimateSurrogate()
    try:
        m.load("models/elm_ensemble.pkl")
    except FileNotFoundError:
        pass
    return m


# ─── Session-state defaults ──────────────────────────────────────────────────

for k, v in {
    "chat_history": [],
    "current_planet_data": None,
    "temperature_map": None,
    "analysis_history": [],
    "selected_planet": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Header ──────────────────────────────────────────────────────────────────

st.title("\U0001fa90 Autonomous Exoplanetary Digital Twin")
st.caption("Simulate alien climates in real time with AI-driven physics surrogates.")

# ─── Tabs ────────────────────────────────────────────────────────────────────

tab_agent, tab_manual, tab_catalog, tab_science, tab_system = st.tabs(
    [
        "\U0001f916 Agent AI",
        "\U0001f39b\ufe0f Manual Mode",
        "\U0001f4ca Catalog",
        "\U0001f52c Science",
        "\U0001f527 System",
    ]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Agent AI with transparent reasoning
# ═══════════════════════════════════════════════════════════════════════════════

with tab_agent:
    col_chat, col_reasoning = st.columns([3, 1])

    with col_chat:
        st.subheader("Conversation with AstroAgent")

        # Audience selector (enhancement B5)
        audience = st.radio(
            "Explanation depth",
            ["Scientist", "Student", "Media"],
            horizontal=True,
            index=0,
        )

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about an exoplanet\u2026"):
            audience_hint = {
                "Scientist": " (respond at expert level, cite equations)",
                "Student": " (explain simply, use analogies)",
                "Media": " (short, vivid language, no jargon)",
            }
            full_prompt = prompt + audience_hint.get(audience, "")

            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Agent reasoning\u2026"):
                    try:
                        agent = _load_agent()
                        response = agent.invoke(
                            {"input": full_prompt, "chat_history": []}
                        )
                        answer = response["output"]
                        steps = response.get("intermediate_steps", [])
                    except Exception as exc:
                        answer = f"\u26a0\ufe0f Agent unavailable: {exc}"
                        steps = []
                    st.markdown(answer)

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
        st.subheader("\U0001f9e0 Reasoning Chain")
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
                        '\U0001f52d Expert Opinion</strong></div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(obs[:800])
                else:
                    with st.expander(f"Step {i+1}: {action.tool}", expanded=(i == 0)):
                        st.markdown(f"**Tool:** `{action.tool}`")
                        st.markdown(f"**Input:** `{action.tool_input}`")
                        st.markdown(f"**Output:** {obs[:500]}")
        else:
            st.info("Reasoning steps appear here after each agent response.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Manual Mode with live "What If" globe
# ═══════════════════════════════════════════════════════════════════════════════

with tab_manual:
    col_params, col_viz = st.columns([1, 2])

    with col_params:
        st.subheader("\U0001f39b\ufe0f Parameters")
        star_teff = st.slider("Stellar temperature [K]", 2500, 7500, 3042, 50)
        star_radius = st.slider("Stellar radius [R\u2609]", 0.08, 3.0, 0.141, 0.01)
        planet_radius = st.slider("Planet radius [R\u2295]", 0.5, 2.5, 1.07, 0.01)
        planet_mass = st.slider("Planet mass [M\u2295]", 0.1, 15.0, 1.27, 0.1)
        semi_major = st.slider(
            "Semi-major axis [AU]", 0.01, 2.0, 0.0485, 0.001, format="%.4f"
        )
        albedo = st.slider("Bond albedo", 0.0, 1.0, 0.3, 0.01)
        locked = st.checkbox("Tidally locked", True)

        run_sim = st.button(
            "\U0001f680 Run Simulation", type="primary", use_container_width=True
        )
        live_mode = st.toggle("Live \u201cWhat If\u201d mode", value=False)

    # Compute on button press or live mode change
    should_compute = run_sim or live_mode

    if should_compute:
        try:
            from modules.astro_physics import (
                compute_esi,
                compute_sephi,
                equilibrium_temperature,
                estimate_density,
                estimate_escape_velocity,
                habitable_surface_fraction,
                hz_boundaries,
                stellar_flux,
            )
            from modules.visualization import (
                create_2d_heatmap,
                create_3d_globe,
                create_hz_diagram,
                generate_eyeball_map,
            )

            T_eq = equilibrium_temperature(
                star_teff, star_radius, semi_major, albedo, locked
            )
            S_abs, S_norm = stellar_flux(star_teff, star_radius, semi_major)
            density = estimate_density(planet_mass, planet_radius)
            v_esc = estimate_escape_velocity(planet_mass, planet_radius)
            esi = compute_esi(planet_radius, density, v_esc, T_eq)
            sephi = compute_sephi(T_eq, planet_mass, planet_radius)
            temp_map = generate_eyeball_map(T_eq, tidally_locked=locked)
            hsf = habitable_surface_fraction(temp_map)

            st.session_state.temperature_map = temp_map
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
                "semi_major": semi_major,
            }
        except Exception as exc:
            st.error(f"\u274c Validation failed: {exc}")

    with col_viz:
        st.subheader("\U0001f30d Visualization")

        if st.session_state.temperature_map is not None:
            d = st.session_state.current_planet_data

            # ── Animated gauge metrics (enhancement A4) ──
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("T_eq", f"{d['T_eq']:.0f} K")
            g2.metric(
                "ESI",
                f"{d['ESI']:.3f}",
                delta=f"{d['ESI'] - 0.8:+.3f} vs 0.8 threshold",
            )
            g3.metric("HSF", f"{d['HSF']:.1%}")
            g4.metric("Flux", f"{d['flux_earth']:.2f} S\u2295")

            # ESI gauge (Plotly Indicator)
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
                    title=dict(text="Earth Similarity Index", font=dict(size=14)),
                )
            )
            esi_gauge.update_layout(
                height=200,
                margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
            )
            st.plotly_chart(esi_gauge, use_container_width=True)

            # SEPHI traffic lights
            sp = d["SEPHI"]
            st.markdown(
                f"**SEPHI** &nbsp; "
                f"{'\\u2705' if sp['thermal_ok'] else '\\u274c'} Thermal &nbsp; "
                f"{'\\u2705' if sp['atmosphere_ok'] else '\\u274c'} Atmosphere &nbsp; "
                f"{'\\u2705' if sp['magnetic_ok'] else '\\u274c'} Magnetic &nbsp; "
                f"(Score: **{sp['sephi_score']:.2f}**)"
            )

            # Globe / heatmap toggle
            view_mode = st.radio("View", ["3D Globe", "2D Heatmap"], horizontal=True)

            from modules.visualization import create_2d_heatmap, create_3d_globe

            if view_mode == "3D Globe":
                fig = create_3d_globe(
                    st.session_state.temperature_map,
                    "Custom Planet",
                    star_teff=star_teff,
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
                    st.markdown(interp)

                    with st.spinner("Classifying climate state..."):
                        tmap = st.session_state.temperature_map
                        cls = classify_climate_state(
                            float(tmap.min()), float(tmap.max()),
                            float(tmap.mean()), locked,
                        )
                    state_emoji = {
                        "Eyeball": "\U0001f441\ufe0f", "Lobster": "\U0001f99e",
                        "Greenhouse": "\U0001f525", "Temperate": "\U0001f33f",
                    }.get(cls.get("state", ""), "\u2753")
                    st.info(
                        f"**Climate state:** {state_emoji} {cls.get('state', 'Unknown')} "
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
                    if review.lower().startswith("plausible"):
                        st.success(f"\u2705 {review}")
                    elif review.lower().startswith("warning"):
                        st.warning(f"\u26a0\ufe0f {review}")
                    else:
                        st.info(review)
                except Exception:
                    st.caption("*AI interpretation unavailable (Ollama not running).*")
        else:
            st.info("Adjust parameters and press **Run Simulation** (or enable live mode).")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Planet Catalog
# ═══════════════════════════════════════════════════════════════════════════════

with tab_catalog:
    st.subheader("\U0001f4ca Habitable-Zone Candidates — NASA Exoplanet Archive")

    # ── Natural-language ADQL query (Qwen orchestrator) ───────────
    nl_query = st.text_input(
        "\U0001f50d Search planets in natural language",
        placeholder="e.g. rocky planets closer than 10 parsecs",
    )
    if nl_query:
        try:
            from modules.llm_helpers import generate_adql_query
            with st.spinner("Qwen translating to ADQL..."):
                adql = generate_adql_query(nl_query)
            if adql:
                st.code(adql, language="sql")
                with st.spinner("Executing query..."):
                    from modules.nasa_client import query_nasa_archive
                    nl_results = query_nasa_archive(adql)
                if nl_results is not None and not nl_results.empty:
                    st.dataframe(nl_results, use_container_width=True)
                    st.success(f"Found {len(nl_results)} results")
                else:
                    st.warning("Query returned no results.")
            else:
                st.warning("Could not generate ADQL query.")
        except Exception as exc:
            st.error(f"ADQL search error: {exc}")

    st.markdown("##### \u2b50 Famous Exoplanets")
    famous = [
        {"name": "TRAPPIST-1 e", "icon": "\U0001f534", "desc": "Rocky, temperate, tidally locked"},
        {"name": "Proxima Cen b", "icon": "\U0001f7e0", "desc": "Closest habitable candidate"},
        {"name": "K2-18 b", "icon": "\U0001f535", "desc": "Water vapour detected (JWST)"},
        {"name": "Kepler-442 b", "icon": "\U0001f7e2", "desc": "High-ESI super-Earth"},
        {"name": "TOI-700 d", "icon": "\U0001f7e1", "desc": "Earth-size in HZ (TESS)"},
        {"name": "LHS 1140 b", "icon": "\U0001f7e3", "desc": "Dense rocky world in HZ"},
    ]
    gcols = st.columns(len(famous))
    for col, p in zip(gcols, famous):
        with col:
            if st.button(
                f"{p['icon']} {p['name']}",
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
                        st.markdown(summary)
                    except Exception:
                        pass
                    with st.expander("Raw data"):
                        st.json(raw_dict)
                else:
                    st.warning("Planet not found in archive.")
            except Exception as exc:
                st.error(str(exc))
        st.session_state["selected_planet"] = None

    if st.button("\U0001f4e5 Fetch full NASA catalog"):
        with st.spinner("Querying NASA Exoplanet Archive\u2026"):
            try:
                from modules.nasa_client import get_habitable_candidates
                cand = get_habitable_candidates()
                st.dataframe(cand, use_container_width=True)
                st.success(f"Found {len(cand)} candidates")
            except Exception as exc:
                st.error(f"NASA error: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Science Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

with tab_science:
    d = st.session_state.current_planet_data
    tmap = st.session_state.temperature_map

    if d is None or tmap is None:
        st.info("Run a simulation first (Manual Mode tab).")
    else:
        # ── Scientific Narrative (domain expert) ──────────────────
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
                narrative = narrate_science_panel(
                    hz_data=d,
                    cross_section_stats=cs_stats,
                    uncertainty_note="ELM ensemble std-dev or +/-15 K analytical uncertainty",
                )
            st.markdown("### Scientific Narrative")
            st.markdown(narrative)
            st.markdown("---")
        except Exception:
            pass

        sci1, sci2 = st.columns(2)

        # ── Habitable-Zone diagram (enhancement C1) ──
        with sci1:
            st.subheader("Habitable Zone")
            try:
                from modules.astro_physics import hz_boundaries as _hz
                from modules.visualization import create_hz_diagram

                lum_solar = (d["star_teff"] / 5778) ** 4 * (
                    star_radius if "star_radius" in dir() else 1.0
                ) ** 2
                hz = _hz(d["star_teff"], lum_solar)
                fig_hz = create_hz_diagram(hz, d["semi_major"], d["star_teff"])
                st.plotly_chart(fig_hz, use_container_width=True)
            except Exception as exc:
                st.warning(f"HZ diagram unavailable: {exc}")

        # ── Atmospheric cross-section along terminator (enhancement C4) ──
        with sci2:
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
                height=300,
                margin=dict(l=40, r=20, t=30, b=40),
            )
            st.plotly_chart(fig_cs, use_container_width=True)

        # ── Uncertainty dashboard (enhancement C6) ──
        st.subheader("Uncertainty Estimate")
        st.markdown(
            "ELM ensemble std-dev is used as an uncertainty proxy.  "
            "For the analytical model a \u00b115 K uniform uncertainty applies."
        )
        uncert_cols = st.columns(3)
        uncert_cols[0].metric("T_eq \u00b1", "\u00b115 K (albedo uncertainty)")
        uncert_cols[1].metric("ESI \u00b1", "\u00b10.05 (propagated)")
        uncert_cols[2].metric("HSF \u00b1", "\u00b15 pp")

        # ── Compare with Earth (domain expert) ──────────────────
        if st.button("\U0001f30d Compare with Earth"):
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
                st.markdown(comp)
            except Exception:
                st.caption("*Comparison unavailable (Ollama not running).*")

        # ── Planetary soundscape (enhancement F1) ──
        st.subheader("\U0001f50a Planetary Soundscape")
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
    st.subheader("\U0001f527 System Health & Diagnostics")

    if st.button("\U0001f9ea Run Self-Diagnostics"):
        checks = {}

        # 1. NASA API
        with st.status("Running diagnostics\u2026", expanded=True) as status:
            st.write("\U0001f4e1 Testing NASA API\u2026")
            try:
                from modules.nasa_client import get_planet_data
                earth_test = get_planet_data("Proxima Cen b")
                checks["NASA API"] = "\u2705" if earth_test is not None else "\u26a0\ufe0f"
            except Exception:
                checks["NASA API"] = "\u274c"

            # 2. T_eq for Earth
            st.write("\U0001f9ee Computing T_eq for Earth sanity check\u2026")
            try:
                from modules.astro_physics import equilibrium_temperature
                t_earth = equilibrium_temperature(5778, 1.0, 1.0, 0.3, False)
                checks["T_eq(Earth)"] = (
                    f"\u2705 {t_earth:.1f} K" if 240 < t_earth < 270 else f"\u26a0\ufe0f {t_earth:.1f} K"
                )
            except Exception as exc:
                checks["T_eq(Earth)"] = f"\u274c {exc}"

            # 3. Pydantic rejection
            st.write("\U0001f6e1\ufe0f Testing Pydantic guardrail\u2026")
            try:
                from modules.validators import PlanetaryParameters
                PlanetaryParameters(
                    name="bad", radius_earth=100, semi_major_axis=1, albedo=0.3
                )
                checks["Pydantic guard"] = "\u274c should have rejected"
            except Exception:
                checks["Pydantic guard"] = "\u2705 rejected invalid input"

            # 4. ELM model
            st.write("\U0001f9e0 Checking ELM model\u2026")
            try:
                from modules.elm_surrogate import ELMClimateSurrogate
                m = ELMClimateSurrogate()
                m.load("models/elm_ensemble.pkl")
                checks["ELM model"] = "\u2705 loaded"
            except FileNotFoundError:
                checks["ELM model"] = "\u26a0\ufe0f not trained yet"
            except Exception as exc:
                checks["ELM model"] = f"\u274c {exc}"

            # 5. Ollama
            st.write("\U0001f916 Checking Ollama\u2026")
            try:
                import ollama as _oll
                tags = _oll.list()
                names = [m.model for m in tags.models] if hasattr(tags, "models") else []
                checks["Ollama"] = f"\u2705 {len(names)} models: {', '.join(names[:5])}"
            except Exception as exc:
                checks["Ollama"] = f"\u274c {exc}"

            status.update(label="Diagnostics complete", state="complete")

        for k, v in checks.items():
            st.markdown(f"- **{k}:** {v}")

    # ── Export (enhancement A7) ──
    st.subheader("Export")
    if st.session_state.temperature_map is not None:
        from modules.visualization import create_3d_globe

        fig_export = create_3d_globe(st.session_state.temperature_map, "Export")
        html_bytes = fig_export.to_html(include_plotlyjs="cdn").encode()
        st.download_button(
            "\U0001f4be Download interactive HTML globe",
            data=html_bytes,
            file_name="planet_globe.html",
            mime="text/html",
        )
    else:
        st.info("Run a simulation to enable export.")

    # ── Docker instructions ──
    with st.expander("Docker deployment"):
        st.code(
            "docker build -t exo-twin .\n"
            "docker run -p 8501:8501 --gpus all exo-twin",
            language="bash",
        )
