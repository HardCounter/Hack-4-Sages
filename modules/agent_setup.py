"""
LangChain agent orchestration — configurable single/dual-model setup.

Runtime modes
-------------
* ``AgentMode.DUAL_LLM``   : Qwen 2.5-14B orchestrator + AstroSage domain expert (default).
* ``AgentMode.SINGLE_LLM`` : astro-agent handles both orchestration and interpretation.
* ``AgentMode.DETERMINISTIC``: No LLM — deterministic tools only (physics + ML).

The active mode is selected at agent-creation time via ``build_agent(mode)``.
"""

import json
import logging
from enum import Enum
from typing import Any, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


# ─── Agent visualization state ────────────────────────────────────────────────

def _update_agent_viz(**kwargs):
    """Accumulate visualization artifacts for the Agent AI tab.

    Stores data in ``st.session_state["_agent_viz"]`` so the Streamlit UI
    can render the same interactive dashboards as Manual Mode.  Silently
    no-ops when called outside a Streamlit context (e.g. in tests).
    """
    try:
        import streamlit as _st
        if "_agent_viz" not in _st.session_state:
            _st.session_state["_agent_viz"] = {}
        _st.session_state["_agent_viz"].update(kwargs)
    except Exception:
        pass


# ─── Runtime modes ─────────────────────────────────────────────────────────────


class AgentMode(str, Enum):
    DUAL_LLM = "dual_llm"
    SINGLE_LLM = "single_llm"
    DETERMINISTIC = "deterministic"


# ─── LLM factory ──────────────────────────────────────────────────────────────

_BASE_URL = "http://localhost:11434"


def _make_llm(model: str) -> ChatOllama:
    return ChatOllama(
        model=model,
        temperature=0.3,
        num_ctx=8192,
        base_url=_BASE_URL,
    )


_primary_llm: Optional[ChatOllama] = None
_domain_llm: Optional[ChatOllama] = None


def _get_primary_llm() -> ChatOllama:
    global _primary_llm
    if _primary_llm is None:
        _primary_llm = _make_llm("qwen2.5:14b")
    return _primary_llm


def _get_domain_llm() -> ChatOllama:
    global _domain_llm
    if _domain_llm is None:
        _domain_llm = _make_llm("astro-agent")
    return _domain_llm


def get_domain_llm_for_mode(mode: AgentMode) -> Optional[ChatOllama]:
    """Return the domain-expert LLM for the given mode, or None."""
    if mode == AgentMode.DETERMINISTIC:
        return None
    if mode == AgentMode.SINGLE_LLM:
        return _get_domain_llm()
    return _get_domain_llm()


# Legacy module-level aliases (kept for backward compat with llm_helpers.py)
primary_llm = property(lambda self: _get_primary_llm())
domain_llm = property(lambda self: _get_domain_llm())


# ─── Tool definitions ─────────────────────────────────────────────────────────

@tool
def query_nasa_archive(planet_name: str) -> str:
    """Fetch observational data for a planet from the NASA Exoplanet Archive.

    Use this whenever the user asks about a specific exoplanet.

    Args:
        planet_name: Catalogue name, e.g. 'Proxima Cen b', 'TRAPPIST-1 e'.
    """
    from modules.nasa_client import get_planet_data

    data = get_planet_data(planet_name)
    if data is None:
        return f"Planet not found: {planet_name}"
    _update_agent_viz(planet_name=planet_name)
    return data.to_json()


@tool
def search_planet_catalog(
    radius_min_earth: float = 0.5,
    radius_max_earth: float = 2.5,
    mass_min_earth: float = 0.0,
    mass_max_earth: float = 20.0,
    eccentricity_min: float = 0.0,
    eccentricity_max: float = 1.0,
    teff_min: int = 2500,
    teff_max: int = 7000,
    max_results: int = 20,
) -> str:
    """Search the NASA Exoplanet Archive catalog with physical criteria.

    Use this when the user asks to FIND or FILTER planets by properties
    (e.g. "terrestrial planets with high eccentricity", "rocky planets
    around M-dwarfs", "super-Earths in the habitable zone").

    Do NOT use query_nasa_archive for catalog searches — that tool only
    looks up a single planet by its exact catalogue name.

    Args:
        radius_min_earth: Minimum planet radius [R_Earth] (default 0.5).
        radius_max_earth: Maximum planet radius [R_Earth] (default 2.5).
        mass_min_earth: Minimum planet mass [M_Earth] (default 0.0).
        mass_max_earth: Maximum planet mass [M_Earth] (default 20.0).
        eccentricity_min: Minimum orbital eccentricity (default 0.0).
        eccentricity_max: Maximum orbital eccentricity (default 1.0).
        teff_min: Minimum host star temperature [K] (default 2500).
        teff_max: Maximum host star temperature [K] (default 7000).
        max_results: Maximum number of results to return (default 20).
    """
    from modules.nasa_client import query_nasa_archive as _nasa_query

    R_JUP_PER_EARTH = 1.0 / 11.209
    M_JUP_PER_EARTH = 1.0 / 317.83

    r_min_jup = radius_min_earth * R_JUP_PER_EARTH
    r_max_jup = radius_max_earth * R_JUP_PER_EARTH
    m_min_jup = mass_min_earth * M_JUP_PER_EARTH
    m_max_jup = mass_max_earth * M_JUP_PER_EARTH

    clauses = [
        f"pl_radj BETWEEN {r_min_jup:.6f} AND {r_max_jup:.6f}",
        f"st_teff BETWEEN {teff_min} AND {teff_max}",
        "pl_radj IS NOT NULL",
        "pl_bmassj IS NOT NULL",
        "st_teff IS NOT NULL",
        "pl_orbsmax IS NOT NULL",
    ]
    if mass_min_earth > 0 or mass_max_earth < 20:
        clauses.append(f"pl_bmassj BETWEEN {m_min_jup:.8f} AND {m_max_jup:.6f}")
    if eccentricity_min > 0 or eccentricity_max < 1:
        clauses.append("pl_orbeccen IS NOT NULL")
        clauses.append(
            f"pl_orbeccen BETWEEN {eccentricity_min:.4f} AND {eccentricity_max:.4f}"
        )

    where = " AND ".join(clauses)
    adql = (
        "SELECT TOP " + str(min(max_results, 50)) + " "
        "pl_name, pl_radj, pl_bmassj, pl_orbsmax, pl_orbper, "
        "pl_orbeccen, pl_insol, pl_eqt, pl_dens, "
        "st_teff, st_rad, st_lum, st_mass "
        f"FROM pscomppars WHERE {where} "
        "ORDER BY pl_insol ASC"
    )

    try:
        df = _nasa_query(adql)
    except Exception as exc:
        return f"NASA catalog search failed: {exc}"

    if df.empty:
        return "No planets matched the search criteria."

    summary_rows = []
    for _, row in df.iterrows():
        r_e = float(row.get("pl_radj", 0)) * 11.209
        m_e = float(row.get("pl_bmassj", 0)) * 317.83
        ecc = row.get("pl_orbeccen", "?")
        summary_rows.append(
            f"- **{row.get('pl_name', '?')}**: "
            f"R={r_e:.2f} R⊕, M={m_e:.2f} M⊕, "
            f"a={row.get('pl_orbsmax', '?')} AU, "
            f"e={ecc}, "
            f"T*={row.get('st_teff', '?')} K"
        )

    return (
        f"Found {len(df)} planets matching criteria:\n"
        + "\n".join(summary_rows)
        + "\n\nUse query_nasa_archive(planet_name) to get full data for any of these, "
        "then compute_habitability and run_climate_simulation for visual analysis."
    )


@tool
def compute_habitability(
    stellar_temp: float,
    stellar_radius: float,
    planet_radius_jup: float,
    planet_mass_jup: float,
    semi_major_axis: float,
    albedo: float = 0.3,
    tidally_locked: bool = True,
    eccentricity: float = 0.0,
) -> str:
    """Compute habitability indices (T_eq, ESI, SEPHI, flux).

    Call this after retrieving NASA data.  Results are automatically
    visualised in the dashboard (ESI gauge, SEPHI badges, HZ diagram, etc.).

    Args:
        stellar_temp: Host star effective temperature [K].
        stellar_radius: Host star radius [R_sun].
        planet_radius_jup: Planet radius [R_Jupiter].
        planet_mass_jup: Planet mass [M_Jupiter].
        semi_major_axis: Orbital semi-major axis [AU].
        albedo: Bond albedo (default 0.3).
        tidally_locked: Whether the planet is tidally locked.
        eccentricity: Orbital eccentricity (default 0.0).
    """
    from modules.astro_physics import (
        assess_co_ratio,
        assess_sulfur_chemistry,
        classify_radius_gap,
        compute_full_analysis,
        hz_boundaries,
    )

    result = compute_full_analysis(
        stellar_temp,
        stellar_radius,
        planet_radius_jup,
        planet_mass_jup,
        semi_major_axis,
        albedo,
        tidally_locked,
        eccentricity,
    )

    _update_agent_viz(
        analysis=result,
        star_teff=stellar_temp,
        star_radius=stellar_radius,
        semi_major=semi_major_axis,
        planet_radius=result["radius_earth"],
        planet_mass=result["mass_earth"],
        albedo=albedo,
        eccentricity=eccentricity,
        tidally_locked=tidally_locked,
    )

    try:
        lum = (stellar_temp / 5778) ** 4 * stellar_radius ** 2
        hz = hz_boundaries(stellar_temp, lum)
        _update_agent_viz(hz_boundaries=hz)
    except Exception:
        pass

    try:
        rg = classify_radius_gap(result["radius_earth"])
        _update_agent_viz(radius_gap=rg)
    except Exception:
        pass

    try:
        sulfur = assess_sulfur_chemistry(result["T_eq_K"], 1.0, "h2_rich")
        co = assess_co_ratio(0.55)
        _update_agent_viz(sulfur=sulfur, co=co)
    except Exception:
        pass

    return json.dumps(result, indent=2, default=str)


@tool
def classify_planet_radius_gap(radius_earth: float) -> str:
    """Classify a planet's radius relative to the Fulton Gap (radius valley).

    Use this when the user asks about planet type, atmosphere retention,
    sub-Neptune vs super-Earth distinction, or the Fulton Gap.

    Args:
        radius_earth: Planet radius in Earth radii.
    """
    from modules.astro_physics import classify_radius_gap
    result = classify_radius_gap(radius_earth)
    _update_agent_viz(radius_gap=result)
    return json.dumps(result, indent=2)


@tool
def predict_sulfur_chemistry(
    t_eq: float,
    surface_pressure_bar: float,
    atmosphere_type: str = "h2_rich",
) -> str:
    """Predict sulfur chemistry species and surface minerals for a planet.

    Use this when the user asks about atmospheric composition, sulfur clouds,
    Venus-like conditions, surface mineralogy, or H2S / H2SO4 formation.

    Args:
        t_eq: Equilibrium temperature in Kelvin.
        surface_pressure_bar: Surface atmospheric pressure in bar.
        atmosphere_type: One of 'h2_rich', 'o2_rich', 'ch4_co2'.
    """
    from modules.astro_physics import assess_sulfur_chemistry
    result = assess_sulfur_chemistry(t_eq, surface_pressure_bar, atmosphere_type)
    _update_agent_viz(sulfur=result)
    return json.dumps(result, indent=2)


@tool
def assess_carbon_oxygen_ratio(co_ratio: float) -> str:
    """Assess planetary composition based on the Carbon-to-Oxygen ratio.

    Use this when the user asks about C/O ratio, water worlds, carbon planets,
    ocean likelihood, or how the C/O ratio changes ESI/habitability.

    Args:
        co_ratio: Carbon-to-Oxygen ratio (solar ≈ 0.55, Earth ≈ 0.50).
    """
    from modules.astro_physics import assess_co_ratio
    result = assess_co_ratio(co_ratio)
    _update_agent_viz(co=result)
    return json.dumps(result, indent=2)


@tool
def run_climate_simulation(
    radius_earth: float,
    mass_earth: float,
    semi_major_axis_au: float,
    star_teff_K: float,
    star_radius_solar: float,
    insol_earth: float,
    albedo: float = 0.3,
    tidally_locked: int = 1,
) -> str:
    """Run the climate surrogate to predict a surface temperature map.

    Uses ELM → PINNFormer → Analytical fallback chain.  The resulting
    temperature map, cloud overlay, and derived metrics are automatically
    displayed as an interactive 3-D globe in the dashboard.

    Args:
        radius_earth: Planet radius [R⊕].
        mass_earth: Planet mass [M⊕].
        semi_major_axis_au: Semi-major axis [AU].
        star_teff_K: Star effective temperature [K].
        star_radius_solar: Star radius [R_sun].
        insol_earth: Instellation [S_Earth].
        albedo: Bond albedo (default 0.3).
        tidally_locked: 1 if tidally locked, 0 otherwise.
    """
    from modules.astro_physics import equilibrium_temperature, habitable_surface_fraction
    from modules.elm_surrogate import ELMClimateSurrogate
    from modules.visualization import generate_eyeball_map

    locked = bool(tidally_locked)
    T_eq = equilibrium_temperature(
        star_teff_K, star_radius_solar, semi_major_axis_au, albedo, locked,
    )

    params = {
        "radius_earth": radius_earth,
        "mass_earth": mass_earth,
        "semi_major_axis_au": semi_major_axis_au,
        "star_teff_K": star_teff_K,
        "star_radius_solar": star_radius_solar,
        "insol_earth": insol_earth,
        "albedo": albedo,
        "tidally_locked": tidally_locked,
    }

    cloud_map = None
    climate_method = "Analytical Fallback"

    try:
        model = ELMClimateSurrogate()
        model.load("models/elm_ensemble.pkl")
        temp_map = model.predict_from_params(params)
        climate_method = "ELM Ensemble"
    except Exception:
        try:
            from modules.pinnformer3d import (
                load_pinnformer,
                sample_cloud_map,
                sample_surface_map,
            )
            pinn = load_pinnformer()
            temp_map = sample_surface_map(pinn, T_eq, tidally_locked=locked)
            cloud_map = sample_cloud_map(
                pinn, n_lat=temp_map.shape[0], n_lon=temp_map.shape[1],
            )
            climate_method = "PINNFormer 3-D"
        except Exception:
            temp_map = generate_eyeball_map(T_eq, tidally_locked=locked)

    hsf = habitable_surface_fraction(temp_map)

    _update_agent_viz(
        temperature_map=temp_map,
        cloud_map=cloud_map,
        climate_method=climate_method,
        T_min=float(temp_map.min()),
        T_max=float(temp_map.max()),
        T_mean=float(temp_map.mean()),
        hsf=hsf,
        star_teff=star_teff_K,
        star_radius=star_radius_solar,
        semi_major=semi_major_axis_au,
        planet_radius=radius_earth,
        planet_mass=mass_earth,
        albedo=albedo,
        tidally_locked=locked,
    )

    result = {
        "T_min_K": round(float(temp_map.min()), 1),
        "T_max_K": round(float(temp_map.max()), 1),
        "T_mean_K": round(float(temp_map.mean()), 1),
        "T_std_K": round(float(temp_map.std()), 1),
        "map_shape": list(temp_map.shape),
        "has_liquid_water": bool(273 <= temp_map.mean() <= 373),
        "HSF": round(hsf, 4),
        "climate_method": climate_method,
    }
    return json.dumps(result, indent=2)


@tool
def consult_domain_expert(question: str, context: Any = "") -> str:
    """Ask the astrophysics domain-expert model for a scientific interpretation.

    Call this after computing habitability metrics or running a
    simulation to get expert-level interpretation of the results.

    Args:
        question: A scientific question to pose to the domain expert.
        context: Optional supporting data (string or JSON) to give the
            expert more context.  Can be a plain string or a dict/list
            which will be serialised automatically.
    """
    if not isinstance(context, str):
        try:
            context = json.dumps(context, indent=2, default=str)
        except Exception:
            context = str(context)

    llm = _get_domain_llm()
    latex_hint = (
        "Use LaTeX math notation for quantities and equations: "
        "$...$ for inline, $$...$$ for display.\n\n"
    )
    full_prompt = latex_hint + question
    if context:
        full_prompt = (
            f"Context data:\n{context}\n\n"
            f"{latex_hint}Question: {question}"
        )
    response = llm.invoke(full_prompt)
    return response.content


@tool
def discover_most_habitable(top_n: int = 5) -> str:
    """Query NASA for habitable-zone candidates, rank them by ESI,
    and ask the domain expert to evaluate the top results.

    This is a multi-step tool that exercises both LLMs.

    Args:
        top_n: How many top planets to return (default 5).
    """
    from modules.nasa_client import get_habitable_candidates
    from modules.astro_physics import (
        compute_esi,
        equilibrium_temperature,
        estimate_density,
        estimate_escape_velocity,
    )

    cand = get_habitable_candidates()
    if cand is None or cand.empty:
        return "No habitable-zone candidates found."

    R_JUP_TO_EARTH = 11.209
    M_JUP_TO_EARTH = 317.83

    scored = []
    for _, row in cand.iterrows():
        try:
            r_e = float(row.get("pl_radj", 0)) * R_JUP_TO_EARTH
            m_e = float(row.get("pl_bmassj", 0)) * M_JUP_TO_EARTH
            a = float(row.get("pl_orbsmax", 0))
            t_star = float(row.get("st_teff", 0))
            r_star = float(row.get("st_rad", 0))
            if r_e <= 0 or m_e <= 0 or a <= 0 or t_star <= 0 or r_star <= 0:
                continue
            T_eq = equilibrium_temperature(t_star, r_star, a, 0.3, True)
            dens = estimate_density(m_e, r_e)
            v_esc = estimate_escape_velocity(m_e, r_e)
            esi = compute_esi(r_e, dens, v_esc, T_eq)
            scored.append({
                "name": row.get("pl_name", "?"),
                "ESI": round(esi, 4),
                "T_eq_K": round(T_eq, 1),
                "R_earth": round(r_e, 2),
                "M_earth": round(m_e, 2),
            })
        except Exception:
            continue

    scored.sort(key=lambda x: x["ESI"], reverse=True)
    top = scored[:top_n]

    expert_prompt = (
        "Rank and evaluate these exoplanet candidates for habitability. "
        "For each planet give one sentence of scientific reasoning. "
        "Use LaTeX math notation for quantities ($T_{eq}$, $R_{\\oplus}$, etc.).\n\n"
        + json.dumps(top, indent=2)
    )
    expert_opinion = _get_domain_llm().invoke(expert_prompt).content
    return f"Top {top_n} by ESI:\n{json.dumps(top, indent=2)}\n\nDomain-expert evaluation:\n{expert_opinion}"


@tool
def compare_two_planets(planet_a_name: str, planet_b_name: str) -> str:
    """Fetch data for two planets, compute indices, and produce a
    comparative habitability analysis using the domain expert.

    Args:
        planet_a_name: First planet catalogue name.
        planet_b_name: Second planet catalogue name.
    """
    from modules.nasa_client import get_planet_data
    from modules.astro_physics import compute_full_analysis

    R_JUP_TO_EARTH = 11.209
    M_JUP_TO_EARTH = 317.83

    results = {}
    for name in (planet_a_name, planet_b_name):
        row = get_planet_data(name)
        if row is None:
            return f"Planet not found: {name}"
        d = row.to_dict()
        try:
            analysis = compute_full_analysis(
                float(d.get("st_teff", 5778)),
                float(d.get("st_rad", 1.0)),
                float(d.get("pl_radj", 0.1)),
                float(d.get("pl_bmassj", 0.003)),
                float(d.get("pl_orbsmax", 1.0)),
                0.3,
                True,
            )
            d.update(analysis)
        except Exception:
            pass
        results[name] = d

    expert_prompt = (
        "Compare these two exoplanets in terms of habitability. "
        "Write 4-5 sentences covering temperature, stellar environment, "
        "size, and overall prospects. "
        "Use LaTeX math notation for quantities ($T_{eq}$, $R_{\\oplus}$, etc.).\n\n"
        f"Planet A ({planet_a_name}):\n{json.dumps(results[planet_a_name], indent=2, default=str)}\n\n"
        f"Planet B ({planet_b_name}):\n{json.dumps(results[planet_b_name], indent=2, default=str)}"
    )
    comparison = _get_domain_llm().invoke(expert_prompt).content
    return comparison


@tool
def detect_anomalous_planets(top_n: int = 5) -> str:
    """Find statistically unusual exoplanets using Isolation Forest.

    Anomalous planets may represent rare habitable candidates or
    data quality issues. Useful for the 'anomaly detection in
    imbalanced datasets' approach.

    Args:
        top_n: Number of top anomalies to return (default 5).
    """
    from modules.nasa_client import get_habitable_candidates
    from modules.anomaly_detection import get_top_anomalies

    cand = get_habitable_candidates()
    if cand is None or cand.empty:
        return "No data available for anomaly detection."
    anomalies = get_top_anomalies(cand, n=top_n)
    cols = ["pl_name", "pl_radj", "pl_bmassj", "pl_orbsmax", "st_teff", "anomaly_score"]
    display_cols = [c for c in cols if c in anomalies.columns]
    return (
        f"Top {top_n} anomalous planets (lower score = more unusual):\n"
        + anomalies[display_cols].to_string(index=False)
    )


@tool
def cite_scientific_literature(query: str, topics: str = "") -> str:
    """Search the indexed scientific literature (~40 peer-reviewed papers) for
    relevant citations using hybrid semantic + keyword search.

    Use this to back up claims with real citations. Returns the most relevant
    paper abstracts, key quantitative findings, and formatted references.

    The corpus covers: habitable zones, ESI/SEPHI, tidal locking, climate
    modeling (GCM, clouds, OHT), biosignatures & false positives, atmospheric
    escape, planetary interiors, stellar context, JWST observations, and
    astrobiology.

    Args:
        query: A scientific topic or question to find citations for.
            Be specific, e.g. "habitable zone boundaries M-dwarf stars"
            rather than just "habitable zone".
        topics: Optional comma-separated topic filter tags. Available tags
            include: habitable_zone, m_dwarf, tidal_locking, biosignatures,
            false_positives, atmospheric_escape, climate_modeling, gcm,
            cloud_feedback, ocean_heat_transport, planetary_interior,
            plate_tectonics, mass_radius, stellar_activity, uv_environment,
            jwst, transit_spectroscopy, astrobiology, photosynthesis.
            Leave empty to search all papers.
    """
    from modules.rag_citations import cite_literature, format_citations_markdown

    topic_list = [t.strip() for t in topics.split(",") if t.strip()] or None
    citations = cite_literature(query, n_results=5, topics=topic_list)
    if not citations:
        return "No relevant citations found."
    details = []
    for c in citations:
        block = (
            f"**{c['authors']} ({c['year']})** — {c['title']}\n"
            f"_{c['journal']}_\n{c['abstract'][:500]}"
        )
        if c.get("key_findings"):
            findings = c["key_findings"]
            if isinstance(findings, list):
                block += "\nKey findings: " + "; ".join(findings[:3])
        details.append(block)
    formatted = format_citations_markdown(citations)
    return "\n\n".join(details) + "\n\n" + formatted


# ─── Tool registry ────────────────────────────────────────────────────────────

tools = [
    query_nasa_archive,
    search_planet_catalog,
    compute_habitability,
    run_climate_simulation,
    classify_planet_radius_gap,
    predict_sulfur_chemistry,
    assess_carbon_oxygen_ratio,
    consult_domain_expert,
    discover_most_habitable,
    compare_two_planets,
    detect_anomalous_planets,
    cite_scientific_literature,
]

# ─── Agent prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are AstroAgent — an autonomous assistant for exoplanet analysis.

CAPABILITIES
1. Retrieve data for a SPECIFIC planet by name (query_nasa_archive)
2. SEARCH the catalog by physical criteria — radius, mass, eccentricity, etc. (search_planet_catalog)
3. Compute habitability indices: T_eq, ESI, SEPHI, stellar flux (compute_habitability)
4. Run a climate surrogate for temperature-map prediction (run_climate_simulation)
5. Consult an astrophysics domain expert for scientific interpretation (consult_domain_expert)
6. Discover the most habitable planets in the archive (discover_most_habitable)
7. Compare two planets side-by-side (compare_two_planets)
8. Detect anomalous planets in the catalog (detect_anomalous_planets)
9. Cite scientific literature to support claims (cite_scientific_literature)
10. Classify a planet's radius relative to the Fulton Gap (classify_planet_radius_gap)
11. Predict sulfur chemistry and surface mineralogy (predict_sulfur_chemistry)
12. Assess planetary composition from the C/O ratio (assess_carbon_oxygen_ratio)

VISUALIZATION
When you call compute_habitability and run_climate_simulation, interactive
visualisations are AUTOMATICALLY generated and displayed to the user below
the conversation.  These include:
- 3-D rotating globe / 2-D heatmap of surface temperature
- ESI gauge, T_eq, HSF, and stellar-flux metrics dashboard
- SEPHI traffic lights, ISA coupling, and false-positive badges
- Composition panel (radius gap, sulfur chemistry, C/O ratio)
- Habitable-zone diagram and terminator cross-section
- CSV export of metrics and temperature map

**Always call BOTH compute_habitability AND run_climate_simulation** when
the user asks for a full analysis, evaluation, or visual output.  Pass
the eccentricity parameter to compute_habitability when available.
The classify_planet_radius_gap, predict_sulfur_chemistry, and
assess_carbon_oxygen_ratio tools also update the dashboard.

PROCEDURE
1. If the user names a specific planet → query_nasa_archive(planet_name).
   If the user wants to FIND/FILTER planets by criteria (radius, mass,
   eccentricity, star type, etc.) → search_planet_catalog with the
   appropriate parameter ranges, then pick the best candidate(s) and
   call query_nasa_archive on each to get full data.
2. From the data → compute habitability metrics (include eccentricity).
3. Run a climate simulation to generate the temperature map and globe.
4. Optionally call classify_planet_radius_gap, predict_sulfur_chemistry,
   or assess_carbon_oxygen_ratio for deeper composition analysis.
5. Consult the domain expert to interpret the combined results.
6. Synthesise the domain expert's opinion with the numbers into a clear answer.
   Always present raw numeric results alongside any narrative.

CITATION POLICY
- Call cite_scientific_literature when presenting habitability analysis
  or substantive scientific claims. Use specific queries, e.g.:
  "habitable zone boundaries for M-dwarf stars" rather than "habitable zone".
- Use the topics parameter for targeted retrieval when appropriate.
- Reference key findings from retrieved papers to ground your statements.
- Cite when relevant — do not force citations where they add no value.

FORMATTING
- Use LaTeX math notation for all equations, variables, and physical quantities.
- Use $...$ for inline math (e.g. $T_{{eq}} = 255\\,\\text{{K}}$, $R = 1.2\\,R_{{\\oplus}}$).
- Use $$...$$ for standalone equations (e.g. the equilibrium temperature formula).
- Common symbols: $R_{{\\oplus}}$, $M_{{\\oplus}}$, $L_{{\\odot}}$, $T_{{eq}}$, $a$ (semi-major axis).
- Do NOT wrap math in markdown code fences; use dollar-sign delimiters only.

RULES
- Always cite the data source (NASA Exoplanet Archive).
- Never invent parameter values — always use tool outputs.
- Express temperatures in Kelvin.
- ESI is in [0, 1]; 1.0 = identical to Earth.
- Flag uncertainties and model limitations explicitly.
- When comparing planets, prefer the compare_two_planets tool.
- For "find habitable" queries, use discover_most_habitable.
"""

_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


# ─── Agent builder ────────────────────────────────────────────────────────────

def build_agent(mode: AgentMode = AgentMode.DUAL_LLM) -> Optional[AgentExecutor]:
    """Construct an AgentExecutor for the requested runtime mode.

    Returns ``None`` when mode is DETERMINISTIC (no LLM needed).
    """
    if mode == AgentMode.DETERMINISTIC:
        logger.info("AgentMode.DETERMINISTIC — no LLM agent created.")
        return None

    if mode == AgentMode.SINGLE_LLM:
        llm = _get_domain_llm()
        max_iter = 5
        logger.info("AgentMode.SINGLE_LLM — astro-agent as sole LLM.")
    else:
        llm = _get_primary_llm()
        max_iter = 7
        logger.info("AgentMode.DUAL_LLM — Qwen orchestrator + astro-agent expert.")

    agent = create_tool_calling_agent(llm, tools, _PROMPT_TEMPLATE)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=max_iter,
        handle_parsing_errors=True,
    )


# Legacy default executor (backward compat with app.py before mode migration)
agent_executor = build_agent(AgentMode.DUAL_LLM)
