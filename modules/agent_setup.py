"""
LangChain agent orchestration — dual-model setup.

* Primary agent  : Qwen2.5-14B (tool-calling orchestrator)
* Domain expert  : AstroAgent (astro-specialised system prompt)
* Tools          : NASA query, habitability computation, climate simulation
"""

import json

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

# ─── LLM instances ────────────────────────────────────────────────────────────

_BASE_URL = "http://localhost:11434"

primary_llm = ChatOllama(
    model="qwen2.5:14b",
    temperature=0.3,
    num_ctx=8192,
    base_url=_BASE_URL,
)

# Fallback / domain-expert model (astro system prompt baked via Modelfile)
domain_llm = ChatOllama(
    model="astro-agent",
    temperature=0.3,
    num_ctx=8192,
    base_url=_BASE_URL,
)


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
    return data.to_json()


@tool
def compute_habitability(
    stellar_temp: float,
    stellar_radius: float,
    planet_radius_jup: float,
    planet_mass_jup: float,
    semi_major_axis: float,
    albedo: float = 0.3,
    tidally_locked: bool = True,
) -> str:
    """Compute habitability indices (T_eq, ESI, SEPHI, flux).

    Call this after retrieving NASA data.

    Args:
        stellar_temp: Host star effective temperature [K].
        stellar_radius: Host star radius [R_sun].
        planet_radius_jup: Planet radius [R_Jupiter].
        planet_mass_jup: Planet mass [M_Jupiter].
        semi_major_axis: Orbital semi-major axis [AU].
        albedo: Bond albedo (default 0.3).
        tidally_locked: Whether the planet is tidally locked.
    """
    from modules.astro_physics import compute_full_analysis

    result = compute_full_analysis(
        stellar_temp,
        stellar_radius,
        planet_radius_jup,
        planet_mass_jup,
        semi_major_axis,
        albedo,
        tidally_locked,
    )
    return json.dumps(result, indent=2, default=str)


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
    """Run the ELM climate surrogate to predict a surface temperature map.

    Use this to generate the 2-D temperature distribution on a planet.

    Args:
        radius_earth: Planet radius [R_Earth].
        mass_earth: Planet mass [M_Earth].
        semi_major_axis_au: Semi-major axis [AU].
        star_teff_K: Star effective temperature [K].
        star_radius_solar: Star radius [R_sun].
        insol_earth: Instellation [S_Earth].
        albedo: Bond albedo (default 0.3).
        tidally_locked: 1 if tidally locked, 0 otherwise.
    """
    from modules.elm_surrogate import ELMClimateSurrogate
    from modules.visualization import generate_eyeball_map
    from modules.astro_physics import equilibrium_temperature

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

    try:
        model = ELMClimateSurrogate()
        model.load("models/elm_ensemble.pkl")
        temp_map = model.predict_from_params(params)
    except FileNotFoundError:
        T_eq = equilibrium_temperature(
            star_teff_K, star_radius_solar, semi_major_axis_au,
            albedo, bool(tidally_locked),
        )
        temp_map = generate_eyeball_map(T_eq, tidally_locked=bool(tidally_locked))

    result = {
        "T_min_K": round(float(temp_map.min()), 1),
        "T_max_K": round(float(temp_map.max()), 1),
        "T_mean_K": round(float(temp_map.mean()), 1),
        "T_std_K": round(float(temp_map.std()), 1),
        "map_shape": list(temp_map.shape),
        "has_liquid_water": bool(273 <= temp_map.mean() <= 373),
    }
    return json.dumps(result, indent=2)


@tool
def consult_domain_expert(question: str, context: str = "") -> str:
    """Ask the astrophysics domain-expert model for a scientific interpretation.

    ALWAYS call this after computing habitability metrics or running a
    simulation — never present raw numbers without expert interpretation.

    Args:
        question: A scientific question to pose to the domain expert.
        context: Optional JSON-encoded structured data (simulation
            results, NASA data) to give the expert more context.
    """
    full_prompt = question
    if context:
        full_prompt = (
            f"Context data:\n{context}\n\n"
            f"Question: {question}"
        )
    response = domain_llm.invoke(full_prompt)
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
        "For each planet give one sentence of scientific reasoning.\n\n"
        + json.dumps(top, indent=2)
    )
    expert_opinion = domain_llm.invoke(expert_prompt).content
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
        "size, and overall prospects.\n\n"
        f"Planet A ({planet_a_name}):\n{json.dumps(results[planet_a_name], indent=2, default=str)}\n\n"
        f"Planet B ({planet_b_name}):\n{json.dumps(results[planet_b_name], indent=2, default=str)}"
    )
    comparison = domain_llm.invoke(expert_prompt).content
    return comparison


# ─── Tool registry ────────────────────────────────────────────────────────────

tools = [
    query_nasa_archive,
    compute_habitability,
    run_climate_simulation,
    consult_domain_expert,
    discover_most_habitable,
    compare_two_planets,
]

# ─── Agent prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are AstroAgent — an autonomous assistant for exoplanet analysis.

CAPABILITIES
1. Retrieve planet data from the NASA Exoplanet Archive (query_nasa_archive)
2. Compute habitability indices: T_eq, ESI, SEPHI, stellar flux (compute_habitability)
3. Run an ELM climate surrogate for temperature-map prediction (run_climate_simulation)
4. Consult an astrophysics domain expert for scientific interpretation (consult_domain_expert)
5. Discover the most habitable planets in the archive (discover_most_habitable)
6. Compare two planets side-by-side (compare_two_planets)

PROCEDURE
1. When the user asks about a planet → first fetch data from NASA.
2. From the data → compute habitability metrics.
3. ALWAYS consult the domain expert after computing habitability metrics.
   Never present raw numbers without expert interpretation. Pass the
   computed results as the `context` parameter.
4. If a climate simulation is requested or relevant → run it, then
   consult the domain expert again to interpret the temperature map.
5. Synthesise the domain expert's opinion with the numbers into a clear answer.

RULES
- Always cite the data source (NASA Exoplanet Archive).
- Never invent parameter values — always use tool outputs.
- Express temperatures in Kelvin.
- ESI is in [0, 1]; 1.0 = identical to Earth.
- Flag uncertainties and model limitations.
- When comparing planets, prefer the compare_two_planets tool.
- For "find habitable" queries, use discover_most_habitable.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# ─── Agent executor ───────────────────────────────────────────────────────────

agent = create_tool_calling_agent(primary_llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    handle_parsing_errors=True,
)
