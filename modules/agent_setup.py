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
def consult_domain_expert(question: str) -> str:
    """Ask the astrophysics domain-expert model for a scientific interpretation.

    Use this for deep scientific reasoning about habitability, climate
    states, or stellar physics.

    Args:
        question: A scientific question to pose to the domain expert.
    """
    response = domain_llm.invoke(question)
    return response.content


# ─── Tool registry ────────────────────────────────────────────────────────────

tools = [
    query_nasa_archive,
    compute_habitability,
    run_climate_simulation,
    consult_domain_expert,
]

# ─── Agent prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are AstroAgent — an autonomous assistant for exoplanet analysis.

CAPABILITIES
1. Retrieve planet data from the NASA Exoplanet Archive (query_nasa_archive)
2. Compute habitability indices: T_eq, ESI, SEPHI, stellar flux (compute_habitability)
3. Run an ELM climate surrogate for temperature-map prediction (run_climate_simulation)
4. Consult an astrophysics domain expert for scientific interpretation (consult_domain_expert)

PROCEDURE
1. When the user asks about a planet → first fetch data from NASA.
2. From the data → compute habitability metrics.
3. If requested → run a climate simulation.
4. For deep scientific explanations → consult the domain expert.
5. Summarise results clearly.

RULES
- Always cite the data source (NASA Exoplanet Archive).
- Never invent parameter values — always use tool outputs.
- Express temperatures in Kelvin.
- ESI is in [0, 1]; 1.0 = identical to Earth.
- Flag uncertainties and model limitations.
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
