"""
Standalone LLM helper functions.

These call Ollama models directly (not through the LangChain agent)
and are imported by individual app tabs for domain-expert
interpretation, classification, narration, and query generation.

All public functions degrade gracefully: if the LLM is unreachable
they return a short fallback string rather than raising.

The active model names are resolved from ``AgentMode`` so that
single-LLM mode routes all requests through the domain-expert model
while dual-LLM mode uses separate models for orchestration vs expertise.
"""

import json
import logging
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    import ollama as _oll
    _HAS_OLLAMA = True
except ImportError:
    _oll = None
    _HAS_OLLAMA = False
    logger.warning("ollama package not installed — LLM helpers will return fallbacks.")

# ─── Model resolution ─────────────────────────────────────────────────────────

_DOMAIN_MODEL = "astrosage"
_ORCHESTRATOR_MODEL = "qwen2.5:14b"
_ACTIVE_MODE: str = "dual_llm"


def set_llm_mode(mode: str) -> None:
    """Set the active LLM mode for helper routing.

    Accepted values: ``"dual_llm"``, ``"single_llm"``, ``"deterministic"``.
    """
    global _ACTIVE_MODE
    _ACTIVE_MODE = mode
    logger.info("LLM helper mode set to %s", mode)


def _resolve_orchestrator_model() -> str:
    if _ACTIVE_MODE == "single_llm":
        return _DOMAIN_MODEL
    return _ORCHESTRATOR_MODEL


# ─── LaTeX post-processing ─────────────────────────────────────────────────────

_LATEX_HINT = (
    "Format physical quantities and equations using LaTeX math notation. "
    "Use $...$ for inline math (e.g. $T_{eq}$, $R_{\\oplus}$, $L/L_{\\odot}$) "
    "and $$...$$ for standalone equations.\n\n"
)


def sanitize_latex(text: str) -> str:
    """Fix common LLM LaTeX issues before rendering with st.markdown()."""
    text = re.sub(r"```latex\s*\n?(.*?)```", r"$$\1$$", text, flags=re.DOTALL)
    # \[ ... \] → $$ ... $$ (Streamlit uses dollar-sign delimiters)
    text = re.sub(r"\\\[\s*", "$$", text)
    text = re.sub(r"\s*\\\]", "$$", text)
    # \( ... \) → $ ... $
    text = re.sub(r"\\\(\s*", "$", text)
    text = re.sub(r"\s*\\\)", "$", text)
    # Drop trailing lone $ to avoid rendering glitches
    inline_count = len(re.findall(r"(?<!\$)\$(?!\$)", text))
    if inline_count % 2 != 0:
        idx = text.rfind("$")
        text = text[:idx] + text[idx + 1:]
    return text


# ─── Low-level callers ────────────────────────────────────────────────────────


def _ask_domain(prompt: str) -> str:
    """Send a single prompt to the AstroSage domain expert."""
    if not _HAS_OLLAMA:
        raise ImportError("ollama package not installed")
    resp = _oll.chat(
        model=_DOMAIN_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp["message"]["content"]


def _ask_orchestrator(prompt: str) -> str:
    """Send a single prompt to the orchestrator (or domain model in single-LLM mode)."""
    if not _HAS_OLLAMA:
        raise ImportError("ollama package not installed")
    model = _resolve_orchestrator_model()
    resp = _oll.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp["message"]["content"]


def _safe(fn, *args, fallback: str = "") -> str:
    """Call *fn* and return its result; on any error return *fallback*."""
    try:
        return fn(*args)
    except Exception:
        return fallback


# ─── Domain-expert helpers (astro-agent) ──────────────────────────────────────

def interpret_simulation(results: Dict) -> str:
    """3-4 sentence scientific interpretation of simulation results.

    Used in Manual Mode after every simulation run.
    """
    prompt = (
        "You are an astrophysics expert. Given the following simulation "
        "results for an exoplanet, write a concise 3-4 sentence scientific "
        "interpretation. Mention the climate state (Eyeball, Lobster, "
        "Greenhouse, or Temperate), habitability prospects, and key risks. "
        "Use Kelvin for temperatures.\n\n"
        + _LATEX_HINT
        + f"Results:\n{json.dumps(results, indent=2, default=str)}"
    )
    return _safe(_ask_domain, prompt,
                 fallback="*AI interpretation unavailable — Ollama not running.*")


def classify_climate_state(
    T_min: float, T_max: float, T_mean: float, tidally_locked: bool
) -> Dict[str, str]:
    """Classify climate topology into Eyeball / Lobster / Greenhouse / Temperate.

    Returns ``{"state": "...", "confidence": "high|medium|low", "reason": "..."}``.
    """
    prompt = (
        "You are an astrophysics classifier. Given the surface temperature "
        "statistics of a planet, classify the climate state as exactly ONE of: "
        "Eyeball, Lobster, Greenhouse, Temperate.\n\n"
        "Respond ONLY with a JSON object: "
        '{"state": "...", "confidence": "high|medium|low", "reason": "one sentence"}. '
        "In the reason field, use LaTeX for quantities (e.g. $T_{mean}$).\n\n"
        f"T_min={T_min:.1f} K, T_max={T_max:.1f} K, T_mean={T_mean:.1f} K, "
        f"tidally_locked={tidally_locked}"
    )
    raw = _safe(_ask_domain, prompt,
                fallback='{"state": "Unknown", "confidence": "low", "reason": "LLM unavailable"}')
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"state": "Unknown", "confidence": "low", "reason": raw[:200]}


def review_elm_output(
    params: Dict, T_min: float, T_max: float, T_mean: float
) -> str:
    """Domain-expert physics plausibility check on ELM output.

    Returns a short verdict: starts with 'Plausible' or 'Warning'.
    """
    prompt = (
        "You are a physics reviewer. Given the input parameters and the "
        "ELM surrogate output below, assess whether the temperature "
        "distribution is physically plausible. Respond in ONE sentence "
        "starting with either 'Plausible:' or 'Warning:'. "
        "Use LaTeX math notation for quantities (e.g. $T_{min}$, $T_{max}$).\n\n"
        f"Input: {json.dumps(params, default=str)}\n"
        f"Output: T_min={T_min:.1f} K, T_max={T_max:.1f} K, T_mean={T_mean:.1f} K"
    )
    return _safe(_ask_domain, prompt, fallback="Review unavailable.")


def summarise_planet_data(planet_data: Dict) -> str:
    """Human-friendly 3-sentence summary of raw NASA planet data.

    Used in the Catalog tab when a planet is selected.
    """
    prompt = (
        "You are an astrophysics communicator. Summarise the following "
        "NASA Exoplanet Archive data in exactly 3 clear sentences for a "
        "general scientific audience. Mention the planet name, key "
        "parameters (radius, mass, temperature, distance), and one "
        "interesting fact.\n\n"
        + _LATEX_HINT
        + f"Data:\n{json.dumps(planet_data, indent=2, default=str)}"
    )
    return _safe(_ask_domain, prompt, fallback="*Summary unavailable.*")


def narrate_science_panel(
    hz_data: Optional[Dict],
    cross_section_stats: Dict,
    uncertainty_note: str,
) -> str:
    """One-paragraph narrative for the Science Dashboard.

    Explains the HZ diagram, terminator cross-section, and
    uncertainty in the context of the current planet.
    """
    prompt = (
        "You are writing the 'Scientific Narrative' section of an "
        "exoplanet analysis dashboard. In one paragraph (4-5 sentences), "
        "explain what the Habitable Zone diagram and the terminator "
        "cross-section reveal about this planet's habitability. "
        "Mention where the planet sits relative to the HZ boundaries, "
        "describe the day-night temperature gradient, and note the "
        "uncertainty caveats.\n\n"
        + _LATEX_HINT
        + f"HZ boundaries (AU): {json.dumps(hz_data, default=str)}\n"
        f"Cross-section stats: {json.dumps(cross_section_stats, default=str)}\n"
        f"Uncertainty: {uncertainty_note}"
    )
    return _safe(_ask_domain, prompt, fallback="*Narrative unavailable.*")


def compare_planets(planet_a: Dict, planet_b: Dict) -> str:
    """Comparative habitability analysis of two planets.

    Used in the Science tab's 'Compare with Earth' feature and
    in the agent's compare_two_planets tool.
    """
    prompt = (
        "You are an astrobiologist. Compare these two planets in terms "
        "of habitability. Write 4-5 sentences covering temperature, "
        "stellar environment, size, and overall habitability prospects. "
        "Be specific with numbers.\n\n"
        + _LATEX_HINT
        + f"Planet A:\n{json.dumps(planet_a, indent=2, default=str)}\n\n"
        f"Planet B:\n{json.dumps(planet_b, indent=2, default=str)}"
    )
    return _safe(_ask_domain, prompt, fallback="*Comparison unavailable.*")


# ─── Orchestrator helpers (Qwen 2.5) ─────────────────────────────────────────

def generate_adql_query(natural_language: str) -> str:
    """Convert a natural-language question into an ADQL query.

    The query targets the ``pscomppars`` table in the NASA Exoplanet
    Archive.  Used in the Catalog tab's search bar.
    """
    prompt = (
        "You are an expert in ADQL (Astronomical Data Query Language). "
        "Convert the following natural-language question into a valid "
        "ADQL query for the NASA Exoplanet Archive table `pscomppars`.\n\n"
        "Available columns include: pl_name, pl_radj (R_Jupiter), "
        "pl_bmassj (M_Jupiter), pl_orbsmax (AU), pl_orbper (days), "
        "pl_insol (S_Earth), pl_eqt (K), pl_dens (g/cm3), "
        "st_teff (K), st_rad (R_sun), st_lum (log L_sun), st_mass (M_sun), "
        "sy_dist (pc), disc_year, discoverymethod.\n\n"
        "Return ONLY the SQL query, no explanation.\n\n"
        f"Question: {natural_language}"
    )
    return _safe(_ask_orchestrator, prompt, fallback="")


def generate_smart_suggestions(conversation_summary: str) -> list:
    """Generate 3 contextual follow-up suggestions from the conversation.

    Returns a list of 3 strings.  Used in the Agent AI tab after
    each response.
    """
    prompt = (
        "Based on the following conversation about exoplanets, suggest "
        "exactly 3 short (under 10 words each) follow-up actions the "
        "user might want. Return them as a JSON array of 3 strings.\n\n"
        f"Conversation:\n{conversation_summary[-2000:]}"
    )
    raw = _safe(_ask_orchestrator, prompt, fallback="[]")
    try:
        start = raw.index("[")
        end = raw.rindex("]") + 1
        items = json.loads(raw[start:end])
        if isinstance(items, list) and len(items) >= 3:
            return [str(s) for s in items[:3]]
    except (ValueError, json.JSONDecodeError):
        pass
    return [
        "Run a climate simulation",
        "Compare with Earth",
        "Find similar planets",
    ]
