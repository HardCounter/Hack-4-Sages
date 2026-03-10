# LLM Agent Tasks ⭐⭐
**File to edit:** `modules/agent_setup.py`

The AstroAgent currently has no knowledge of Fulton Gap, sulfur chemistry, or C/O ratio. This task adds three new `@tool` functions so the agent can reason about and explain these features to users.

**Prerequisite:** Math engine tasks must be done first — the new `astro_physics.py` functions must exist before the agent can call them.

---

## Task 1 — Add `classify_planet_radius_gap` tool

In `agent_setup.py`, paste a new `@tool` after the existing `compute_habitability` tool:

```python
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
    return json.dumps(result, indent=2)
```

---

## Task 2 — Add `predict_sulfur_chemistry` tool

```python
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
    return json.dumps(result, indent=2)
```

---

## Task 3 — Add `assess_carbon_oxygen_ratio` tool

```python
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
    return json.dumps(result, indent=2)
```

---

## Task 4 — Register the new tools

Find the line where the tools list is assembled (look for a list like `tools = [query_nasa_archive, compute_habitability, ...]`) and add the three new tools:

```python
tools = [
    query_nasa_archive,
    compute_habitability,
    run_climate_simulation,
    classify_planet_radius_gap,       # NEW
    predict_sulfur_chemistry,          # NEW
    assess_carbon_oxygen_ratio,        # NEW
]
```

---

## Task 5 — Update the system prompt

Find the system prompt string (the long string passed to `ChatPromptTemplate`) and append this paragraph to it:

```
You also have access to three new tools: classify_planet_radius_gap, predict_sulfur_chemistry,
and assess_carbon_oxygen_ratio. Use classify_planet_radius_gap whenever a planet's radius
suggests it may sit in the Fulton Gap (1.5–2.0 Earth radii). Use predict_sulfur_chemistry
when discussing surface conditions, clouds, or mineralogy. Use assess_carbon_oxygen_ratio
when the user provides or asks about a C/O ratio, or when discussing whether a planet
could host liquid water vs. being a dry carbon world.
```

---

## What the agent should now be able to do

| User question | Tool called |
|---|---|
| "Is this planet a super-Earth or sub-Neptune?" | `classify_planet_radius_gap` |
| "Could there be sulfur clouds on Kepler-22b?" | `predict_sulfur_chemistry` |
| "What happens if I set C/O to 0.9?" | `assess_carbon_oxygen_ratio` |
| "Why is the ESI adjusted lower?" | `assess_carbon_oxygen_ratio` (explain modifier) |
| "What minerals are on the surface?" | `predict_sulfur_chemistry` |

---

## Checklist
- [ ] `classify_planet_radius_gap` tool added to `agent_setup.py`
- [ ] `predict_sulfur_chemistry` tool added to `agent_setup.py`
- [ ] `assess_carbon_oxygen_ratio` tool added to `agent_setup.py`
- [ ] All three tools added to the `tools = [...]` list
- [ ] System prompt updated with tool usage guidance
