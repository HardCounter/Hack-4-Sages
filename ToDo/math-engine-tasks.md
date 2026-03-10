# Math Engine Tasks ⭐⭐⭐
**File to edit:** `modules/astro_physics.py`
**Also edit:** `app.py` (UI wiring)

This is the highest-priority implementation. All three missing features are physics/classification functions that live here.

---

## Task 1 — Add `classify_radius_gap()` to `astro_physics.py`

Paste this function after the existing `estimate_density` function:

```python
# ─── Fulton Gap / Radius Valley ───────────────────────────────────────────────

def classify_radius_gap(radius_earth: float) -> Dict[str, object]:
    """Classify planet into radius valley regimes (Fulton et al. 2017).

    The scarcity of planets between 1.5–2.0 R⊕ separates rocky
    super-Earths (atmosphere stripped by photoevaporation) from
    gas-envelope-retaining sub-Neptunes.
    """
    GAP_LOW = 1.5
    GAP_HIGH = 2.0
    GAP_MID = (GAP_LOW + GAP_HIGH) / 2.0

    if radius_earth < GAP_LOW:
        classification = "rocky_super_earth"
        label = "Rocky Super-Earth"
        atmosphere_retention = "likely_lost"
    elif radius_earth <= GAP_HIGH:
        classification = "radius_gap"
        label = "Radius Gap (unstable)"
        atmosphere_retention = "uncertain"
    elif radius_earth <= 3.5:
        classification = "sub_neptune"
        label = "Sub-Neptune"
        atmosphere_retention = "likely_retained"
    else:
        classification = "giant"
        label = "Gas Giant"
        atmosphere_retention = "retained"

    gap_proximity = max(0.0, 1.0 - abs(radius_earth - GAP_MID) / (GAP_MID - GAP_LOW))

    return {
        "classification": classification,
        "label": label,
        "atmosphere_retention": atmosphere_retention,
        "gap_proximity": round(float(gap_proximity), 3),
    }
```

---

## Task 2 — Add `assess_sulfur_chemistry()` to `astro_physics.py`

Paste after `classify_radius_gap`:

```python
# ─── Sulfur Chemistry ─────────────────────────────────────────────────────────

def assess_sulfur_chemistry(
    t_eq: float,
    surface_pressure_bar: float,
    atmosphere_type: str = "h2_rich",
) -> Dict[str, object]:
    """Predict sulfur speciation from T_eq, pressure, and atmosphere type.

    Three atmosphere regimes: 'h2_rich', 'o2_rich', 'ch4_co2'.
    Based on Zahnle et al. (2016) sulfur photochemistry constraints.
    """
    # Determine dominant gas phase
    h2so4_condensation = t_eq > 400 and surface_pressure_bar > 10
    h2s_condensation = t_eq < 300 and surface_pressure_bar < 1

    if h2so4_condensation:
        dominant_gas = "H2SO4"
        cloud_condensates = ["H2SO4"]
        regime = "oxidising"
    elif t_eq > 300 and surface_pressure_bar > 1:
        dominant_gas = "SO2"
        cloud_condensates = ["SO2"]
        regime = "mixed"
    else:
        dominant_gas = "H2S"
        cloud_condensates = ["H2S"] if h2s_condensation else []
        regime = "reducing"

    # Surface mineral mapping
    mineral_map = {
        "h2_rich":  ["FeS", "FeS2"],
        "o2_rich":  ["CaSO4", "FeSO4"],
        "ch4_co2":  ["FeS", "CaSO4"],
    }
    surface_minerals = mineral_map.get(atmosphere_type, ["FeS"])

    notes_parts = []
    if h2so4_condensation:
        notes_parts.append("Venus-like H2SO4 cloud deck expected.")
    if h2s_condensation:
        notes_parts.append("H2S condensation at surface level.")
    if not notes_parts:
        notes_parts.append("Moderate sulfur activity, SO2 dominated.")

    return {
        "dominant_gas": dominant_gas,
        "cloud_condensates": cloud_condensates,
        "surface_minerals": surface_minerals,
        "regime": regime,
        "h2s_condensation": h2s_condensation,
        "h2so4_condensation": h2so4_condensation,
        "atmosphere_type": atmosphere_type,
        "notes": " ".join(notes_parts),
    }
```

---

## Task 3 — Add `assess_co_ratio()` to `astro_physics.py`

Paste after `assess_sulfur_chemistry`:

```python
# ─── Carbon-to-Oxygen Ratio ───────────────────────────────────────────────────

def assess_co_ratio(co_ratio: float) -> Dict[str, object]:
    """Classify planetary composition from the C/O ratio.

    Solar C/O ≈ 0.55. Earth C/O ≈ 0.50.
    Above 0.8 → carbon planet (graphite/carbide surface, no oceans).
    """
    if co_ratio < 0.55:
        classification = "water_world_candidate"
        label = "Water World Candidate"
        ocean_likelihood = "high"
        dominant_surface = "silicates_water"
        atmosphere_bias = "CO2_H2O"
        habitability_modifier = 0.15
    elif co_ratio <= 0.80:
        classification = "transitional"
        label = "Solar-like Composition"
        ocean_likelihood = "moderate"
        dominant_surface = "mixed"
        atmosphere_bias = "mixed"
        habitability_modifier = 0.0
    else:
        classification = "carbon_planet"
        label = "Carbon Planet"
        ocean_likelihood = "low"
        dominant_surface = "graphite_carbides"
        atmosphere_bias = "CH4_dominated"
        habitability_modifier = -0.4

    return {
        "classification": classification,
        "label": label,
        "ocean_likelihood": ocean_likelihood,
        "dominant_surface": dominant_surface,
        "atmosphere_bias": atmosphere_bias,
        "habitability_modifier": round(habitability_modifier, 3),
        "co_ratio": round(co_ratio, 3),
    }
```

---

## Task 4 — Wire into `app.py`

### 4a. Add sliders to the sidebar
Find the existing planet parameter sliders block and add:

```python
co_ratio = st.sidebar.slider("C/O Ratio", 0.1, 1.5, 0.55, 0.01,
    help="Carbon-to-Oxygen ratio. Solar ≈ 0.55")
surface_pressure = st.sidebar.slider("Surface Pressure (bar)", 0.001, 100.0, 1.0, 0.1,
    help="Surface atmospheric pressure in bar")
atm_type = st.sidebar.selectbox("Atmosphere Type",
    ["h2_rich", "o2_rich", "ch4_co2"],
    help="Dominant atmospheric regime for sulfur chemistry")
```

### 4b. Call the new functions inside the simulation pipeline
After the `T_eq` is computed, add:

```python
from modules.astro_physics import assess_co_ratio, assess_sulfur_chemistry, classify_radius_gap

rg = classify_radius_gap(planet_radius)
sulfur = assess_sulfur_chemistry(T_eq, surface_pressure, atm_type)
co = assess_co_ratio(co_ratio)
```

Store them in `st.session_state.current_planet_data`.

### 4c. Display in the planet card
Add three compact metric/badge rows:

```python
# Radius Gap
st.markdown(f"**Radius Class:** {rg['label']}  —  atmosphere: {rg['atmosphere_retention']}")
if rg['classification'] == 'radius_gap':
    st.progress(rg['gap_proximity'], text="Gap proximity")

# Sulfur Chemistry  
st.markdown(f"**Sulfur:** dominant gas `{sulfur['dominant_gas']}` | regime: `{sulfur['regime']}`")
if sulfur['h2so4_condensation']:
    st.warning("⚠️ Venus-like H₂SO₄ clouds predicted")

# C/O
adjusted_esi = esi + co['habitability_modifier']
st.markdown(f"**C/O:** {co['label']} | ESI adjusted: `{adjusted_esi:.3f}`")
if co['classification'] == 'carbon_planet':
    st.error("Carbon-rich composition — liquid water unlikely")
```

---

## Checklist
- [ ] `classify_radius_gap()` added to `astro_physics.py`
- [ ] `assess_sulfur_chemistry()` added to `astro_physics.py`
- [ ] `assess_co_ratio()` added to `astro_physics.py`
- [ ] Three sliders added to sidebar in `app.py`
- [ ] Functions called in simulation pipeline
- [ ] Results stored in `session_state`
- [ ] Results displayed in planet card
