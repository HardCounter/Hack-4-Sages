# Implementation Guide: Missing Features

Three features are currently **documented but not implemented**. This file tells you exactly what to build, where to put it, and what the output should look like.

---

## 1. Fulton Gap / Radius Valley Classification

### What it does
Given a planet's radius (in Earth radii), classify it into one of three regimes based on the observed scarcity of planets between 1.5–2.0 R⊕.

### Where to add it
`modules/astro_physics.py` — add a new function:

```python
def classify_radius_gap(radius_earth: float) -> dict:
    """
    Classify a planet based on the Fulton Gap (radius valley).

    Parameters
    ----------
    radius_earth : float
        Planet radius in Earth radii.

    Returns
    -------
    dict with keys:
        - "classification": str  →  "rocky_super_earth" | "radius_gap" | "sub_neptune" | "giant"
        - "label": str           →  human-readable label
        - "gap_proximity": float →  0.0–1.0, how close to the gap centre (1.0 = dead centre)
        - "atmosphere_retention": str → "likely_lost" | "uncertain" | "likely_retained"
    """
```

### Boundaries
| Radius (R⊕) | Classification |
|---|---|
| < 1.5 | `rocky_super_earth` — atmosphere likely stripped |
| 1.5 – 2.0 | `radius_gap` — uncertain, atmospherically unstable |
| 2.0 – 3.5 | `sub_neptune` — gas envelope retained |
| > 3.5 | `giant` — full gas planet |

### Where to display it in app.py
- Add a badge/metric next to planet radius in the main planet card
- Use colour coding: green = rocky or sub-neptune, orange = gap zone
- Show `gap_proximity` as a small progress bar if the planet is in the gap zone

---

## 2. Sulfur Chemistry Constraints

### What it does
Given equilibrium temperature (`T_eq` in Kelvin) and surface pressure (in bar), predict:
1. Which sulfur **gas species** dominate the atmosphere
2. Which sulfur **surface minerals** are stable
3. What **atmospheric regime** the planet is in

### Where to add it
`modules/astro_physics.py` — add a new function:

```python
def assess_sulfur_chemistry(
    t_eq: float,
    surface_pressure_bar: float,
    atmosphere_type: str = "h2_rich",  # "h2_rich" | "o2_rich" | "ch4_co2"
) -> dict:
    """
    Predict sulfur speciation based on temperature, pressure, and atmosphere type.

    Returns
    -------
    dict with keys:
        - "dominant_gas": str        →  "H2S" | "H2SO4" | "SO2" | "S2"
        - "cloud_condensates": list  →  e.g. ["H2S", "H2SO4"]
        - "surface_minerals": list   →  e.g. ["FeS", "FeS2", "CaSO4"]
        - "regime": str              →  "reducing" | "oxidising" | "mixed"
        - "h2s_condensation": bool   →  True if H2S clouds form
        - "h2so4_condensation": bool →  True if H2SO4 clouds form
        - "notes": str               →  short human-readable summary
    """
```

### Logic rules
| Condition | Result |
|---|---|
| `T_eq > 400 K` AND `P > 10 bar` | H₂SO₄ clouds form (Venus-like) |
| `T_eq < 300 K` AND `P < 1 bar` | H₂S gas dominates |
| `atmosphere_type == "h2_rich"` | Surface minerals: FeS, FeS₂ |
| `atmosphere_type == "o2_rich"` | Surface minerals: CaSO₄, FeSO₄ |
| `atmosphere_type == "ch4_co2"` | Surface minerals: mixed FeS + CaSO₄ |

### Where to display it in app.py
- Add a **"Sulfur Chemistry"** panel in the scientific detail section (alongside the existing ISA panel)
- Show `dominant_gas` and `cloud_condensates` as tags/chips
- Show `surface_minerals` as a bullet list
- Use a warning icon if `h2so4_condensation` is True (Venus-like runaway)
- The panel should **react live** when the user changes `T_eq` or surface pressure sliders

---

## 3. Carbon-to-Oxygen (C/O) Ratio

### What it does
Given the C/O ratio of the planet/system, classify the planet's bulk composition and shift its habitability interpretation.

### Where to add it
`modules/astro_physics.py` — add a new function:

```python
def assess_co_ratio(co_ratio: float) -> dict:
    """
    Classify a planet's composition based on the Carbon-to-Oxygen ratio.

    Parameters
    ----------
    co_ratio : float
        Carbon-to-Oxygen ratio (solar value ≈ 0.55).

    Returns
    -------
    dict with keys:
        - "classification": str   →  "water_world_candidate" | "transitional" | "carbon_planet"
        - "label": str            →  human-readable label
        - "ocean_likelihood": str →  "high" | "moderate" | "low"
        - "dominant_surface": str →  "silicates_water" | "mixed" | "graphite_carbides"
        - "atmosphere_bias": str  →  "CO2_H2O" | "mixed" | "CH4_dominated"
        - "habitability_modifier": float → multiplicative factor applied to ESI score (-1.0 to +1.0)
    """
```

### Boundaries
| C/O | Classification | Interpretation |
|---|---|---|
| < 0.55 | `water_world_candidate` | Oxygen-rich, water/silicate surface, Earth-like |
| 0.55 – 0.80 | `transitional` | Solar-like, mixed composition |
| > 0.80 | `carbon_planet` | Carbon-rich, dry, graphite/carbide surface, no oceans |

Solar C/O ≈ 0.55. Earth C/O ≈ 0.5.

### Where to display it in app.py
- Add a **C/O Ratio** input slider in the planet parameter sidebar (range 0.1 – 1.5, default 0.55)
- Display the classification as a badge in the planet card
- Apply `habitability_modifier` to the ESI score display (show both raw and adjusted)
- If `carbon_planet`, add a red warning: *"Carbon-rich composition — liquid water unlikely"*

---

## Integration Checklist

- [ ] Add `classify_radius_gap()` to `modules/astro_physics.py`
- [ ] Add `assess_sulfur_chemistry()` to `modules/astro_physics.py`
- [ ] Add `assess_co_ratio()` to `modules/astro_physics.py`
- [ ] Add C/O ratio slider to sidebar in `app.py`
- [ ] Add surface pressure slider to sidebar in `app.py` (needed for sulfur chemistry)
- [ ] Add Radius Gap badge to planet card in `app.py`
- [ ] Add Sulfur Chemistry panel to scientific detail section in `app.py`
- [ ] Add C/O classification badge + ESI modifier to planet card in `app.py`
- [ ] Write unit tests in `tests/` for all three new functions
