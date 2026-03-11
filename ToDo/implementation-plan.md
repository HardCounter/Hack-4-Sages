# LLM Agent — Detailed Implementation Plan

**Source spec:** `ToDo/llm-agent-tasks.md`  
**Files to modify:**
- `modules/astro_physics.py` — add three backend functions (prerequisite)
- `modules/agent_setup.py` — add three `@tool` wrappers, register them, update system prompt

---

## Current state (as of audit)

### `modules/astro_physics.py`
- ✅ Has: `equilibrium_temperature`, `stellar_flux`, `compute_esi`, `compute_sephi`, `hz_boundaries`, `habitable_surface_fraction`, `estimate_outgassing_rate`, `estimate_isa_interaction`, `estimate_uv_flux`, `assess_biosignature_false_positives`, `compute_full_analysis`
- ❌ Missing: `classify_radius_gap`, `assess_sulfur_chemistry`, `assess_co_ratio`

### `modules/agent_setup.py`
- ✅ Has tools: `query_nasa_archive`, `compute_habitability`, `run_climate_simulation`, `consult_domain_expert`, `discover_most_habitable`, `compare_two_planets`, `detect_anomalous_planets`, `cite_scientific_literature`
- ❌ Missing tools: `classify_planet_radius_gap`, `predict_sulfur_chemistry`, `assess_carbon_oxygen_ratio`
- ❌ `tools = [...]` list does not include the three new tools
- ❌ `SYSTEM_PROMPT` has no mention of the three new tools

---

## Step 0 — Prerequisite: add backend functions to `astro_physics.py`

**Location:** append after `assess_biosignature_false_positives` (before `compute_full_analysis`), around line ~310.

### 0-A — `classify_radius_gap(radius_earth: float) -> dict`

**Purpose:** classify a planet as super-Earth, sub-Neptune, or gap-straddler based on the Fulton radius gap (~1.5–2.0 R⊕).

**Logic to implement:**
```
if radius_earth < 1.5:
    category = "super-Earth"
    description = "Below the radius gap — likely rocky, atmosphere may have been stripped"
elif 1.5 <= radius_earth <= 2.0:
    category = "radius gap (Fulton Gap)"
    description = "Within the valley — uncertain composition, transitional regime"
elif 2.0 < radius_earth <= 4.0:
    category = "sub-Neptune"
    description = "Above the radius gap — likely retains a volatile H/He envelope"
else:
    category = "Neptune-size or larger"
    description = "Large volatile-envelope planet"
```

**Return dict keys:** `radius_earth`, `category`, `description`, `atmosphere_retention_likely` (bool: True if < 1.5 or > 2.0 R⊕ with mass support), `fulton_gap_note`.

---

### 0-B — `assess_sulfur_chemistry(t_eq: float, surface_pressure_bar: float, atmosphere_type: str = "h2_rich") -> dict`

**Purpose:** predict dominant sulfur species and surface mineralogy based on temperature, pressure, and atmospheric redox state.

**Logic to implement (three atmosphere types):**

| atmosphere_type | dominant sulfur species                | surface minerals                                           |
| --------------- | -------------------------------------- | ---------------------------------------------------------- |
| `h2_rich`       | H₂S (cool), SO₂ (>600 K)               | pyrite (FeS₂) if cool; elemental S if mid; SO₂ ice if cold |
| `o2_rich`       | SO₂, H₂SO₄ aerosols (T>400 K + high P) | anhydrite/gypsum (CaSO₄)                                   |
| `ch4_co2`       | COS, H₂S, trace SO₂                    | siderite, mixed sulfides                                   |

**Temperature thresholds (approximate):**
- `h2_rich`: T < 400 K → H₂S dominant; 400–700 K → mixed; > 700 K → SO₂ dominant
- Venus-like condition: `o2_rich` + T > 700 K + P > 50 bar → H₂SO₄ cloud layers likely

**Return dict keys:** `t_eq_K`, `surface_pressure_bar`, `atmosphere_type`, `dominant_sulfur_species` (list), `surface_minerals` (list), `venus_like_conditions` (bool), `h2so4_clouds_likely` (bool), `notes`.

---

### 0-C — `assess_co_ratio(co_ratio: float) -> dict`

**Purpose:** interpret the Carbon-to-Oxygen ratio and its effect on planetary composition, water likelihood, and ESI modifier.

**Logic to implement:**
```
if co_ratio < 0.4:
    regime = "water-rich / oxygen-rich"
    description = "Excess oxygen → abundant silicates, water likely"
    esi_modifier = +0.02
elif 0.4 <= co_ratio <= 0.65:
    regime = "solar-like"
    description = "Near-solar C/O — silicate-dominated, Earth-like mineralogy expected"
    esi_modifier = 0.0
elif 0.65 < co_ratio < 1.0:
    regime = "carbon-enhanced"
    description = "Elevated carbon → graphite crust, carbide minerals, water scarcer"
    esi_modifier = -0.05
else:  # >= 1.0
    regime = "carbon planet"
    description = "C/O ≥ 1 — silicon carbide / graphite interior, water extremely unlikely"
    esi_modifier = -0.15
```

**Return dict keys:** `co_ratio`, `regime`, `description`, `esi_modifier`, `water_likelihood` (one of "high", "moderate", "low", "negligible"), `dominant_minerals` (list), `habitability_note`.

---

## Step 1 — Add `classify_planet_radius_gap` tool to `agent_setup.py`

**Location:** insert after the `compute_habitability` tool (after its closing parenthesis `)`), before `run_climate_simulation`.

**Exact code to insert:**
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

## Step 2 — Add `predict_sulfur_chemistry` tool to `agent_setup.py`

**Location:** insert after `classify_planet_radius_gap`, before `run_climate_simulation`.

**Exact code to insert:**
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

## Step 3 — Add `assess_carbon_oxygen_ratio` tool to `agent_setup.py`

**Location:** insert after `predict_sulfur_chemistry`, before `run_climate_simulation`.

**Exact code to insert:**
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

## Step 4 — Register the three new tools in `tools = [...]`

**Location:** `agent_setup.py`, the `tools` list under `# ─── Tool registry ───`.

**Current list (lines ~340–350):**
```python
tools = [
    query_nasa_archive,
    compute_habitability,
    run_climate_simulation,
    consult_domain_expert,
    discover_most_habitable,
    compare_two_planets,
    detect_anomalous_planets,
    cite_scientific_literature,
]
```

**Target list:**
```python
tools = [
    query_nasa_archive,
    compute_habitability,
    run_climate_simulation,
    classify_planet_radius_gap,       # NEW
    predict_sulfur_chemistry,          # NEW
    assess_carbon_oxygen_ratio,        # NEW
    consult_domain_expert,
    discover_most_habitable,
    compare_two_planets,
    detect_anomalous_planets,
    cite_scientific_literature,
]
```

**Placement rationale:** grouped with the physics-computation tools, before the multi-step / expert tools.

---

## Step 5 — Update `SYSTEM_PROMPT` in `agent_setup.py`

**Location:** the `SYSTEM_PROMPT` string, specifically:
- In the `CAPABILITIES` numbered list — append items 9, 10, 11
- Add a new `NEW TOOL USAGE GUIDANCE` section before the final `"""` closing quote

**Add to CAPABILITIES list (after item 8):**
```
9. Classify a planet's radius relative to the Fulton Gap (classify_planet_radius_gap)
10. Predict sulfur chemistry and surface mineralogy (predict_sulfur_chemistry)
11. Assess planetary composition from the C/O ratio (assess_carbon_oxygen_ratio)
```

**Add new section at end of SYSTEM_PROMPT (before closing `"""`):**
```
NEW TOOL USAGE GUIDANCE
- Use classify_planet_radius_gap whenever a planet's radius suggests it may sit
  in the Fulton Gap (1.5–2.0 Earth radii), or when the user asks about
  sub-Neptune vs super-Earth distinction, or atmosphere retention.
- Use predict_sulfur_chemistry when discussing surface conditions, clouds,
  or mineralogy, or when the user asks about H2S, H2SO4, or Venus-like environments.
- Use assess_carbon_oxygen_ratio when the user provides or asks about a C/O ratio,
  or when discussing whether a planet could host liquid water vs. being a dry
  carbon world. The returned esi_modifier field explains any ESI adjustment.
```

---

## Dependency order (must be respected)

```
Step 0-A  →  Step 0-B  →  Step 0-C    (all in astro_physics.py, can be done in one edit)
                ↓
Step 1  →  Step 2  →  Step 3           (all in agent_setup.py, tool definitions)
                ↓
         Step 4                         (tool registry list)
                ↓
         Step 5                         (system prompt)
```

Steps 0-A/B/C can be done in a single append to `astro_physics.py`.  
Steps 1/2/3 can be done in a single insert block in `agent_setup.py`.  
Steps 4 and 5 are small targeted edits in `agent_setup.py`.

---

---

## Step 6 — UI wiring in `app.py`

**Source spec:** `ToDo/math-engine-tasks.md` Tasks 4a–4c

### 6-A — Add three input widgets to the sidebar (col_params block)

**Location:** `app.py`, inside `with tab_manual:` → `with col_params:`, after the existing `locked = st.checkbox(...)` slider and before the `run_sim = st.button(...)` call (approximately line 240).

**Widgets to add:**
```python
co_ratio = st.sidebar.slider("C/O Ratio", 0.1, 1.5, 0.55, 0.01,
    help="Carbon-to-Oxygen ratio. Solar ≈ 0.55")
surface_pressure = st.sidebar.slider("Surface Pressure (bar)", 0.001, 100.0, 1.0, 0.1,
    help="Surface atmospheric pressure in bar")
atm_type = st.sidebar.selectbox("Atmosphere Type",
    ["h2_rich", "o2_rich", "ch4_co2"],
    help="Dominant atmospheric regime for sulfur chemistry")
```

> **Note:** The spec uses `st.sidebar.*` but the existing sliders use plain `st.*` inside `with col_params:`. For visual consistency, use plain `st.slider` / `st.selectbox` (no `.sidebar`) so the widgets appear inside `col_params`, matching the existing layout.

---

### 6-B — Call the three new functions inside the simulation pipeline

**Location:** inside the `if should_compute:` block → inside the `with st.status(...)` block, after the line `sephi = compute_sephi(T_eq, planet_mass, planet_radius)` (approximately lines 275–280).

**Add to the existing import block** (the `from modules.astro_physics import ...` at the top of the pipeline):
```python
from modules.astro_physics import (
    ...existing imports...,
    assess_co_ratio,
    assess_sulfur_chemistry,
    classify_radius_gap,
)
```

**Add computation calls** after `sephi = compute_sephi(...)`:
```python
_pipeline.write("🪐 Classifying radius gap, sulfur chemistry, C/O ratio…")
rg   = classify_radius_gap(planet_radius)
sulfur = assess_sulfur_chemistry(T_eq, surface_pressure, atm_type)
co   = assess_co_ratio(co_ratio)
```

---

### 6-C — Store new results in `st.session_state.current_planet_data`

**Location:** inside the same pipeline block, in the `st.session_state.current_planet_data = { ... }` dict assignment (approximately lines 295–313).

**Add three new keys:**
```python
st.session_state.current_planet_data = {
    ...existing keys...,
    "radius_gap":    rg,
    "sulfur":        sulfur,
    "co":            co,
}
```

---

### 6-D — Display results in the planet card (`col_viz`)

**Location:** inside `with col_viz:`, after the existing ISA Coupling / False-Positive Risk badge block (after the `except Exception: pass` at approximately line 440), before the `view_mode = st.radio(...)` globe toggle.

**Code to insert:**
```python
# ── Radius Gap / Fulton Valley classification ─────────────────────────
if "radius_gap" in d:
    rg = d["radius_gap"]
    sulfur = d["sulfur"]
    co = d["co"]

    rg_col, sulfur_col, co_col = st.columns(3)

    with rg_col:
        st.markdown(f"**Radius Class:** {rg['label']}")
        st.caption(f"Atmosphere retention: `{rg['atmosphere_retention']}`")
        if rg["classification"] == "radius_gap":
            st.progress(rg["gap_proximity"], text="Gap proximity")

    with sulfur_col:
        st.markdown(f"**Sulfur:** `{sulfur['dominant_gas']}` | `{sulfur['regime']}`")
        st.caption(f"Surface minerals: {', '.join(sulfur['surface_minerals'])}")
        if sulfur["h2so4_condensation"]:
            st.warning("⚠️ Venus-like H₂SO₄ clouds predicted")

    with co_col:
        adjusted_esi = d["ESI"] + co["habitability_modifier"]
        st.markdown(f"**C/O:** {co['label']}")
        st.caption(f"ESI adjusted: `{adjusted_esi:.3f}` (modifier: {co['habitability_modifier']:+.3f})")
        if co["classification"] == "carbon_planet":
            st.error("Carbon-rich — liquid water unlikely")
```

> **Placement note:** inserting before the globe toggle keeps all scalar badges grouped together, above the visual heavy elements.

---

### 6-E — Glossary additions

**Location:** `app.py`, the `_GLOSSARY` dict (approximately lines 63–71).

**Add three new entries:**
```python
"Fulton Gap":  "Scarcity of planets between 1.5–2.0 R⊕ separating rocky super-Earths from sub-Neptunes.",
"C/O Ratio":   "Carbon-to-Oxygen ratio controlling mineralogy; solar ≈ 0.55, C/O ≥ 1 → carbon planet.",
"Sulfur Chemistry": "Dominant sulfur species (H₂S, SO₂, H₂SO₄) determined by temperature, pressure, and redox state.",
```

---

---

## Step 7 — UI Polish: Modern, Responsive, Properly Spaced

**File to edit:** `app.py`  
**Goal:** make the entire UI production-quality — consistent visual language, fluid responsive layout for desktop and mobile, no overflowing or cramped elements.

---

### 7-A — Global CSS overhaul

**Location:** replace the existing `st.markdown("""<style>...</style>""", unsafe_allow_html=True)` block (lines ~33–57).

**Problems with current CSS:**
- No mobile breakpoints — columns collapse badly on narrow screens
- Font sizes are fixed — don't scale with viewport
- No consistent spacing tokens (`--gap`, `--radius`, `--card-bg`)
- Sidebar has no max-width cap — explodes on ultrawide monitors
- Chart containers have hardcoded pixel heights

**Required additions / fixes:**

```css
/* ── CSS custom properties (design tokens) ── */
:root {
  --gap-xs: 0.4rem;
  --gap-sm: 0.75rem;
  --gap-md: 1.25rem;
  --gap-lg: 2rem;
  --radius-card: 12px;
  --card-bg: rgba(10, 14, 39, 0.7);
  --border-subtle: 1px solid rgba(66, 165, 245, 0.2);
  --accent: #00d4ff;
  --accent-dim: #1a237e;
  --text-primary: #e0e0e0;
  --text-muted: #90a4ae;
}

/* ── Sidebar width clamp ── */
[data-testid="stSidebar"] {
  min-width: 220px !important;
  max-width: 320px !important;
}

/* ── All metric cards: consistent padding + border ── */
[data-testid="stMetric"] {
  background: var(--card-bg);
  border: var(--border-subtle);
  border-radius: var(--radius-card);
  padding: var(--gap-sm) var(--gap-md) !important;
}

/* ── Expander headers ── */
[data-testid="stExpanderDetails"] {
  background: var(--card-bg);
  border-radius: 0 0 var(--radius-card) var(--radius-card);
  padding: var(--gap-md);
}

/* ── Tab list: pill-style active tab ── */
.stTabs [aria-selected="true"] {
  background: rgba(0, 212, 255, 0.12) !important;
  border-radius: 8px !important;
  padding: 4px 12px !important;
}

/* ── Button row: always full-width on narrow screens ── */
@media (max-width: 640px) {
  .stButton > button { width: 100% !important; }
  [data-testid="column"] { min-width: 100% !important; flex: unset !important; }
  [data-testid="stSidebar"] { min-width: 100% !important; max-width: 100% !important; }
  .stTabs [data-baseweb="tab"] { font-size: 0.75rem; padding: 6px 8px; }
}

/* ── Plotly chart: respect parent width ── */
.js-plotly-plot, .plot-container { max-width: 100% !important; }

/* ── Chat bubbles: max-width + word-wrap ── */
div[data-testid="stChatMessage"] {
  max-width: 100%;
  word-wrap: break-word;
  margin-bottom: var(--gap-sm);
}

/* ── st.info / st.warning / st.error: rounded + spaced ── */
div[data-testid="stAlert"] {
  border-radius: var(--radius-card) !important;
  margin-bottom: var(--gap-sm);
}

/* ── Slider label + value: readable on dark bg ── */
[data-testid="stSlider"] label, [data-testid="stSlider"] span {
  color: var(--text-primary) !important;
}

/* ── Section dividers (replace plain <hr>) ── */
hr { border-color: rgba(66, 165, 245, 0.15) !important; margin: var(--gap-lg) 0; }

/* ── Progress bar: use accent colour ── */
[data-testid="stProgressBar"] > div > div { background: var(--accent) !important; }
```

---

### 7-B — Header & caption spacing

**Location:** after `st.set_page_config(...)`, around lines 23–24 (the `st.title` / `st.caption` calls).

**Current:**
```python
st.title("🪐 Autonomous Exoplanetary Digital Twin")
st.caption("Simulate alien climates in real time with AI-driven physics surrogates.")
```

**Replace with:**
```python
st.markdown(
    '<h1 style="font-family:\'Orbitron\',sans-serif;'
    'font-size:clamp(1.4rem,3vw,2.2rem);letter-spacing:.04em;'
    'color:#e0e0e0;margin-bottom:0">🪐 Autonomous Exoplanetary Digital Twin</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color:#90a4ae;font-size:clamp(.85rem,1.5vw,1rem);'
    'margin-top:.25rem;margin-bottom:1.5rem">'
    'Simulate alien climates in real time with AI-driven physics surrogates.</p>',
    unsafe_allow_html=True,
)
```

**Why:** `clamp()` scales title font fluidly from mobile (1.4 rem) to desktop (2.2 rem) without media-query hacks.

---

### 7-C — Manual Mode (Tab 2) layout fixes

#### 7-C-1 — Column ratio fix for small screens

**Current:**
```python
col_params, col_viz = st.columns([1, 2])
```

**Replace with:**
```python
col_params, col_viz = st.columns([1, 2], gap="large")
```

The `gap="large"` keyword adds Streamlit's built-in gutter spacing. On narrow viewports Streamlit already stacks columns; the gap prevents them from butting against each other.

#### 7-C-2 — Parameter widgets: add vertical rhythm

Wrap the existing slider block in a styled container:
```python
with st.container(border=True):
    st.markdown("##### ⚙️ Planet Parameters")
    star_teff = st.slider(...)
    ...
    locked = st.checkbox(...)
```

`border=True` (Streamlit ≥ 1.28) draws a subtle border card — matches the metric card style.

#### 7-C-3 — Run button + live toggle: side by side, full width

**Current:** button and toggle are on separate lines.

**Replace with:**
```python
btn_col, toggle_col = st.columns([3, 2], gap="small")
with btn_col:
    run_sim = st.button("🚀 Run Simulation", type="primary", use_container_width=True)
with toggle_col:
    live_mode = st.toggle('Live "What If" mode', value=False)
```

#### 7-C-4 — Metric row: make labels human-readable

**Current:** raw metric labels like `T_eq`, `HSF`.

**Replace with full labels:**
```python
g1.metric("Equilibrium Temp.", f"{d['T_eq']:.0f} K", help="Radiative equilibrium surface temperature")
g2.metric("Earth Similarity (ESI)", f"{d['ESI']:.3f}", delta=..., help="1.0 = identical to Earth")
g3.metric("Habitable Surface", f"{d['HSF']:.1%}", help="Fraction of surface with 273–373 K")
g4.metric("Stellar Flux", f"{d['flux_earth']:.2f} S⊕", help="Relative to Earth's solar constant")
```

#### 7-C-5 — ESI gauge: height responsive via `use_container_width`

Replace hardcoded `height=200` with:
```python
esi_gauge.update_layout(
    height=None,          # let Plotly auto-size
    autosize=True,
    margin=dict(l=10, r=10, t=40, b=10),
    ...
)
```
And pass `use_container_width=True` (already done — keep it).

---

### 7-D — Radius Gap / Sulfur / C/O badge block (Step 6-D) — spacing

The three-column block added in Step 6-D should be wrapped in a container:
```python
with st.container(border=True):
    st.markdown("##### 🔬 Composition & Atmospheric Classification")
    rg_col, sulfur_col, co_col = st.columns(3, gap="medium")
    ...
```

On mobile (< 640 px) the `@media` rule from 7-A collapses these to full-width stacked blocks automatically.

---

### 7-E — Catalog (Tab 3) — famous-planet card grid

**Current:** famous-planet buttons use `st.columns(len(famous))` = 6 fixed columns. This overflows on mobile.

**Replace with a 3×2 responsive grid:**
```python
row1_cols = st.columns(3, gap="small")
row2_cols = st.columns(3, gap="small")
grid_cols = row1_cols + row2_cols
for col, p in zip(grid_cols, famous):
    with col:
        if st.button(f"{p['icon']} {p['name']}", use_container_width=True, help=p["desc"]):
            st.session_state["selected_planet"] = p["name"]
```

---

### 7-F — Science Dashboard (Tab 4) — chart height uniformity

All Plotly figures should share a consistent height token. Create a module-level constant at the top of `app.py`:
```python
_CHART_H = 320   # px — consistent height for all science charts
```

Apply to every `update_layout(height=...)` call in Tab 4:
- `fig_hz.update_layout(height=_CHART_H, ...)`
- `fig_cs.update_layout(height=_CHART_H, ...)`

---

### 7-G — System tab (Tab 5) — export button styling

**Current:** download button has no visual prominence.

**Wrap in a highlighted container:**
```python
with st.container(border=True):
    st.markdown("#### 💾 Download Results")
    st.caption("Export the interactive 3D globe as a self-contained HTML file.")
    st.download_button(
        "📥 Download interactive HTML globe",
        ...
        use_container_width=True,
    )
```

---

### 7-H — Sidebar global improvements

**Add a logo / branding strip at the very top of the sidebar** (before any widgets):
```python
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:1rem 0 .5rem">'
        '<span style="font-family:Orbitron,sans-serif;font-size:1.1rem;'
        'color:#00d4ff;letter-spacing:.08em">🪐 EXOTWIN</span></div>',
        unsafe_allow_html=True,
    )
    st.divider()
```

**Add info footer at the bottom of the sidebar:**
```python
    st.sidebar.markdown("---")
    st.sidebar.caption("Data: NASA Exoplanet Archive · Models: Qwen 2.5-14B, astro-agent · v0.1")
```

---

### 7-I — Typography & colour consistency audit

| Element                            | Current issue                              | Fix                                 |
| ---------------------------------- | ------------------------------------------ | ----------------------------------- |
| `st.subheader` in col_viz          | "Visualization" — generic                  | Change to "🌍 Climate Simulation"    |
| `st.subheader` in col_params       | "🎛️ Parameters" — already good              | Keep                                |
| `st.markdown(f"**SEPHI** ...")`    | Bare markdown, no visual grouping          | Wrap in `st.container(border=True)` |
| ISA / FP badge block               | `isa_col, fp_col = st.columns(2)` — no gap | Add `gap="medium"`                  |
| Tab 4 `sci1, sci2 = st.columns(2)` | No gap                                     | Add `gap="large"`                   |
| Tab 4 `sci3, sci4 = st.columns(2)` | `sci3` is unused dead code                 | Remove or use for Fulton-Gap plot   |

---

### 7-J — Accessibility & usability

- All `st.slider` calls should have `help=` text (tooltip) if not already present.
- All `st.button` calls that trigger expensive operations should show a `st.spinner`.
- `st.warning` / `st.error` messages must be dismissible — add `:dismissible` where Streamlit supports it (≥ 1.33).
- Ensure tab labels use **Unicode symbols + text**, not symbols alone (already done — verify).
- Add `st.set_page_config(..., menu_items={"About": "Autonomous Exoplanetary Digital Twin — Hack4Sages 2026"})` to the `set_page_config` call.

---

### 7-K — Performance: avoid re-rendering on every slider tick

**Current issue:** `live_mode = st.toggle(...)` causes the entire pipeline to re-run on every slider move.

**Fix:** wrap the heavy simulation block in a debounce guard:
```python
# Only re-run in live mode if at least one param changed from previous run
_prev = st.session_state.get("_last_params", {})
_curr = dict(star_teff=star_teff, star_radius=star_radius, planet_radius=planet_radius,
             planet_mass=planet_mass, semi_major=semi_major, albedo=albedo, locked=locked)
_params_changed = (_curr != _prev)

should_compute = run_sim or (live_mode and _params_changed)
if should_compute:
    st.session_state["_last_params"] = _curr
    # ... rest of pipeline
```

---

### 7-L — Remove all emoji characters from the UI

**Location:** entire `app.py`

The current codebase uses emoji extensively — in tab labels, button text, subheaders, status messages, metric labels, and inline markdown. Emoji renders inconsistently across operating systems and screen readers, breaks visual rhythm on professional displays, and looks out of place in a scientific tool. Replace every emoji with either nothing (when purely decorative) or a short text equivalent (when it carries meaning).

**Preserved by explicit design decision:**
- ✅ (`\u2705`) and ❌ (`\u274c`) — kept everywhere they appear (SEPHI, ISA, FP badges, diagnostics output, `_ok` / `_fail` variables)
- Any emoji inside `st.code(...)` or `st.json(...)` blocks — those are data, not UI
- Planetary symbol in the `_GLOSSARY` keys — tooltip text only, not rendered UI


**Full inventory of emoji to remove/replace:**

| Location                                           | Current                                                                       | Replace with                                                                    |
| -------------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `st.set_page_config(page_icon=...)`                | `"\U0001fa90"` (🪐)                                                            | `"ET"` or `"EX"`                                                                |
| `st.title(...)`                                    | `"🪐 Autonomous..."`                                                           | `"Autonomous Exoplanetary Digital Twin"`                                        |
| Tab labels                                         | `"🤖 Agent AI"`, `"🎛️ Manual Mode"`, `"📊 Catalog"`, `"🔬 Science"`, `"🔧 System"` | `"Agent AI"`, `"Manual Mode"`, `"Catalog"`, `"Science"`, `"System"`             |
| Sidebar branding (7-H)                             | `"🪐 EXOTWIN"`                                                                 | `"EXOTWIN"`                                                                     |
| `run_sim` button                                   | `"🚀 Run Simulation"`                                                          | `"Run Simulation"`                                                              |
| `st.subheader("🧠 Reasoning Chain")`                | leading emoji                                                                 | `"Reasoning Chain"`                                                             |
| SEPHI traffic-lights                               | `"\u2705"` / `"\u274c"` checkmarks                                            | **KEEP as-is** — ✅ / ❌ are preserved by design decision                         |
| ISA / FP badges                                    | `"\u2705"` / `"\u274c"` / `"\u26a0\ufe0f"`                                    | **KEEP** ✅ / ❌; replace `"\u26a0\ufe0f"` (⚠️) with `"WARN"`                      |
| `_ok = "\u2705"` / `_fail = "\u274c"` variables    | Unicode emoji                                                                 | **KEEP** — both ✅ and ❌ are intentionally retained                              |
| `state_emoji` dict in classify_climate_state block | all emoji other than ✅/❌                                                      | Remove all entries except any that map to ✅/❌; omit other icons                 |
| Famous-planet icons                                | `"\U0001f534"` etc.                                                           | Remove icon column entirely, keep planet name only                              |
| `discover_most_habitable` button                   | no emoji — already clean                                                      | —                                                                               |
| `st.warning("⚠️ Venus-like...")`                    | leading `⚠️`                                                                   | `st.warning("Venus-like H2SO4 clouds predicted")` — Streamlit adds its own icon |
| `st.error("Carbon-rich...")`                       | no leading emoji here                                                         | already clean                                                                   |
| Status pipeline writes                             | `"\U0001f6e1\ufe0f Validating..."` etc.                                       | `"Validating parameters..."` etc.                                               |
| `st.subheader("📊 Habitable-Zone Candidates")`      | leading emoji                                                                 | `"Habitable-Zone Candidates — NASA Exoplanet Archive"`                          |
| `st.markdown("##### ⭐ Famous Exoplanets")`         | `⭐`                                                                           | `"##### Famous Exoplanets"`                                                     |
| `st.subheader("🔊 Planetary Soundscape")`           | `🔊`                                                                           | `"Planetary Soundscape"`                                                        |
| Generate sound button                              | `"Generate sound"` — clean                                                    | —                                                                               |
| Export download button                             | `"📥 Download..."`                                                             | `"Download interactive HTML globe"`                                             |
| Diagnostics `st.button`                            | `"🧪 Run Self-Diagnostics"`                                                    | `"Run Self-Diagnostics"`                                                        |
| Diagnostics `st.write` calls                       | all have leading emoji                                                        | strip emoji, keep text                                                          |
| Compare with Earth button                          | `"🌍 Compare with Earth"`                                                      | `"Compare with Earth"`                                                          |
| Fetch NASA catalog button                          | `"📥 Fetch full NASA catalog"`                                                 | `"Fetch full NASA catalog"`                                                     |
| `architecture diagram` expander                    | heading text clean                                                            | —                                                                               |
| Docker expander                                    | heading text clean                                                            | —                                                                               |


**Search pattern to catch everything else:**

```bash
grep -Pn '[\x{1F300}-\x{1FAFF}\x{2600}-\x{27BF}\x{FE00}-\x{FE0F}\x{2300}-\x{23FF}]' app.py
```

Verify each hit manually — skip lines containing `\u2705` or `\u274c`.

---

### 7-M — Add "About Us" section

**Locations:** two places — the System tab (`tab_system`) in `app.py`, and the sidebar footer (7-H).

#### 7-M-1 — System tab: "About" expander at the bottom

**Location:** `app.py`, inside `with tab_system:`, after the Docker deployment expander (last item in the tab, approximately line 975).

**Code to insert:**
```python
with st.expander("About this project", expanded=False):
    st.markdown("""
**Autonomous Exoplanetary Digital Twin** was built for the Hack4Sages 2026 hackathon
by the HardCounter team.

It combines a dual-LLM agent (Qwen 2.5-14B + astro-agent), an ELM climate surrogate,
PINNFormer 3D physics-informed neural network, CTGAN data augmentation, and the
NASA Exoplanet Archive to deliver real-time exoplanet habitability analysis.

**Links:**
- GitHub: [HardCounter/Hack-4-Sages](https://github.com/HardCounter/Hack-4-Sages)
- LinkedIn: [HardCounter Team](https://www.linkedin.com/company/hardcounter-team)
""")
```

#### 7-M-2 — Sidebar footer update (extends 7-H)

Replace the plain-text sidebar footer added in 7-H with one that includes the links:

**Current (from 7-H):**
```python
st.sidebar.caption("Data: NASA Exoplanet Archive · Models: Qwen 2.5-14B, astro-agent · v0.1")
```

**Replace with:**
```python
st.sidebar.markdown(
    "<div style='font-size:0.72rem;color:#607d8b;padding-top:0.5rem;line-height:1.6'>"
    "Data: NASA Exoplanet Archive<br>"
    "Models: Qwen 2.5-14B, astro-agent<br>"
    "<a href='https://github.com/HardCounter/Hack-4-Sages' "
    "style='color:#42a5f5' target='_blank'>GitHub</a> &nbsp;|&nbsp; "
    "<a href='https://www.linkedin.com/company/hardcounter-team' "
    "style='color:#42a5f5' target='_blank'>LinkedIn</a>"
    "</div>",
    unsafe_allow_html=True,
)
```

**Why two locations:** the sidebar footer is always visible regardless of which tab is active, giving persistent branding. The System tab expander provides a fuller project description for judges and technical reviewers.

---

### Summary of files touched by Step 7

| Sub-step | Location in `app.py`                | Change type                         |
| -------- | ----------------------------------- | ----------------------------------- |
| 7-A      | Global CSS block (~line 33)         | Replace entire `<style>` block      |
| 7-B      | Title/caption (~line 140)           | Replace 2 lines                     |
| 7-C-1    | `st.columns([1,2])` (~line 225)     | Add `gap="large"`                   |
| 7-C-2    | Slider block (~line 228)            | Wrap in `st.container(border=True)` |
| 7-C-3    | `run_sim` / `live_mode` (~line 242) | New 2-col layout                    |
| 7-C-4    | Metric labels (~line 360)           | Update 4 `st.metric` calls          |
| 7-C-5    | ESI gauge layout (~line 375)        | Remove `height=200`                 |
| 7-D      | Radius Gap badge (~Step 6-D insert) | Wrap in container                   |
| 7-E      | Famous-planet grid (~line 555)      | 3×2 grid instead of 1×6             |
| 7-F      | Chart heights (Tab 4)               | `_CHART_H` constant + apply         |
| 7-G      | Export section (~line 900)          | Wrap in container                   |
| 7-H      | Sidebar (top + bottom)              | Add branding + footer               |
| 7-I      | Various column/subheader labels     | Minor label/gap fixes               |
| 7-J      | All sliders/buttons                 | Add `help=`, accessibility attrs    |
| 7-K      | `should_compute` guard (~line 245)  | Param-change debounce               |
| **7-L**  | Entire `app.py`                     | Strip emoji except ✅/❌              |
| **7-M**  | System tab + sidebar footer         | Add About Us section                |

---

## Updated dependency order

```
Step 0-A/B/C  (astro_physics.py — three new functions)
      ↓
Step 1/2/3    (agent_setup.py — three @tool wrappers)     ║  Step 6-A  (app.py — sidebar sliders)
      ↓                                                    ↓
Step 4        (agent_setup.py — tools list)        Step 6-B/C  (app.py — pipeline calls + session_state)
      ↓                                                    ↓
Step 5        (agent_setup.py — system prompt)     Step 6-D/E  (app.py — display block + glossary)
                                                           ↓
                                                   Step 7-A–K  (app.py — UI polish, can be done after
                                                               Steps 6-A–E since 7-D wraps the 6-D block)··• pip install -r requirements.txt 
Defaulting to user installation because normal site-packages is not writeable
Collecting streamlit==1.41.0 (from -r requirements.txt (line 1))
  Using cached streamlit-1.41.0-py2.py3-none-any.whl.metadata (8.5 kB)
Collecting pandas==2.2.3 (from -r requirements.txt (line 2))
  Using cached pandas-2.2.3.tar.gz (4.4 MB)
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Installing backend dependencies ... done
  Preparing metadata (pyproject.toml) ... error
  error: subprocess-exited-with-error
  
  × Preparing metadata (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> [140 lines of output]
      + meson setup /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0 /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/.mesonpy-05ydv4z0/build -Dbuildtype=release -Db_ndebug=if-release -Db_vscrt=md --vsenv --native-file=/tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/.mesonpy-05ydv4z0/build/meson-python-native-file.ini
      The Meson build system
      Version: 1.2.1
      Source dir: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0
      Build dir: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/.mesonpy-05ydv4z0/build
      Build type: native build
      Project name: pandas
      Project version: 2.2.3
      C compiler for the host machine: cc (gcc 15.2.1 "cc (GCC) 15.2.1 20260123 (Red Hat 15.2.1-7)")
      C linker for the host machine: cc ld.bfd 2.45.1-4
      C++ compiler for the host machine: c++ (gcc 15.2.1 "c++ (GCC) 15.2.1 20260123 (Red Hat 15.2.1-7)")
      C++ linker for the host machine: c++ ld.bfd 2.45.1-4
      Cython compiler for the host machine: cython (cython 3.0.12)
      Host machine cpu family: x86_64
      Host machine cpu: x86_64
      Program python found: YES (/usr/bin/python3)
      Found pkg-config: /usr/bin/pkg-config (2.3.0)
      Run-time dependency python found: YES 3.14
      Build targets in project: 53
      
      pandas 2.2.3
      
        User defined options
          Native files: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/.mesonpy-05ydv4z0/build/meson-python-native-file.ini
          buildtype   : release
          vsenv       : True
          b_ndebug    : if-release
          b_vscrt     : md
      
      Found ninja-1.13.0.git.kitware.jobserver-pipe-1 at /tmp/pip-build-env-v491pk58/normal/bin/ninja
      
      Visual Studio environment is needed to run Ninja. It is recommended to use Meson wrapper:
      /tmp/pip-build-env-v491pk58/overlay/bin/meson compile -C .
      + /tmp/pip-build-env-v491pk58/normal/bin/ninja
      [1/151] Generating pandas/_libs/algos_common_helper_pxi with a custom command
      [2/151] Generating pandas/_libs/index_class_helper_pxi with a custom command
      [3/151] Generating pandas/_libs/algos_take_helper_pxi with a custom command
      [4/151] Generating pandas/_libs/intervaltree_helper_pxi with a custom command
      [5/151] Generating pandas/_libs/khash_primitive_helper_pxi with a custom command
      [6/151] Generating pandas/_libs/hashtable_class_helper_pxi with a custom command
      [7/151] Generating pandas/_libs/sparse_op_helper_pxi with a custom command
      [8/151] Generating pandas/_libs/hashtable_func_helper_pxi with a custom command
      [9/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/base.pyx
      [10/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/ccalendar.pyx
      [11/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/nattype.pyx
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:188:38: noexcept clause is ignored for function returning Python object
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:193:40: noexcept clause is ignored for function returning Python object
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/nattype.pyx:79:0: Global name __nat_unpickle matched from within class scope in contradiction to to Python 'class private name' rules. This may change in a future release.
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/nattype.pyx:79:0: Global name __nat_unpickle matched from within class scope in contradiction to to Python 'class private name' rules. This may change in a future release.
      [12/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/dtypes.pyx
      [13/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/np_datetime.pyx
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:188:38: noexcept clause is ignored for function returning Python object
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:193:40: noexcept clause is ignored for function returning Python object
      [14/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/arrays.pyx
      [15/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/fields.pyx
      [16/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/indexing.pyx
      [17/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/tzconversion.pyx
      [18/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/vectorized.pyx
      [19/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/conversion.pyx
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:188:38: noexcept clause is ignored for function returning Python object
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:193:40: noexcept clause is ignored for function returning Python object
      [20/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/period.pyx
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:188:38: noexcept clause is ignored for function returning Python object
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:193:40: noexcept clause is ignored for function returning Python object
      [21/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/offsets.pyx
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:188:38: noexcept clause is ignored for function returning Python object
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:193:40: noexcept clause is ignored for function returning Python object
      [22/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/timezones.pyx
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:188:38: noexcept clause is ignored for function returning Python object
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:193:40: noexcept clause is ignored for function returning Python object
      [23/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/timedeltas.pyx
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:188:38: noexcept clause is ignored for function returning Python object
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:193:40: noexcept clause is ignored for function returning Python object
      [24/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/parsing.pyx
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:188:38: noexcept clause is ignored for function returning Python object
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:193:40: noexcept clause is ignored for function returning Python object
      [25/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/strptime.pyx
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:188:38: noexcept clause is ignored for function returning Python object
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:193:40: noexcept clause is ignored for function returning Python object
      [26/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/hashing.pyx
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:188:38: noexcept clause is ignored for function returning Python object
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:193:40: noexcept clause is ignored for function returning Python object
      [27/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/timestamps.pyx
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:188:38: noexcept clause is ignored for function returning Python object
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:193:40: noexcept clause is ignored for function returning Python object
      [28/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/ops_dispatch.pyx
      [29/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/properties.pyx
      [30/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/byteswap.pyx
      [31/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/missing.pyx
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:188:38: noexcept clause is ignored for function returning Python object
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:193:40: noexcept clause is ignored for function returning Python object
      [32/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/internals.pyx
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:188:38: noexcept clause is ignored for function returning Python object
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:193:40: noexcept clause is ignored for function returning Python object
      [33/151] Compiling C object pandas/_libs/tslibs/base.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_tslibs_base.pyx.c.o
      FAILED: [code=1] pandas/_libs/tslibs/base.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_tslibs_base.pyx.c.o
      cc -Ipandas/_libs/tslibs/base.cpython-314-x86_64-linux-gnu.so.p -Ipandas/_libs/tslibs -I../../pandas/_libs/tslibs -I../../../../pip-build-env-v491pk58/overlay/lib64/python3.14/site-packages/numpy/_core/include -I../../pandas/_libs/include -I/usr/include/python3.14 -fvisibility=hidden -fdiagnostics-color=always -DNDEBUG -D_FILE_OFFSET_BITS=64 -w -std=c11 -O3 -DNPY_NO_DEPRECATED_API=0 -DNPY_TARGET_VERSION=NPY_1_21_API_VERSION -fPIC -MD -MQ pandas/_libs/tslibs/base.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_tslibs_base.pyx.c.o -MF pandas/_libs/tslibs/base.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_tslibs_base.pyx.c.o.d -o pandas/_libs/tslibs/base.cpython-314-x86_64-linux-gnu.so.p/meson-generated_pandas__libs_tslibs_base.pyx.c.o -c pandas/_libs/tslibs/base.cpython-314-x86_64-linux-gnu.so.p/pandas/_libs/tslibs/base.pyx.c
      pandas/_libs/tslibs/base.cpython-314-x86_64-linux-gnu.so.p/pandas/_libs/tslibs/base.pyx.c:16:10: fatal error: Python.h: No such file or directory
         16 | #include "Python.h"
            |          ^~~~~~~~~~
      compilation terminated.
      [34/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/testing.pyx
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:188:38: noexcept clause is ignored for function returning Python object
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:193:40: noexcept clause is ignored for function returning Python object
      [35/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/sas.pyx
      [36/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/ops.pyx
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:188:38: noexcept clause is ignored for function returning Python object
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:193:40: noexcept clause is ignored for function returning Python object
      [37/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/reshape.pyx
      [38/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/index.pyx
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:188:38: noexcept clause is ignored for function returning Python object
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:193:40: noexcept clause is ignored for function returning Python object
      [39/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/parsers.pyx
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:188:38: noexcept clause is ignored for function returning Python object
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:193:40: noexcept clause is ignored for function returning Python object
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/parsers.pyx:1605:18: noexcept clause is ignored for function returning Python object
      [40/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/window/indexers.pyx
      [41/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/writers.pyx
      [42/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslib.pyx
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:188:38: noexcept clause is ignored for function returning Python object
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:193:40: noexcept clause is ignored for function returning Python object
      [43/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/lib.pyx
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:188:38: noexcept clause is ignored for function returning Python object
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:193:40: noexcept clause is ignored for function returning Python object
      [44/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/window/aggregations.pyx
      [45/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/interval.pyx
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:188:38: noexcept clause is ignored for function returning Python object
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:193:40: noexcept clause is ignored for function returning Python object
      [46/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/sparse.pyx
      [47/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/join.pyx
      [48/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/algos.pyx
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:188:38: noexcept clause is ignored for function returning Python object
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:193:40: noexcept clause is ignored for function returning Python object
      [49/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/groupby.pyx
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:188:38: noexcept clause is ignored for function returning Python object
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:193:40: noexcept clause is ignored for function returning Python object
      [50/151] Compiling Cython source /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/hashtable.pyx
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:188:38: noexcept clause is ignored for function returning Python object
      warning: /tmp/pip-install-qxwszpxa/pandas_911c4b0266f5494c963ea574a52cb4e0/pandas/_libs/tslibs/util.pxd:193:40: noexcept clause is ignored for function returning Python object
      ninja: build stopped: subcommand failed.
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed

× Encountered error while generating package metadata.
╰─> See above for output.

note: This is an issue with the package mentioned above, not pip.
hint: See above for details.
```

Steps 1–5 (agent wiring) and Steps 6-A–E (UI wiring) are independent of each other but both depend on Step 0.  
Step 7 is largely independent but 7-D must follow 6-D (it wraps the badge block added there).

---

## Checklist

- [ ] `classify_radius_gap` function added to `astro_physics.py`
- [ ] `assess_sulfur_chemistry` function added to `astro_physics.py`
- [ ] `assess_co_ratio` function added to `astro_physics.py`
- [ ] `classify_planet_radius_gap` tool added to `agent_setup.py`
- [ ] `predict_sulfur_chemistry` tool added to `agent_setup.py`
- [ ] `assess_carbon_oxygen_ratio` tool added to `agent_setup.py`
- [ ] All three tools added to the `tools = [...]` list (after `run_climate_simulation`)
- [ ] `SYSTEM_PROMPT` CAPABILITIES list extended with items 9–11
- [ ] `SYSTEM_PROMPT` NEW TOOL USAGE GUIDANCE section appended
- [ ] No import errors: verify `from modules.astro_physics import classify_radius_gap, assess_sulfur_chemistry, assess_co_ratio` resolves
- [ ] Three sidebar widgets added to `col_params` in `app.py` (C/O ratio, surface pressure, atmosphere type)
- [ ] Three new functions called in simulation pipeline after `compute_sephi`
- [ ] `radius_gap`, `sulfur`, `co` keys stored in `st.session_state.current_planet_data`
- [ ] Three-column badge block (`rg_col`, `sulfur_col`, `co_col`) inserted before globe toggle in `col_viz`
- [ ] Three glossary entries added to `_GLOSSARY` dict
- [ ] **[UI]** Global CSS block replaced with design-token system + mobile breakpoints (7-A)
- [ ] **[UI]** Title/caption use `clamp()` fluid typography (7-B)
- [ ] **[UI]** `col_params / col_viz` columns use `gap="large"` (7-C-1)
- [ ] **[UI]** Parameter sliders wrapped in `st.container(border=True)` (7-C-2)
- [ ] **[UI]** Run button + live toggle placed side-by-side in 2-col row (7-C-3)
- [ ] **[UI]** Metric labels updated to full human-readable strings with `help=` text (7-C-4)
- [ ] **[UI]** ESI gauge removes hardcoded `height=200`, uses `autosize=True` (7-C-5)
- [ ] **[UI]** Radius/sulfur/C/O badge block wrapped in `st.container(border=True)` with heading (7-D)
- [ ] **[UI]** Famous-planet grid changed from 1×6 to 3×2 layout (7-E)
- [ ] **[UI]** `_CHART_H = 320` constant added, applied to all Tab 4 Plotly charts (7-F)
- [ ] **[UI]** Export section wrapped in `st.container(border=True)` with caption (7-G)
- [ ] **[UI]** Sidebar branding strip + footer caption added (7-H)
- [ ] **[UI]** Column gaps, dead code (`sci3`), and subheader label fixes applied (7-I)
- [ ] **[UI]** All sliders have `help=` text; `set_page_config` has `menu_items` (7-J)
- [ ] **[UI]** Param-change debounce guard prevents redundant re-runs in live mode (7-K)
- [ ] **[UI]** Emoji stripped from `app.py` (tab labels, buttons, subheaders, status writes, planet-card icons) — ✅/❌ intentionally kept (7-L)
- [ ] **[UI]** "About Us" section added to System tab and sidebar footer with GitHub + LinkedIn links (7-M)
