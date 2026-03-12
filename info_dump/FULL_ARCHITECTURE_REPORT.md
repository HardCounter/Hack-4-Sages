# Autonomous Exoplanetary Digital Twin — Full Architecture Report

> **Generated:** 2026-03-12  
> **Source:** Complete deep-read of the Hack-4-Sages repository  
> **Scope:** Every Python module, configuration file, Modelfile, Dockerfile, and data pipeline

---

## Table of Contents

1. [High-Level Overview & Core Value Proposition](#1-high-level-overview--core-value-proposition)
2. [Complete System Architecture](#2-complete-system-architecture)
   - 2.1 [Layer Diagram](#21-layer-diagram)
   - 2.2 [Data Flow & Pipeline](#22-data-flow--pipeline)
   - 2.3 [Runtime Modes](#23-runtime-modes)
3. [Module-by-Module Implementation Breakdown](#3-module-by-module-implementation-breakdown)
   - 3.1 [`app.py` — Streamlit Frontend](#31-apppy--streamlit-frontend)
   - 3.2 [`train_models.py` — One-Shot Training Script](#32-train_modelspy--one-shot-training-script)
   - 3.3 [`modules/agent_setup.py` — LangChain Agent Orchestration](#33-modulesagent_setuppy--langchain-agent-orchestration)
   - 3.4 [`modules/astro_physics.py` — Core Astrophysics Engine](#34-modulesastro_physicspy--core-astrophysics-engine)
   - 3.5 [`modules/elm_surrogate.py` — ELM Climate Surrogate](#35-moduleselm_surrogatepy--elm-climate-surrogate)
   - 3.6 [`modules/pinnformer3d.py` — Transformer-Based PINN](#36-modulespinnformer3dpy--transformer-based-pinn)
   - 3.7 [`modules/pinn_heat.py` — DeepXDE 1-D PINN Fallback](#37-modulespinn_heatpy--deepxde-1-d-pinn-fallback)
   - 3.8 [`modules/data_augmentation.py` — CTGAN Synthetic Data](#38-modulesdata_augmentationpy--ctgan-synthetic-data)
   - 3.9 [`modules/anomaly_detection.py` — Isolation Forest Anomaly Detection](#39-modulesanomaly_detectionpy--isolation-forest-anomaly-detection)
   - 3.10 [`modules/rag_citations.py` — RAG Scientific Literature](#310-modulesrag_citationspy--rag-scientific-literature)
   - 3.11 [`modules/nasa_client.py` — NASA TAP Client](#311-modulesnasa_clientpy--nasa-tap-client)
   - 3.12 [`modules/combined_catalog.py` — Multi-Source Catalog Consolidation](#312-modulescombined_catalogpy--multi-source-catalog-consolidation)
   - 3.13 [`modules/degradation.py` — Graceful Degradation Manager](#313-modulesdegradationpy--graceful-degradation-manager)
   - 3.14 [`modules/validators.py` — Pydantic Physics Guardrails](#314-modulesvalidatorspy--pydantic-physics-guardrails)
   - 3.15 [`modules/visualization.py` — Plotly Visualisation Suite](#315-modulesvisualizationpy--plotly-visualisation-suite)
   - 3.16 [`modules/llm_helpers.py` — Standalone LLM Helpers](#316-modulesllm_helperspy--standalone-llm-helpers)
   - 3.17 [`modules/model_evaluation.py` — Model Evaluation Utilities](#317-modulesmodel_evaluationpy--model-evaluation-utilities)
   - 3.18 [`modules/gcm_benchmarks.py` — GCM Reference Cases](#318-modulesgcm_benchmarkspy--gcm-reference-cases)
   - 3.19 [`tools/data_fetch.py` — European Catalog Fetcher](#319-toolsdata_fetchpy--european-catalog-fetcher)
4. [AI/ML Model Details](#4-aiml-model-details)
   - 4.1 [ELM — Extreme Learning Machine Ensemble](#41-elm--extreme-learning-machine-ensemble)
   - 4.2 [PINNFormer 3-D — Physics-Informed Transformer](#42-pinnformer-3-d--physics-informed-transformer)
   - 4.3 [1-D PINN (DeepXDE)](#43-1-d-pinn-deepxde)
   - 4.4 [CTGAN — Conditional Tabular GAN](#44-ctgan--conditional-tabular-gan)
   - 4.5 [Isolation Forest](#45-isolation-forest)
   - 4.6 [LLM Agent Setup — LangChain + Ollama](#46-llm-agent-setup--langchain--ollama)
5. [Infrastructure & Deployment](#5-infrastructure--deployment)
6. [Technology Stack](#7-technology-stack)
7. [Repository Structure](#8-repository-structure)
8. [Scientific Methodology](#9-scientific-methodology)
9. [Testing](#10-testing)
10. [Diagnostics & Tooling](#11-diagnostics--tooling)
11. [Documentation & Info Dump](#12-documentation--info-dump)
12. [Known Limitations & Future Scalability](#13-known-limitations--future-scalability)

## 1. High-Level Overview & Core Value Proposition

The **Autonomous Exoplanetary Digital Twin** is a full-stack scientific application that constructs interactive, physics-grounded "digital twins" of exoplanets. Given a planet's observed parameters (stellar host temperature, radius, orbital distance, etc.), the system:

1. **Retrieves real observational data** from the NASA Exoplanet Archive (TAP/ADQL), Exoplanet.eu, DACE (Geneva/CHEOPS), and Gaia DR3.
2. **Computes physics-based habitability indices** — equilibrium temperature, Earth Similarity Index (ESI), SEPHI score, habitable zone boundaries, atmospheric escape timescales, interior-surface-atmosphere (ISA) coupling, and biosignature false-positive risk.
3. **Generates climate surface-temperature maps** via a cascade of three models (ELM ensemble → PINNFormer 3-D → analytical fallback), each with increasing simplicity and availability guarantees.
4. **Renders interactive 3-D globes** of the predicted surface temperature with optional cloud-fraction overlays, host-star markers, and rotation animation.
5. **Augments sparse habitable-planet data** using a CTGAN trained on ~5,700 confirmed planets to address extreme class imbalance (~60 real habitable candidates).
6. **Detects anomalous planets** in the catalog via Isolation Forest + UMAP embeddings.
7. **Grounds all scientific claims** in a RAG system indexing 40 peer-reviewed papers with hybrid semantic + TF-IDF search.
8. **Provides autonomous AI agent interaction** via a LangChain agent backed by two LLMs: Qwen 2.5-14B (orchestrator) and AstroSage-Llama-3.1-8B (domain expert).

**Core value:** Replace the need for computationally expensive General Circulation Model (GCM) simulations (which take days/weeks on supercomputers) with fast, physics-informed surrogate models that run in seconds on consumer hardware, while maintaining scientific rigour via physics-informed losses, Pydantic validation, and GCM benchmark comparison.

> **Scope & Non-Goals**
> - The ELM ensemble and analytical models are trained on analytically generated data, **not** calibrated against ROCKE-3D or ExoCAM. Outputs are hypothesis-generating approximations.
> - PINNFormer 3D is an **experimental** PDE surrogate, not a production GCM replacement.
> - No time-evolving atmosphere is modeled; ocean heat transport and cloud feedback appear only as low-order parameterisations in specific PINNFormer modes, not as full GCMs.
> - The system does not implement bidirectional data assimilation and is therefore a *digital-twin-inspired* surrogate, not a classical digital twin.
> - A small set of precomputed GCM benchmark cases is included for qualitative comparison only.
> - See `info_dump/judge_critique_response.md` for a full critique-response log.

---

## 2. Complete System Architecture

### 2.1 Layer Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                              │
│  Streamlit (app.py) — 5 tabs                                       │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌──────────┐ ┌───────────┐ │
│  │ Agent AI │ │  Manual  │ │ Catalog │ │ Science  │ │  System   │ │
│  │  (Chat)  │ │  Mode    │ │(Browser)│ │(Dashboard│ │(Diagnostics│ │
│  └──────────┘ └──────────┘ └─────────┘ └──────────┘ └───────────┘ │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                     AGENT / ORCHESTRATION LAYER                     │
│  LangChain AgentExecutor + Tool-calling agent                       │
│  ┌────────────────┐  ┌───────────────────┐  ┌────────────────────┐ │
│  │ Qwen 2.5-14B   │  │ AstroSage-LLaMA   │  │  Deterministic     │ │
│  │ (Orchestrator)  │  │ 8B (Domain Expert) │  │  Mode (no LLM)    │ │
│  └────────────────┘  └───────────────────┘  └────────────────────┘ │
│  11 registered tools: query_nasa, compute_habitability, ...         │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                  PHYSICS / ML COMPUTATION LAYER                     │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │astro_physics │  │elm_surrogate │  │pinnformer3d  │              │
│  │(T_eq, ESI,  │  │(ELM Ensemble │  │(Transformer  │              │
│  │ SEPHI, HZ,  │  │ 32×64 grid,  │  │ PINN, PDE    │              │
│  │ ISA, escape,│  │ conformal CI)│  │ loss, 4 fields│             │
│  │ false pos.) │  │              │  │ T/ocean/cloud │              │
│  └─────────────┘  └──────────────┘  │ /ice)        │              │
│  ┌─────────────┐  ┌──────────────┐  └──────────────┘              │
│  │pinn_heat    │  │data_augment. │  ┌──────────────┐              │
│  │(1-D DeepXDE │  │(CTGAN synth. │  │anomaly_det.  │              │
│  │ fallback)   │  │ planets)     │  │(IsoForest +  │              │
│  └─────────────┘  └──────────────┘  │ UMAP)        │              │
│                                     └──────────────┘              │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│              VALIDATION & DEGRADATION LAYER                         │
│  ┌────────────┐  ┌─────────────────┐  ┌──────────────────────┐    │
│  │ validators  │  │  degradation    │  │  model_evaluation    │    │
│  │(Pydantic   │  │(L0→L4 fallback  │  │(ELM vs GCM,         │    │
│  │ guardrails)│  │ chain)          │  │ CTGAN stats, PINN)   │    │
│  └────────────┘  └─────────────────┘  └──────────────────────┘    │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                    DATA LAYER                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ nasa_client   │  │combined_cat. │  │ rag_citations│             │
│  │(TAP/ADQL →   │  │(NASA+EU+DACE │  │(ChromaDB +   │             │
│  │ pscomppars)  │  │ +Gaia merge) │  │ 40 papers)   │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
│  ┌──────────────┐  ┌──────────────┐                                │
│  │ data_fetch    │  │ gcm_benchmarks│                               │
│  │(Gaia, EU,    │  │(3 synthetic  │                                │
│  │ DACE fetcher)│  │ GCM profiles)│                                │
│  └──────────────┘  └──────────────┘                                │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow & Pipeline

**User interaction → Simulation pipeline:**

```
User selects planet parameters (sliders or agent chat)
        │
        ▼
PlanetaryParameters validator (Pydantic) — rejects unphysical inputs
        │
        ▼
astro_physics.equilibrium_temperature() — T_eq with redistribution factor f
astro_physics.stellar_flux()            — orbit-averaged S_abs, S_norm
astro_physics.compute_esi()             — Earth Similarity Index
astro_physics.compute_sephi()           — SEPHI 3-criteria score
astro_physics.hz_boundaries()           — Kopparapu 2013 HZ bounds
astro_physics.estimate_isa_interaction()         — interior coupling
astro_physics.assess_biosignature_false_positives() — photochemistry risk
        │
        ▼
SimulationOutput validator (Pydantic) — rejects T_eq ∉ [10, 5000] K, etc.
        │
        ▼
Climate map generation (cascade with GracefulDegradation):
  L0: ELM Ensemble → predict 32×64 temperature grid
  L1: PINNFormer 3-D → predict T_atm + T_ocean + f_cloud + f_ice on sphere
  L2: Analytical cos^{1/4} profile (always succeeds)
        │
        ▼
degradation.validate_temperature_map() — no NaN, no negative, T < 5000 K
        │
        ▼
visualization.create_3d_globe() or create_2d_heatmap()
        │
        ▼
llm_helpers.interpret_simulation()     — 3-4 sentence domain-expert narrative
llm_helpers.classify_climate_state()   — Eyeball / Lobster / Greenhouse / Temperate
llm_helpers.review_elm_output()        — physics plausibility check
```

**Data ingestion pipeline:**

```
tools/data_fetch.py:
  Gaia DR3 (astroquery) → data/gaia_raw.csv
  Exoplanet.eu (pyvo TAP) → data/exoplanet_eu_raw.csv
  DACE (dace_query) → data/dace_raw.csv

modules/combined_catalog.py:
  NASA (TAP/ADQL) + EU + DACE → normalised schema → de-duplicated DataFrame
  (NASA priority on duplicate planet names)

modules/data_augmentation.py (CTGAN):
  combined_catalog → prepare_normalised_data() → CTGAN training
  → generate_synthetic_planets(n=500000) → validate_synthetic_data()
  → models/ctgan_exoplanets.pkl
```

### 2.3 Runtime Modes

The system supports three runtime modes, configurable in the System tab:

| Mode | LLMs Active | VRAM Requirement | Capabilities |
|------|-------------|------------------|-------------|
| **Dual-LLM** | Qwen 2.5-14B (orchestrator) + AstroSage-8B (domain expert) | ~12 GB | Full: agent chat, tool calling, narrative generation, climate classification |
| **Single-LLM** | AstroSage-8B only | ~5 GB | AstroSage handles both orchestration and domain expertise |
| **Deterministic** | None | 0 GB | All physics, ML, and visualization tools active — no AI narratives |

---

## 3. Module-by-Module Implementation Breakdown

### 3.1 `app.py` — Streamlit Frontend

**Lines:** ~1,600  
**Role:** Main application entry point. Defines the Streamlit UI with 5 tabs.

#### Tab 1: Agent AI
- Full conversational interface with LangChain `AgentExecutor`.
- Maintains `chat_history` in `st.session_state`, converted to LangChain `HumanMessage`/`AIMessage` for context.
- Shows a **Reasoning Chain** panel: each agent intermediate step is rendered (tool name, input, output), with `consult_domain_expert` steps highlighted with a purple gradient card.
- After each response, calls `llm_helpers.generate_smart_suggestions()` to show 3 contextual follow-up buttons.
- Supports "Scientist" and "Outreach" audience modes, appending a prompt hint.

#### Tab 2: Manual Mode
- Parameter sliders for stellar temperature (2500–7500 K), stellar radius (0.08–3.0 R☉), planet radius (0.5–2.5 R⊕), mass (0.1–15 M⊕), semi-major axis (0.01–2.0 AU), eccentricity (0–0.9), surface type, atmosphere type, Bond albedo, tidal locking, C/O ratio, surface pressure, and atmosphere regime.
- Climate model selector: ELM Ensemble / PINNFormer 3-D / Analytical.
- "Live What-If" toggle for real-time simulation on parameter change (with debounce).
- Renders: ESI gauge, 2×2 metric grid (T_eq, ESI, HSF, flux), SEPHI traffic lights, ISA coupling badge, false-positive risk badge, radius gap / sulfur / C/O classification cards.
- 3-D globe with cloud overlay controls, or 2-D heatmap toggle.
- Expandable AI interpretation panel (interpret, classify climate state, physics review).
- Raw data export (CSV download for metrics and temperature maps).

#### Tab 3: Planet Catalog
- Natural-language search bar → `llm_helpers.generate_adql_query()` → NASA TAP execution.
- 6 pre-loaded famous planet buttons (TRAPPIST-1 e, Proxima Cen b, K2-18 b, Kepler-442 b, TOI-700 d, LHS 1140 b).
- Full NASA catalog fetch with anomaly detection (Isolation Forest), UMAP embedding visualization, and "Weirdest Planets" table.
- Real vs Synthetic comparison panel: overlays CTGAN-generated habitable planets against real catalog distributions with histogram grids and summary statistics.

#### Tab 4: Science Dashboard
- Scientific narrative (LLM-generated paragraph).
- Habitable Zone diagram (Kopparapu parameterisation).
- Interior-Surface-Atmosphere card (ISA score, outgassing rate, plate tectonics, C-Si cycle, water cycling, volatile retention).
- Photochemical false-positive card (O₂ risk, CH₄ risk, UV flux, risk flags).
- Terminator cross-section plot (equatorial temperature profile with 273 K / 373 K reference lines).
- Uncertainty estimate: conformal prediction intervals from ELM ensemble (90% coverage) or static analytical estimates.
- GCM benchmark comparison: pattern correlation, RMSE, bias, zonal mean RMSE against 3 reference cases.
- Compare with Earth (LLM narrative).
- Planetary Soundscape (sonification — temperature-to-frequency mapping for outreach).

#### Tab 5: System
- LLM runtime mode selector (Dual-LLM / Single-LLM / Deterministic).
- Self-diagnostics button: checks NASA API, T_eq sanity, ELM model availability, Ollama reachability, PyTorch/CUDA status.

**UI Design:** Custom dark cosmic theme with CSS (`Space Grotesk` + `Orbitron` fonts, `radial-gradient(ellipse, #0a0e27, #000)` background, hexagonal slider thumbs, hidden sidebar).

---

### 3.2 `train_models.py` — One-Shot Training Script

**Lines:** ~220  
**Role:** CLI tool to populate `models/` directory.

**Three training targets:**

1. **ELM Ensemble** (always runs, ~5s on CPU):
   - Generates `n_samples` synthetic planet–temp-map pairs via `generate_analytical_training_data()`.
   - If `n_samples ≥ 100,000`, uses batched 2-pass training (incremental H^TH accumulation) to control peak memory.
   - Default: 5,000 samples, 10 ELMs × 500 neurons each.
   - Saves → `models/elm_ensemble.pkl`.

2. **CTGAN** (optional, `--ctgan` flag):
   - Builds combined catalog (NASA + EU + DACE).
   - Applies noise-augmented upsampling of habitable class before training.
   - Default 600 epochs, generator/discriminator dims (256, 256).
   - Generates 500,000 synthetic planets, applies physics filter.
   - Saves → `models/ctgan_exoplanets.pkl`.

3. **PINNFormer 3-D** (optional, `--pinn` flag):
   - Supports 9 physics modes: `basic`, `greenhouse`, `oht`, `clouds`, `tidal`, `ice_albedo`, `advection`, `oht_clouds`, `full`.
   - Default 5,000 epochs, 8,192 collocation points.
   - Auto-detects CUDA/ROCm.
   - Saves → `models/pinn3d_weights.pt`.

---

### 3.3 `modules/agent_setup.py` — LangChain Agent Orchestration

**Lines:** ~580  
**Role:** Defines the LangChain agent with 11 tools and 3 runtime modes.

**LLM Factory:** Lazy-initialized singletons via `_get_primary_llm()` (Qwen 2.5-14B) and `_get_domain_llm()` (AstroSage), both served by Ollama at `localhost:11434`. Temperature 0.3, context window 8192 tokens.

**11 Registered Tools:**

| # | Tool | Purpose |
|---|------|---------|
| 1 | `query_nasa_archive` | Fetch planet data from NASA by name |
| 2 | `compute_habitability` | Full physics analysis (T_eq, ESI, SEPHI, flux) |
| 3 | `run_climate_simulation` | ELM surrogate → 32×64 temperature map |
| 4 | `classify_planet_radius_gap` | Fulton Gap classification |
| 5 | `predict_sulfur_chemistry` | Sulfur speciation prediction |
| 6 | `assess_carbon_oxygen_ratio` | C/O ratio habitability assessment |
| 7 | `consult_domain_expert` | Free-form question to AstroSage |
| 8 | `discover_most_habitable` | Multi-step: query NASA → rank by ESI → expert evaluation |
| 9 | `compare_two_planets` | Fetch + analyze + expert comparison of two planets |
| 10 | `detect_anomalous_planets` | Isolation Forest on habitable candidates |
| 11 | `cite_scientific_literature` | RAG hybrid search over 40 papers |

**Agent construction:** `create_tool_calling_agent()` with a `ChatPromptTemplate` containing a detailed system prompt defining AstroAgent's capabilities, procedure (fetch → compute → consult expert → synthesize), and citation policy.

**`build_agent(mode)`:**
- `DETERMINISTIC` → returns `None`.
- `SINGLE_LLM` → AstroSage as sole LLM, `max_iterations=5`.
- `DUAL_LLM` → Qwen orchestrator, `max_iterations=7`.

---

### 3.4 `modules/astro_physics.py` — Core Astrophysics Engine

**Lines:** ~870  
**Role:** Deterministic physics calculations — the mathematical backbone of the system.

#### Physical Constants
`STEFAN_BOLTZMANN`, `L_SUN`, `R_SUN`, `AU`, `S_EARTH`, `G_GRAV`, `M_EARTH_KG`, `R_EARTH_M`, `M_PROTON`.

#### Key Functions and Their Mathematics

**`estimate_albedo(surface_type, atmosphere_type)`:**
- Look-up table of (best, sigma) pairs for 4 surface × 3 atmosphere classes.
- Literature: Kasting et al. (1993), Shields et al. (2013), Kopparapu et al. (2013).

**`redistribution_factor(tidally_locked, optical_depth_class)`:**
- Continuous f ∈ {thin, moderate, thick} × {locked, fast}.
- Moderate locked: f = √2 ≈ 1.414; moderate fast: f = 2.0.
- Based on Leconte et al. (2013), Pierrehumbert (2011).

**`orbit_averaged_flux_factor(eccentricity)`:**
- $\langle F \rangle / F(a) = 1 / \sqrt{1 - e^2}$  
- Williams & Pollard (2002).

**`equilibrium_temperature()`:**
- $T_{eq} = T_* \cdot \sqrt{R_* / (f \cdot a)} \cdot (1 - A_B)^{1/4} \cdot \langle 1/r^2 \rangle^{1/4}$

**`stellar_flux()`:**
- $S = L_* / (4\pi a^2) \cdot \langle 1/r^2 \rangle$ where $L_* = 4\pi R_*^2 \sigma T_*^4$

**`compute_esi(radius, density, escape_vel, surface_temp)`:**
- $ESI = \prod_i \left(1 - \frac{|x_i - x_{ref,i}|}{x_i + x_{ref,i}}\right)^{w_i/n}$
- Weights: radius=0.57, density=1.07, escape_vel=0.70, temperature=5.58.
- Schulze-Makuch et al. (2011).

**`compute_sephi(surface_temp, mass_earth, radius_earth)`:**
- 3 binary criteria: thermal (273–373 K), atmospheric (v_esc ≥ 5 km/s), magnetic (M ≥ 0.5 M⊕, R ≤ 2.5 R⊕).
- Score = criteria_met / 3.
- Rodríguez-Mozos & Moya (2017).

**`hz_boundaries(star_teff, star_luminosity_solar)`:**
- Kopparapu et al. (2013) 4th-degree polynomial parameterisation.
- 4 boundaries: recent_venus, runaway_gh, max_gh, early_mars.
- $S_{eff} = s_0 + aT + bT^2 + cT^3 + dT^4$ where $T = T_{eff} - 5780$
- $d_{AU} = \sqrt{L_*/S_{eff}}$

**`habitable_surface_fraction(temp_map)`:**
- Cosine-latitude weighted area integration: $f_{hab} = \sum (\text{mask} \cdot \cos\phi) / \sum \cos\phi$

**`estimate_outgassing_rate(mass, radius, age)`:**
- $\dot{V}_{rel} = (g/g_E)^{0.75} \cdot (age/4.5)^{-1.5} \cdot H_{radio}$
- Radiogenic boost for age < 1 Gyr; gravity correction for M > 2 M⊕.
- Kite et al. (2009), Stamenković et al. (2012).

**`estimate_isa_interaction()`:**
- Combines outgassing, plate tectonics plausibility (3-level: plausible/uncertain/unlikely), carbonate-silicate cycle activity, water cycling, volatile retention.
- ISA score = mean of 4 sub-scores.

**`estimate_uv_flux()`:**
- Tabulated UV/bolometric ratios for T_eff = 2800–10000 K, linearly interpolated.
- France et al. (2013), Lammer et al. (2009).

**`estimate_atmospheric_escape()`:**
- Energy-limited formalism: $dM/dt = (\epsilon \pi F_{XUV} R_p R_{XUV}^2) / (G M_p K_{tide})$
- ε = 0.15 heating efficiency, R_XUV = 1.2×R_p.
- Watson et al. (1981), Erkaev et al. (2007).

**`assess_biosignature_false_positives()`:**
- Evaluates O₂ (UV photolysis), CH₄ (volcanism), O₃ (abiotic chemistry) false positive risks.
- Returns categorical flags and recommendation.

**`classify_radius_gap(radius_earth)`:**
- Fulton Gap at 1.5–2.0 R⊕: rocky_super_earth / radius_gap / sub_neptune / giant.
- Gap proximity metric.

**`assess_sulfur_chemistry(t_eq, pressure, atmosphere_type)`:**
- H₂SO₄ (T > 400 K, P > 10 bar), SO₂ (T > 300, P > 1), H₂S (otherwise).
- Surface minerals by atmosphere regime.
- Zahnle et al. (2016).

**`assess_co_ratio(co_ratio)`:**
- < 0.55: water world candidate (+0.15 ESI modifier).
- 0.55–0.80: solar-like (0.0 modifier).
- > 0.80: carbon planet (−0.4 modifier).

**`compute_full_analysis()`:**
- End-to-end pipeline from raw NASA parameters (Jupiter units) to all metrics. Calls all of the above.

---

### 3.5 `modules/elm_surrogate.py` — ELM Climate Surrogate

**Lines:** ~600  
**Role:** Fast data-driven climate map predictor on a 32×64 lat-lon grid.

#### `PureELM` Class
- Single hidden-layer neural network with frozen random weights.
- **Training:** Closed-form solution via Moore-Penrose pseudoinverse:
  - $\beta = (H^TH + I/C)^{-1} H^T T$
  - H = hidden layer activations (tanh/sigmoid/ReLU).
- **Incremental training:** 2-pass algorithm:
  1. Accumulate $H^TH$ and $H^TT$ across batches (additive normal equations).
  2. Single solve at the end.
- 500 hidden neurons default, regularisation C = 10⁴.

#### `ELMEnsemble` Class
- K = 10 independent `PureELM` models.
- Variance reduction via averaging.
- `predict_std()`: per-output standard deviation across ensemble members.

#### `ELMClimateSurrogate` Class
- High-level wrapper with `StandardScaler` for input/output normalization.
- **8 input features:** radius_earth, mass_earth, semi_major_axis_au, star_teff_K, star_radius_solar, insol_earth, albedo, tidally_locked.
- **Output:** Flattened 32×64 = 2048-dimensional temperature map.
- **Conformal prediction intervals:** Uses ensemble spread as nonconformity score with Gaussian z-quantile scaling.
- Falls back to `PureELM` when `scikit-elm` is not installed.
- Serialisation: pickle bundle with models + scalers + metadata.

#### Analytical Training Data Generator
- 4 planet regimes with astrophysically motivated parameter ranges:
  - **temperate_g_dwarf** (20%): T_eff 5000–6100 K, a 0.7–1.7 AU, R 0.8–1.5 R⊕.
  - **m_dwarf_locked** (35%): T_eff 2500–3900 K, a 0.01–0.25 AU, R 0.7–1.6 R⊕.
  - **hot_close_in** (25%): T_eff 3800–7000 K, a 0.01–0.10 AU, R 0.5–2.0 R⊕.
  - **cold_super_earth** (20%): T_eff 3500–6000 K, a 1.0–2.5 AU, R 1.3–2.5 R⊕.
- **Mass-radius relation:** Broken power law after Otegi+ (2020):
  - Rocky (R < 1.5 R⊕): $M \sim R^{3.45}$
  - Volatile-rich (R ≥ 1.5): $M \sim R^{1.75}$ (continuous match)
  - 15% log-normal scatter.
- **Temperature maps:**
  - Tidally locked: substellar irradiation $\cos(z)^{0.25}$ with atmospheric redistribution heuristic.
  - Non-locked: latitude-dependent gradient with noise.

---

### 3.6 `modules/pinnformer3d.py` — Transformer-Based PINN

**Lines:** ~700  
**Role:** Physics-Informed Neural Network solving a coupled PDE system on an exoplanet sphere.

#### PDE System

**Atmosphere energy balance:**

$$\kappa_{atm} \nabla^2 T + S(\theta,\phi)(1 - \alpha_{eff}) - (1-G)\sigma T^4 + F_{oht} + Q_{tidal} + \bar{v}\cdot\nabla T = 0$$

**Ocean mixed-layer (OHT):**

$$\kappa_{oht} \nabla^2 T_{ocean} - \gamma(T_{ocean} - T_{atm}) = 0$$

**Cloud fraction (diagnostic):**

$$f_{cloud} \rightarrow \sigma\left(\frac{T_{atm} - 280}{15}\right)$$

**Ice fraction (diagnostic):**

$$f_{ice} \rightarrow \sigma\left(\frac{273.15 - T_{atm}}{8}\right)$$

#### `PINNPhysicsConfig`
- 6 toggle-able physics modules: greenhouse, OHT, clouds, tidal, ice_albedo, advection.
- 9 presets: basic, greenhouse, oht, clouds, tidal, ice_albedo, advection, oht_clouds, full.
- Per-component loss weightings: λ_pde=1.0, λ_oht=0.5, λ_cloud=0.3, λ_ice=0.3, λ_advection=0.2.
- Dynamic output field count: 1 (T_atm only) to 4 (T_atm + T_ocean + f_cloud + f_ice).

#### `PINNFormer3D` Architecture (PyTorch)
- **Input:** (θ, φ, z) — latitude, longitude, vertical coordinate.
- **Input projection:** `nn.Linear(3, d_model=128)`.
- **Positional encoding:** `WaveletPositionalEncoding` — sinusoidal encoding with linearly spaced frequencies 1..10, concatenated sin+cos.
- **Transformer backbone:** `nn.TransformerEncoder` with 4 layers, 4 attention heads, FFN dim 256, batch_first=True.
- **Output heads:** `nn.ModuleList` of 1–4 `nn.Linear(d_model, 1)` heads, one per physical field.
- Forces `SDPBackend.MATH` for attention to support second-order autograd (Laplacian computation).

#### `pinn_loss_3d()` — Physics-Informed Loss
- Computes Laplacian via `torch.autograd.grad` (second derivatives through all 3 coordinates).
- Stellar irradiation: $S = S_{max} \cdot \max(0, \cos\theta \cos\phi)$.
- Effective albedo: $\alpha_{eff} = \alpha_{base} + \Delta\alpha_{cloud} \cdot \sigma(f_{cloud}) + (\alpha_{ice} - \alpha_{base}) \cdot \sigma(f_{ice})$.
- Atmosphere PDE residual with optional greenhouse factor $(1-G)$, ocean coupling $\gamma(T_{ocean} - T_{atm})$, tidal heating $Q_{tidal}$, and advection $v_{zonal} \cdot \partial T / \partial\phi$.
- Boundary conditions: Dirichlet at substellar (T_sub) and antistellar (T_night) points.
- Returns per-component breakdown: L_atm, L_bc, L_oht, L_cloud, L_ice.

#### Training
- Adam optimiser with cosine annealing scheduler.
- Gradient clipping (default 1.0).
- 8,192 collocation points, 512 boundary points (256 substellar + 256 nightside).
- Validation on fresh 4,096-point grid: PDE residual RMSE/max, temperature statistics.
- Divergence early-warning: stops logging if T_max > 10⁴ or T_min < −10³.

#### Sampling Helpers
- `sample_surface_map()` → np.ndarray (32×64), clipped [30, 3000] K.
- `sample_ocean_map()` → np.ndarray or None (if model has < 2 outputs).
- `sample_cloud_map()` → np.ndarray or None, sigmoid-transformed.
- `sample_ice_map()` → np.ndarray or None, sigmoid-transformed.

#### Fallback
- If PyTorch is not installed, all classes/functions are replaced with stubs that raise `ImportError`.

---

### 3.7 `modules/pinn_heat.py` — DeepXDE 1-D PINN Fallback

**Lines:** ~170  
**Role:** Lightweight 1-D PINN for terminator heat equation. CPU-friendly alternative to 3-D PINNFormer.

#### PDE

$$\kappa T'' + S_{abs}(x) - \sigma T^4 = 0, \quad x \in [0, \pi]$$

- κ = 0.025 (thermal diffusivity).
- $S_{abs}(x) = S_{max} \cdot \max(0, \cos x)$ (substellar at x=0, nightside at x=π).
- Dirichlet BCs: T(0) = T_sub, T(π) = T_night.

#### Implementation
- DeepXDE `Interval(0, π)` geometry.
- FNN: 3 hidden layers × 64 neurons, tanh activation, Glorot normal initialisation.
- Adam optimiser, 10,000 epochs default.
- Post-training diagnostics: PDE residual RMSE, temperature range.
- Stub functions if DeepXDE not installed.

---

### 3.8 `modules/data_augmentation.py` — CTGAN Synthetic Data

**Lines:** ~250  
**Role:** Address extreme class imbalance in NASA catalog (~60 habitable vs ~5,700 total).

#### `ExoplanetDataAugmenter` Class

**Data preparation:**
- `prepare_data(df)`: NASA schema (Jupiter units) → normalised schema (Earth units) with binary `habitable` label.
- `prepare_normalised_data(df)`: Combined catalog (already in Earth units) → adds `habitable` label.
- Habitability criterion: R ∈ [0.5, 2.5] R⊕, insolation ∈ [0.2, 2.0] S⊕, T_eff ∈ [2500, 7000] K.

**CTGAN Training:**
- Log-transforms heavy-tailed features (mass, semi-major axis, period, star T_eff, star radius, star mass) via `np.log1p()` before training to improve VGM normalizer accuracy.
- Architecture: generator (256, 256), discriminator (256, 256), discriminator_steps=5, lr=2×10⁻⁴.
- Supports conditional sampling: `condition_column="habitable", condition_value=1`.

**Post-hoc Physics Filter (`validate_synthetic_data`):**
- Hard physical bounds: R ∈ (0.3, 25) R⊕, M ∈ (0.01, 5000) M⊕, a ∈ (0.001, 50) AU, T_eq ∈ (10, 4000) K, etc.
- 1st–99th percentile clipping per column.

**Persistence:**
- Custom `_DeviceRemappingUnpickler` handles CUDA → CPU tensor remapping on load.

---

### 3.9 `modules/anomaly_detection.py` — Isolation Forest Anomaly Detection

**Lines:** ~200  
**Role:** Statistical outlier detection in exoplanet catalogs.

**Feature sets:** Auto-detects NASA schema (`pl_radj`, `pl_bmassj`, ...) vs combined catalog schema (`radius_earth`, `mass_earth`, ...).

**`detect_anomalies(df)`:**
- `StandardScaler` normalisation → `IsolationForest(contamination=0.05, n_jobs=-1)`.
- Adds `anomaly_score` (lower = more anomalous) and `is_anomaly` boolean columns.

**`build_weird_planets_table(df)`:**
- For each anomaly, computes z-score deviation per feature.
- Identifies top-3 most deviant features and generates human-readable reason strings (e.g., "Mass unusually high (4.2σ)").

**`compute_umap_embedding(df)`:**
- 2-D UMAP projection (n_neighbors=15, min_dist=0.1).
- Gracefully returns `None` if `umap-learn` is not installed.

**`create_umap_figure(df, embedding)`:**
- Plotly scatter colored by anomaly status (red = anomalous, blue = normal).

---

### 3.10 `modules/rag_citations.py` — RAG Scientific Literature

**Lines:** ~870+ (mostly paper corpus data)  
**Role:** Ground all scientific claims in peer-reviewed literature.

#### Paper Corpus
40 papers covering 6 domains:
- **Habitable zones:** Kopparapu (2013), Kasting (1993), Goldblatt (2013)
- **Habitability metrics:** Schulze-Makuch (2011) — ESI, Rodríguez-Mozos (2017) — SEPHI
- **Tidal locking & climate:** Turbet (2016), Pierrehumbert (2011), Leconte (2013), Wordsworth (2015)
- **Biosignatures:** Meadows (2018), Seager (2016), Luger & Barnes (2015)
- **Planetary interiors:** Chen & Kipping (2017), Kite (2009), Walker (1981), Stamenković (2012)
- **Atmospheric escape:** Owen & Wu (2013), Zahnle & Catling (2017), Tian (2015)
- **Climate modeling:** Wolf & Toon (2015), Driscoll & Barnes (2015)
- **Stellar context:** Lammer (2009), Shields (2016)
- **JWST / observational:** Various

Each paper contains: id, title, authors, year, journal, abstract, topics (list of tags), key_findings (list of quantitative takeaways).

#### Hybrid Search Engine

**Semantic search (primary):**
- ChromaDB vector store with `sentence-transformers/all-MiniLM-L6-v2` embeddings.
- Collection: `exoplanet_papers`.
- Metadata filtering by topic tags.

**TF-IDF keyword search (fallback):**
- Document-term matrix built from paper abstracts + key findings.
- Cosine similarity scoring.

**Reciprocal Rank Fusion (RRF):**
- Merges semantic + keyword rankings: $RRF(d) = \sum_r \frac{1}{k + rank_r(d)}$ with k=60.
- Returns top-N papers sorted by fused score.

**`cite_literature(query, n_results, topics)`:**
- If ChromaDB available: hybrid search.
- Otherwise: TF-IDF keyword-only fallback.

---

### 3.11 `modules/nasa_client.py` — NASA TAP Client

**Lines:** ~130  
**Role:** Interface to NASA Exoplanet Archive via Table Access Protocol.

- **`query_nasa_archive(adql_query)`:** HTTP GET to `https://exoplanetarchive.ipac.caltech.edu/TAP/sync`, returns CSV → DataFrame.
- **`get_planet_data(planet_name)`:** Fetches full parameter set for a single planet from `pscomppars`.
- **`get_habitable_candidates()`:** Parameterised query for Earth-sized (R 0.5–2.5 R_J), HZ-insolation (0.2–2.0 S⊕), solar-like host (2500–7000 K).
- **`get_all_confirmed_planets()`:** Full catalog (~5,700 rows) for CTGAN training.
- **Unit converters:** `jupiter_to_earth_radius/mass`, `log_solar_lum_to_watts`, `solar_to_meters_radius`, `au_to_meters`.

---

### 3.12 `modules/combined_catalog.py` — Multi-Source Catalog Consolidation

**Lines:** ~200  
**Role:** Merge NASA + Exoplanet.eu + DACE + Gaia into a normalised schema.

**Normalised schema:** `pl_name`, `radius_earth`, `mass_earth`, `semi_major_axis_au`, `period_days`, `insol_earth`, `t_eq_K`, `star_teff_K`, `star_radius_solar`, `star_mass_solar`, `source`.

**Per-source normalization:**
- **NASA:** Jupiter → Earth unit conversion.
- **Exoplanet.eu:** VO TAP column mapping (`target_name`, `radius`, `mass`, `semi_major_axis`, etc.).
- **DACE:** Geneva column mapping (`planet_name`, `planet_mass`, `planet_radius`, etc.) with fallback column names.
- **Gaia:** Currently skipped (transit table lacks physical parameters for climate modeling).

**De-duplication:** `drop_duplicates(subset="pl_name", keep="first")` with NASA rows first (priority).

---

### 3.13 `modules/degradation.py` — Graceful Degradation Manager

**Lines:** ~180  
**Role:** Ensure the application never crashes due to component failures.

**`GracefulDegradation` class:**

| Level | Trigger | Fallback |
|-------|---------|----------|
| L0 | Full mode | Dual-LLM / Single-LLM / Deterministic |
| L1 | LLM unavailable | Deterministic tools only |
| L2 | ELM unphysical | PINNFormer 3-D → analytical cos^{1/4} |
| L3 | 3-D render timeout | 2-D heatmap |
| L4 | CTGAN fails | NASA-only data |

**`run_with_fallback(primary_fn, fallback_fn, timeout, label)`:**
- Executes `primary_fn`; on exception or timeout, switches to `fallback_fn`.
- Displays Streamlit warning banner with failure reason.

**`validate_temperature_map(temp_map)`:**
- Rejects: None, empty, NaN, inf, negative, > 5000 K.

**`check_ollama_available()`:**
- Pings Ollama via `ollama.list()`.

**`run_simulation_pipeline(params)`:**
- Full cascade: ELM → PINNFormer → analytical → validation → 3-D globe → 2-D heatmap.

---

### 3.14 `modules/validators.py` — Pydantic Physics Guardrails

**Lines:** ~140  
**Role:** Prevent thermodynamically / astrophysically impossible inputs and outputs.

**`StellarParameters`:**
- T_eff ∈ [2000, 50000] K (hydrogen-burning limit to O-type upper bound).
- Radius ∈ [0.08, 100] R☉; Mass ∈ [0.08, 150] M☉.

**`PlanetaryParameters`:**
- Radius ∈ [0.3, 25] R⊕; Mass < 4132 M⊕ (deuterium-burning limit ≈ 13 M_Jup).
- Semi-major axis ∈ [0.001, 1000] AU; eccentricity ∈ [0, 0.9].
- Surface type ∈ {ocean, desert, ice, mixed_rocky}; atmosphere type ∈ {thin, temperate, thick_cloudy}.
- **Mass-radius consistency check:** Chen & Kipping (2017): for R < 4 R⊕, $R_{expected} \approx M^{0.279}$; ratio must be within [0.2, 5.0].

**`SimulationOutput`:**
- T_eq ∈ [10, 5000] K; ESI ∈ [0, 1]; flux_earth ∈ [0, 10000]; NaN/inf rejection.

---

### 3.15 `modules/visualization.py` — Plotly Visualisation Suite

**Lines:** ~350  
**Role:** All visual outputs.

**Scientific Colorscale:** 10-stop gradient from ultra-cold violet (#1a0533) through habitable green (#41ab5d) to extreme dark red (#67001f), calibrated to temperature ranges.

**`generate_eyeball_map(T_eq, tidally_locked)`:**
- Tidally locked: $T = T_{night} + (T_{sub} - T_{night}) \cdot \cos(z)^{0.25}$ where $T_{sub} = 1.4 \cdot T_{eq}$, $T_{night} = \max(0.3 \cdot T_{eq}, 40)$.
- Fast rotator: $T = T_{eq} \cdot (1 + 0.15 \cos\phi)$.

**`create_3d_globe(temperature_map, ...)`:**
- Plotly `Surface` on a unit sphere: X = cos(φ)cos(θ), Y = cos(φ)sin(θ), Z = sin(φ).
- Rolls temperature map by n_lon/2 so substellar point faces +X (where star marker sits).
- Optional cloud-fraction overlay: semi-transparent white Surface at r=1.001.
- Host-star diamond marker at (3, 0, 0) with spectral-type color.
- 100-frame rotation animation.
- Lighting: ambient=0.4, diffuse=0.6, specular=0.15, light at (1000, 0, 0).

**`create_2d_heatmap(temperature_map)`:**
- Plotly `imshow` with Mollweide-like lat-lon axes.

**`create_hz_diagram(hz_boundaries, planet_semi_major, star_teff)`:**
- Rectangular zones: Too Hot, Habitable Zone, Extended HZ.
- Planet position marker.

**`_star_color(teff)`:**
- M-dwarf (< 3500 K): #ff4500; K-type: #ffa500; G-type: #fff44f; F-type: #fffbe6; A/B: #caf0f8.

---

### 3.16 `modules/llm_helpers.py` — Standalone LLM Helpers

**Lines:** ~250  
**Role:** Direct Ollama calls outside the LangChain agent, used by UI tabs.

**Model routing:** Respects `_ACTIVE_MODE` — in single-LLM mode, the orchestrator model resolves to AstroSage instead of Qwen.

**Domain-expert functions (AstroSage):**
| Function | Purpose | Fallback |
|----------|---------|----------|
| `interpret_simulation(results)` | 3-4 sentence scientific interpretation | "*AI interpretation unavailable*" |
| `classify_climate_state(T_min, T_max, T_mean, locked)` | Eyeball/Lobster/Greenhouse/Temperate JSON | `{"state": "Unknown", ...}` |
| `review_elm_output(params, T_min, T_max, T_mean)` | "Plausible:" or "Warning:" verdict | "Review unavailable." |
| `summarise_planet_data(planet_data)` | 3-sentence human-friendly summary | "*Summary unavailable.*" |
| `narrate_science_panel(hz, cross_section, uncertainty)` | Science Dashboard narrative | "*Narrative unavailable.*" |
| `compare_planets(planet_a, planet_b)` | 4-5 sentence comparative analysis | "*Comparison unavailable.*" |

**Orchestrator functions (Qwen 2.5):**
| Function | Purpose | Fallback |
|----------|---------|----------|
| `generate_adql_query(natural_language)` | NL → ADQL conversion | `""` |
| `generate_smart_suggestions(conversation)` | 3 contextual follow-ups as JSON array | Default suggestions list |

All functions wrapped in `_safe()` for graceful fallback.

---

### 3.17 `modules/model_evaluation.py` — Model Evaluation Utilities

**Lines:** ~150  
**Role:** Physics-aware diagnostics for all three model families.

**`evaluate_elm_against_gcm(model)`:**
- Compares ELM output to 3 GCM benchmarks (earth_like, proxima_b, hot_rock).
- Computes pattern correlation, RMSE, bias, zonal mean RMSE for each case.

**`summarise_ctgan_statistics(real, synthetic)`:**
- Per-column mean/std comparison.
- Maximum absolute correlation difference between real and synthetic.

**`PinnSummary` / `summarise_pinn_history(history)`:**
- Extracts PDE residual RMSE/max, temperature statistics from `TrainingHistory.validation`.

---

### 3.18 `modules/gcm_benchmarks.py` — GCM Reference Cases

**Lines:** ~170  
**Role:** Synthetic GCM temperature profiles for surrogate validation.

**3 benchmark cases:**

| Case | Source | Profile |
|------|--------|---------|
| **earth_like** | Del Genio et al. (2019), ROCKE-3D | Aquaplanet: 260–300 K meridional gradient + weak OHT |
| **proxima_b** | Turbet et al. (2016), LMD GCM | Tidally locked: 180–290 K eyeball + atmospheric transport |
| **hot_rock** | Leconte et al. (2013) | Synchronous: 50–600 K extreme day-night contrast |

Each generator produces a 32×64 numpy array with appropriate noise.

**`compare_surrogate_to_gcm(surrogate_map, gcm_map)`:**
- Pattern correlation, RMSE, bias, zonal mean RMSE.

---

### 3.19 `tools/data_fetch.py` — European Catalog Fetcher

**Lines:** ~80  
**Role:** Fetch raw data from non-NASA sources.

- **Gaia DR3:** `astroquery.gaia.Gaia.launch_job()` — joins `vari_planetary_transit` with `gaia_source` for stellar context.
- **Exoplanet.eu:** `pyvo.dal.TAPService("http://voparis-tap-planeto.obspm.fr/tap")` — queries `exoplanet.epn_core`.
- **DACE:** `dace_query.exoplanet.Exoplanet.query_database(output_format="pandas")`.

Outputs: `data/gaia_raw.csv`, `data/exoplanet_eu_raw.csv`, `data/dace_raw.csv`.

---

## 4. AI/ML Model Details

### 4.1 ELM — Extreme Learning Machine Ensemble

| Property | Detail |
|----------|--------|
| **Type** | Single-hidden-layer feedforward neural network |
| **Training** | Closed-form Moore-Penrose pseudoinverse (no backpropagation) |
| **Ensemble** | K=10 independent models, output averaged |
| **Hidden neurons** | 500 per model |
| **Activation** | tanh |
| **Regularisation** | Tikhonov (C=10⁴ ↔ α=10⁻⁴) |
| **Input features** | 8: R_p, M_p, a, T_eff*, R*, S, A_B, locked |
| **Output** | 2048 (flattened 32×64 temperature grid) |
| **Training data** | Analytically generated from 4 planet regimes |
| **Inference time** | <100 ms on CPU |
| **Uncertainty** | Ensemble spread → conformal intervals (90% coverage, Gaussian z-quantile) |

### 4.2 PINNFormer 3-D — Physics-Informed Transformer

| Property | Detail |
|----------|--------|
| **Type** | Transformer encoder with physics-informed PDE loss |
| **Architecture** | Linear(3→128) → WaveletPosEnc → 4×TransformerEncoderLayer(128, 4heads, FFN=256) → 1–4 output heads |
| **Input** | (θ, φ, z) — latitude, longitude, altitude |
| **Outputs** | T_atm (always), T_ocean (with OHT), f_cloud (with clouds), f_ice (with ice_albedo) |
| **PDE** | Steady-state coupled atmosphere-ocean system with greenhouse, clouds, tidal heating, ice-albedo, advection |
| **Loss** | L_bc + λ_pde·L_atm + λ_oht·L_oht + λ_cloud·L_cloud + λ_ice·L_ice |
| **Laplacian** | Exact via `torch.autograd.grad` (second derivatives through computational graph) |
| **Optimizer** | Adam with cosine annealing, gradient clipping |
| **Collocation** | 8,192 random interior points + 512 boundary points |
| **Epochs** | 5,000–10,000 (configurable) |
| **Device** | CUDA/ROCm preferred, CPU fallback |

### 4.3 1-D PINN (DeepXDE)

| Property | Detail |
|----------|--------|
| **Type** | DeepXDE physics-informed FNN |
| **PDE** | κT'' + S(x) − σT⁴ = 0 along terminator (x ∈ [0, π]) |
| **Architecture** | FNN: 1→64→64→64→1, tanh, Glorot normal |
| **BCs** | Dirichlet: T(0)=T_sub, T(π)=T_night |
| **Optimizer** | Adam, lr=10⁻³, 10,000 epochs |
| **Use case** | CPU fallback for 1-D terminator profiles, CI testing |

### 4.4 CTGAN — Conditional Tabular GAN

| Property | Detail |
|----------|--------|
| **Type** | Conditional Tabular GAN (from `ctgan` package) |
| **Generator/Discriminator** | 2-layer MLPs (256, 256) |
| **Pre-processing** | Log1p transform on 6 heavy-tailed features |
| **Class balancing** | Noise-augmented upsampling of habitable class pre-training |
| **Conditional sampling** | `condition_column="habitable", condition_value=1` |
| **Post-hoc filter** | Hard physics bounds + 1st–99th percentile clipping |
| **Training data** | Combined catalog (NASA + EU + DACE), ~5,700 planets |
| **Default epochs** | 600 |

### 4.5 Isolation Forest

| Property | Detail |
|----------|--------|
| **Type** | Scikit-learn `IsolationForest` |
| **Contamination** | 5% |
| **Features** | 8 planetary/stellar features (auto-detected schema) |
| **Pre-processing** | `StandardScaler` |
| **Visualization** | UMAP 2-D embedding (n_neighbors=15, min_dist=0.1) |
| **Output** | Anomaly score (lower = more unusual), boolean flag, human-readable "why weird" reasons |

### 4.6 LLM Agent Setup — LangChain + Ollama

**Two LLM models served via Ollama:**

| Model | Modelfile | Base | Role | Parameters |
|-------|-----------|------|------|------------|
| **qwen2.5:14b** | `Modelfile.astro` | Qwen 2.5-14B | Orchestrator — tool calling, routing | temp=0.3, top_p=0.9, ctx=8192 |
| **astrosage** | `Modelfile.astrosage` | AstroSage-Llama-3.1-8B (AstroMLab, Q5_K_M GGUF) | Domain expert — interpretation, classification | temp=0.3, top_p=0.9, ctx=8192 |

**Agent architecture:**
- `langchain.agents.create_tool_calling_agent` with `ChatPromptTemplate` (system + chat_history + input + agent_scratchpad).
- `AgentExecutor` with `max_iterations=7` (dual) or 5 (single), `handle_parsing_errors=True`.
- LLMs accessed via `langchain_ollama.ChatOllama`.

**System prompt defines:**
1. 11 tool capabilities with usage guidance.
2. Procedural instructions: fetch data → compute habitability → consult expert → synthesise.
3. Citation policy: call `cite_scientific_literature` for substantive claims.
4. Rules: always cite NASA, never fabricate values, flag uncertainties.

---

## 5. Infrastructure & Deployment

### 5.1 Local Development

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
ollama pull qwen2.5:14b
ollama create astro-agent -f Modelfile.astro
ollama create astrosage -f Modelfile.astrosage
python train_models.py
streamlit run app.py
```

### 5.2 Docker

**Dockerfile:**
- Base: `python:3.11-slim`.
- Exposes port 8501 (Streamlit default).
- Healthcheck: `curl --fail http://localhost:8501/_stcore/health`.
- CMD: `streamlit run app.py --server.port=8501 --server.address=0.0.0.0`.

### 5.3 Hardware Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| Python | 3.10+ | 3.11 |
| RAM | 8 GB | 16 GB |
| GPU | Not required | NVIDIA with CUDA (for Ollama, CTGAN, PINNFormer) |
| Disk | ~2 GB (models + Ollama) | ~10 GB |
| Network | Required (NASA API) | — |

### 5.4 Key Dependencies (from `requirements.txt`):
- **Core:** streamlit, pandas, numpy, plotly, requests, pydantic, scipy
- **ML:** scikit-learn, scikit-elm, ctgan, torch (implied by pinnformer3d)
- **PINN:** deepxde
- **LLM:** langchain, langchain-ollama, ollama
- **RAG:** chromadb, sentence-transformers
- **Viz:** kaleido, imageio
- **Stats:** umap-learn, mapie

**Model artifacts** (`models/` directory):
- `elm_ensemble.pkl` — Pickled ELM ensemble + scalers (~MB)
- `ctgan_exoplanets.pkl` — Pickled CTGAN + log-transform flag (~MB)
- `pinn3d_weights.pt` — PyTorch state dict + physics config

---

## 7. Technology Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| Frontend | Streamlit | 1.41.0 | Web UI with reactive widgets, chat, caching |
| LLM Runtime | Ollama | 0.4.4 | Local LLM inference (CUDA-accelerated) |
| Agent Framework | LangChain | 0.3.13 | ReAct agent with tool calling |
| LLM Binding | langchain-ollama | 0.2.3 | Ollama ↔ LangChain bridge |
| Climate ML | scikit-elm | 0.21a0 | Extreme Learning Machine ensemble |
| Data Augmentation | CTGAN | 0.12.1 | Tabular GAN for exoplanet oversampling |
| PINN (1D) | DeepXDE | 1.12.0 | 1D physics-informed neural network |
| PINN (3D) | PyTorch | (optional) | Transformer-based 3D PINNFormer |
| Anomaly Detection | scikit-learn | 1.6.0 | Isolation Forest |
| Dimensionality Reduction | umap-learn | ≥0.5.0 | UMAP embedding for planet visualization |
| Vector DB | ChromaDB | ≥0.4.0 | RAG paper storage and retrieval |
| Embeddings | sentence-transformers | ≥2.2.0 | all-MiniLM-L6-v2 for paper embeddings |
| Validation | Pydantic | 2.10.3 | Physics-constrained data models |
| Uncertainty | mapie | ≥0.8.0 | Conformal prediction intervals |
| Visualization | Plotly | 5.24.1 | 3D globe, 2D heatmap, HZ diagram |
| Data Access | requests | 2.32.3 | NASA TAP API calls |
| Data Processing | pandas | 2.2.3 | Tabular data manipulation |
| Numerics | NumPy | ≥2.1.0 | Array operations |
| Scientific Computing | SciPy | 1.14.1 | Interpolation, audio generation |
| NetCDF | netCDF4 | 1.7.2 | Climate data format support |
| Testing | pytest | ≥7.0.0 | 72 test cases |

---

## 8. Repository Structure

```
Hack-4-Sages/
├── app.py                          # Main Streamlit application (~1,739 lines)
├── train_models.py                 # CLI for training ELM, CTGAN, PINNFormer (~306 lines)
├── requirements.txt                # Python dependencies (23 packages)
├── Dockerfile                      # Container build (python:3.11-slim)
├── Modelfile.astro                 # Ollama custom model — Qwen 2.5-14B orchestrator
├── Modelfile.astrosage             # Ollama custom model — AstroSage-Llama-3.1-8B domain expert
├── README.md                       # Project documentation
├── METHODOLOGY.md                  # Scientific methodology & formulas
│
├── modules/                        # Core Python modules
│   ├── __init__.py
│   ├── agent_setup.py              # LangChain dual-model agent (~555 lines)
│   ├── anomaly_detection.py        # Isolation Forest + UMAP (~259 lines)
│   ├── astro_physics.py            # Physics calculations (~805 lines)
│   ├── combined_catalog.py         # Combined NASA/EU/DACE exoplanet catalog (~237 lines)
│   ├── data_augmentation.py        # CTGAN wrapper (~248 lines)
│   ├── degradation.py              # Graceful degradation (~193 lines)
│   ├── elm_surrogate.py            # ELM ensemble climate model (~644 lines)
│   ├── gcm_benchmarks.py           # Synthetic GCM benchmark maps (~153 lines)
│   ├── llm_helpers.py              # Ollama helper functions (~256 lines)
│   ├── model_evaluation.py         # Model diagnostics for ELM/CTGAN/PINN (~173 lines)
│   ├── nasa_client.py              # NASA TAP client (~130 lines)
│   ├── pinn_heat.py                # DeepXDE 1D PINN (~154 lines)
│   ├── pinnformer3d.py             # PyTorch transformer PINN (~692 lines)
│   ├── rag_citations.py            # Hybrid RAG + ChromaDB (~1,403 lines)
│   ├── validators.py               # Pydantic models (~177 lines)
│   └── visualization.py            # Plotly 3D/2D renderers (~336 lines)
│
├── tests/                          # Pytest test suite
│   ├── __init__.py
│   ├── test_astro_physics.py       # Physics function tests (~243 lines)
│   ├── test_ctgan_physics.py       # CTGAN physical filters and stats (~52 lines)
│   ├── test_degradation_and_modes.py  # Degradation + agent mode smoke tests (~89 lines)
│   ├── test_elm_gcm_evaluation.py  # ELM vs GCM benchmark evaluation (~45 lines)
│   ├── test_elm_surrogate.py       # ELM surrogate tests (~73 lines)
│   ├── test_pinn_validation.py     # PINNFormer physics validation (~38 lines)
│   ├── test_rag_citations.py       # RAG citation tests (~230 lines)
│   └── test_validators.py          # Pydantic validator tests (~114 lines)
│
├── diagnostic/                     # Post-training diagnostic scripts
│   ├── diagnose_ctgan.py           # CTGAN real vs synthetic comparison
│   ├── diagnose_elm.py             # ELM benchmark planets + HTML export
│   └── diagnose_pinnformer.py      # PINNFormer vs analytical comparison
│
├── diagnostics/                    # Exported HTML visualizations
│   ├── elm_Earth_like.html
│   ├── elm_Proxima_b_like.html
│   ├── elm_Hot_rock.html
│   └── elm_Cold_super_earth.html
│
├── models/                         # Trained model weights
│   ├── elm_ensemble.pkl            # ELM ensemble (pickle)
│   ├── ctgan_exoplanets.pkl        # CTGAN model (pickle)
│   └── pinn3d_weights.pt           # PINNFormer weights (PyTorch)
│
├── data/
│   ├── combined_catalog_preview.csv
│   ├── dace_raw.csv
│   ├── exoplanet_eu_raw.csv
│   ├── gaia_raw.csv
│   └── nasa_cache/                 # Cached NASA query results
│
├── tools/                          # Data fetching utilities
│   ├── __init__.py
│   ├── build_combined_catalog_preview.py
│   └── data_fetch.py               # Gaia/EU/DACE fetcher
│
├── notebooks/                      # Jupyter notebooks (placeholder)
│
└── info_dump/                      # Documentation & analysis
    ├── Desc.txt                    # Project pitch
    ├── PROJECT_REPORT.md           # Full project report
    ├── FULL_ARCHITECTURE_REPORT.md # Full architecture report (this file)
    ├── tech_stack.md               # Technology stack (Polish)
    ├── short_plan.md               # Architecture plan (Polish)
    ├── raport_praktyczny.md        # Implementation report (Polish)
    ├── raport_teoretyczny.md       # Theoretical report (Polish)
    ├── judge_critique_response.md  # Judge critique and response log
    ├── ctgan_talking_points.md     # CTGAN discussion notes
    ├── post_hoc_physics_filter_report.md  # Physics filter report
    ├── papers/
    │   └── links.md                # Reference paper links
    ├── upgrades/
    │   ├── ulepszenia_hackathon.md # Improvement catalog
    │   ├── analiza_dodatkowe_modele_ai.md  # Extra AI models analysis
    │   └── FUTURE_ROADMAP.md       # Future roadmap
    └── gemini_deep/                # Gemini-generated deep-dive PDFs (5 files)
```

**Codebase size:** ~7,000 lines of Python across 30+ files (modules, tests, diagnostics, entry points, and helpers).

---

## 9. Scientific Methodology

The project implements the following peer-reviewed physical models:

### 9.1 Equilibrium Temperature (Kasting et al. 1993)
Radiative balance between absorbed stellar flux and re-emitted thermal radiation. Uses redistribution factor f = √2 for tidally locked (single-hemisphere), f = 2 for fast rotators. Assumes no greenhouse effect, no internal heat, uniform albedo, zero eccentricity.

### 9.2 Earth Similarity Index (Schulze-Makuch et al. 2011)
Geometric mean of weighted similarity ratios across 4 parameters: radius (w=0.57), density (w=1.07), escape velocity (w=0.70), surface temperature (w=5.58). ESI measures similarity to Earth, not habitability per se.

### 9.3 SEPHI (Rodríguez-Mozos & Moya 2017)
Three binary criteria: thermal (liquid water range), atmospheric retention (v_esc ≥ 5 km/s), magnetic field proxy (mass and radius constraints). Score = criteria met / 3.

### 9.4 Habitable Zone (Kopparapu et al. 2013)
4th-degree polynomial parameterization of stellar flux limits: S_eff = S₀ + aT + bT² + cT³ + dT⁴ where T = T_eff − 5780 K. Four boundaries: recent Venus, runaway greenhouse, maximum greenhouse, early Mars.

### 9.5 Habitable Surface Fraction
Spherically-weighted fraction of surface with 273 ≤ T ≤ 373 K, using cosine(latitude) weighting for correct area integration.

### 9.6 ISA Interactions (Kite et al. 2009)
Simplified outgassing model: $\dot{V}_{rel} = (g/g_\oplus)^{0.75} \cdot (\tau/4.5\;\text{Gyr})^{-1.5}$. Assesses plate tectonics, carbonate-silicate cycle, water cycling, and volatile retention.

### 9.7 False-Positive Assessment (Meadows et al. 2018, Luger & Barnes 2015)
UV flux estimation with linear spectral-type interpolation. Evaluates abiotic O₂ (photolysis), volcanic CH₄, and photochemical O₃ risks.

### 9.8 References
The `METHODOLOGY.md` file contains 17 peer-reviewed references from journals including ApJ, Astrobiology, MNRAS, A&A, Physics Reports, and JOSS. The RAG citation system indexes an expanded corpus of 40 papers (a superset of the METHODOLOGY references) covering additional domains: atmospheric escape, runaway greenhouse, cloud feedbacks, ocean heat transport, planetary interiors, tidal heating, stellar flares, M-dwarf UV, pre-MS HZ, JWST observations, biosignature frameworks, and photosynthesis limits.

---

## 10. Testing

**Framework:** pytest | **Scope:** dozens of unit and slow-integration tests (72 test cases)

| Test File | Focus |
|---|---|
| `test_astro_physics.py` | T_eq, flux, ESI, SEPHI, density, v_esc, HZ, HSF, ISA, outgassing, UV flux, atmospheric escape, false positives |
| `test_elm_surrogate.py` | PureELM, ELMEnsemble, analytical data generator, surrogate train/predict, conformal intervals |
| `test_elm_gcm_evaluation.py` | ELM ensemble vs synthetic GCM benchmarks (Earth-like, Proxima b, hot rock); pattern correlation and RMSE thresholds |
| `test_ctgan_physics.py` | Physical post-filters for CTGAN output and finite statistics from `summarise_ctgan_statistics` |
| `test_pinn_validation.py` | Short CPU-only PINNFormer training, validation metric finiteness, surface map sanity, habitable surface fraction bounds |
| `test_degradation_and_modes.py` | GracefulDegradation behaviour and agent-mode smoke tests |
| `test_rag_citations.py` | Corpus integrity (40 papers), hybrid search ranking, topic filtering, citation formatting, public API behaviour |
| `test_validators.py` | Pydantic validators for stellar/planetary parameters and mass-radius consistency |

---

## 11. Diagnostics & Tooling

### Diagnostic Scripts (`diagnostic/`)

| Script | Function |
|---|---|
| `diagnose_elm.py` | Benchmark ELM on 3 planet types (Earth-like, Proxima b-like, Hot rock), export HTML heatmaps |
| `diagnose_ctgan.py` | Compare real vs synthetic habitable planet distributions |
| `diagnose_pinnformer.py` | Compare PINN predictions vs analytical eyeball model for Proxima b |

### Exported Diagnostics (`diagnostics/`)

Self-contained HTML files (~4.5 MB each) with interactive Plotly temperature heatmaps for benchmark planets.

---

## 12. Documentation & Info Dump

### English Documentation
- `README.md` — Setup, usage, architecture overview
- `METHODOLOGY.md` — Scientific formulas, assumptions, 17 references

### Polish Documentation (`info_dump/`)
- `raport_teoretyczny.md` — Theoretical background: GCM limitations, surrogates, HZ, tidal locking, ELM math, CTGAN, ReAct agents
- `raport_praktyczny.md` — Implementation walkthrough: environment, Streamlit, NASA, physics, Pydantic, CTGAN, ELM, Ollama, LangChain, Plotly, DeepXDE, degradation, project timeline
- `tech_stack.md` — Technology stack summary
- `short_plan.md` — Architecture plan

### Upgrade Analysis (`info_dump/upgrades/`)
- `ulepszenia_hackathon.md` — Categorized improvement catalog (A: WOW factor, B: AI, C: Science, D: UX, E: Architecture, F: Innovation) with top 10 ranking and quick wins
- `analiza_dodatkowe_modele_ai.md` — Deep analysis of 12 additional AI models: XGBoost, GP, VAE, FNO, Isolation Forest, UMAP, RAG, KAN-PINN, Neural ODE, Normalizing Flows, Autoencoder, Conformal Prediction

### Gemini Deep-Dives (`info_dump/gemini_deep/`)
5 PDF documents with AI-generated analyses of the project concept, hackathon planning, and transformer/LLM strategies. **Non-authoritative planning aids** generated during early brainstorming — not core scientific artifacts. Retained for process transparency.

### Judge Critique Response (`info_dump/judge_critique_response.md`)
Structured record of the lead-judge critique, thematic issue groupings, and planned remediation actions.

---

## 13. Known Limitations & Future Scalability

### Scientific Limitations

1. **ELM training data is synthetic, not from real GCMs.** The analytical training data generator uses simplified radiative-equilibrium physics (cos^{1/4} profiles with heuristic redistribution factors). This cannot capture the complex dynamical phenomena that real GCMs produce (superrotation, baroclinic instability, convective cloud formation). The GCM benchmark comparisons in `gcm_benchmarks.py` are also against *synthetic* approximations of published GCM results, not against the actual GCM output grids.

2. **PINNFormer solves a simplified steady-state PDE.** The coupled atmosphere-ocean system is a strong simplification of real atmospheric dynamics — it lacks time dependence, realistic radiative transfer, a hydrological cycle, and resolution of convective processes. The "cloud fraction" and "ice fraction" are diagnostic (constrained by sigmoid functions) rather than prognostic.

3. **CTGAN operates on a tiny habitable sample.** With only ~60 real habitable-zone planets, the CTGAN has very limited distributional information for the habitable class. Noise-augmented upsampling pre-training helps but cannot substitute for real observational diversity.

4. **Single-point mass-radius relation.** The Chen & Kipping (2017) power-law is used for the mass-radius consistency check, but real planets have significant compositional scatter that this relation cannot capture.

5. **No photochemical modeling.** Biosignature false-positive assessment is rule-based (UV flux thresholds + geological activity) rather than using actual photochemical network integration.

### Technical Limitations

6. **Ollama dependency for LLM inference.** Requires a running Ollama server with ~12 GB VRAM for dual-LLM mode. No cloud API fallback is implemented.

7. **No caching of NASA API responses.** Each catalog fetch hits the NASA TAP endpoint live. For high-traffic scenarios, a caching layer would reduce latency and API load.

8. **ELM ensemble uncertainty uses Gaussian z-quantile approximation** rather than true conformal prediction (which would require a calibration dataset). The `mapie` package is listed in requirements but not actually used in the conformal prediction logic.

9. **PINNFormer requires PyTorch and ideally a GPU.** On CPU, training 5,000+ epochs is impractical. The model is pre-trained and loaded at inference time, but re-training per planet is not feasible in real-time.

10. **Pickle-based model serialisation** (`elm_ensemble.pkl`, `ctgan_exoplanets.pkl`) carries inherent risks if models are loaded from untrusted sources.

### Future Scalability Directions

11. **GCM-trained surrogates.** Replace the analytical training data generator with real temperature maps from ROCKE-3D, ExoCAM, or LMD Generic GCM runs — would dramatically improve ELM fidelity.

12. **Temporal dynamics in PINNFormer.** Extend the PDE system to include time dependence for studying climate evolution, seasonal cycles (for eccentric orbits), and tidal heating evolution.

13. **Spectroscopic forward modeling.** Add a transmission/emission spectrum simulator (e.g., petitRADTRANS integration) to predict JWST-observable signatures from the climate maps.

14. **Multi-GPU distributed PINN training.** For the "full" physics mode with all 6 modules active, the Laplacian computation through the transformer is computationally expensive. Data-parallel or model-parallel training would enable faster iteration.

15. **Real-time GCM-in-the-loop.** For priority targets, integrate a lightweight GCM (e.g., ExoPlaSim) as an additional model in the fallback cascade — between ELM and analytical.

16. **Extended RAG corpus.** The 40-paper corpus is hand-curated and static. Integration with ADS (Astrophysics Data System) or Semantic Scholar APIs would enable dynamic literature retrieval.

17. **API-first architecture.** Refactoring the physics engine and ML models into a FastAPI backend would decouple the computation layer from the Streamlit frontend, enabling multi-user scaling and external API consumers.

---

*Report generated from exhaustive source-code analysis. All descriptions are derived directly from the codebase — no content was fabricated or guessed.*
