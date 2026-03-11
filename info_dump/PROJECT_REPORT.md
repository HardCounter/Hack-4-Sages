# Autonomous Exoplanetary Digital Twin — Full Project Report

> **Repository:** Hack-4-Sages  
> **Context:** HACK-4-SAGES hackathon (72-hour sprint)  
> **Stack:** Python 3.10+ / Streamlit / Ollama / LangChain / scikit-elm / CTGAN / Plotly  
> **Generated:** 2026-03-11

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [System Architecture](#3-system-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Module-by-Module Breakdown](#5-module-by-module-breakdown)
   - 5.1 [Main Application (`app.py`)](#51-main-application-apppy)
   - 5.2 [Astrophysics Engine (`astro_physics.py`)](#52-astrophysics-engine-astro_physicspy)
   - 5.3 [LLM Agent System (`agent_setup.py`)](#53-llm-agent-system-agent_setuppy)
   - 5.4 [LLM Helpers (`llm_helpers.py`)](#54-llm-helpers-llm_helperspy)
   - 5.5 [NASA Client (`nasa_client.py`)](#55-nasa-client-nasa_clientpy)
   - 5.6 [ELM Climate Surrogate (`elm_surrogate.py`)](#56-elm-climate-surrogate-elm_surrogatepy)
   - 5.7 [PINNFormer 3D (`pinnformer3d.py`)](#57-pinnformer-3d-pinnformer3dpy)
   - 5.8 [1D PINN Heat (`pinn_heat.py`)](#58-1d-pinn-heat-pinn_heatpy)
   - 5.9 [CTGAN Data Augmentation (`data_augmentation.py`)](#59-ctgan-data-augmentation-data_augmentationpy)
   - 5.10 [Anomaly Detection (`anomaly_detection.py`)](#510-anomaly-detection-anomaly_detectionpy)
   - 5.11 [RAG Citations (`rag_citations.py`)](#511-rag-citations-rag_citationspy)
   - 5.12 [Visualization (`visualization.py`)](#512-visualization-visualizationpy)
   - 5.13 [Validators (`validators.py`)](#513-validators-validatorspy)
   - 5.14 [Graceful Degradation (`degradation.py`)](#514-graceful-degradation-degradationpy)
   - 5.15 [Model Training (`train_models.py`)](#515-model-training-train_modelspy)
6. [Data Flow & Pipeline](#6-data-flow--pipeline)
7. [Scientific Methodology](#7-scientific-methodology)
8. [AI/ML Model Details](#8-aiml-model-details)
9. [User Interface (5 Tabs)](#9-user-interface-5-tabs)
10. [Deployment & Infrastructure](#10-deployment--infrastructure)
11. [Testing](#11-testing)
12. [Diagnostics & Tooling](#12-diagnostics--tooling)
13. [Documentation & Info Dump](#13-documentation--info-dump)
14. [Known Limitations](#14-known-limitations)
15. [Dependencies](#15-dependencies)

---

## 1. Project Overview

The **Autonomous Exoplanetary Digital Twin** is a browser-based application that simulates alien climates in near real-time. It replaces computationally expensive General Circulation Models (GCMs, e.g. NASA ROCKE-3D, which require days of supercomputer time) with fast ML surrogates that produce results in seconds.

**Core value proposition:** A user types a natural-language question about an exoplanet; an AI agent autonomously queries NASA, computes habitability indices, runs a climate simulation, consults a domain-expert LLM, and returns an interpreted answer with a 3D interactive globe — all within seconds.

### Key Capabilities

| Capability | Implementation |
|---|---|
| Real-time exoplanet data | NASA Exoplanet Archive via TAP/ADQL |
| Habitability assessment | T_eq, ESI, SEPHI, HZ boundaries, HSF |
| Interior-Surface-Atmosphere coupling | ISA interaction model (Kite et al. 2009) |
| Biosignature false-positive analysis | UV flux, abiotic O₂/CH₄/O₃ risk assessment |
| Climate simulation | ELM ensemble surrogate (2D temperature maps) |
| Physics-informed modeling | PINNFormer 3D (transformer-based PINN) |
| Data augmentation | CTGAN for habitable planet oversampling |
| Anomaly detection | Isolation Forest + UMAP visualization |
| Scientific citations | Hybrid RAG over 40 peer-reviewed papers (ChromaDB + TF-IDF RRF) |
| Natural language interface | Dual-model LangChain agent (Qwen 2.5 + astro-agent) |
| 3D visualization | Plotly globe with temperature mapping |
| Input validation | Pydantic physics guardrails |
| Fault tolerance | Multi-level graceful degradation |

---

## 2. Repository Structure

```
Hack-4-Sages/
├── app.py                          # Main Streamlit application (751 lines)
├── train_models.py                 # CLI for training ELM, CTGAN, PINNFormer (218 lines)
├── requirements.txt                # Python dependencies (23 packages)
├── Dockerfile                      # Container build (python:3.11-slim)
├── Modelfile.astro                 # Ollama custom model definition
├── README.md                       # Project documentation
├── METHODOLOGY.md                  # Scientific methodology & formulas
├── Judges-requirements.md          # Hackathon judging criteria
├── lead-judge.md                   # Judge-specific targeting notes
├── .gitignore
│
├── modules/                        # Core Python modules (14 files)
│   ├── __init__.py
│   ├── agent_setup.py              # LangChain dual-model agent (439 lines)
│   ├── anomaly_detection.py        # Isolation Forest + UMAP (133 lines)
│   ├── astro_physics.py            # Physics calculations (429 lines)
│   ├── data_augmentation.py        # CTGAN wrapper (124 lines)
│   ├── degradation.py              # Graceful degradation (123 lines)
│   ├── elm_surrogate.py            # ELM ensemble climate model (331 lines)
│   ├── llm_helpers.py              # Ollama helper functions (226 lines)
│   ├── nasa_client.py              # NASA TAP client (131 lines)
│   ├── pinn_heat.py                # DeepXDE 1D PINN (76 lines)
│   ├── pinnformer3d.py             # PyTorch transformer PINN (239 lines)
│   ├── rag_citations.py            # Hybrid RAG + ChromaDB (1404 lines)
│   ├── validators.py               # Pydantic models (122 lines)
│   └── visualization.py            # Plotly 3D/2D renderers (297 lines)
│
├── tests/                          # Pytest test suite (72 tests)
│   ├── __init__.py
│   ├── test_astro_physics.py       # Physics function tests (131 lines)
│   ├── test_elm_surrogate.py       # ELM surrogate tests (74 lines)
│   ├── test_pinnformer3d.py        # PINNFormer tests (101 lines)
│   ├── test_rag_citations.py       # RAG citation tests (231 lines)
│   └── test_validators.py          # Pydantic validator tests (56 lines)
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
│   ├── pinn_Proxima_b_analytical.html
│   └── pinn_Proxima_b_PINNFormer.html
│
├── models/                         # Trained model weights
│   ├── elm_ensemble.pkl            # ELM ensemble (pickle)
│   ├── ctgan_exoplanets.pkl        # CTGAN model (pickle)
│   ├── pinn3d_weights.pt           # PINNFormer weights (PyTorch)
│   └── .gitkeep
│
├── data/
│   ├── nasa_cache/                 # Cached NASA query results
│   │   └── .gitkeep
│   └── chroma_db/                  # Persistent ChromaDB vector store (auto-generated, gitignored)
│
├── notebooks/                      # Jupyter notebooks (placeholder)
│   └── .gitkeep
│
└── info_dump/                      # Documentation & analysis
    ├── Desc.txt                    # Project pitch
    ├── tech_stack.md               # Technology stack (Polish)
    ├── short_plan.md               # Architecture plan (Polish)
    ├── raport_praktyczny.md        # Implementation report (Polish, 1766 lines)
    ├── raport_teoretyczny.md       # Theoretical report (Polish, 438 lines)
    ├── Hackathon_Questions.pdf
    ├── papers/
    │   └── links.md                # Reference paper links
    ├── upgrades/
    │   ├── ulepszenia_hackathon.md # Improvement catalog (500 lines)
    │   └── analiza_dodatkowe_modele_ai.md  # Extra AI models analysis (2363 lines)
    ├── use less slop/
    │   ├── analiza_sprzet_i_rozszerzenia.md  # RTX 3050 Ti hardware analysis
    │   └── raport_rx7600xt_analiza.md        # RX 7600 XT hardware analysis
    └── gemini_deep/                # Gemini-generated deep-dive PDFs (5 files)
```

**Codebase size:** ~5,500 lines of Python across 25 files (14 modules, 5 test files, 3 diagnostics, 2 entry points, 1 init).

---

## 3. System Architecture

The system follows a layered architecture with five distinct layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                        │
│              Streamlit UI (5 tabs, dark theme)               │
│   Agent AI │ Manual Mode │ Catalog │ Science │ System        │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                    ORCHESTRATION LAYER                       │
│              LangChain AgentExecutor (ReAct)                 │
│   ┌────────────────┐    ┌──────────────────┐                │
│   │  Qwen 2.5-14B  │    │  astro-agent     │                │
│   │  (Orchestrator) │◄──►│  (Domain Expert) │                │
│   └────────────────┘    └──────────────────┘                │
│              8 tools registered for function calling         │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                    COMPUTATION LAYER                         │
│  ┌───────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │ ELM       │ │PINNFormer│ │ CTGAN    │ │ Isolation    │  │
│  │ Ensemble  │ │ 3D       │ │          │ │ Forest+UMAP  │  │
│  └───────────┘ └──────────┘ └──────────┘ └──────────────┘  │
│  ┌───────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │ Physics   │ │ HZ       │ │ ISA      │ │ False-Pos    │  │
│  │ (T_eq,ESI)│ │ Boundary │ │ Coupling │ │ Assessment   │  │
│  └───────────┘ └──────────┘ └──────────┘ └──────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                    VALIDATION LAYER                          │
│          Pydantic models with physics constraints            │
│   StellarParameters │ PlanetaryParameters │ SimulationOutput │
│              + Graceful Degradation Manager                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                    DATA LAYER                                │
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ NASA Exoplanet  │  │ ChromaDB     │  │ Local Model    │  │
│  │ Archive (TAP)   │  │ (40 papers)  │  │ Storage (.pkl) │  │
│  └─────────────────┘  └──────────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Dual-LLM Design

The system uses two LLM instances via Ollama:

1. **Qwen 2.5-14B (Orchestrator):** General-purpose model with tool-calling capability. Decides which tools to invoke, sequences multi-step analyses, and synthesizes final answers.

2. **astro-agent (Domain Expert):** Same Qwen 2.5 base model but with a specialized astrophysics system prompt baked via `Modelfile.astro`. Interprets raw numerical results, classifies climate states, reviews physics plausibility, and provides scientific narratives.

The orchestrator always consults the domain expert after computing metrics — raw numbers are never presented without expert interpretation.

---

## 4. Technology Stack

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

## 5. Module-by-Module Breakdown

### 5.1 Main Application (`app.py`)

**Lines:** 751 | **Role:** Streamlit entry point

The application is organized into 5 tabs with a cosmic dark theme (custom CSS with Space Grotesk and Orbitron fonts, dark radial gradient background).

**Key architectural patterns:**
- `@st.cache_resource` for expensive singletons (agent, ELM model, PINNFormer)
- `st.session_state` for chat history, planet data, temperature maps, and analysis history
- `st.status` for pipeline progress tracking
- Lazy imports inside functions to avoid loading unused modules
- Tooltip glossary for scientific terms (ESI, T_eq, Albedo, etc.)

**Cached resources:**
- `_load_agent()` → LangChain AgentExecutor
- `_load_elm()` → ELMClimateSurrogate with model weights
- `_load_pinn()` → PINNFormer3D with weights (returns None if unavailable)

### 5.2 Astrophysics Engine (`astro_physics.py`)

**Lines:** 429 | **Role:** Core physics calculations

This is the scientific backbone of the project, implementing peer-reviewed formulations:

| Function | Formula / Source | Output |
|---|---|---|
| `equilibrium_temperature()` | T_eq = T★ √(R★ / f·a) · (1-A)^(1/4) | Temperature [K] |
| `stellar_flux()` | S = σT★⁴ · (R★/a)² | Flux [W/m²] and [S⊕] |
| `compute_esi()` | Schulze-Makuch et al. 2011 | ESI ∈ [0, 1] |
| `estimate_density()` | ρ = M / (4/3 π R³) | Density [g/cm³] |
| `estimate_escape_velocity()` | v_esc = √(2GM/R) | Velocity [km/s] |
| `compute_sephi()` | Rodríguez-Mozos & Moya 2017 | SEPHI score + 3 criteria |
| `hz_boundaries()` | Kopparapu et al. 2013 | 4 HZ distances [AU] |
| `habitable_surface_fraction()` | Cosine-weighted 273–373 K fraction | HSF ∈ [0, 1] |
| `estimate_outgassing_rate()` | Kite et al. 2009 | Relative volcanic rate |
| `estimate_isa_interaction()` | Plate tectonics + C-Si cycle + water cycling + volatile retention | ISA assessment dict |
| `estimate_uv_flux()` | Linear UV fraction interpolation | UV flux [W/m²] |
| `assess_biosignature_false_positives()` | Meadows et al. 2018, Luger & Barnes 2015 | O₂, CH₄, O₃ risk levels |
| `compute_full_analysis()` | Orchestrates all above | Combined analysis dict |

**Key implementation details:**
- The redistribution factor `f` is √2 for tidally locked planets (hemisphere re-emission) and 2 for fast rotators.
- ESI uses a geometric mean with weighted similarity ratios across 4 parameters (radius, density, escape velocity, temperature).
- SEPHI evaluates 3 binary criteria: thermal (273–373 K), atmospheric retention (v_esc ≥ 5 km/s), and magnetic field proxy.
- HZ boundaries use a 4th-degree polynomial in T_eff with coefficients from Kopparapu 2013 for 4 limits: recent Venus, runaway greenhouse, maximum greenhouse, early Mars.
- ISA assessment checks plate tectonics feasibility (0.5 ≤ M/M⊕ ≤ 5.0, R ≤ 2.0 R⊕), carbonate-silicate cycling, water cycling, and volatile retention.
- False-positive assessment evaluates UV-driven photolysis risks for abiotic O₂, volcanic CH₄, and photochemical O₃.

### 5.3 LLM Agent System (`agent_setup.py`)

**Lines:** 439 | **Role:** LangChain agent with 8 tools

Uses `create_tool_calling_agent` with a ReAct-style loop (max 10 iterations). Both LLMs connect to Ollama at `http://localhost:11434`.

**8 Registered Tools:**

| Tool | Description | Multi-step? |
|---|---|---|
| `query_nasa_archive` | Fetch planet data from NASA | No |
| `compute_habitability` | Calculate T_eq, ESI, SEPHI, flux | No |
| `run_climate_simulation` | ELM surrogate → temperature map stats | No |
| `consult_domain_expert` | Ask astro-agent for interpretation | No |
| `discover_most_habitable` | NASA → rank by ESI → expert evaluation | Yes (3 steps) |
| `compare_two_planets` | Fetch 2 planets → compute → expert comparison | Yes (3 steps) |
| `detect_anomalous_planets` | NASA → Isolation Forest → top anomalies | Yes (2 steps) |
| `cite_scientific_literature` | Hybrid RAG (semantic + TF-IDF RRF) → 40-paper corpus → formatted references with key findings; accepts optional topic filter tags | No |

**System prompt enforces a procedure:**
1. Fetch data from NASA first
2. Compute habitability metrics
3. Always consult domain expert (never present raw numbers)
4. Run climate simulation if requested
5. Synthesize into a clear answer

**Citation policy (new):** The system prompt includes a dedicated CITATION POLICY section requiring the agent to always cite after habitability analysis, use topic-specific queries, include 2-3 citations per substantive claim, and cross-reference false-positive literature when discussing biosignatures.

### 5.4 LLM Helpers (`llm_helpers.py`)

**Lines:** 226 | **Role:** Domain-expert and orchestrator helper functions for UI tabs

Functions called directly by the Streamlit UI (not through the agent):

| Function | LLM Used | Purpose |
|---|---|---|
| `interpret_simulation()` | astro-agent | Narrative interpretation of simulation results |
| `classify_climate_state()` | astro-agent | Classify as Eyeball/Lobster/Greenhouse/Temperate |
| `review_elm_output()` | astro-agent | Physics plausibility check |
| `summarise_planet_data()` | astro-agent | Natural-language summary of NASA data |
| `narrate_science_panel()` | astro-agent | Science dashboard narrative |
| `compare_planets()` | astro-agent | Side-by-side planet comparison |
| `generate_adql_query()` | Qwen 2.5 | Natural language → ADQL SQL translation |
| `generate_smart_suggestions()` | Qwen 2.5 | Proactive follow-up suggestions |

All functions use a `_safe()` wrapper for graceful fallback when Ollama is unavailable.

### 5.5 NASA Client (`nasa_client.py`)

**Lines:** 131 | **Role:** NASA Exoplanet Archive TAP interface

Queries the NASA Exoplanet Archive via the Table Access Protocol (TAP) at `https://exoplanetarchive.ipac.caltech.edu/TAP/sync`.

| Function | Query |
|---|---|
| `query_nasa_archive(adql)` | Execute arbitrary ADQL query |
| `get_planet_data(name)` | Fetch single planet from `pscomppars` |
| `get_habitable_candidates()` | Filter: R < 2.5 R_Jup, T_eq 200–400 K |
| `get_all_confirmed_planets()` | Full catalog with key columns |

Includes unit conversion helpers: Jupiter radii → Earth radii (×11.209), Jupiter masses → Earth masses (×317.83), AU → meters.

### 5.6 ELM Climate Surrogate (`elm_surrogate.py`)

**Lines:** 331 | **Role:** Primary climate prediction model

The Extreme Learning Machine (Huang et al. 2006) is the main climate surrogate, mapping planetary parameters to 2D temperature maps.

**Architecture:**
- `PureELM`: Single hidden layer with random frozen input weights and Moore-Penrose pseudoinverse solution: β = (HᵀH + λI)⁻¹ HᵀT
- `ELMEnsemble`: K=10 independent PureELMs for variance reduction
- `ELMClimateSurrogate`: High-level wrapper with training, prediction, save/load, and conformal intervals

**Input features (8):** radius_earth, mass_earth, semi_major_axis_au, star_teff_K, star_radius_solar, insol_earth, albedo, tidally_locked

**Output:** 32×64 temperature map (latitude × longitude)

**Training data generation:** `generate_analytical_training_data()` creates synthetic temperature maps using a simplified radiative model with day-night temperature contrast for tidally locked planets.

**Uncertainty quantification:** Conformal prediction intervals using ensemble spread as nonconformity score. At significance level α:
- CI = [ŷ − z_{α/2} · σ, ŷ + z_{α/2} · σ] where σ = ensemble standard deviation

Supports both `skelm` (optimized C library) and pure NumPy fallback.

### 5.7 PINNFormer 3D (`pinnformer3d.py`)

**Lines:** 239 | **Role:** Experimental physics-informed climate model

A transformer-based Physics-Informed Neural Network solving the steady-state heat equation on a tidally locked planet:

**PDE:** κ ∇²T + S(θ, φ) − σ T⁴ = 0

**Architecture:**
- `WaveletPositionalEncoding`: Multi-frequency sinusoidal encoding of spherical coordinates
- `PINNFormer3D`: PyTorch `nn.TransformerEncoder` with 4 heads, 3 layers, d_model=128
- Input: (θ, φ, z) — latitude, longitude, depth
- Output: Temperature T

**Training:**
- Collocation points sampled on the sphere
- Loss = PDE residual + boundary condition penalties
- Uses `sdpa_kernel(SDPBackend.MATH)` for second-order autograd compatibility

**Key functions:**
- `train_pinnformer()`: Full training loop
- `sample_surface_map()`: Generate 2D temperature map from trained model
- `save_pinnformer()` / `load_pinnformer()`: Weight persistence

Falls back to stub functions if PyTorch is not installed.

### 5.8 1D PINN Heat (`pinn_heat.py`)

**Lines:** 76 | **Role:** Fallback 1D PINN for terminator profile

Uses DeepXDE to solve κ T'' + S(x) − σ T⁴ = 0 along the terminator (day-night boundary). Provides a 1D temperature profile as a simpler alternative to the full 3D PINNFormer.

### 5.9 CTGAN Data Augmentation (`data_augmentation.py`)

**Lines:** 124 | **Role:** Synthetic habitable planet generation

Wraps the CTGAN library to generate synthetic exoplanets, targeting class imbalance in habitable-zone candidates (which are rare in the real catalog).

**Pipeline:**
1. `prepare_data()` — Select and clean columns from NASA catalog
2. `train()` — Train CTGAN with mode-specific normalization
3. `generate_synthetic_planets()` — Sample synthetic rows
4. `validate_synthetic_data()` — Compare distributions (real vs synthetic)

Trained model saved as `models/ctgan_exoplanets.pkl`.

### 5.10 Anomaly Detection (`anomaly_detection.py`)

**Lines:** 133 | **Role:** Identify unusual exoplanets

**Pipeline:**
1. `detect_anomalies()` — Isolation Forest (Liu et al. 2008) on multi-dimensional parameter space
   - Features: radius, mass, orbital distance, period, instellation, T_eq, stellar temperature, stellar radius
   - Adds `anomaly_score` and `is_anomaly` columns
2. `get_top_anomalies()` — Sort by anomaly score, return top N
3. `compute_umap_embedding()` — UMAP (McInnes et al. 2018) for 2D projection
4. `create_umap_figure()` — Plotly scatter with anomaly coloring

Anomalous planets may represent: rare habitable candidates, data quality issues, or genuinely unusual systems.

### 5.11 RAG Citations (`rag_citations.py`)

**Lines:** 1404 | **Role:** Hybrid literature retrieval and citation

Indexes 40 peer-reviewed papers in a persistent ChromaDB vector store (`data/chroma_db`) using `all-MiniLM-L6-v2` sentence embeddings over composite documents (abstract + key findings).

**Search architecture:** Hybrid retrieval combining:
1. ChromaDB semantic search (sentence-transformer cosine similarity)
2. TF-IDF weighted keyword search (IDF with stop-word removal)
3. Reciprocal Rank Fusion (Cormack et al. 2009, k=60) to merge rankings

**Paper coverage (40 papers across 6 domains):**

| Domain | Papers | Example Authors |
|---|---|---|
| HZ & Habitability Metrics (4) | HZ boundaries, ESI, SEPHI | Kopparapu 2013, Schulze-Makuch 2011, Kasting 1993, Rodríguez-Mozos 2017 |
| Tidal Locking & Climate States (5) | Eyeball/lobster states, atmospheric collapse, GCM | Turbet 2016, Shields 2016, Leconte 2013, Pierrehumbert 2011, Wordsworth 2015 |
| Biosignatures & False Positives (5) | O₂/CH₄ false positives, biosignature framework | Meadows 2018, Seager 2016, Luger 2015, Schwieterman 2018, Catling 2018 |
| Atmospheric Science (5) | Escape, runaway greenhouse, cosmic shoreline, clouds | Owen & Wu 2013, Goldblatt 2013, Zahnle & Catling 2017, Tian 2015, Wolf & Toon 2015 |
| Planetary Interiors (5) | Mass-radius, plate tectonics, tidal heating, outgassing | Chen & Kipping 2017, Kite 2009, Zeng 2019, Stamenković 2012, Walker 1981, Driscoll & Barnes 2015 |
| Stellar Context (4) | M-dwarf UV, flares, pre-MS HZ | Lammer 2009, Ramirez & Kaltenegger 2014, Segura 2010, France 2013 |
| Observational / JWST (4) | K2-18 b, LHS 475 b, TRAPPIST-1 b | Madhusudhan 2023, Lustig-Yaeger 2023, Greene 2023, Benneke 2019 |
| Climate Modeling (4) | Cloud feedback, OHT, ROCKE-3D | Yang 2013, Hu & Yang 2014, Joshi 1997, Del Genio 2019 |
| Astrobiology (4) | Habitability definition, photosynthesis limits | Cockell 2016, Raven & Cockell 2006, Petkowski 2020 |

**Per-paper schema (8 fields):** `id`, `title`, `authors`, `year`, `journal`, `abstract` (100–180 words with numerical results), `topics` (list of domain tags), `key_findings` (2–4 quantitative bullet points).

**Topic filtering:** Optional `topics` parameter on `cite_literature()` filters papers matching any requested tag (union semantics). Available tags include: `habitable_zone`, `m_dwarf`, `tidal_locking`, `biosignatures`, `false_positives`, `atmospheric_escape`, `climate_modeling`, `gcm`, `cloud_feedback`, `ocean_heat_transport`, `planetary_interior`, `plate_tectonics`, `mass_radius`, `stellar_activity`, `uv_environment`, `jwst`, `transit_spectroscopy`, `astrobiology`, `photosynthesis`.

**Persistent ChromaDB:** Uses `PersistentClient(path="data/chroma_db")` instead of in-memory. Auto-re-seeds when `collection.count() != len(_PAPERS)` (handles corpus expansions).

**Fallback:** If ChromaDB/sentence-transformers are unavailable, falls back to TF-IDF keyword search with log-weighted IDF and stop-word removal.

**Functions:**
- `cite_literature(query, n_results=5, topics=None)` → List of matching papers with metadata + key_findings
- `format_citations_markdown(citations)` → Formatted reference list
- `_hybrid_search(query, n_results, topics)` → Reciprocal Rank Fusion of semantic + keyword
- `_fallback_keyword_search(query, n_results, topics)` → TF-IDF weighted fallback

### 5.12 Visualization (`visualization.py`)

**Lines:** 297 | **Role:** Plotly-based scientific visualizations

| Function | Output |
|---|---|
| `generate_eyeball_map()` | Analytical temperature map for tidally locked (eyeball) or fast-rotating planets |
| `create_3d_globe()` | Interactive Plotly 3D sphere with temperature coloring and star marker |
| `create_2d_heatmap()` | Longitude-latitude heatmap with water zone annotations |
| `create_hz_diagram()` | Habitable zone distance diagram with 4 boundaries |

Uses a custom `SCIENCE_COLORSCALE` (blue → cyan → green → yellow → orange → red) and `_star_color()` for realistic stellar coloring based on T_eff.

The 3D globe maps temperature data onto a spherical surface using spherical coordinate transforms, with mesh lighting and a dark cosmic background.

### 5.13 Validators (`validators.py`)

**Lines:** 122 | **Role:** Pydantic physics guardrails

Three Pydantic models enforce physical constraints:

**`StellarParameters`:**
- teff: 2000–50000 K
- radius_solar: 0.08–100 R☉
- mass_solar: 0.08–150 M☉

**`PlanetaryParameters`:**
- radius_earth: 0.1–25 R⊕
- mass_earth: 0.01–13000 M⊕ (optional)
- semi_major_axis: 0.001–1000 AU
- albedo: 0.0–1.0
- Mass-radius consistency check using Chen & Kipping 2017 power law

**`SimulationOutput`:**
- T_eq_K: 10–5000 K
- ESI: 0.0–1.0
- flux_earth: > 0

These prevent physically impossible inputs from propagating through the pipeline.

### 5.14 Graceful Degradation (`degradation.py`)

**Lines:** 123 | **Role:** Fault tolerance

`GracefulDegradation` class provides `run_with_fallback(primary, fallback, timeout, label)` — attempts the primary function with a timeout, falls back on failure.

`run_simulation_pipeline()` implements a 3-tier fallback chain:
1. **ELM ensemble** → 2D temperature map (preferred)
2. **Analytical eyeball model** → Simplified radiative map (fallback)
3. **3D globe** → 2D heatmap (rendering fallback)

Also includes `validate_temperature_map()` to reject maps with NaN, infinite values, or physically unreasonable temperatures (< 2.7 K CMB floor or > 5000 K).

### 5.15 Model Training (`train_models.py`)

**Lines:** 218 | **Role:** CLI training entry point

```bash
python train_models.py                    # ELM only (~5 seconds)
python train_models.py --ctgan            # + CTGAN augmentation
python train_models.py --pinn             # + PINNFormer 3D
python train_models.py --ctgan --pinn     # All models
```

**CLI arguments:**
- `--elm-samples` (default: 2000) — Training samples for ELM
- `--elm-neurons` (default: 500) — Hidden neurons per ELM
- `--elm-models` (default: 10) — Ensemble size
- `--ctgan-epochs` (default: 300) — CTGAN training epochs
- `--pinn-epochs` (default: 5000) — PINNFormer training epochs

Outputs are saved to `models/` directory.

---

## 6. Data Flow & Pipeline

### Agent Query Flow (Tab 1)

```
User question
    │
    ▼
Qwen 2.5 (orchestrator) ──► Decides tool sequence
    │
    ├──► query_nasa_archive(planet) ──► NASA TAP API ──► planet data
    │
    ├──► compute_habitability(data) ──► astro_physics ──► T_eq, ESI, SEPHI, HZ
    │
    ├──► run_climate_simulation(params) ──► ELM ensemble ──► temperature map stats
    │
    ├──► consult_domain_expert(metrics) ──► astro-agent ──► interpretation
    │
    ├──► cite_scientific_literature(query, topics) ──► Hybrid RAG (semantic + TF-IDF RRF) ──► 40-paper corpus ──► references + key findings
    │
    └──► Synthesize final answer with citations
```

### Manual Simulation Flow (Tab 2)

```
User adjusts sliders
    │
    ▼
PlanetaryParameters validation (Pydantic)
    │
    ▼
equilibrium_temperature() + stellar_flux() + ESI + SEPHI
    │
    ▼
GracefulDegradation.run_with_fallback():
    ├── Try: PINNFormer 3D (if selected & tidally locked)
    ├── Try: ELM ensemble prediction
    └── Fallback: Analytical eyeball map
    │
    ▼
validate_temperature_map() ── if invalid ──► analytical fallback
    │
    ▼
habitable_surface_fraction()
    │
    ▼
Visualization (3D globe or 2D heatmap)
    │
    ▼
LLM interpretation (climate state, plausibility review)
```

### Catalog Flow (Tab 3)

```
Natural language query ──► generate_adql_query() (Qwen) ──► ADQL SQL
    │
    ▼
query_nasa_archive(adql) ──► NASA TAP ──► DataFrame
    │
    ▼
detect_anomalies() ──► Isolation Forest ──► anomaly scores
    │
    ▼
compute_umap_embedding() ──► UMAP 2D projection
    │
    ▼
create_umap_figure() ──► Interactive scatter plot
```

---

## 7. Scientific Methodology

The project implements the following peer-reviewed physical models:

### 7.1 Equilibrium Temperature (Kasting et al. 1993)
Radiative balance between absorbed stellar flux and re-emitted thermal radiation. Uses redistribution factor f = √2 for tidally locked (single-hemisphere), f = 2 for fast rotators. Assumes no greenhouse effect, no internal heat, uniform albedo, zero eccentricity.

### 7.2 Earth Similarity Index (Schulze-Makuch et al. 2011)
Geometric mean of weighted similarity ratios across 4 parameters: radius (w=0.57), density (w=1.07), escape velocity (w=0.70), surface temperature (w=5.58). ESI measures similarity to Earth, not habitability per se.

### 7.3 SEPHI (Rodríguez-Mozos & Moya 2017)
Three binary criteria: thermal (liquid water range), atmospheric retention (v_esc ≥ 5 km/s), magnetic field proxy (mass and radius constraints). Score = criteria met / 3.

### 7.4 Habitable Zone (Kopparapu et al. 2013)
4th-degree polynomial parameterization of stellar flux limits: S_eff = S₀ + aT + bT² + cT³ + dT⁴ where T = T_eff - 5780 K. Four boundaries: recent Venus, runaway greenhouse, maximum greenhouse, early Mars.

### 7.5 Habitable Surface Fraction
Spherically-weighted fraction of surface with 273 ≤ T ≤ 373 K, using cosine(latitude) weighting for correct area integration.

### 7.6 ISA Interactions (Kite et al. 2009)
Simplified outgassing model: V̇_rel = (g/g⊕)^0.75 · (τ/4.5 Gyr)^(-1.5). Assesses plate tectonics, carbonate-silicate cycle, water cycling, and volatile retention.

### 7.7 False-Positive Assessment (Meadows et al. 2018, Luger & Barnes 2015)
UV flux estimation with linear spectral-type interpolation. Evaluates abiotic O₂ (photolysis), volcanic CH₄, and photochemical O₃ risks.

### 7.8 References

The METHODOLOGY.md file contains 17 peer-reviewed references from journals including ApJ, Astrobiology, MNRAS, A&A, Physics Reports, and JOSS. The RAG citation system indexes an expanded corpus of 40 papers (a superset of the METHODOLOGY references) with extended abstracts and key findings, covering additional domains: atmospheric escape (Owen & Wu 2013, Zahnle & Catling 2017), runaway greenhouse (Goldblatt 2013), cloud feedbacks (Yang 2013, Wolf & Toon 2015), ocean heat transport (Hu & Yang 2014), planetary interiors (Walker 1981, Zeng 2019, Stamenković 2012), tidal heating (Driscoll & Barnes 2015), stellar flares (Segura 2010), M-dwarf UV (France 2013), pre-MS HZ (Ramirez & Kaltenegger 2014), JWST observations (Madhusudhan 2023, Greene 2023, Lustig-Yaeger 2023, Benneke 2019), biosignature framework (Schwieterman 2018, Catling 2018), habitability definition (Cockell 2016), and photosynthesis limits (Raven & Cockell 2006).

---

## 8. AI/ML Model Details

### 8.1 ELM Ensemble (Primary Climate Model)

| Property | Value |
|---|---|
| Architecture | Single hidden layer, random frozen weights |
| Training | Closed-form: β = (HᵀH + λI)⁻¹ HᵀT |
| Ensemble size | K=10 independent ELMs |
| Input dimension | 8 features |
| Output dimension | 32×64 = 2048 (temperature grid) |
| Training data | Analytically generated (not GCM) |
| Training time | ~5 seconds |
| Uncertainty | Conformal prediction via ensemble spread |

### 8.2 PINNFormer 3D (Experimental)

| Property | Value |
|---|---|
| Architecture | PyTorch TransformerEncoder (4 heads, 3 layers, d=128) |
| PDE solved | κ ∇²T + S(θ,φ) − σT⁴ = 0 |
| Input | (θ, φ, z) spherical coordinates |
| Positional encoding | Wavelet (multi-frequency sinusoidal) |
| Training | Collocation-point PDE residual minimization |
| Constraint | Requires PyTorch, optional CUDA |

### 8.3 CTGAN (Data Augmentation)

| Property | Value |
|---|---|
| Architecture | Conditional Tabular GAN |
| Purpose | Oversample habitable-zone planets |
| Input | NASA catalog columns |
| Training epochs | Default 300 |
| Validation | Distribution comparison (real vs synthetic) |

### 8.4 Isolation Forest (Anomaly Detection)

| Property | Value |
|---|---|
| Algorithm | Liu et al. 2008 |
| Features | 8 planetary/stellar parameters |
| Output | Anomaly score + binary label |
| Visualization | UMAP 2D projection |

### 8.5 RAG System

| Property | Value |
|---|---|
| Vector DB | ChromaDB (persistent, `data/chroma_db`) |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers, 384-dim) |
| Corpus | 40 peer-reviewed papers across 6 scientific domains |
| Per-paper content | Extended abstract (100–180 words) + 2–4 key findings |
| Search strategy | Hybrid: semantic (ChromaDB) + TF-IDF keyword, fused via Reciprocal Rank Fusion (k=60) |
| Topic filtering | Optional domain-tag filter (union semantics, ~20 tags) |
| Default results | 5 (increased from 3) |
| Fallback | TF-IDF weighted keyword search with IDF + stop-word removal |
| Domains covered | HZ, tidal locking, biosignatures, atmospheric escape, interiors, stellar context, JWST, climate modeling, astrobiology |

---

## 9. User Interface (5 Tabs)

### Tab 1: Agent AI
- Full chat interface with conversation history
- Audience selector (Scientist / Student / Media) adjusting explanation depth
- Transparent reasoning chain panel showing each tool invocation, input, and output
- Domain expert opinions highlighted with purple gradient styling
- LLM-generated proactive follow-up suggestions

### Tab 2: Manual Mode
- 7 parameter sliders (stellar temp, stellar radius, planet radius, planet mass, semi-major axis, albedo, tidal locking)
- PINNFormer 3D toggle (experimental)
- "Run Simulation" button + "Live What If" toggle for real-time updates
- ESI gauge (Plotly indicator with red/yellow/green zones)
- SEPHI traffic lights (thermal, atmosphere, magnetic)
- ISA coupling badge and false-positive risk badge
- 3D globe / 2D heatmap toggle
- AI interpretation expandable (climate state classification, physics plausibility review)

### Tab 3: Planet Catalog
- Natural language search → LLM-generated ADQL query → NASA results
- Famous exoplanets gallery (TRAPPIST-1 e, Proxima Cen b, K2-18 b, Kepler-442 b, TOI-700 d, LHS 1140 b)
- Planet detail view with domain expert summary
- Full NASA catalog fetch
- Anomaly detection with top anomalies table
- UMAP scatter plot visualization

### Tab 4: Science Dashboard
- Scientific narrative (LLM-generated)
- Habitable Zone diagram with 4 boundaries and planet position
- ISA interaction detail (score, outgassing rate, 4 boolean criteria)
- Photochemical false-positive assessment (O₂, CH₄, UV flux metrics + risk flags)
- Terminator cross-section plot (equatorial temperature profile with 273 K / 373 K markers)
- Uncertainty dashboard (conformal prediction intervals from ELM ensemble, or static estimates)
- Earth comparison (domain expert)
- Planetary soundscape (temperature → frequency sonification, 5-second WAV)

### Tab 5: System
- Self-diagnostics (NASA API, T_eq sanity check, Pydantic guardrail, ELM model, Ollama status)
- HTML globe export (interactive Plotly with CDN)
- System architecture diagram (Mermaid flowchart)
- Docker deployment instructions

---

## 10. Deployment & Infrastructure

### Local Development

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
ollama pull qwen2.5:14b
ollama create astro-agent -f Modelfile.astro
python train_models.py
streamlit run app.py
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Ollama Custom Model

```
FROM qwen2.5:14b
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
SYSTEM """You are AstroAgent, an expert astrophysics assistant..."""
```

### Hardware Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| Python | 3.10+ | 3.11 |
| RAM | 8 GB | 16 GB |
| GPU | Not required | NVIDIA with CUDA (for Ollama, CTGAN, PINNFormer) |
| Disk | ~2 GB (models + Ollama) | ~10 GB |
| Network | Required (NASA API) | — |

---

## 11. Testing

**Framework:** pytest | **Total tests:** 72

| Test File | Tests | Coverage |
|---|---|---|
| `test_astro_physics.py` | ~15 | T_eq, flux, ESI, SEPHI, density, v_esc, HZ, HSF, ISA, false positives |
| `test_elm_surrogate.py` | ~10 | PureELM, ELMEnsemble, analytical data, surrogate train/predict, conformal |
| `test_pinnformer3d.py` | ~8 | Forward pass, loss, short training, surface map (skip if no PyTorch) |
| `test_rag_citations.py` | ~34 | Corpus integrity (40 papers, required fields, unique IDs, topics, key_findings), tokenisation, TF-IDF scoring, fallback search (results, ranking, topic filter), topic filtering (single, multi, domain coverage), composite documents, citation formatting, public API (n_results, topic restriction), domain coverage (atmospheric escape, JWST, clouds, carbonate-silicate, OHT) |
| `test_validators.py` | ~5 | Valid/invalid inputs, mass-radius consistency |

---

## 12. Diagnostics & Tooling

### Diagnostic Scripts (`diagnostic/`)

| Script | Function |
|---|---|
| `diagnose_elm.py` | Benchmark ELM on 3 planet types (Earth-like, Proxima b-like, Hot rock), export HTML heatmaps |
| `diagnose_ctgan.py` | Compare real vs synthetic habitable planet distributions |
| `diagnose_pinnformer.py` | Compare PINN predictions vs analytical eyeball model for Proxima b |

### Exported Diagnostics (`diagnostics/`)

5 self-contained HTML files (~4.5 MB each) with interactive Plotly temperature heatmaps for benchmark planets.

---

## 13. Documentation & Info Dump

### English Documentation
- `README.md` — Setup, usage, architecture overview
- `METHODOLOGY.md` — Scientific formulas, assumptions, 17 references
- `Judges-requirements.md` — Hackathon judging criteria (7 requirements)
- `lead-judge.md` — Judge-specific notes and targeting strategy

### Polish Documentation (`info_dump/`)
- `raport_teoretyczny.md` (438 lines) — Theoretical background: GCM limitations, surrogates, HZ, tidal locking, ELM math, CTGAN, ReAct agents
- `raport_praktyczny.md` (1766 lines) — Implementation walkthrough: environment, Streamlit, NASA, physics, Pydantic, CTGAN, ELM, Ollama, LangChain, Plotly, DeepXDE, degradation, project timeline
- `tech_stack.md` — Technology stack summary
- `short_plan.md` — Architecture plan

### Upgrade Analysis (`info_dump/upgrades/`)
- `ulepszenia_hackathon.md` (500 lines) — Categorized improvement catalog (A: WOW factor, B: AI, C: Science, D: UX, E: Architecture, F: Innovation) with top 10 ranking and quick wins
- `analiza_dodatkowe_modele_ai.md` (2363 lines) — Deep analysis of 12 additional AI models: XGBoost, GP, VAE, FNO, Isolation Forest, UMAP, RAG, KAN-PINN, Neural ODE, Normalizing Flows, Autoencoder, Conformal Prediction

### Hardware Analysis (`info_dump/use less slop/`)
- `analiza_sprzet_i_rozszerzenia.md` — NVIDIA RTX 3050 Ti (4 GB) feasibility analysis
- `raport_rx7600xt_analiza.md` — AMD RX 7600 XT (16 GB) feasibility analysis with ROCm/Vulkan considerations

### Research Papers (`info_dump/papers/`)
- Links to 3D GCM multi-model simulation papers (arXiv)

### Gemini Deep-Dives (`info_dump/gemini_deep/`)
- 5 PDF documents with AI-generated analyses of the project concept, hackathon planning, and transformer/LLM strategies

---

## 14. Known Limitations

1. **No GCM validation** — ELM is trained on analytically generated data, not calibrated against 3D climate models (ExoCAM, ROCKE-3D).
2. **Static atmosphere** — No photochemistry, cloud feedback, or atmospheric evolution modeling.
3. **Fixed albedo** — User-specified rather than self-consistently computed from atmospheric composition.
4. **Circular orbits only** — No eccentricity support; eccentric orbits can significantly affect habitability.
5. **Simplified ISA** — Outgassing model is highly simplified; real mantle convection depends on rheology, composition, and thermal history.
6. **Linear UV fraction** — Real stellar spectra are complex; should use model atmospheres (PHOENIX, BT-Settl).
7. **No atmospheric escape** — Volatile retention is a simple threshold, not time-integrated escape modeling.
8. **Ollama dependency** — AI interpretation features require a running Ollama instance with ~8 GB VRAM for Qwen 2.5-14B.
9. **Network dependency** — NASA TAP queries require internet connectivity.

---

## 15. Dependencies

Full dependency list from `requirements.txt`:

```
streamlit==1.41.0
pandas==2.2.3
numpy>=2.1.0
plotly==5.24.1
requests==2.32.3
pydantic==2.10.3
scikit-learn==1.6.0
scikit-elm==0.21a0
ctgan==0.12.1
langchain==0.3.13
langchain-ollama==0.2.3
langchain-core>=0.3.0
ollama==0.4.4
scipy==1.14.1
netCDF4==1.7.2
deepxde==1.12.0
kaleido==0.2.1
imageio==2.36.1
chromadb>=0.4.0
sentence-transformers>=2.2.0
umap-learn>=0.5.0
mapie>=0.8.0
pytest>=7.0.0
```

**Optional (not in requirements.txt):**
- PyTorch (for PINNFormer 3D and CUDA acceleration)
- NVIDIA CUDA toolkit (for GPU-accelerated inference)

---

*End of report.*
