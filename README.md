# Autonomous Exoplanetary Digital Twin

A real-time, browser-based platform that simulates alien climates using AI-driven physics surrogates. Built for the HACK-4-SAGES hackathon.

## What it does

- **Query NASA** — pull real observational data for any confirmed exoplanet via the TAP protocol.
- **Compute habitability** — equilibrium temperature, Earth Similarity Index (ESI), SEPHI, habitable-zone boundaries (Kopparapu 2013), habitable surface fraction.
- **Predict climates** — an ensemble of Extreme Learning Machines (ELM) predicts 2-D surface temperature maps in milliseconds, replacing days of supercomputer GCM runs.
- **Augment data** — a CTGAN synthesises thousands of physically plausible habitable-planet configurations to fix the extreme class imbalance in real catalogs.
- **3-D visualisation** — interactive Plotly globe with scientific colour-mapping, host-star marker, habitable-zone overlay, and 2-D heatmap fallback.
- **LLM agent** — a dual-model LangChain agent (Qwen 2.5 + astro-specialist) autonomously reasons, calls tools, and explains results in natural language.
- **Physics guardrails** — Pydantic validators reject any output that violates thermodynamic or astrophysical constraints.
- **Graceful degradation** — every module has a fallback path so the app never crashes.

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10 | 3.11 |
| RAM | 8 GB | 16 GB |
| GPU | — | NVIDIA RTX with 8+ GB VRAM (CUDA) |
| Disk | 5 GB | 15 GB (includes LLM weights) |
| Ollama | latest | latest |

## Quick start

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd Hack-4-Sages

python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

If you have an NVIDIA GPU and want CUDA-accelerated training (CTGAN, PINNFormer), install PyTorch with CUDA **before** the requirements:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 3. Install and configure Ollama

Download Ollama from <https://ollama.com/download> and install it.

```bash
# Pull the primary agent model (~9 GB)
ollama pull qwen2.5:14b

# (Optional) Pull a lighter backup model (~4.5 GB)
ollama pull qwen2.5:7b

# Create the astro-specialist model from the included Modelfile
ollama create astro-agent -f Modelfile.astro

# Verify
ollama list
```

Ollama automatically starts a local API server on `http://localhost:11434`. On NVIDIA GPUs it uses the CUDA backend by default.

### 4. Train the models

A single script trains everything and populates the `models/` directory:

```bash
# Fast default — trains ELM only (~5 seconds on CPU)
python train_models.py

# Full suite — ELM + CTGAN + PINNFormer (needs internet + GPU)
python train_models.py --ctgan --pinn
```

| Flag | What it trains | Time | Requirements |
|------|---------------|------|-------------|
| *(none)* | ELM ensemble | ~5 s | CPU |
| `--ctgan` | + CTGAN augmenter | ~5–15 min | Internet (downloads NASA catalog), GPU helps |
| `--pinn` | + PINNFormer 3-D | ~1–2 h | PyTorch, GPU strongly recommended, no internet needed |

The app works even without trained models — it falls back to an analytical model for climate maps. But running at least `python train_models.py` (ELM) gives much better predictions.

### 5. Launch the application

```bash
streamlit run app.py
```

The browser opens automatically at **http://localhost:8501**.

## How to use the app

The interface has five tabs:

### Agent AI

Type a question in natural language, for example:

- *"Analyze the habitability of TRAPPIST-1 e"*
- *"What conditions exist on Proxima Centauri b?"*
- *"Compare Kepler-442 b with Earth"*

The agent autonomously fetches NASA data, computes indices, and optionally runs the climate surrogate. The **Reasoning Chain** panel on the right shows every tool call the agent makes.

Use the **Explanation depth** toggle (Scientist / Student / Media) to control how technical the response is.

### Manual Mode

Drag the sliders to set stellar and planetary parameters, then press **Run Simulation**. Enable **Live "What If" mode** to see the globe update in real time as you move sliders.

The panel shows:
- ESI gauge (0–1 scale with colour zones)
- SEPHI traffic lights (thermal / atmosphere / magnetic criteria)
- Habitable Surface Fraction (HSF)
- Interactive 3-D globe or 2-D heatmap

### Catalog

Browse the NASA Exoplanet Archive. Click one of the six famous-planet buttons for instant data, or press **Fetch full NASA catalog** to load all habitable-zone candidates.

### Science

Available after running a simulation:
- **Habitable Zone diagram** — planet position relative to HZ boundaries
- **Terminator cross-section** — temperature profile along the day–night boundary with freezing/boiling lines
- **Uncertainty estimates** — propagated from ELM ensemble variance
- **Planetary Soundscape** — sonification of the equatorial temperature profile

### System

- **Self-Diagnostics** — tests NASA connectivity, T_eq sanity, Pydantic guardrail, ELM loading, and Ollama status
- **Export** — download the 3-D globe as a standalone interactive HTML file
- **Docker** — deployment instructions

## Advanced training options

The `train_models.py` script accepts extra tuning flags:

```bash
python train_models.py --elm-samples 10000        # more training data for ELM
python train_models.py --ctgan --ctgan-epochs 500  # longer CTGAN training
python train_models.py --pinn --pinn-epochs 10000  # longer PINNFormer training
```

You can also train the lightweight 1-D PINN fallback (DeepXDE, CPU-friendly):

```python
from modules.pinn_heat import train_1d_pinn
model = train_1d_pinn(epochs=10000)
```

## Project structure

```
Hack-4-Sages/
├── app.py                      # Streamlit application (5 tabs)
├── requirements.txt            # Python dependencies
├── Modelfile.astro             # Ollama custom model config
├── Dockerfile                  # Container deployment
├── modules/
│   ├── nasa_client.py          # NASA TAP API client
│   ├── astro_physics.py        # T_eq, ESI, SEPHI, HZ, full pipeline
│   ├── validators.py           # Pydantic physics guardrails
│   ├── elm_surrogate.py        # ELM ensemble (scikit-elm + NumPy)
│   ├── data_augmentation.py    # CTGAN augmentation pipeline
│   ├── agent_setup.py          # LangChain dual-model agent
│   ├── visualization.py        # Plotly 3-D globe, 2-D heatmap, HZ diagram
│   ├── degradation.py          # Graceful-degradation manager
│   ├── pinnformer3d.py         # PINNFormer 3-D (PyTorch)
│   └── pinn_heat.py            # DeepXDE 1-D PINN fallback
├── models/                     # Trained model weights (.pkl, .pt)
├── data/nasa_cache/            # Cached NASA query results
├── notebooks/                  # Exploration notebooks
└── info_dump/                  # Research documents and reports
```

## Docker deployment

```bash
docker build -t exo-twin .
docker run -p 8501:8501 --gpus all exo-twin
```

Then open **http://localhost:8501**.

Note: Ollama must run separately or be added as a service in a `docker-compose.yml`.

## Tech stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| LLM hosting | Ollama (CUDA) |
| Agent framework | LangChain |
| Climate surrogate | ELM (scikit-elm / NumPy) |
| Data augmentation | CTGAN |
| PINN | DeepXDE + custom PINNFormer (PyTorch) |
| Validation | Pydantic |
| Visualisation | Plotly |
| Data source | NASA Exoplanet Archive (TAP) |
