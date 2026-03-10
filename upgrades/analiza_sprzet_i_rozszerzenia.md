# Analiza Możliwości Sprzętowych i Rozszerzeń Systemu

## Sprzęt: NVIDIA RTX 3050 Ti (4 GB VRAM)
**Benchmark referencyjny:** RNN-CNN 3.5M parametrów na spektrogramach — trening 30 s.

---

## 1. AstroLLaMA — Ocena Wykonalności

### 1.1 Problem: Rozmiar Modelu vs. VRAM

| Model | Parametry | Kwantyzacja | VRAM (inference) | VRAM (fine-tune) |
|---|---|---|---|---|
| AstroLLaMA-3-8B | 8B | FP16 | **~16 GB** | ~32 GB |
| AstroLLaMA-3-8B | 8B | Q4_K_M (4-bit) | **~5.0 GB** | — |
| AstroLLaMA-3-8B | 8B | Q3_K_S (3-bit) | **~3.8 GB** | — |
| AstroLLaMA-2-7B (stary) | 7B | Q4_K_M | **~4.5 GB** | — |

### 1.2 Werdykt: AstroLLaMA na RTX 3050 Ti

**🔴 NIE ZMIEŚCI SIĘ w wersji GPU-only.** Nawet przy agresywnej kwantyzacji Q4 model 8B zajmuje ~5 GB, a masz 4 GB VRAM.

**Rozwiązania:**

| Opcja | Jak | Wydajność | Rekomendacja |
|---|---|---|---|
| **Ollama CPU-only** | `OLLAMA_NUM_GPU=0 ollama run astrollama` | ~2–5 tok/s na i7 | ✅ Działa, ale wolne |
| **Ollama partial offload** | Automatyczny split CPU + GPU | ~5–10 tok/s | ✅ **Najlepsza opcja** |
| **Mniejszy model** | Qwen2.5-3B / Phi-3-mini-3.8B / LLaMA-3.2-3B | ~15 tok/s (Q4 na GPU) | ✅ **Najszybsza opcja** |
| **API zdalne** | Groq / Together.ai / OpenRouter (darmowe) | ~80 tok/s | ✅ Najszybciej, ale wymaga internetu |

### 1.3 Rekomendowana Strategia

```
Priorytet 1: Qwen2.5-3B (Q4_K_M) → mieści się w 4 GB VRAM
             Świetny function calling, szybki (~15 tok/s)
             ollama pull qwen2.5:3b

Priorytet 2: Ollama partial offload z modelem 7-8B
             Automatycznie wrzuca ile się da na GPU, resztę na CPU
             ollama pull llama3.2:8b   (a potem custom Modelfile z system prompt astro)

Priorytet 3: API zdalne (Groq cloud — darmowe 14k req/dzień)
             pip install langchain-groq
             Modele: llama-3.1-70b-versatile (za darmo, ~100 tok/s)
```

### 1.4 Specjalizacja Bez AstroLLaMA

AstroLLaMA wyróżnia się niską perpleksją na żargonie astrofizycznym, ale **nie jest konieczna** jako fundament agenta. Specjalizację astronomiczną osiągniemy inaczej:

1. **System prompt z wiedzą domenową** — wbudowujemy terminologię i ograniczenia fizyczne
2. **Narzędzia (tools)** — agent nie musi „wiedzieć" fizyki, bo deleguje do `compute_habitability()` i `run_elm_prediction()`
3. **RAG (Retrieval-Augmented Generation)** — opcjonalnie podłączamy bazę wiedzy z fragmentami artykułów naukowych
4. **Pydantic guardrails** — fizyka jest egzekwowana programistycznie, nie przez LLM

**Wniosek:** Qwen2.5-3B + dobry system prompt + narzędzia > AstroLLaMA 8B offline na CPU.

---

## 2. PINNFormer — Ocena Wykonalności

### 2.1 Czym jest PINNFormer?

PINNFormer (Zhao et al., 2023) to architektura Transformer zaprojektowana dla Physics-Informed Neural Networks. Zastępuje standardowy MLP w PINNach wielowarstwowym Transformerem z:
- Multi-head self-attention na kolokacyjnych punktach PDE
- Wavelet positional encoding zamiast standardowego sinusoidalnego
- Loss = $\mathcal{L}_{\text{data}} + \lambda \cdot \mathcal{L}_{\text{PDE}}$

### 2.2 Wymagania vs. Twój Sprzęt

| Aspekt | PINNFormer (typowy) | Twoje możliwości (RTX 3050 Ti) |
|---|---|---|
| Parametry | ~0.5–2M | ✅ OK (twój benchmark: 3.5M w 30s) |
| VRAM (trening) | ~1–2 GB (1D PDE) | ✅ Mieści się |
| VRAM (trening) | ~3–6 GB (2D PDE, 256 kolokcji) | ⚠️ Graniczne |
| Czas treningu (1D ciepło, 10k epok) | ~2–5 min | ✅ Akceptowalne |
| Czas treningu (2D ciepło, 50k epok) | ~30–120 min | ⚠️ Dużo na hackathon |

### 2.3 Werdykt: PINNFormer na RTX 3050 Ti

**🟡 WYKONALNE, ale z ograniczeniami:**

| Wariant | Feasibility | Uwagi |
|---|---|---|
| **1D równanie ciepła na terminatorze** | ✅ TAK | ~1 GB VRAM, trening 2–5 min, 256 punktów kolokacji |
| **2D mapa temperatury (lat × lon)** | ⚠️ RYZYKOWNE | ~3–4 GB VRAM, trening 30–60 min, wymaga batch'owania |
| **Pełny 3D (klimat atmosferyczny)** | 🔴 NIE | Przekracza VRAM i czas hackathonu |

### 2.4 Rekomendacja PINNFormer

**Użyj DeepXDE z prostszą architekturą (MLP-PINN) zamiast pełnego PINNFormera:**

```python
# Prosty PINN (MLP) — pewna ścieżka, działa na 4 GB
# 1D równanie ciepła: κ·T'' + S(x) - σ·T⁴ = 0
# DeepXDE MLP: [1] + [64]*3 + [1] = ~10k parametrów
# Trening: ~1 min na RTX 3050 Ti
# VRAM: < 500 MB

# PINNFormer — ambitna ścieżka, jako "upgrade"
# Wymaga custom implementacji (brak gotowej biblioteki)
# Trening 1D: ~5 min
# Więcej pracy kodowej na hackathonie
```

**Strategia:** Zaimplementuj PINN-MLP (DeepXDE) jako baseline, a PINNFormer jako opcjonalny upgrade jeśli czas pozwoli.

### 2.5 Implementacja Lekkiego PINN-MLP (Bezpieczna Ścieżka)

```python
import deepxde as dde
import numpy as np
import torch

# Wymuszenie GPU
dde.config.set_default_float("float32")  # float32 zamiast float64 → 2× mniej VRAM

# 1D stacjonarne równanie ciepła na terminatorze
SIGMA = 5.670374419e-8
KAPPA = 0.025
S_MAX = 900.0

def pde(x, T):
    dT_xx = dde.grad.hessian(T, x)
    S = S_MAX * torch.clamp(torch.cos(x), min=0)
    return KAPPA * dT_xx + S - SIGMA * T**4

geom = dde.geometry.Interval(0, np.pi)
bc_sub = dde.icbc.DirichletBC(geom, lambda x: 320.0, lambda x, on: np.isclose(x[0], 0))
bc_anti = dde.icbc.DirichletBC(geom, lambda x: 80.0,  lambda x, on: np.isclose(x[0], np.pi))

data = dde.data.PDE(geom, pde, [bc_sub, bc_anti], num_domain=256, num_boundary=2)

# Lekka sieć — 10k parametrów, ~200 MB VRAM
net = dde.nn.FNN([1] + [64]*3 + [1], "tanh", "Glorot normal")

model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
model.train(epochs=10000)  # ~1 min na RTX 3050 Ti
```

---

## 3. Dodatkowe Ulepszenia i Funkcje (Gdy Czas Pozwoli)

Uporządkowane od **najwyższego** do **najniższego** impact/effort ratio:

### 🏆 Tier 1: Wysokie Szanse na Realizację (2–4 h każde)

#### 3.1 RAG (Retrieval-Augmented Generation) dla Agenta
**Impact:** Ogromny — agent cytuje prawdziwe artykuły naukowe.
**Effort:** 2–3 h.

```python
# Implementacja z ChromaDB (wektorowa baza)
pip install chromadb langchain-chroma sentence-transformers

from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Załaduj abstrakty z arXiv (JSON/TXT)
# 2. Podziel na chunki
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(raw_docs)

# 3. Embeddingi + zapis
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # ~80 MB
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="data/chroma_db")

# 4. Retriever jako narzędzie agenta
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

**Efekt:** Gdy użytkownik pyta „co mówi literatura o klimacie TRAPPIST-1e?", agent znajduje i cytuje fragmenty artykułów.

#### 3.2 Porównanie Planet (Comparative Mode)
**Impact:** Duży — interaktywne porównanie dwóch planet obok siebie.
**Effort:** 2 h.

```python
# W Streamlit — dwie kolumny z dwoma globami
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(create_3d_globe(map_A, "TRAPPIST-1e"))
with col2:
    st.plotly_chart(create_3d_globe(map_B, "Proxima Cen b"))

# Tabela porównawcza
comparison = pd.DataFrame({
    "Parametr": ["T_eq", "ESI", "Naświetlenie", "Habitabilność"],
    "TRAPPIST-1e": [230, 0.85, 0.66, "✅"],
    "Proxima Cen b": [234, 0.87, 0.65, "✅"]
})
st.table(comparison)
```

#### 3.3 Eksport PDF / Raport Automatyczny
**Impact:** Średni-wysoki — naukowiec dostaje gotowy PDF.
**Effort:** 2 h.

```python
pip install fpdf2

from fpdf import FPDF

def generate_report(planet_data: dict, fig_path: str) -> str:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(text=f"Raport: {planet_data['name']}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    pdf.cell(text=f"T_eq = {planet_data['T_eq']} K")
    pdf.cell(text=f"ESI = {planet_data['ESI']}")
    pdf.image(fig_path, w=180)  # Screenshot globu
    path = f"reports/{planet_data['name']}_report.pdf"
    pdf.output(path)
    return path

# W Streamlit
if st.button("📄 Generuj raport PDF"):
    path = generate_report(data, "temp_globe.png")
    with open(path, "rb") as f:
        st.download_button("⬇️ Pobierz PDF", f, file_name="raport.pdf")
```

#### 3.4 Animacja Ewolucji Klimatu (Time-Lapse)
**Impact:** WOW-factor na prezentacji.
**Effort:** 3–4 h.

```python
import plotly.graph_objects as go

# Generuj mapy dla różnych wartości naświetlenia (symulacja zmian orbity)
frames = []
insol_range = np.linspace(0.3, 2.0, 30)

for i, insol in enumerate(insol_range):
    temp_map = elm_model.predict_from_params({**params, "insol_earth": insol})
    frames.append(go.Frame(
        data=[go.Surface(z=Z, x=X, y=Y, surfacecolor=temp_map)],
        name=f"S={insol:.2f}"
    ))

fig = go.Figure(data=frames[0].data, frames=frames)
fig.update_layout(
    updatemenus=[{
        "type": "buttons",
        "buttons": [
            {"label": "▶️ Play", "method": "animate",
             "args": [None, {"frame": {"duration": 200}, "transition": {"duration": 100}}]},
            {"label": "⏸️ Pause", "method": "animate",
             "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]}
        ]
    }],
    sliders=[{"steps": [{"args": [[f.name]], "label": f.name, "method": "animate"}
                         for f in frames]}]
)
```

### 🥈 Tier 2: Ambitne Rozszerzenia (4–8 h każde)

#### 3.5 Klasyfikator Typ Klimatu (Eyeball / Lobster / Greenhouse)
**Impact:** Naukowy — automatyczna klasyfikacja topologii.
**Effort:** 4–5 h.

```python
from sklearn.ensemble import RandomForestClassifier

# Etykiety z analizy map temperatur
# 0 = Eyeball, 1 = Lobster, 2 = Greenhouse, 3 = Snowball
labels = classify_topology(temperature_maps)  # Ręczne / heurystyczne

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_features, labels)

# Predykcja
climate_type = clf.predict(new_planet_features)
climate_names = {0: "🔵 Gałka Oczna", 1: "🦞 Homar", 2: "🔴 Cieplarnia", 3: "❄️ Śnieżka"}
st.info(f"Przewidywany typ klimatu: {climate_names[climate_type[0]]}")
```

#### 3.6 Integracja z danymi JWST (Widma Transmisyjne)
**Impact:** Cutting-edge nauka.
**Effort:** 6–8 h.

```python
# Pobieranie widm z MAST (Mikulski Archive for Space Telescopes)
from astroquery.mast import Observations

obs = Observations.query_criteria(
    target_name="TRAPPIST-1 e",
    instrument_name="NIRSPEC",
    obs_collection="JWST"
)

# Analiza biosygnatur (O3, CH4, CO2, H2O)
# Porównanie zmierzonego widma z modelami atmosferycznymi
```

#### 3.7 Uncertainty Quantification (Mapy Niepewności)
**Impact:** Naukowa rzetelność — pokazuje gdzie model jest pewny, a gdzie nie.
**Effort:** 3–4 h.

```python
# Zespół ELM naturalnie daje niepewność!
predictions = np.array([m.predict(X) for m in ensemble.models])
mean_map = predictions.mean(axis=0)
std_map = predictions.std(axis=0)   # ← MAPA NIEPEWNOŚCI

# Wizualizacja
fig_uncertainty = create_3d_globe(
    std_map.reshape(32, 64),
    planet_name="Niepewność predykcji",
    colorscale="Reds"
)
```

#### 3.8 Multi-Language Support (EN/PL)
**Impact:** UX, prezentacja.
**Effort:** 2 h.

```python
LANG = st.sidebar.selectbox("🌐 Język", ["PL", "EN"])
T = {
    "PL": {"title": "Cyfrowy Bliźniak Egzoplanetarny", "run": "Uruchom symulację"},
    "EN": {"title": "Exoplanetary Digital Twin", "run": "Run Simulation"}
}
st.title(T[LANG]["title"])
```

### 🥉 Tier 3: Stretch Goals (8+ h)

| Funkcja | Opis | Effort |
|---|---|---|
| **Fourier Neural Operator (FNO)** | Alternatywa dla ELM — rozwiązuje PDE w przestrzeni Fouriera | 10+ h |
| **Multiplayer** | Wielu użytkowników eksploruje różne planety jednocześnie | 8 h |
| **VR/AR mode** | WebXR — sfera planety w VR | 12+ h |
| **Orbital mechanics** | Symulacja orbity z biblioteką poliastro | 6 h |
| **Biosignature detection** | ML-owy detektor biosygnatur z widm JWST | 10+ h |

---

## 4. Optymalizacja Pamięci GPU — Poradnik RTX 3050 Ti

### 4.1 Budżet VRAM (4 GB)

```
TOTAL VRAM:                       4096 MB
├── System/driver overhead:        ~300 MB
├── Ollama (Qwen2.5-3B Q4):      ~2000 MB  ← LUB
├── Ollama (LLaMA 3.2-1B Q8):     ~1500 MB ← mniejszy
├── ELM (scikit-elm, CPU):            0 MB  ← CPU only!
├── CTGAN (trening):               ~500 MB  ← jednorazowo
├── PINN (DeepXDE, 1D):            ~200 MB
├── Plotly (rendering):             ~100 MB  ← przeglądarka, nie GPU
└── Zapas bezpieczeństwa:          ~500 MB
```

### 4.2 Kluczowe Zasady

1. **ELM trenuj na CPU** — scikit-elm używa NumPy (CPU), nie potrzebuje GPU. Pseudoodwrotność Moore'a-Penrose'a jest operacją CPU.
2. **CTGAN trenuj PRZED uruchomieniem Ollama** — nie ładuj obu jednocześnie. Zapisz model CTGAN, zwolnij GPU, potem startuj Ollama.
3. **Ollama: użyj Q4_K_M** — najlepszy stosunek jakość/VRAM.
4. **PINN: użyj float32** — `dde.config.set_default_float("float32")` — 2× mniej VRAM niż float64.
5. **Nie ładuj modeli jednocześnie** — Ollama + trening CTGAN = crash.

### 4.3 Sekwencja Operacji (Memory-Safe)

```
Faza 1 (offline): Trening CTGAN → zapisz .pkl → zwolnij GPU
Faza 2 (offline): Trening ELM (CPU) → zapisz .pkl
Faza 3 (offline): Trening PINN (opcja) → zapisz model → zwolnij GPU
Faza 4 (runtime): Ollama start → załaduj ELM (CPU) → Streamlit app
```

---

## 5. Podsumowanie Decyzji

| Komponent | Decyzja | Powód |
|---|---|---|
| **LLM** | Qwen2.5-3B (Q4) przez Ollama | Mieści się w 4 GB VRAM, dobry function calling |
| **LLM fallback** | Groq API (LLaMA 3.1-70B, darmowe) | Gdy lokalne zbyt wolne |
| **AstroLLaMA** | ❌ Porzucone | Nie mieści się; specjalizację osiągamy przez tools + prompt |
| **PINNFormer** | ⚠️ Opcjonalny | Zamiast tego: PINN-MLP (DeepXDE) — prostszy, stabilny |
| **Surogat** | ELM (CPU) | Trening CPU, predykcja CPU — GPU wolne dla LLM |
| **CTGAN** | ✅ Tak | Trening jednorazowy (~500 MB), zwolnienie GPU przed runtime |
| **Top ulepszenie** | RAG + porównanie planet + PDF export | Najwyższy impact/effort ratio |
