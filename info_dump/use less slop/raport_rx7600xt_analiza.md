# Analiza Możliwości Projektu na AMD Radeon RX 7600 XT (16 GB VRAM)

## Porównanie z Poprzednią Konfiguracją

| Parametr | NVIDIA RTX 3050 Ti (stara) | AMD RX 7600 XT (nowa) | Zmiana |
|---|---|---|---|
| **VRAM** | 4 GB GDDR6 | **16 GB GDDR6** | **4× więcej** |
| **Przepustowość pamięci** | 192 GB/s | **288 GB/s** | 1.5× więcej |
| **Architektura** | Ampere (CUDA) | RDNA 3 (ROCm) | Zmiana ekosystemu |
| **FP32 TFLOPS** | ~5.3 | ~12.4 | ~2.3× więcej |
| **FP16 TFLOPS** | ~10.6 | ~24.8 | ~2.3× więcej |
| **TDP** | 60 W | 150 W | Więcej mocy |
| **Sterowniki ML** | CUDA (pełne wsparcie) | ROCm (ograniczone na Windows) | ⚠️ Krytyczne |

**Kluczowe wnioski:**
- 16 GB VRAM to **game-changer** — otwiera drzwi do modeli niedostępnych na 4 GB
- Zmiana z NVIDIA na AMD wprowadza **istotne problemy z kompatybilnością oprogramowania**
- Surowa moc obliczeniowa jest ~2× lepsza

---

## 1. KRYTYCZNY PROBLEM: Ekosystem AMD vs NVIDIA na Windows

### 1.1 Stan Wsparcia ROCm (marzec 2026)

| Aspekt | NVIDIA (CUDA) | AMD (ROCm) |
|---|---|---|
| **PyTorch** | ✅ Natywne, stabilne | ⚠️ ROCm oficjalnie **tylko Linux** |
| **PyTorch na Windows** | ✅ `pip install torch` | 🔴 **Brak oficjalnego ROCm na Windows** |
| **DirectML (Windows)** | N/A | ⚠️ Alternatywa, ale ograniczona |
| **Ollama** | ✅ Pełne GPU | ✅ Wspiera AMD (Vulkan/ROCm) |
| **llama.cpp** | ✅ CUDA | ✅ Vulkan backend (działa na AMD) |
| **DeepXDE** | ✅ CUDA | 🔴 Wymaga PyTorch GPU → problem |
| **CTGAN** | ✅ CUDA | 🔴 Wymaga PyTorch GPU → problem |
| **scikit-elm** | ✅ CPU (NumPy) | ✅ CPU (NumPy) — brak zmian |

### 1.2 Możliwe Ścieżki Obejścia (Windows)

| Opcja | Jak | Trudność | Rekomendacja |
|---|---|---|---|
| **WSL2 + ROCm** | Ubuntu w WSL2, instalacja ROCm 6.x + PyTorch ROCm | Średnia | ✅ **Najlepsza opcja dla PyTorch** |
| **DirectML** | `pip install torch-directml` | Łatwa | ⚠️ Działa, ale wolniejsze i mniej kompatybilne |
| **Vulkan (llama.cpp/Ollama)** | Natywne wsparcie w Ollama | Łatwa | ✅ **Dla LLM — działa od razu** |
| **Dual-boot Linux** | Instalacja Ubuntu obok Windows | Duża | ✅ Pełne wsparcie, ale wymaga czasu |
| **CPU-only PyTorch** | `pip install torch` (domyślnie CPU) | Łatwa | ⚠️ Brak akceleracji GPU dla CTGAN/PINN |

### 1.3 Rekomendowana Konfiguracja

```
WARIANT A (Pragmatyczny — hackathon):
├── LLM (Ollama): Natywne wsparcie AMD przez Vulkan → pełne 16 GB VRAM ✅
├── ELM (scikit-elm): CPU — bez zmian ✅ 
├── CTGAN: CPU-only PyTorch (wolniejsze, ale działa) ⚠️
├── PINN/DeepXDE: CPU-only PyTorch ⚠️
└── Plotly/Streamlit: Bez zmian ✅

WARIANT B (Optymalny — z przygotowaniem):
├── WSL2 + Ubuntu 22.04 + ROCm 6.x
├── PyTorch ROCm → pełna akceleracja GPU  
├── LLM: Ollama w WSL2 lub natywnie Windows
├── Wszystkie modele ML: GPU-accelerated ✅
└── Wymaga: ~2-3h konfiguracji WSL2 + ROCm
```

---

## 2. AstroLLaMA — Nowa Analiza z 16 GB VRAM

### 2.1 Tabela Wykonalności (zaktualizowana)

| Model | Parametry | Kwantyzacja | VRAM (inference) | VRAM (fine-tune) | RX 7600 XT? |
|---|---|---|---|---|---|
| AstroLLaMA-3-8B | 8B | **FP16** | ~16 GB | ~32 GB | ✅ **MIEŚCI SIĘ!** (na styk) |
| AstroLLaMA-3-8B | 8B | Q8_0 (8-bit) | ~8.5 GB | — | ✅ Komfortowo |
| AstroLLaMA-3-8B | 8B | Q5_K_M (5-bit) | ~5.7 GB | — | ✅ Dużo zapasu |
| AstroLLaMA-3-8B | 8B | Q4_K_M (4-bit) | ~5.0 GB | — | ✅ Dużo zapasu |
| AstroLLaMA-2-7B | 7B | FP16 | ~14 GB | ~28 GB | ✅ Mieści się |
| LLaMA-3.1-8B | 8B | Q8_0 | ~8.5 GB | — | ✅ Komfortowo |
| Qwen2.5-7B | 7B | Q8_0 | ~7.5 GB | — | ✅ Komfortowo |
| **Qwen2.5-14B** | **14B** | **Q4_K_M** | **~9 GB** | — | ✅ **NOWA OPCJA!** |
| **LLaMA-3.1-70B** | **70B** | **Q2_K** | **~25 GB** | — | 🔴 Nie mieści się |

### 2.2 Werdykt: AstroLLaMA na RX 7600 XT

**🟢 WYKONALNE! AstroLLaMA 8B w pełni mieści się w 16 GB VRAM.**

To fundamentalna zmiana w stosunku do RTX 3050 Ti (4 GB), gdzie AstroLLaMA była całkowicie wykluczona.

**Osiągalna wydajność przez Ollama (Vulkan backend):**

| Model | Kwantyzacja | VRAM | Szacowane tok/s | Rekomendacja |
|---|---|---|---|---|
| AstroLLaMA-3-8B | Q4_K_M | ~5 GB | ~25-35 tok/s | ✅ **Szybka + lekka** |
| AstroLLaMA-3-8B | Q5_K_M | ~5.7 GB | ~20-30 tok/s | ✅ Lepsza jakość |
| AstroLLaMA-3-8B | Q8_0 | ~8.5 GB | ~15-22 tok/s | ✅ Najlepsza jakość offline |
| AstroLLaMA-3-8B | FP16 | ~16 GB | ~10-15 tok/s | ⚠️ Na limicie VRAM |
| Qwen2.5-14B | Q4_K_M | ~9 GB | ~12-18 tok/s | ✅ **Większy model!** |

### 2.3 Nowa Strategia LLM (Priorytetyzacja)

```
Priorytet 1: AstroLLaMA-3-8B (Q5_K_M) → 5.7 GB VRAM
             Specjalistyczna wiedza astrofizyczna, niska perpleksja
             Wystarczające performance (~25 tok/s)
             ollama pull astrollama   (lub custom Modelfile)
             
             ZOSTAJE 10.3 GB VRAM NA INNE MODELE!

Priorytet 2: Qwen2.5-14B (Q4_K_M) → ~9 GB VRAM 
             Znacznie lepszy reasoning i function calling niż modele 7-8B
             Możliwy JEDNOCZEŚNIE z PINN jeśli PINN na CPU
             ollama pull qwen2.5:14b

Priorytet 3: Dual-model setup:
             AstroLLaMA-8B (Q4, ~5 GB) dla wiedzy domenowej
             + Qwen2.5-7B (Q4, ~4.5 GB) dla function calling
             Razem: ~9.5 GB — mieści się z zapasem!

Priorytet 4: API zdalne jako fallback (Groq — darmowe)
             Nie wymaga VRAM, ~100 tok/s
```

### 2.4 AstroLLaMA — Czy Warto?

Z 16 GB VRAM argument "nie mieści się" **odpada**. Pytanie staje się: czy AstroLLaMA daje wystarczającą przewagę?

| Aspekt | AstroLLaMA-3-8B | Qwen2.5-7B/14B |
|---|---|---|
| Perpleksja astro | **~6.1** (specjalistyczna) | ~25-32 (ogólna) |
| Function calling | ⚠️ Słabszy (base model) | ✅ **Dedykowane wsparcie** |
| Żargon naukowy | ✅ Natywny | ⚠️ Wymaga system prompt |
| Dostępność w Ollama | ⚠️ Wymaga custom Modelfile | ✅ Gotowy `ollama pull` |
| Reasoning/planning | ⚠️ Bazowy | ✅ Lepszy (instruction-tuned) |

**Rekomendacja:** 
- **Jeśli priorytetem jest demos "wow" z żargonem naukowym** → AstroLLaMA
- **Jeśli priorytetem jest działający agent z function calling** → Qwen2.5-14B (Q4)
- **Najlepsze z obu światów**: Użyj Qwen2.5-14B jako agenta orkiestracji, a AstroLLaMA jako "konsultanta naukowego" wywoływanego jako narzędzie (tool) do interpretacji wyników

---

## 3. PINNFormer 3D — Nowa Analiza z 16 GB VRAM

### 3.1 Czym jest PINNFormer?

PINNFormer (Zhao et al., 2023) zastępuje MLP w Physics-Informed Neural Networks transformerem z:
- Multi-head self-attention na punktach kolokacji PDE
- Wavelet positional encoding
- Loss = $\mathcal{L}_{\text{data}} + \lambda \cdot \mathcal{L}_{\text{PDE}}$

### 3.2 Wymagania 3D PINNFormer vs. RX 7600 XT

| Wariant PDE | Parametry modelu | Punkty kolokacji | VRAM (trening) | Czas treningu | RX 7600 XT? |
|---|---|---|---|---|---|
| 1D ciepło (terminator) | ~0.5M | 256 | ~0.5-1 GB | 2-5 min | ✅ Łatwe |
| 2D mapa (lat × lon) | ~1-2M | 2048 | ~2-4 GB | 15-30 min | ✅ **TERAZ WYKONALNE!** |
| **3D atmosfera (lat × lon × alt)** | ~2-5M | 8192-32768 | **~6-12 GB** | 1-4 h | ✅ **TERAZ WYKONALNE!** |
| 3D pełny klimat (z cyrkulacją) | ~10-20M | 65536+ | ~15-30 GB | 8+ h | 🔴 Graniczne/Nie |

### 3.3 Werdykt: 3D PINNFormer na RX 7600 XT

**🟢 3D PINNFormer JEST WYKONALNY na RX 7600 XT!**

To kluczowa zmiana — na RTX 3050 Ti nawet 2D była ryzykowna. Z 16 GB VRAM:

| Wymiar PDE | Stara karta (4 GB) | RX 7600 XT (16 GB) |
|---|---|---|
| 1D ciepło | ✅ OK | ✅ Trywialne |
| 2D mapa temperatur | ⚠️ Graniczne | ✅ **Komfortowe** |
| **3D profil atmosferyczny** | 🔴 Niemożliwe | ✅ **Wykonalne!** |
| 3D pełny GCM | 🔴 Niemożliwe | ⚠️ Graniczne |

### 3.4 Problem: PyTorch na AMD Windows

**UWAGA KRYTYCZNA:** PINNFormer wymaga PyTorch z GPU. Na AMD + Windows to nie jest trywialne.

| Ścieżka | Jak uruchomić PyTorch GPU na AMD | Status |
|---|---|---|
| **ROCm (WSL2)** | WSL2 → Ubuntu → `pip install torch --index-url https://download.pytorch.org/whl/rocm6.2` | ✅ Działa |
| **DirectML** | `pip install torch-directml` → `device = torch_directml.device()` | ⚠️ Ograniczone — nie wszystkie operatory |
| **CPU-only** | Standardowy PyTorch, trening na CPU | ⚠️ 10-50× wolniejszy |

**Dla 3D PINNFormer na hackathonie:**

```
Jeśli WSL2 + ROCm skonfigurowane:
  → Trening GPU: 3D w ~1-2h ✅
  → VRAM: ~8-10 GB, zostaje ~6 GB na Ollama

Jeśli tylko Windows (bez WSL2):
  → Trening CPU: 3D w ~10-20h 🔴 Za długo na hackathon
  → Alternatywa: DirectML (jeśli kompatybilne) → ~3-6h ⚠️
  → Alternatywa: Google Colab (GPU darmowe) → ~1-2h ✅
```

### 3.5 Implementacja 3D PINNFormer

```python
# 3D stacjonarne równanie ciepła: κ∇²T + S(θ,φ) - σT⁴ = 0
# Wymiary: θ (szerokość), φ (długość), z (wysokość atmosfery)

import torch
import torch.nn as nn
import numpy as np

class PINNFormer3D(nn.Module):
    """
    Transformer-based PINN for 3D heat equation.
    Architektura: Embedding → N × TransformerEncoder → Linear → T(θ,φ,z)
    """
    def __init__(self, d_model=128, nhead=4, num_layers=4, dim_feedforward=256):
        super().__init__()
        self.input_proj = nn.Linear(3, d_model)  # (θ, φ, z) → d_model
        
        # Wavelet positional encoding
        self.pos_enc = WaveletPositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, 1)  # → T
    
    def forward(self, x):
        # x: (batch, 3) → θ, φ, z
        h = self.input_proj(x.unsqueeze(1))  # (batch, 1, d_model)
        h = self.pos_enc(h)
        h = self.transformer(h)
        T = self.output_proj(h.squeeze(1))
        return T


class WaveletPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_freq=10):
        super().__init__()
        freqs = torch.linspace(1, max_freq, d_model // 2)
        self.register_buffer('freqs', freqs)
    
    def forward(self, x):
        # Wavelet-based encoding (Morlet-like)
        pos = torch.arange(x.size(1), device=x.device).float().unsqueeze(-1)
        enc = torch.cat([
            torch.sin(pos * self.freqs),
            torch.cos(pos * self.freqs)
        ], dim=-1)
        return x + enc.unsqueeze(0)


# ═══ Loss z fizyką ═══
SIGMA = 5.670374419e-8
KAPPA = 0.025

def pinn_loss_3d(model, x_colloc, x_bc, T_bc, lambda_pde=1.0):
    """
    Physics-informed loss:
    L = L_data (boundary) + λ · L_PDE (interior)
    
    PDE: κ∇²T + S(θ,φ) - σT⁴ = 0
    """
    x_colloc.requires_grad_(True)
    T_pred = model(x_colloc)
    
    # Oblicz laplacjan ∇²T
    grad_T = torch.autograd.grad(T_pred.sum(), x_colloc, create_graph=True)[0]
    laplacian = 0
    for i in range(3):
        grad_Ti = torch.autograd.grad(grad_T[:, i].sum(), x_colloc, create_graph=True)[0]
        laplacian += grad_Ti[:, i]
    
    # Źródło ciepła S(θ,φ) — profil substelarny
    theta, phi, z = x_colloc[:, 0], x_colloc[:, 1], x_colloc[:, 2]
    S = 900.0 * torch.clamp(torch.cos(theta) * torch.cos(phi), min=0)
    
    # Residuum PDE
    residual = KAPPA * laplacian + S - SIGMA * T_pred.squeeze()**4
    L_pde = torch.mean(residual**2)
    
    # Loss na warunkach brzegowych
    T_bc_pred = model(x_bc)
    L_bc = torch.mean((T_bc_pred.squeeze() - T_bc)**2)
    
    return L_bc + lambda_pde * L_pde


# ═══ Estymacja zasobów ═══
# Model: d_model=128, nhead=4, layers=4 → ~2.1M parametrów
# Kolokacja: 8192 punktów 3D
# VRAM: ~4-6 GB (trening z gradientami drugiego rzędu)
# Czas na RX 7600 XT (ROCm): ~30-60 min (10k epok)
# Czas na CPU: ~8-15h (NIE REKOMENDOWANE na hackathon)
```

### 3.6 Porównanie Ścieżek PINN

| Ścieżka | Model | VRAM | Czas treningu | Złożoność implementacji | Wartość naukowa |
|---|---|---|---|---|---|
| **A: PINN-MLP 1D (DeepXDE)** | MLP [1,64,64,64,1] | ~200 MB | ~1 min | ✅ Niska (gotowa biblioteka) | Średnia |
| **B: PINN-MLP 2D (DeepXDE)** | MLP [2,128,128,128,1] | ~500 MB | ~5-10 min | ✅ Niska | Wysoka |
| **C: PINNFormer 2D** | Transformer 2D | ~2-4 GB | ~15-30 min | ⚠️ Średnia (custom) | Bardzo wysoka |
| **D: PINNFormer 3D** | Transformer 3D | ~6-10 GB | ~1-2h (GPU) | 🔴 Wysoka (custom) | **Maksymalna** |

**Rekomendacja hackathonowa:**
```
Dzień 1: Zaimplementuj ścieżkę A (1D DeepXDE) → 1h → działa
Dzień 2: Upgrade do B (2D DeepXDE) → 2h
Dzień 2-3: Jeśli czas pozwoli → C lub D (PINNFormer) → 4-8h

Z RX 7600 XT ścieżka C (PINNFormer 2D) jest realistyczna na hackathon!
Ścieżka D (3D) jest ambitna ale technicznie wykonalna.
```

---

## 4. Budżet VRAM — Nowy Plan (16 GB)

### 4.1 Scenariusze Równoległego Użycia

**Scenariusz 1: Maksymalny LLM + ELM (Runtime)**
```
TOTAL VRAM:                        16384 MB
├── System/driver overhead:          ~500 MB
├── Ollama (AstroLLaMA-8B Q5):    ~5700 MB
├── ELM (scikit-elm, CPU):              0 MB  ← CPU only
├── Plotly (rendering):              ~100 MB  ← przeglądarka
└── WOLNE:                         ~10084 MB  ← ogromny zapas!
```

**Scenariusz 2: Duży LLM + PINN jednocześnie**
```
TOTAL VRAM:                        16384 MB
├── System/driver overhead:          ~500 MB
├── Ollama (Qwen2.5-14B Q4):      ~9000 MB
├── PINNFormer 2D (trening):       ~3000 MB
├── ELM (CPU):                          0 MB
└── WOLNE:                          ~3884 MB  ← wystarczające
```

**Scenariusz 3: AstroLLaMA + PINNFormer 3D**
```
TOTAL VRAM:                        16384 MB
├── System/driver overhead:          ~500 MB
├── Ollama (AstroLLaMA-8B Q4):    ~5000 MB
├── PINNFormer 3D (trening):       ~8000 MB
├── ELM (CPU):                          0 MB
└── WOLNE:                          ~2884 MB  ← ciasno, ale wykonalne
```

**Scenariusz 4: Pełny trening offline (bez LLM)**
```
TOTAL VRAM:                        16384 MB
├── System/driver overhead:          ~500 MB
├── CTGAN (trening):               ~1500 MB
├── PINNFormer 3D (trening):       ~8000 MB
└── WOLNE:                          ~6384 MB  ← komfortowo
```

### 4.2 Porównanie ze Starą Kartą

| Scenariusz | RTX 3050 Ti (4 GB) | RX 7600 XT (16 GB) |
|---|---|---|
| AstroLLaMA-8B FP16 inference | 🔴 Niemożliwe | ✅ Mieści się |
| AstroLLaMA-8B Q4 + PINN 2D | 🔴 Niemożliwe | ✅ Jednocześnie! |
| Qwen2.5-14B inference | 🔴 Niemożliwe | ✅ Nowa opcja |
| PINNFormer 3D | 🔴 Niemożliwe | ✅ Wykonalne |
| CTGAN + LLM jednocześnie | 🔴 Crash | ✅ OK |
| Dual-model (2× LLM) | 🔴 Niemożliwe | ✅ Wykonalne |

---

## 5. Rozwiązania Chmurowe — Analiza Alternatyw

### 5.1 Google Colab

| Aspekt | Colab Free | Colab Pro ($12/mies.) | Colab Pro+ ($50/mies.) |
|---|---|---|---|
| **GPU** | T4 (16 GB) | T4/A100 (16-40 GB) | A100 (40-80 GB) |
| **CUDA** | ✅ Pełne | ✅ Pełne | ✅ Pełne |
| **Czas sesji** | ~4h (nieciągłe) | ~12h | ~24h |
| **RAM** | 12 GB | 25 GB | 52 GB |
| **Dysk** | ~78 GB | ~225 GB | ~225 GB |
| **Przydatność** | ⚠️ Odpada po ~4h | ✅ Dobra | ✅ Idealna |

**Colab jako uzupełnienie RX 7600 XT:**

| Zadanie | Gdzie wykonać | Powód |
|---|---|---|
| **Trening CTGAN** | Colab (CUDA) lub lokalnie (CPU) | CUDA gwarantuje szybkość |
| **Trening PINNFormer 3D** | Colab (CUDA) ✅ | Omija problem ROCm na Windows |
| **LLM inference** | Lokalnie (Ollama + AMD) ✅ | Vulkan działa natywnie |
| **ELM trening** | Lokalnie (CPU) | scikit-elm = CPU only |
| **Streamlit app** | Lokalnie ✅ | Zawsze dostępne |
| **Fine-tuning LLM** | Colab Pro (A100) | 16 GB to za mało na full fine-tune |

### 5.2 Inne Platformy Chmurowe

| Platforma | GPU | Koszt | PyTorch CUDA | Uwagi |
|---|---|---|---|---|
| **Google Colab Free** | T4 16 GB | Darmowe | ✅ | Limit czasu ~4h |
| **Google Colab Pro** | T4/V100/A100 | $12/mies. | ✅ | **Najlepszy stosunek cena/wartość** |
| **Kaggle Notebooks** | T4×2 (30h/tyg) | Darmowe | ✅ | Dobra alternatywa |
| **Lightning AI** | T4/A10G | Darmowe (22h/mies.) | ✅ | Dobre free tier |
| **Paperspace Gradient** | M4000/RTX4000 | Od $8/mies. | ✅ | Persistent VM |
| **Groq Cloud (API)** | LPU | Darmowe (14k req/d) | N/A | **Tylko LLM inference** |
| **Together.ai (API)** | — | Darmowe (limitowane) | N/A | Tylko LLM inference |

### 5.3 Strategia Hybrydowa (Rekomendowana)

```
LOKALNIE (RX 7600 XT):
├── Ollama + AstroLLaMA/Qwen2.5 → inference LLM (Vulkan)
├── scikit-elm → trening i predykcja ELM (CPU)
├── Streamlit → UI/frontend
├── Plotly → wizualizacja 3D
└── Pydantic → walidacja

CHMURA (Colab/Kaggle):
├── CTGAN trening (PyTorch CUDA) → eksport .pkl → skopiuj lokalnie
├── PINNFormer trening (PyTorch CUDA) → eksport wag → skopiuj lokalnie
└── Opcjonalnie: fine-tuning LLM z LoRA (jeśli A100 dostępne)
```

**Workflow:**
1. Trenuj CTGAN i PINNFormer w Colab (CUDA) — eksportuj modele
2. Pobierz .pkl / .pt na dysk lokalny
3. Uruchom Streamlit + Ollama lokalnie (RX 7600 XT)
4. Załaduj wytrenowane modele do inference (CPU/GPU)

---

## 6. Porównanie: Lokalne vs Chmurowe vs Hybrydowe

| Kryterium | 100% Lokalne (RX 7600 XT) | 100% Chmurowe (Colab) | Hybrydowe (Rekomendacja) |
|---|---|---|---|
| **LLM inference** | ✅ 25-35 tok/s (Vulkan) | ⚠️ Wymaga GPU runtime | ✅ Lokalnie (szybko) |
| **CTGAN trening** | ⚠️ CPU-only (wolne) | ✅ GPU (szybko) | ✅ Chmura → eksport |
| **PINNFormer 3D** | ⚠️ Wymaga WSL2+ROCm | ✅ GPU CUDA (pewne) | ✅ Chmura → eksport |
| **ELM** | ✅ CPU (szybkie) | ✅ Też CPU | ✅ Lokalnie |
| **Streamlit UI** | ✅ Localhost | ⚠️ Ngrok/tunnel | ✅ Lokalnie |
| **Offline** | ✅ Brak internetu OK | 🔴 Wymaga internet | ⚠️ Po treningu — offline |
| **Konfiguracja** | ⚠️ ROCm setup | ✅ Gotowe | ✅ Niewiele setup |
| **Stabilność** | ✅ Nie zależy od sesji | ⚠️ Sesja może wygasnąć | ✅ Stabilne runtime |

---

## 7. Konkretne Kroki Konfiguracji

### 7.1 Ollama na AMD RX 7600 XT (Windows — natywne)

```powershell
# Ollama automatycznie wykrywa AMD GPU i używa Vulkan backend
# 1. Pobierz i zainstaluj Ollama z https://ollama.com/download
# 2. Sprawdź wykrywanie GPU:

ollama --version

# 3. Pobierz model (AstroLLaMA lub Qwen2.5)
ollama pull qwen2.5:14b       # 14B Q4 — ~9 GB VRAM
# LUB
ollama pull qwen2.5:7b        # 7B Q4 — ~4.5 GB VRAM

# 4. Dla AstroLLaMA — przygotuj custom Modelfile:
```

**Custom Modelfile dla AstroLLaMA-like zachowania:**
```dockerfile
# Modelfile.astro
FROM qwen2.5:14b

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 8192

SYSTEM """You are AstroAgent, a specialized astrophysics AI assistant for exoplanetary 
climate simulation. You operate as an autonomous agent with access to tools for querying 
the NASA Exoplanet Archive, computing equilibrium temperatures, running ELM climate 
surrogates, and generating 3D visualizations.

Your domain expertise includes:
- Exoplanetary habitability assessment (ESI, SEPHI indices)
- Tidal locking and climate topology (Eyeball, Lobster, Greenhouse states)
- Stellar flux calculations and habitable zone boundaries (Kopparapu 2013)
- Physics-Informed Neural Networks for heat equation solving
- Radiative equilibrium (Stefan-Boltzmann law)

Always reason step-by-step. When analyzing a planet:
1. Retrieve observational data from NASA archive
2. Compute fundamental parameters (T_eq, flux, ESI)
3. Run ELM climate surrogate for temperature distribution
4. Validate results against physical constraints
5. Generate 3D visualization and interpretation

Use precise scientific terminology. Cite relevant equations. Express temperatures in 
Kelvin. Flag uncertainties and model limitations."""
```

```powershell
# Tworzenie modelu z Modelfile
ollama create astro-agent -f Modelfile.astro

# Test
ollama run astro-agent "Analyze the habitability of TRAPPIST-1e"
```

### 7.2 WSL2 + ROCm (opcjonalne — dla PyTorch GPU)

```powershell
# W PowerShell (jako administrator):
wsl --install -d Ubuntu-22.04

# Po restarcie, w WSL2:
```

```bash
# W terminalu Ubuntu WSL2:

# 1. Instalacja ROCm 6.2
wget https://repo.radeon.com/amdgpu-install/6.2/ubuntu/jammy/amdgpu-install_6.2.60200-1_all.deb
sudo apt install ./amdgpu-install_6.2.60200-1_all.deb
sudo amdgpu-install --usecase=rocm

# 2. Weryfikacja
rocminfo | grep "Name:"

# 3. PyTorch z ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# 4. Test
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# 5. Instalacja DeepXDE i zależności  
pip install deepxde scikit-elm ctgan
```

### 7.3 Google Colab Setup (dla treningu CTGAN + PINNFormer)

```python
# W Colab (cell 1):
!pip install ctgan deepxde scikit-elm torch

# Cell 2 — sprawdzenie GPU:
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# Cell 3 — trening modeli...
# Cell N — eksport:
import pickle
# CTGAN
with open("/content/drive/MyDrive/hack4sages/ctgan_model.pkl", "wb") as f:
    pickle.dump(ctgan_model, f)
# PINNFormer
torch.save(pinn_model.state_dict(), "/content/drive/MyDrive/hack4sages/pinn3d_weights.pt")
```

---

## 8. Zaktualizowana Tabela Decyzji

| Komponent | Stara decyzja (RTX 3050 Ti) | **Nowa decyzja (RX 7600 XT)** | Powód zmiany |
|---|---|---|---|
| **LLM** | Qwen2.5-3B (Q4) — jedyne co się mieściło | **Qwen2.5-14B (Q4)** lub **AstroLLaMA-8B (Q5)** | 16 GB → można użyć 4× większe modele |
| **AstroLLaMA** | ❌ Porzucone | **✅ WYKONALNE** (Q5/Q8 inference) | 16 GB VRAM wystarczy |
| **PINNFormer** | ⚠️ Opcjonalny (tylko 1D MLP) | **✅ 2D/3D wykonalne** | 16 GB pozwala na duże siatki |
| **CTGAN trening** | GPU (jednorazowy, 500 MB) | **Colab (CUDA)** lub CPU lokalne | Brak natywnego PyTorch GPU na AMD Windows |
| **ELM** | CPU | CPU (bez zmian) | scikit-elm = NumPy |
| **Dual-model** | ❌ Niemożliwe | **✅ Dwa modele jednocześnie** | Np. AstroLLaMA 5 GB + Qwen 4.5 GB = 9.5 GB |
| **Fine-tuning** | ❌ Niemożliwe | **⚠️ QLoRA na 8B modelu** (eksperymentalne) | ~12-14 GB VRAM z QLoRA |
| **Strategia trening** | Sekwencyjne ładowanie, unikanie crashy | **Równoległe** — LLM + PINN jednocześnie | Dużo VRAM zapasu |

---

## 9. Sekwencja Operacji na Hackathonie (72h Plan)

### Faza 0: Przygotowanie (przed hackathon, 2-3h)

```
□ Zainstaluj Ollama na Windows
□ ollama pull qwen2.5:14b (lub qwen2.5:7b jako backup)
□ Przygotuj Modelfile.astro → ollama create astro-agent
□ (Opcja A) Skonfiguruj WSL2 + ROCm (dla PyTorch GPU)
□ (Opcja B) Przygotuj notebook Colab z CTGAN + PINNFormer
□ pip install streamlit pandas numpy plotly requests pydantic scikit-learn scikit-elm langchain langchain-ollama ollama scipy
□ Przetestuj: ollama run astro-agent "test"
```

### Faza 1: Trening Offline (6-8h)

```
Szyny równoległe:

Ścieżka CHMURA (Colab):          Ścieżka LOKALNA:
├── Trening CTGAN (CUDA)          ├── Implementacja astro_physics.py
├── Trening PINNFormer 2D/3D      ├── Implementacja validators.py  
├── Eksport .pkl / .pt            ├── Implementacja nasa_client.py
└── Download do projektu           ├── Budowa Streamlit UI
                                   └── Testowanie Ollama + agent
```

### Faza 2: Integracja (12-16h)

```
Lokalnie (RX 7600 XT, 16 GB VRAM):
├── Ollama (AstroLLaMA/Qwen2.5-14B) → ~9 GB VRAM → działa
├── Załaduj wytrenowane modele (CTGAN .pkl, PINN .pt)
├── Orkiestracja LangChain / smolagents
├── Implementacja function calling
├── Wizualizacja 3D (Plotly)
└── Graceful degradation
```

### Faza 3: Polish i Demo (4-6h)

```
├── RAG (ChromaDB + embeddingi)
├── Porównanie planet (dual-globe)
├── PDF export
├── Testy end-to-end
├── Przygotowanie prezentacji
└── Backup plan (demo offline)
```

---

## 10. Podsumowanie Końcowe

### Co zmienia RX 7600 XT 16 GB w stosunku do RTX 3050 Ti 4 GB?

| Aspekt | Wpływ | Ocena |
|---|---|---|
| **AstroLLaMA dostępne** | Można użyć specjalistycznego modelu astrofizycznego | 🟢 **Ogromny** |
| **Większe modele LLM** | Qwen2.5-14B zamiast 3B → drastycznie lepszy reasoning | 🟢 **Ogromny** |
| **PINNFormer 3D** | Z niemożliwego → wykonalny | 🟢 **Ogromny** |
| **Równoległe ładowanie** | LLM + PINN jednocześnie w VRAM | 🟢 **Duży** |
| **CTGAN trening** | Szybszy (jeśli ROCm/WSL2) lub chmura | 🟡 **Średni** |
| **Ekosystem AMD** | PyTorch wymaga obejść (WSL2/DirectML) | 🔴 **Problem** |
| **Ollama na AMD** | Działa natywnie (Vulkan) | 🟢 **Pozytywne** |

### Główne ryzyka i mitygacje

| Ryzyko | Prawdopodobieństwo | Mitygacja |
|---|---|---|
| PyTorch nie działa na AMD Windows | Wysokie | WSL2 + ROCm lub Colab |
| ROCm setup zbyt czasochłonne | Średnie | Fallback na Colab + CPU lokalne |
| AstroLLaMA Modelfile nie działa | Niskie | Qwen2.5-14B z astro system prompt |
| Ollama nie wykrywa AMD GPU | Niskie | Sprawdzić przed hackathon |
| Colab sesja wygasa w trakcie treningu | Średnie | Kaggle jako backup, Colab Pro |

### Ostateczna rekomendacja

**RX 7600 XT 16 GB VRAM to doskonała karta do tego projektu**, z jednym zastrzeżeniem: **ekosystem AMD na Windows wymaga dodatkowej pracy** (WSL2 lub chmura dla PyTorch GPU).

**Optymalna strategia:**
1. **LLM** → Ollama + Qwen2.5-14B (Q4) z astro system prompt → **lokalnie, Vulkan** → ~14-18 tok/s
2. **CTGAN** → Trening w **Google Colab** (CUDA) → eksport .pkl
3. **PINNFormer 2D/3D** → Trening w **Google Colab** (CUDA) → eksport .pt
4. **ELM** → Trening **lokalny CPU** (scikit-elm)
5. **Runtime** → Streamlit + Ollama + załadowane modele → **100% lokalne, offline**
6. **Backup LLM** → Groq API (darmowe, ~100 tok/s) jeśli Ollama się zacina

**Z tą kartą projekt jest w pełni realizowalny na hackathonie, z możliwością implementacji ambitnych rozszerzeń (PINNFormer 3D, AstroLLaMA, dual-model) które były niemożliwe na 4 GB VRAM.**
