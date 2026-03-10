# Analiza Dodatkowych Modeli AI — Cyfrowy Bliźniak Egzoplanetarny

## Kontekst

Obecny stos AI projektu składa się z czterech modeli:

| Model | Rola w systemie | Typ architektury |
|---|---|---|
| **LLM** (Qwen2.5 / AstroLLaMA) | Orkiestracja agentowa, synteza tekstowa | Generatywny językowy (Transformer) |
| **CTGAN** | Augmentacja danych tabelarycznych z NASA | Generatywny adversarialny (GAN) |
| **ELM Ensemble** | Surogat klimatyczny — predykcja mapy temperatury | Bezgradientowa regresja (pseudoodwrotność) |
| **PINNFormer / PINN-MLP** | Rozwiązywanie równania propagacji ciepła (PDE) | Sieci informowane fizyką (PINN) |

Poniżej przedstawiono **12 dodatkowych modeli AI**, które mogą zostać dołączone do architektury. Każdy model jest opisany pod kątem: teorii, roli w pipeline, wymagań sprzętowych, implementacji, ograniczeń i referencji naukowych.

---

## Mapa Modeli w Pipeline

```
DANE NASA ──→ [5] Isolation Forest (anomalie)
         │
         ├──→ [CTGAN] (augmentacja) ←──→ [10] Normalizing Flow (alternatywa)
         │            │
         │            ▼
         ├──→ [6] UMAP + HDBSCAN (klasteryzacja / wizualizacja)
         │
         ▼
   PARAMETRY PLANETY
         │
         ├──→ [1] XGBoost (klasyfikacja topologii klimatu)
         │
         ├──→ [ELM Ensemble] (mapa temperatury)
         │           ├──→ [12] Conformal Prediction (formalna uncertainty)
         │           └──→ [11] Autoencoder (kompresja / podobieństwa)
         │
         ├──→ [2] Gaussian Process Regression (surogat + natywna niepewność)
         │
         ├──→ [PINNFormer / PINN-MLP] (rozwiązanie PDE ciepła)
         ├──→ [8] KAN-PINN (alternatywna architektura PINN)
         ├──→ [4] FNO (operator Fouriera — uczenie operatora PDE)
         ├──→ [9] Neural ODE (ewolucja czasowa klimatu)
         │
         ▼
   WYNIKI ──→ [3] VAE (latent space planet)
         ──→ [7] Sentence Transformers + ChromaDB (RAG)
         ──→ [LLM Agent] (orkiestracja + synteza)
```

---

## 1. XGBoost / LightGBM — Klasyfikator Topologii Klimatu

### 1.1 Teoria

XGBoost (eXtreme Gradient Boosting, Chen & Guestrin, 2016) to algorytm uczenia zespołowego oparty na sekwencyjnym budowaniu drzew decyzyjnych, w którym każde kolejne drzewo koryguje błędy poprzedników. Funkcja celu:

$$\mathcal{L}(\phi) = \sum_{i} l(\hat{y}_i, y_i) + \sum_{k} \Omega(f_k)$$

gdzie $l$ to różniczkowalna funkcja kosztu, a $\Omega(f_k) = \gamma T + \frac{1}{2}\lambda \|\mathbf{w}\|^2$ to regularyzacja (liczba liści $T$, wagi liści $\mathbf{w}$).

LightGBM (Ke et al., 2017) to wariant z Gradient-based One-Side Sampling (GOSS) i Exclusive Feature Bundling (EFB), przyspieszający trening na dużych zbiorach.

### 1.2 Rola w Projekcie

Klasyfikacja typu klimatu planety na podstawie parametrów wejściowych — **zanim** ELM wygeneruje pełną mapę temperatury. Cztery klasy:

| Klasa | Nazwa | Opis fizyczny |
|---|---|---|
| 0 | 🔵 Gałka Oczna (Eyeball) | Okrągły ocean w punkcie substelarnym otoczony globalnym lodem |
| 1 | 🦞 Homar (Lobster) | Równikowy pas oceanu rozciągający się symetrycznie |
| 2 | 🔴 Cieplarnia (Greenhouse) | Globalny ocean, brak lodu, silny efekt cieplarniany |
| 3 | ❄️ Śnieżka (Snowball) | Globalna pokrywa lodowa, brak ciekłej wody |

### 1.3 Dane Treningowe

Etykiety generowane heurystycznie z map temperatur ELM/analitycznych:

```python
def classify_topology(temp_map: np.ndarray) -> int:
    """
    Heurystyczna klasyfikacja topologii klimatu z mapy temperatur.
    
    Logika:
    - Snowball: >90% pikseli < 273K
    - Greenhouse: >90% pikseli > 273K
    - Eyeball: ciepły region okrągły, skupiony w punkcie substelarnym
    - Lobster: ciepły region wydłużony wzdłuż równika
    """
    habitable_mask = (temp_map >= 273) & (temp_map <= 373)
    habitable_frac = habitable_mask.sum() / temp_map.size
    cold_frac = (temp_map < 273).sum() / temp_map.size
    hot_frac = (temp_map > 373).sum() / temp_map.size
    
    if cold_frac > 0.90:
        return 3  # Snowball
    if hot_frac > 0.50:
        return 2  # Greenhouse
    
    # Analiza kształtu regionu habitabilnego
    n_lat, n_lon = temp_map.shape
    equator_row = n_lat // 2
    equator_strip = habitable_mask[equator_row - 2 : equator_row + 2, :]
    equator_frac = equator_strip.sum() / equator_strip.size
    
    # Stosunek szerokości do wysokości regionu ciepłego
    lat_extent = habitable_mask.any(axis=1).sum() / n_lat
    lon_extent = habitable_mask.any(axis=0).sum() / n_lon
    
    aspect_ratio = lon_extent / (lat_extent + 1e-8)
    
    if aspect_ratio > 2.0 and equator_frac > 0.5:
        return 1  # Lobster (wydłużony wzdłuż równika)
    else:
        return 0  # Eyeball (skupiony, okrągły)
```

### 1.4 Implementacja

```python
# climate_classifier.py
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import pickle


class ClimateTopologyClassifier:
    """
    Klasyfikator topologii klimatu egzoplanetarnego.
    Wejście: parametry planetarne → Wyjście: typ klimatu (0-3).
    """
    
    CLIMATE_NAMES = {
        0: "🔵 Gałka Oczna (Eyeball)",
        1: "🦞 Homar (Lobster)",
        2: "🔴 Cieplarnia (Greenhouse)",
        3: "❄️ Śnieżka (Snowball)"
    }
    
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=4,
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42
        )
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Trening klasyfikatora.
        
        Args:
            X: Macierz cech [N, 8] — parametry planetarne
               (radius_earth, mass_earth, semi_major_axis_au, star_teff_K,
                star_radius_solar, insol_earth, albedo, tidally_locked)
            y: Etykiety [N] — 0/1/2/3
        """
        # Walidacja krzyżowa
        scores = cross_val_score(self.model, X, y, cv=5, scoring="accuracy")
        print(f"CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        
        self.model.fit(X, y)
        print("Model XGBoost wytrenowany.")
    
    def predict(self, X: np.ndarray) -> dict:
        """
        Predykcja z prawdopodobieństwami klas.
        
        Returns:
            dict z kluczami: 'label', 'name', 'probabilities'
        """
        proba = self.model.predict_proba(X)[0]
        label = int(np.argmax(proba))
        
        return {
            "label": label,
            "name": self.CLIMATE_NAMES[label],
            "probabilities": {
                self.CLIMATE_NAMES[i]: round(float(p), 4)
                for i, p in enumerate(proba)
            },
            "confidence": round(float(proba[label]), 4)
        }
    
    def feature_importance(self) -> dict:
        """Zwraca ważność cech — przydatne do wizualizacji."""
        names = [
            "radius_earth", "mass_earth", "semi_major_axis_au",
            "star_teff_K", "star_radius_solar", "insol_earth",
            "albedo", "tidally_locked"
        ]
        importances = self.model.feature_importances_
        return dict(sorted(
            zip(names, importances), key=lambda x: x[1], reverse=True
        ))
    
    def save(self, path: str = "models/climate_classifier.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
    
    def load(self, path: str = "models/climate_classifier.pkl"):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
```

### 1.5 Integracja z Agentem

```python
@tool
def classify_climate_type(
    radius_earth: float,
    mass_earth: float,
    semi_major_axis_au: float,
    star_teff_K: float,
    star_radius_solar: float,
    insol_earth: float,
    albedo: float = 0.3,
    tidally_locked: int = 1
) -> str:
    """
    Klasyfikuje typ klimatu egzoplanety (Eyeball/Lobster/Greenhouse/Snowball).
    Użyj po obliczeniu parametrów planetarnych, przed symulacją ELM.
    """
    clf = load_classifier()
    X = np.array([[radius_earth, mass_earth, semi_major_axis_au,
                    star_teff_K, star_radius_solar, insol_earth,
                    albedo, tidally_locked]])
    result = clf.predict(X)
    return json.dumps(result, indent=2, ensure_ascii=False)
```

### 1.6 Wymagania Sprzętowe

| Zasób | Wartość |
|---|---|
| **VRAM** | 0 MB (CPU-only) |
| **RAM** | ~50 MB |
| **Trening** | < 10 sek na 5000 próbek |
| **Inference** | < 1 ms |
| **Biblioteka** | `xgboost` lub `lightgbm` + `scikit-learn` |

### 1.7 Ograniczenia

- Etykiety heurystyczne mogą być niedokładne — zależą od jakości reguły `classify_topology()`
- Klasy Eyeball/Lobster mogą być trudne do rozdzielenia bez precyzyjnych danych GCM
- Model nie generalizuje na typy klimatu nieobjęte treningiem (np. "super-rotation state")

### 1.8 Referencje

- Chen, T., Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD 2016.
- Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017.

### 1.9 Szacowany Effort: ~2h | Impact: ⭐⭐⭐⭐

---

## 2. Gaussian Process Regression (GPR) — Surogat z Natywną Niepewnością

### 2.1 Teoria

Gaussian Process (Rasmussen & Williams, 2006) definiuje rozkład prawdopodobieństwa nad funkcjami:

$$f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$$

gdzie $m(\mathbf{x})$ to funkcja średniej, a $k(\mathbf{x}, \mathbf{x}')$ to funkcja kowariancji (kernel).

Dla danych treningowych $(\mathbf{X}, \mathbf{y})$ i nowego punktu $\mathbf{x}_*$, predykcja posterior to:

$$\mu(\mathbf{x}_*) = \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{y}$$

$$\sigma^2(\mathbf{x}_*) = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_*$$

gdzie $\mathbf{K}$ to macierz Grama (kowariancji danych treningowych), $\mathbf{k}_*$ to wektor kowariancji między punktem testowym a treningowymi.

**Kluczowa przewaga:** Model natywnie zwraca zarówno predykcję $\mu$ (wartość), jak i niepewność $\sigma$ (odchylenie standardowe). Nie wymaga ensemble'u.

### 2.2 Rola w Projekcie

Uzupełniający surogat klimatyczny obok ELM z wbudowaną **kwantyfikacją niepewności**:

1. **Predykcja + confidence interval:** Dla każdego piksela mapy temperatury — T ± σ
2. **Second opinion:** Jeśli ELM i GPR się zgadzają → wysoka pewność; jeśli się różnią → flagowanie
3. **Aktywne uczenie:** GPR wskazuje, w jakich regionach przestrzeni parametrów brakuje danych (wysoka σ) — wskazówka, jakie planety syntetyzować za CTGAN

### 2.3 Implementacja

```python
# gpr_surrogate.py
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
from typing import Tuple


class GPRClimateSurrogate:
    """
    Gaussian Process Regression jako surogat klimatyczny z natywną niepewnością.
    
    OGRANICZENIE: GPR skaluje się O(N³) — nie nadaje się do dużych zbiorów.
    Rozwiązanie: PCA na wyjściach (mapy temperatur) redukuje wymiarowość.
    
    Pipeline:
        Parametry [8D] → GPR → Latent [16D] → PCA inverse → Mapa [32×64]
    """
    
    N_LAT = 32
    N_LON = 64
    N_COMPONENTS = 16  # Wymiarowość latent space (PCA)
    
    def __init__(self, kernel=None, n_restarts: int = 5, alpha: float = 1e-6):
        """
        Args:
            kernel: Kernel GP (domyślnie: Matern(5/2) + WhiteKernel)
            n_restarts: Liczba restartów optymalizacji hiperparametrów
            alpha: Szum obserwacyjny (regularyzacja diagonali)
        """
        if kernel is None:
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3)) *
                Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) +
                WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e1))
            )
        
        self.scaler_X = StandardScaler()
        self.pca = PCA(n_components=self.N_COMPONENTS)
        self.scaler_y = StandardScaler()
        
        # Osobny GPR per komponent PCA (multi-output przez N osobnych GPR)
        self.gpr_models = [
            GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=n_restarts,
                alpha=alpha,
                normalize_y=True
            )
            for _ in range(self.N_COMPONENTS)
        ]
    
    def train(self, X: np.ndarray, y_maps: np.ndarray):
        """
        Trening GPR na zredukowanych mapach temperatur.
        
        Args:
            X: Macierz cech [N, 8]
            y_maps: Macierz map temperatur [N, N_LAT * N_LON]
        """
        # Normalizacja wejść
        X_scaled = self.scaler_X.fit_transform(X)
        
        # PCA na mapach → redukcja 2048D → 16D
        y_latent = self.pca.fit_transform(y_maps)
        y_latent_scaled = self.scaler_y.fit_transform(y_latent)
        
        explained = self.pca.explained_variance_ratio_.sum()
        print(f"PCA: {self.N_COMPONENTS} komponentów wyjaśnia {explained:.1%} wariancji")
        
        # Trening osobnego GPR per komponent
        for i, gpr in enumerate(self.gpr_models):
            gpr.fit(X_scaled, y_latent_scaled[:, i])
            print(f"  GPR {i+1}/{self.N_COMPONENTS} — log-likelihood: "
                  f"{gpr.log_marginal_likelihood_value_:.3f}")
        
        print("GPR Surrogate wytrenowany.")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predykcja z niepewnością.
        
        Returns:
            mean_map: Średnia mapa temperatur [N_LAT, N_LON] w K
            std_map:  Mapa odchylenia standardowego [N_LAT, N_LON] w K
        """
        X_scaled = self.scaler_X.transform(X)
        
        latent_means = []
        latent_stds = []
        
        for gpr in self.gpr_models:
            mu, sigma = gpr.predict(X_scaled, return_std=True)
            latent_means.append(mu)
            latent_stds.append(sigma)
        
        # Odwrócenie normalizacji i PCA
        latent_mean = np.array(latent_means).T  # [1, N_COMPONENTS]
        latent_mean = self.scaler_y.inverse_transform(latent_mean)
        mean_map = self.pca.inverse_transform(latent_mean)
        mean_map = mean_map.reshape(self.N_LAT, self.N_LON)
        
        # Propagacja niepewności przez PCA (przybliżona)
        latent_std = np.array(latent_stds).T
        latent_std_unscaled = latent_std * self.scaler_y.scale_
        # Niepewność w przestrzeni oryginalnej ≈ ||V · σ_latent||
        std_in_original = np.sqrt(
            (self.pca.components_.T ** 2 @ (latent_std_unscaled.T ** 2)).T
        )
        std_map = std_in_original.reshape(self.N_LAT, self.N_LON)
        
        return mean_map, std_map
    
    def predict_from_params(self, params: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Predykcja bezpośrednio z parametrów planetarnych."""
        X = np.array([[
            params["radius_earth"],
            params["mass_earth"],
            params["semi_major_axis_au"],
            params["star_teff_K"],
            params["star_radius_solar"],
            params["insol_earth"],
            params.get("albedo", 0.3),
            float(params.get("tidally_locked", 1))
        ]])
        return self.predict(X)
    
    def agreement_with_elm(
        self,
        elm_map: np.ndarray,
        gpr_mean: np.ndarray,
        gpr_std: np.ndarray,
        threshold_sigma: float = 2.0
    ) -> dict:
        """
        Porównanie predykcji ELM i GPR.
        Flaguje piksele, gdzie ELM wykracza poza 2σ GPR.
        """
        diff = np.abs(elm_map - gpr_mean)
        disagreement_mask = diff > threshold_sigma * gpr_std
        disagreement_frac = disagreement_mask.sum() / disagreement_mask.size
        
        return {
            "agreement_fraction": round(1.0 - float(disagreement_frac), 4),
            "mean_abs_diff_K": round(float(diff.mean()), 2),
            "max_disagreement_K": round(float(diff.max()), 2),
            "n_flagged_pixels": int(disagreement_mask.sum()),
            "flagged_fraction": round(float(disagreement_frac), 4)
        }
    
    def save(self, path: str = "models/gpr_surrogate.pkl"):
        bundle = {
            "gpr_models": self.gpr_models,
            "scaler_X": self.scaler_X,
            "scaler_y": self.scaler_y,
            "pca": self.pca
        }
        with open(path, "wb") as f:
            pickle.dump(bundle, f)
    
    def load(self, path: str = "models/gpr_surrogate.pkl"):
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        self.gpr_models = bundle["gpr_models"]
        self.scaler_X = bundle["scaler_X"]
        self.scaler_y = bundle["scaler_y"]
        self.pca = bundle["pca"]
```

### 2.4 Wizualizacja Niepewności

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def visualize_gpr_with_uncertainty(mean_map, std_map, planet_name):
    """Dwa globy obok siebie: predykcja + mapa niepewności."""
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "surface"}, {"type": "surface"}]],
        subplot_titles=[f"{planet_name} — Temperatura [K]",
                        f"{planet_name} — Niepewność [K]"]
    )
    # ... renderowanie dwóch sfer z surfacecolor ...
    return fig
```

### 2.5 Wymagania Sprzętowe

| Zasób | Wartość |
|---|---|
| **VRAM** | 0 MB (CPU-only, scikit-learn) |
| **RAM** | ~200–500 MB (zależne od N) |
| **Trening** | ~1–5 min (N ≤ 2000), O(N³) |
| **Inference** | ~10–50 ms per predykcja |
| **Biblioteka** | `scikit-learn` (GaussianProcessRegressor) |

### 2.6 Ograniczenia

- **Skalowanie $O(N^3)$:** Trening na > 2000 próbkach staje się wolny. Rozwiązania: Sparse GPR, Random Fourier Features, lub ograniczenie do małego podzbioru.
- **Multi-output:** GPR jest naturalnie single-output. Obsługujemy multi-output przez osobne GPR per komponent PCA — nie uwzględnia korelacji między komponentami.
- **Propagacja niepewności przez PCA** jest przybliżona (zakłada niezależność komponentów).

### 2.7 Referencje

- Rasmussen, C. E., Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning.* MIT Press.
- Quinonero-Candela, J., Rasmussen, C. E. (2005). *A Unifying View of Sparse Approximate Gaussian Process Regression.* JMLR 6.

### 2.8 Szacowany Effort: ~3–4h | Impact: ⭐⭐⭐⭐

---

## 3. Variational Autoencoder (VAE) — Eksploracja Przestrzeni Latentnej Planet

### 3.1 Teoria

Variational Autoencoder (Kingma & Welling, 2014) to generatywny model uczący się latentnej reprezentacji danych. Składa się z:

**Encoder** $q_\phi(\mathbf{z}|\mathbf{x})$: Mapuje dane wejściowe na parametry rozkładu w przestrzeni latentnej:

$$q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}_\phi(\mathbf{x}), \mathrm{diag}(\boldsymbol{\sigma}_\phi^2(\mathbf{x})))$$

**Decoder** $p_\theta(\mathbf{x}|\mathbf{z})$: Rekonstruuje dane z latentnej reprezentacji.

**Funkcja kosztu (ELBO):**

$$\mathcal{L}(\theta, \phi; \mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{\mathrm{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

gdzie pierwszy człon to rekonstrukcja, a drugi to regularyzacja KL-divergence wymuszająca strukturę przestrzeni latentnej.

### 3.2 Rola w Projekcie

1. **Mapa przestrzeni planetarnej:** Kompresja N-wymiarowych parametrów planetarnych do 2D/3D latent space. Wizualizacja: scatter plot z kolorami wg ESI/typu klimatu.
2. **Interpolacja między planetami:** "Pokaż planety pośrednie między TRAPPIST-1e a Proxima Cen b" przez liniowe chodzenie po latent space.
3. **Generowanie nowych planet:** Próbkowanie z latent space generuje fizycznie spójne konfiguracje — uzupełnienie CTGAN.
4. **Detekcja anomalii:** Planety z niskim ELBO (słaba rekonstrukcja) to outlierzy.

### 3.3 Implementacja

```python
# vae_planets.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class PlanetaryVAE(nn.Module):
    """
    Variational Autoencoder dla konfiguracji planetarnych.
    
    Wejście: wektor parametrów [8D] (radius, mass, semi_major, star_teff, 
             star_radius, insol, albedo, tidal_lock)
    Latent:  z ∈ R^latent_dim (domyślnie 2D lub 3D dla wizualizacji)
    Wyjście: rekonstrukcja parametrów [8D]
    """
    
    def __init__(self, input_dim: int = 8, latent_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + σ ⊙ ε, ε ~ N(0,I)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    @staticmethod
    def loss_function(recon_x, x, mu, logvar, beta: float = 1.0):
        """
        β-VAE loss = Reconstruction + β · KL divergence.
        β > 1 → bardziej rozplecione (disentangled) latent space.
        """
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl_loss


def interpolate_planets(vae, planet_a_params, planet_b_params, n_steps=10):
    """
    Interpolacja w przestrzeni latentnej między dwiema planetami.
    Zwraca n_steps pośrednich konfiguracji planetarnych.
    """
    vae.eval()
    with torch.no_grad():
        x_a = torch.FloatTensor(planet_a_params).unsqueeze(0)
        x_b = torch.FloatTensor(planet_b_params).unsqueeze(0)
        
        mu_a, _ = vae.encode(x_a)
        mu_b, _ = vae.encode(x_b)
        
        interpolated = []
        for alpha in np.linspace(0, 1, n_steps):
            z = (1 - alpha) * mu_a + alpha * mu_b
            decoded = vae.decode(z)
            interpolated.append(decoded.squeeze().numpy())
        
        return np.array(interpolated)


def train_vae(X_train: np.ndarray, latent_dim: int = 2, epochs: int = 200,
              lr: float = 1e-3, beta: float = 1.0):
    """
    Trening VAE na danych planetarnych.
    
    Args:
        X_train: Znormalizowane parametry planet [N, 8]
        latent_dim: Wymiarowość latent space (2 lub 3 dla wizualizacji)
        epochs: Liczba epok
        lr: Learning rate
        beta: Współczynnik β-VAE (regularyzacja)
    """
    dataset = torch.FloatTensor(X_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    
    model = PlanetaryVAE(input_dim=X_train.shape[1], latent_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            recon, mu, logvar = model(batch)
            loss = PlanetaryVAE.loss_function(recon, batch, mu, logvar, beta)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} — Loss: {total_loss/len(dataset):.4f}")
    
    return model
```

### 3.4 Wizualizacja Latent Space

```python
import plotly.express as px

def plot_latent_space(vae, X_data, labels, label_name="ESI"):
    """Scatter plot latent space z kolorami wg ESI lub typu klimatu."""
    vae.eval()
    with torch.no_grad():
        mu, _ = vae.encode(torch.FloatTensor(X_data))
        z = mu.numpy()
    
    fig = px.scatter(
        x=z[:, 0], y=z[:, 1],
        color=labels,
        color_continuous_scale="Viridis",
        labels={"x": "Latent dim 1", "y": "Latent dim 2", "color": label_name},
        title="Przestrzeń latentna planet (VAE)",
        hover_name=[f"Planet {i}" for i in range(len(labels))]
    )
    return fig
```

### 3.5 Wymagania Sprzętowe

| Zasób | Wartość |
|---|---|
| **VRAM** | ~200–500 MB (trening GPU) lub 0 (CPU) |
| **RAM** | ~100 MB |
| **Trening** | ~1–5 min (CPU), < 30 sek (GPU) |
| **Inference** | < 1 ms |
| **Biblioteka** | PyTorch |

### 3.6 Ograniczenia

- Mały zbiór danych NASA (~2000–5000 po filtracji) może ograniczyć jakość latent space
- 2D latent space traci informację — kompromis wizualizacja vs. jakość
- Wymaga normalizacji danych (różne skale parametrów)

### 3.7 Referencje

- Kingma, D. P., Welling, M. (2014). *Auto-Encoding Variational Bayes.* ICLR 2014.
- Higgins, I., et al. (2017). *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.* ICLR 2017.

### 3.8 Szacowany Effort: ~4–5h | Impact: ⭐⭐⭐⭐ (WOW-factor wizualizacji)

---

## 4. Fourier Neural Operator (FNO) — Uczenie Operatorów PDE

### 4.1 Teoria

Fourier Neural Operator (Li et al., 2021) uczy się mapowania między przestrzeniami funkcji:

$$\mathcal{G}_\theta: \mathcal{A} \to \mathcal{U}$$

gdzie $\mathcal{A}$ to przestrzeń warunków początkowych/parametrów, a $\mathcal{U}$ to przestrzeń rozwiązań PDE.

Kluczowy element — **warstwa Fouriera (Spectral Convolution):**

$$v_{l+1}(x) = \sigma\left(W \cdot v_l(x) + \mathcal{F}^{-1}\left(R_\phi \cdot \mathcal{F}(v_l)\right)(x)\right)$$

gdzie:
- $\mathcal{F}$ i $\mathcal{F}^{-1}$ to FFT i odwrotna FFT
- $R_\phi$ to uczony filtr w dziedzinie Fouriera (parametryzowany tensorowo)
- $W$ to lokalna transformacja liniowa
- $\sigma$ to nieliniowość (np. GELU)

**Kluczowa różnica vs. PINN:** PINN rozwiązuje jedno konkretne PDE (uczy się rozwiązania). FNO uczy się **operatora** — po wytrenowaniu daje rozwiązanie dla dowolnych nowych warunków brzegowych **bez re-treningu**.

### 4.2 Rola w Projekcie

FNO jako trzeci paradygmat surogatów fizycznych obok ELM i PINN:

| Cecha | ELM | PINN | FNO |
|---|---|---|---|
| Co uczy się | Regresja danych | Rozwiązanie jednego PDE | **Operator PDE** |
| Generalizacja | Na nowe parametry | Wymaga re-treningu | **Na nowe parametry i warunki** |
| Fizyka w treningu | Nie (data-driven) | Tak (PDE w loss) | Opcjonalna (physics-informed FNO) |
| Szybkość inference | ~1 ms | ~10 ms | ~5 ms |
| Trening | Sekundy | Minuty/godziny | ~10–30 min |

### 4.3 Implementacja

```python
# fno_surrogate.py
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class SpectralConv2d(nn.Module):
    """
    Warstwa konwolucji spektralnej 2D (Fourier Layer).
    Operuje w dziedzinie Fouriera — uczone wagi filtrują mody.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Liczba modów Fouriera (oś lat)
        self.modes2 = modes2  # Liczba modów Fouriera (oś lon)
        
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
    
    def compl_mul2d(self, input, weights):
        """Mnożenie złożone (batchowe)."""
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    def forward(self, x):
        batchsize = x.shape[0]
        
        # FFT 2D
        x_ft = torch.fft.rfft2(x)
        
        # Mnożenie z uczonymi wagami w dziedzinie Fouriera
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        # Inverse FFT
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    """
    Fourier Neural Operator 2D dla predykcji map temperatur.
    
    Wejście: parametry planety zakodowane jako stałe pola 2D
             (każdy parametr powtórzony na siatce lat×lon)
    Wyjście: mapa temperatury [N_lat, N_lon]
    """
    def __init__(self, modes1=12, modes2=12, width=32, n_params=8,
                 n_lat=32, n_lon=64):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_lat = n_lat
        self.n_lon = n_lon
        
        # Input: n_params kanałów (parametry jako pola) + 2 (meshgrid lat/lon)
        self.fc0 = nn.Linear(n_params + 2, width)
        
        # 4 warstwy Fouriera
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)
        
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        
        # Output
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            params: [batch, n_params] — parametry planetarne
        Returns:
            temp_map: [batch, n_lat, n_lon] — mapa temperatur
        """
        batch_size = params.shape[0]
        
        # Rozciagnij parametry na siatkę 2D
        # params [B, 8] → [B, 8, N_lat, N_lon]
        grid_params = params.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, self.n_lat, self.n_lon
        )
        
        # Dodaj współrzędne siatki (meshgrid)
        lat = torch.linspace(-1, 1, self.n_lat, device=params.device)
        lon = torch.linspace(-1, 1, self.n_lon, device=params.device)
        LAT, LON = torch.meshgrid(lat, lon, indexing='ij')
        grid = torch.stack([LAT, LON]).unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Concat: [B, n_params + 2, N_lat, N_lon]
        x = torch.cat([grid_params, grid], dim=1)
        
        # Permute → [B, N_lat, N_lon, n_params + 2] → FC → [B, N_lat, N_lon, width]
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # [B, width, N_lat, N_lon]
        
        # 4 warstwy Fouriera z residual connections
        x1 = self.conv0(x) + self.w0(x)
        x = torch.nn.functional.gelu(x1)
        
        x1 = self.conv1(x) + self.w1(x)
        x = torch.nn.functional.gelu(x1)
        
        x1 = self.conv2(x) + self.w2(x)
        x = torch.nn.functional.gelu(x1)
        
        x1 = self.conv3(x) + self.w3(x)
        x = torch.nn.functional.gelu(x1)
        
        # Output: [B, width, N_lat, N_lon] → [B, N_lat, N_lon, 1]
        x = x.permute(0, 2, 3, 1)
        x = torch.nn.functional.gelu(self.fc1(x))
        x = self.fc2(x)
        
        return x.squeeze(-1)  # [B, N_lat, N_lon]
```

### 4.4 Wymagania Sprzętowe

| Zasób | Wartość |
|---|---|
| **VRAM** | ~1–3 GB (trening 2D, 32×64) |
| **RAM** | ~500 MB |
| **Trening** | ~10–30 min (GPU), ~2–4h (CPU) |
| **Inference** | ~5 ms per mapa |
| **Biblioteka** | PyTorch + opcjonalnie `neuraloperator` |

### 4.5 Ograniczenia

- Wymaga PyTorch GPU → Colab / WSL2+ROCm na AMD
- Potrzebuje większego zbioru danych treningowych niż ELM (~5000+ par parametry→mapa)
- FFT wymaga regularnej siatki (lat/lon grid — OK w naszym przypadku)

### 4.6 Referencje

- Li, Z., et al. (2021). *Fourier Neural Operator for Parametric Partial Differential Equations.* ICLR 2021.
- Pathak, J., et al. (2022). *FourCastNet: A Global Data-driven High-resolution Weather Forecasting Model.* arXiv:2202.11214.

### 4.7 Szacowany Effort: ~6–8h | Impact: ⭐⭐⭐⭐⭐ (state-of-the-art)

---

## 5. Isolation Forest — Detektor Anomalii w Danych NASA

### 5.1 Teoria

Isolation Forest (Liu et al., 2008) bazuje na założeniu, że anomalie są **łatwe do izolacji**. Algorytm buduje las losowych drzew binarnych (isolation trees), w których dane partycjonowane są losowymi podziałami cech. Anomalie — jako rzadkie i odległe od reszty — wymagają mniej podziałów do izolacji.

**Anomaly score:**

$$s(\mathbf{x}, n) = 2^{-\frac{E[h(\mathbf{x})]}{c(n)}}$$

gdzie $h(\mathbf{x})$ to średnia głębokość izolacji próbki w lesie, a $c(n)$ to normalizacja zależna od rozmiaru danych.

Wartości bliskie 1 → anomalia, bliskie 0.5 → normalna, bliskie 0 → gęsty klaster.

### 5.2 Rola w Projekcie

Automatyczna identyfikacja **nietypowych planet** w katalogu NASA:
- "Ta planeta ma ekstremalnie niskie albedo jak na swój promień — anomalia"
- "Kombinacja parametrów jest statystycznie niespotykana — obiekt warty zbadania"
- Agent jako narzędzie: `find_anomalous_planets()` → top-5 z wyjaśnieniem

### 5.3 Implementacja

```python
# anomaly_detector.py
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple


class ExoplanetAnomalyDetector:
    """
    Detekcja anomalii w parametrach egzoplanet.
    Identyfikuje planety o nietypowych konfiguracjach — potencjalnie
    najciekawsze obiekty badawcze.
    """
    
    FEATURE_COLUMNS = [
        "radius_earth", "mass_earth", "semi_major_axis_au",
        "star_teff_K", "star_radius_solar", "insol_earth"
    ]
    
    def __init__(self, contamination: float = 0.05, n_estimators: int = 200):
        """
        Args:
            contamination: Oczekiwana frakcja anomalii (domyślnie 5%)
            n_estimators: Liczba drzew izolacyjnych
        """
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
    
    def fit(self, df: pd.DataFrame):
        """Trening na kompletnym katalogu planet."""
        X = df[self.FEATURE_COLUMNS].dropna().values
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        print(f"Isolation Forest wytrenowany na {len(X)} planetach.")
    
    def detect(self, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Wykrywa anomalie i zwraca top-N najbardziej nietypowych planet.
        
        Returns:
            DataFrame z dodanymi kolumnami: anomaly_score, is_anomaly
        """
        X = df[self.FEATURE_COLUMNS].dropna()
        X_scaled = self.scaler.transform(X.values)
        
        scores = self.model.decision_function(X_scaled)  # Im niższy → bardziej anomalny
        labels = self.model.predict(X_scaled)              # -1 = anomalia, 1 = normalna
        
        result = df.loc[X.index].copy()
        result["anomaly_score"] = -scores  # Odwracamy, żeby wyższy = bardziej anomalny
        result["is_anomaly"] = (labels == -1)
        
        return result.nlargest(top_n, "anomaly_score")
    
    def explain_anomaly(self, planet_row: pd.Series, all_data: pd.DataFrame) -> List[str]:
        """
        Wyjaśnia, DLACZEGO planeta jest anomalią.
        Porównuje każdy parametr z percentylami populacji.
        """
        explanations = []
        for col in self.FEATURE_COLUMNS:
            val = planet_row[col]
            if pd.isna(val):
                continue
            
            percentile = (all_data[col].dropna() < val).mean() * 100
            
            if percentile > 97:
                explanations.append(
                    f"  ⚠️ {col} = {val:.3f} — w 97. percentylu "
                    f"(wyższe niż {percentile:.1f}% populacji)"
                )
            elif percentile < 3:
                explanations.append(
                    f"  ⚠️ {col} = {val:.3f} — w 3. percentylu "
                    f"(niższe niż {100-percentile:.1f}% populacji)"
                )
        
        if not explanations:
            explanations.append(
                "  ℹ️ Anomalność wynika z nietypowej KOMBINACJI parametrów, "
                "nie z ekstremalności pojedynczych cech."
            )
        
        return explanations
```

### 5.4 Integracja z Agentem

```python
@tool
def find_anomalous_planets(top_n: int = 5) -> str:
    """
    Znajduje najbardziej nietypowe egzoplanety w katalogu NASA.
    Zwraca top-N anomalii z wyjaśnieniem, dlaczego są nietypowe.
    """
    detector = load_anomaly_detector()
    all_data = load_nasa_catalog()
    anomalies = detector.detect(all_data, top_n=top_n)
    
    results = []
    for _, planet in anomalies.iterrows():
        explanation = detector.explain_anomaly(planet, all_data)
        results.append({
            "name": planet.get("pl_name", "Unknown"),
            "anomaly_score": round(planet["anomaly_score"], 4),
            "reasons": explanation
        })
    
    return json.dumps(results, indent=2, ensure_ascii=False)
```

### 5.5 Wymagania Sprzętowe

| Zasób | Wartość |
|---|---|
| **VRAM** | 0 MB (CPU-only) |
| **RAM** | ~20 MB |
| **Trening** | < 1 sek |
| **Inference** | < 10 ms |
| **Biblioteka** | `scikit-learn` (IsolationForest) |

### 5.6 Referencje

- Liu, F. T., Ting, K. M., Zhou, Z.-H. (2008). *Isolation Forest.* ICDM 2008.

### 5.7 Szacowany Effort: ~1–2h | Impact: ⭐⭐⭐

---

## 6. UMAP + HDBSCAN — Wizualizacja i Klasteryzacja Przestrzeni Planet

### 6.1 Teoria

**UMAP** (Uniform Manifold Approximation and Projection, McInnes et al., 2018) to algorytm redukcji wymiarowości oparty na teorii rozmaitości topologicznych i optymalizacji entropi krzyżowej w grafie sąsiedztwa. Daje lepsze zachowanie globalnej struktury niż t-SNE przy podobnej jakości lokalizacji.

**HDBSCAN** (Hierarchical Density-Based Spatial Clustering, Campello et al., 2013) to rozszerzenie DBSCAN z automatycznym doborem parametru epsilon. Znajduje klastry o zmiennej gęstości i oznacza szum (noise) — idealne dla heterogenicznych danych astronomicznych.

### 6.2 Rola w Projekcie

1. **Taksonomia planet:** Automatyczna klasyfikacja planet w bezetykietowe grupy (np. "Klaster A: gorące Jowisze", "Klaster B: skaliste w HZ")
2. **Interaktywna wizualizacja:** 2D scatter plot UMAP z kolorami klasterów w Streamlit
3. **Wyszukiwanie podobnych:** Po wybraniu planety → podświetlenie sąsiadów w UMAP space

### 6.3 Implementacja

```python
# planet_clustering.py
import numpy as np
import pandas as pd
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler
import plotly.express as px


class PlanetaryClustering:
    """
    Klasteryzacja i wizualizacja przestrzeni egzoplanet.
    UMAP (redukcja wymiarowości) + HDBSCAN (klasteryzacja gęstościowa).
    """
    
    FEATURE_COLS = [
        "radius_earth", "mass_earth", "semi_major_axis_au",
        "star_teff_K", "insol_earth", "t_eq_K"
    ]
    
    def __init__(self, n_components: int = 2, min_cluster_size: int = 15):
        self.scaler = StandardScaler()
        self.reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=30,
            min_dist=0.1,
            metric="euclidean",
            random_state=42
        )
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=5,
            gen_min_span_tree=True,
            prediction_data=True
        )
        self.embedding = None
        self.labels = None
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline: normalizacja → UMAP → HDBSCAN → etykiety klasterów.
        
        Returns:
            DataFrame z kolumnami: umap_x, umap_y, cluster
        """
        X = df[self.FEATURE_COLS].dropna()
        X_scaled = self.scaler.fit_transform(X.values)
        
        # Redukcja wymiarowości
        self.embedding = self.reducer.fit_transform(X_scaled)
        
        # Klasteryzacja
        self.labels = self.clusterer.fit_predict(self.embedding)
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = (self.labels == -1).sum()
        
        print(f"UMAP + HDBSCAN: {n_clusters} klasterów, {n_noise} outlierów")
        
        result = df.loc[X.index].copy()
        result["umap_x"] = self.embedding[:, 0]
        result["umap_y"] = self.embedding[:, 1]
        result["cluster"] = self.labels
        
        return result
    
    def describe_clusters(self, df_clustered: pd.DataFrame) -> dict:
        """
        Generuje opisy statystyczne każdego klastera.
        Przydatne do automatycznego etykietowania przez LLM.
        """
        descriptions = {}
        for cluster_id in sorted(df_clustered["cluster"].unique()):
            if cluster_id == -1:
                continue
            subset = df_clustered[df_clustered["cluster"] == cluster_id]
            desc = {}
            for col in self.FEATURE_COLS:
                if col in subset.columns:
                    desc[col] = {
                        "mean": round(float(subset[col].mean()), 3),
                        "std": round(float(subset[col].std()), 3),
                        "min": round(float(subset[col].min()), 3),
                        "max": round(float(subset[col].max()), 3),
                    }
            desc["n_planets"] = len(subset)
            descriptions[f"Cluster_{cluster_id}"] = desc
        
        return descriptions
    
    def find_neighbors(self, planet_idx: int, k: int = 5) -> np.ndarray:
        """Znajduje k najbliższych sąsiadów w UMAP space."""
        point = self.embedding[planet_idx]
        distances = np.linalg.norm(self.embedding - point, axis=1)
        indices = np.argsort(distances)[1:k+1]  # Pomijamy samą siebie (index 0)
        return indices
    
    def plot_interactive(self, df_clustered: pd.DataFrame,
                         highlight_planet: str = None) -> px.scatter:
        """Interaktywny scatter plot UMAP w Plotly."""
        fig = px.scatter(
            df_clustered,
            x="umap_x", y="umap_y",
            color="cluster",
            hover_name="pl_name" if "pl_name" in df_clustered.columns else None,
            hover_data=self.FEATURE_COLS,
            title="Przestrzeń Egzoplanet (UMAP + HDBSCAN)",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"umap_x": "UMAP 1", "umap_y": "UMAP 2", "cluster": "Klaster"},
            template="plotly_dark"
        )
        
        # Podświetlenie wybranej planety
        if highlight_planet and "pl_name" in df_clustered.columns:
            planet = df_clustered[df_clustered["pl_name"] == highlight_planet]
            if not planet.empty:
                fig.add_scatter(
                    x=planet["umap_x"], y=planet["umap_y"],
                    mode="markers",
                    marker=dict(size=20, color="red", symbol="star", line=dict(width=2, color="white")),
                    name=highlight_planet
                )
        
        fig.update_layout(width=800, height=600)
        return fig
```

### 6.4 Wymagania Sprzętowe

| Zasób | Wartość |
|---|---|
| **VRAM** | 0 MB (CPU-only) |
| **RAM** | ~200 MB |
| **Trening** | ~10–30 sek (N ≤ 5000) |
| **Inference** | < 100 ms |
| **Biblioteka** | `umap-learn`, `hdbscan`, `plotly` |

### 6.5 Referencje

- McInnes, L., Healy, J., Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.* arXiv:1802.03426.
- Campello, R. J. G. B., Moulavi, D., Sander, J. (2013). *Density-Based Clustering Based on Hierarchical Density Estimates.* PAKDD 2013.

### 6.6 Szacowany Effort: ~2–3h | Impact: ⭐⭐⭐⭐

---

## 7. Sentence Transformers + ChromaDB — RAG dla Agenta

### 7.1 Teoria

Retrieval-Augmented Generation (Lewis et al., 2020) łączy model generatywny (LLM) z bazą wiedzy. Zamiast polegać wyłącznie na wagach modelu, system:

1. **Retrieval:** Zamienia zapytanie na embedding → szuka najbliższych fragmentów w bazie wektorowej
2. **Augmentation:** Dołącza znalezione fragmenty do kontekstu LLM
3. **Generation:** LLM odpowiada na podstawie zapytania + pobranej wiedzy

**Sentence Transformers** (Reimers & Gurevych, 2019): Modele BERT fine-tunowane na zadaniach semantic similarity. `all-MiniLM-L6-v2` to lekki (~80 MB), szybki model z dobrą jakością embeddingów.

**ChromaDB:** Wektorowa baza danych z wbudowanym persistence i filtrowaniem metadanych.

### 7.2 Rola w Projekcie

Agent cytuje **prawdziwe artykuły naukowe** w odpowiedziach:

- "Wg Turbet et al. (2016), Proxima Cen b prawdopodobnie ma stan klimatyczny Eyeball..."
- "Kopparapu et al. (2013) definiują granice HZ jako..."
- "Obserwacje JWST (Lustig-Yaeger et al., 2023) nie wykryły atmosfery na TRAPPIST-1b..."

### 7.3 Implementacja

```python
# rag_knowledge_base.py
import os
import json
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class AstroKnowledgeBase:
    """
    RAG knowledge base z artykułami astronomicznymi.
    Pipeline: artykuły → chunki → embeddingi → ChromaDB → retrieval.
    """
    
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # ~80 MB, 384D embeddingi
    PERSIST_DIR = "data/chroma_db"
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
        self.vectorstore = None
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " "]
        )
    
    def ingest_documents(self, texts: List[dict]):
        """
        Indeksuje dokumenty do bazy wektorowej.
        
        Args:
            texts: Lista dict z kluczami: 'content', 'source', 'title'
        """
        documents = []
        for item in texts:
            chunks = self.splitter.split_text(item["content"])
            for chunk in chunks:
                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": item.get("source", "unknown"),
                        "title": item.get("title", ""),
                    }
                ))
        
        self.vectorstore = Chroma.from_documents(
            documents,
            self.embeddings,
            persist_directory=self.PERSIST_DIR
        )
        print(f"Zaindeksowano {len(documents)} chunków z {len(texts)} dokumentów.")
    
    def query(self, question: str, k: int = 3) -> List[dict]:
        """
        Semantyczne wyszukiwanie w bazie wiedzy.
        
        Returns:
            Lista dict z kluczami: 'content', 'source', 'title', 'score'
        """
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                persist_directory=self.PERSIST_DIR,
                embedding_function=self.embeddings
            )
        
        results = self.vectorstore.similarity_search_with_score(question, k=k)
        
        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "title": doc.metadata.get("title", ""),
                "score": round(float(score), 4)
            }
            for doc, score in results
        ]
    
    def as_retriever(self, k: int = 3):
        """Zwraca retriever kompatybilny z LangChain."""
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                persist_directory=self.PERSIST_DIR,
                embedding_function=self.embeddings
            )
        return self.vectorstore.as_retriever(search_kwargs={"k": k})


# Przygotowanie bazy wiedzy (jednorazowe)
ASTRO_KNOWLEDGE = [
    {
        "title": "Turbet et al. 2016 — Proxima Centauri b climate",
        "source": "A&A 596, A112",
        "content": """Turbet et al. (2016) performed 3D GCM simulations of Proxima Centauri b 
        using the LMD Generic Model. They explored various atmospheric compositions 
        (N2-dominated, CO2-dominated) and found that the planet can sustain liquid water 
        on its surface for a wide range of conditions. For a tidally locked configuration, 
        the climate state resembles an 'eyeball' topology with a substellar ocean surrounded 
        by ice. The dayside mean temperature ranges from 250-320 K depending on atmospheric 
        pressure and CO2 concentration."""
    },
    {
        "title": "Kopparapu et al. 2013 — Habitable Zone boundaries",
        "source": "ApJ 765, 131",
        "content": """Kopparapu et al. (2013) provided updated habitable zone boundaries 
        based on 1D radiative-convective climate models. The inner edge (runaway greenhouse) 
        corresponds to a stellar flux of ~1.1 S_Earth for a Sun-like star, while the outer 
        edge (maximum greenhouse) is at ~0.36 S_Earth. For M-dwarfs (T_eff ~ 3000K), 
        these boundaries shift inward. The empirical HZ boundaries based on Venus and 
        early Mars extend the conservative limits."""
    },
    # ... dodatkowe artykuły ...
]
```

### 7.4 Integracja z Agentem

```python
@tool
def search_scientific_literature(query: str) -> str:
    """
    Przeszukuje bazę wiedzy z artykułami astronomicznymi.
    Użyj, gdy potrzebujesz cytatu naukowego lub kontekstu z literatury.
    
    Args:
        query: Pytanie lub temat do wyszukania (np. "TRAPPIST-1e climate simulations")
    """
    kb = load_knowledge_base()
    results = kb.query(query, k=3)
    
    formatted = []
    for r in results:
        formatted.append(f"📄 {r['title']}\n   Źródło: {r['source']}\n   {r['content'][:300]}...")
    
    return "\n\n".join(formatted)
```

### 7.5 Wymagania Sprzętowe

| Zasób | Wartość |
|---|---|
| **VRAM** | 0 MB (embedding model na CPU) |
| **RAM** | ~200 MB (model + baza) |
| **Indeksowanie** | ~10 sek (100 dokumentów) |
| **Query** | < 100 ms |
| **Dysk** | ~50–200 MB (ChromaDB persisted) |
| **Biblioteki** | `chromadb`, `langchain-chroma`, `sentence-transformers` |

### 7.6 Referencje

- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020.
- Reimers, N., Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019.

### 7.7 Szacowany Effort: ~2–3h | Impact: ⭐⭐⭐⭐⭐

---

## 8. Kolmogorov-Arnold Networks (KAN) — Alternatywna Architektura PINN

### 8.1 Teoria

Kolmogorov-Arnold Networks (Liu et al., 2024) opierają się na twierdzeniu Kołmogorowa-Arnolda (1957):

> Każda ciągła funkcja wielowymiarowa $f: [0,1]^n \to \mathbb{R}$ może być przedstawiona jako:
> $$f(\mathbf{x}) = \sum_{q=0}^{2n} \Phi_q\left(\sum_{p=1}^{n} \phi_{q,p}(x_p)\right)$$

W odróżnieniu od MLP (gdzie aktywacje są na węzłach i stałe), w KAN **funkcje aktywacji są na krawędziach** i są uczone (parametryzowane B-spline'ami).

**KAN vs MLP:**

| Cecha | MLP | KAN |
|---|---|---|
| Aktywacje | Na węzłach (stałe: ReLU, tanh) | Na krawędziach (uczone splines) |
| Parametry | Wagi macierzy $W$ | Współczynniki B-spline na krawędziach |
| Interpolacja | Zmienna (zależna od warstw) | Doskonała (splines gładkie) |
| Interpretowalność | Niska | Wyższa (splines można zwizualizować) |
| Gładkie PDE | Wymaga wielu neuronów | Naturalne (splines gładkie) |

### 8.2 Rola w Projekcie

Trzecia architektura PINN obok MLP-PINN i PINNFormer. Rozwiązuje **to samo** równanie ciepła, ale z innym podejściem architektonicznym — pozwala na **porównanie trzech paradygmatów:**

1. **PINN-MLP (DeepXDE):** Klasyczne, proste, szybkie
2. **PINNFormer:** Transformer + attention, state-of-the-art
3. **KAN-PINN:** Uczone aktywacje, doskonała interpolacja gładkich funkcji

### 8.3 Implementacja

```python
# kan_pinn.py
import torch
import torch.nn as nn
import numpy as np


class BSplineActivation(nn.Module):
    """
    B-spline activation function (uczony na krawędzi).
    Parametryzowana przez współczynniki kontrolne B-spline'a.
    """
    def __init__(self, num_knots: int = 10, degree: int = 3,
                 grid_range: tuple = (-2, 2)):
        super().__init__()
        self.degree = degree
        self.num_knots = num_knots
        
        # Siatka węzłów (knots)
        knots = torch.linspace(grid_range[0], grid_range[1], num_knots + degree + 1)
        self.register_buffer('knots', knots)
        
        # Uczone współczynniki kontrolne
        self.coefficients = nn.Parameter(torch.randn(num_knots) * 0.1)
    
    def forward(self, x):
        """Ewaluacja B-spline'a w punktach x."""
        # Uproszczona implementacja — pełna ewaluacja de Boor
        # W praktyce lepiej użyć biblioteki torchspline lub pykan
        # Tu: aproksymacja przez liniową kombinację funkcji bazowych
        basis = self._basis_functions(x)
        return (basis * self.coefficients).sum(dim=-1)
    
    def _basis_functions(self, x):
        """Oblicza funkcje bazowe B-spline stopnia 0 (i rekurencyjnie wyższe)."""
        # Uproszczenie: interpolacja liniowa między węzłami
        x_expanded = x.unsqueeze(-1)
        knots = self.knots[:-1].unsqueeze(0)
        widths = (self.knots[1:] - self.knots[:-1]).unsqueeze(0)
        
        # Clamped linear basis
        basis = torch.clamp(1.0 - torch.abs(x_expanded - knots) / (widths + 1e-8), min=0)
        
        # Normalizacja
        basis = basis / (basis.sum(dim=-1, keepdim=True) + 1e-8)
        return basis[:, :self.coefficients.shape[0]]


class KANLayer(nn.Module):
    """
    Warstwa KAN: każda krawędź (połączenie) ma własny spline.
    """
    def __init__(self, in_features, out_features, num_knots=10):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Osobny spline per krawędź (in × out splines)
        self.splines = nn.ModuleList([
            nn.ModuleList([
                BSplineActivation(num_knots=num_knots)
                for _ in range(in_features)
            ])
            for _ in range(out_features)
        ])
    
    def forward(self, x):
        """
        x: [batch, in_features]
        output: [batch, out_features]
        
        output_j = Σ_i spline_{j,i}(x_i)
        """
        outputs = []
        for j in range(self.out_features):
            activation_sum = sum(
                self.splines[j][i](x[:, i])
                for i in range(self.in_features)
            )
            outputs.append(activation_sum)
        
        return torch.stack(outputs, dim=-1)


class KANPINN(nn.Module):
    """
    KAN-PINN: Kolmogorov-Arnold Network jako solver PDE.
    
    Rozwiązuje: κ∇²T + S(x) - σT⁴ = 0 (1D ciepło na terminatorze)
    """
    def __init__(self, layers=[1, 16, 16, 1], num_knots=10):
        super().__init__()
        self.kan_layers = nn.ModuleList([
            KANLayer(layers[i], layers[i+1], num_knots=num_knots)
            for i in range(len(layers) - 1)
        ])
    
    def forward(self, x):
        for layer in self.kan_layers:
            x = layer(x)
        return x


def train_kan_pinn(model, n_colloc=256, n_epochs=5000, lr=1e-3):
    """
    Trening KAN-PINN na równaniu ciepła 1D.
    
    PDE: κ · d²T/dx² + S(x) - σ · T⁴ = 0
    BC:  T(0) = 320 K (substelarny), T(π) = 80 K (antystelarny)
    """
    SIGMA = 5.670374419e-8
    KAPPA = 0.025
    S_MAX = 900.0
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Punkty kolokacji
    x_colloc = torch.linspace(0.01, np.pi - 0.01, n_colloc, requires_grad=True).unsqueeze(-1)
    x_bc_left = torch.tensor([[0.0]])
    x_bc_right = torch.tensor([[np.pi]])
    T_bc_left = torch.tensor([[320.0]])
    T_bc_right = torch.tensor([[80.0]])
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # PDE residual
        T = model(x_colloc)
        dT = torch.autograd.grad(T.sum(), x_colloc, create_graph=True)[0]
        d2T = torch.autograd.grad(dT.sum(), x_colloc, create_graph=True)[0]
        
        S = S_MAX * torch.clamp(torch.cos(x_colloc), min=0)
        residual = KAPPA * d2T + S - SIGMA * T ** 4
        loss_pde = torch.mean(residual ** 2)
        
        # Boundary conditions
        loss_bc = (
            (model(x_bc_left) - T_bc_left) ** 2 +
            (model(x_bc_right) - T_bc_right) ** 2
        ).sum()
        
        loss = loss_pde + 100 * loss_bc
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} — PDE: {loss_pde.item():.6f}, "
                  f"BC: {loss_bc.item():.6f}")
    
    return model
```

### 8.4 Wymagania Sprzętowe

| Zasób | Wartość |
|---|---|
| **VRAM** | ~200–500 MB (trening GPU) lub 0 (CPU) |
| **RAM** | ~100 MB |
| **Trening 1D** | ~2–5 min (GPU), ~15–30 min (CPU) |
| **Inference** | < 10 ms |
| **Biblioteka** | PyTorch (custom) lub `pykan` |

### 8.5 Referencje

- Liu, Z., et al. (2024). *KAN: Kolmogorov-Arnold Networks.* arXiv:2404.19756.
- Kolmogorov, A. N. (1957). *On the Representation of Continuous Functions of Many Variables by Superposition of Continuous Functions of One Variable and Addition.* Doklady.
- Shukla, K., et al. (2024). *A Comprehensive and FAIR Comparison between MLP and KAN Representations for Differential Equations and Operator Networks.* arXiv:2406.02917.

### 8.6 Szacowany Effort: ~3–4h | Impact: ⭐⭐⭐⭐ (cutting-edge architecture)

---

## 9. Neural ODE — Modelowanie Ewolucji Czasowej Klimatu

### 9.1 Teoria

Neural ODE (Chen et al., 2018) modeluje ciągłą dynamikę systemów za pomocą ODE parametryzowanego siecią neuronową:

$$\frac{d\mathbf{h}(t)}{dt} = f_\theta(\mathbf{h}(t), t)$$

gdzie $f_\theta$ to sieć neuronowa. Rozwiązanie uzyskujemy przez solver ODE (np. Dormand-Prince, RK45):

$$\mathbf{h}(t_1) = \mathbf{h}(t_0) + \int_{t_0}^{t_1} f_\theta(\mathbf{h}(t), t) \, dt$$

Gradienty obliczane są **adjoint method** — bez przechowywania pośrednich stanów w pamięci, co jest memory-efficient.

### 9.2 Rola w Projekcie

Modelowanie **ewolucji klimatu w czasie** — jak planeta ewoluuje od stanu początkowego (gorąca magma / losowa temperatura) do stanu równowagi (Eyeball / Lobster / Snowball):

1. **Animation input:** Generuje sekwencję map temperatur w czasie → animacja Plotly Frames
2. **Czas relaksacji:** Jak szybko planeta osiąga równowagę klimatyczną?
3. **Scenariusze "what-if":** Jak zmieni się klimat, jeśli albedo nagle się zmieni? (np. wulkanizm)

### 9.3 Implementacja

```python
# neural_ode_climate.py
import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint


class ClimateODEFunc(nn.Module):
    """
    Sieć definiująca dynamikę klimatu: dT/dt = f_θ(T, params).
    
    Uczy się, jak temperatura ewoluuje na planecie w czasie.
    Wejście: stan T(t) [N_lat * N_lon] + parametry planety
    Wyjście: dT/dt [N_lat * N_lon]
    """
    def __init__(self, state_dim: int, param_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + param_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.params = None  # Ustawiane przed integracją
    
    def set_params(self, params: torch.Tensor):
        """Ustawia parametry planety (przekazywane jako kontekst)."""
        self.params = params
    
    def forward(self, t, state):
        """
        Args:
            t: czas (skalar)
            state: [batch, state_dim] — spłaszczona mapa temperatur
        """
        # Concat stanu z parametrami planety
        x = torch.cat([state, self.params.expand(state.shape[0], -1)], dim=-1)
        return self.net(x)


class ClimateEvolutionModel:
    """
    Neural ODE do modelowania ewolucji klimatu planety.
    
    Workflow:
    1. Ustaw stan początkowy (np. jednorodna temperatura = T_eq)
    2. Integruj ODE w czasie → sekwencja map temperatur
    3. Wizualizuj jako animację
    """
    
    N_LAT = 16   # Mniejsza rozdzielczość (dla szybkości ODE)
    N_LON = 32
    
    def __init__(self, param_dim: int = 8, hidden_dim: int = 128):
        state_dim = self.N_LAT * self.N_LON
        self.ode_func = ClimateODEFunc(state_dim, param_dim, hidden_dim)
        self.state_dim = state_dim
    
    def evolve(self, initial_state: np.ndarray, planet_params: np.ndarray,
               t_span: tuple = (0, 10), n_steps: int = 50) -> np.ndarray:
        """
        Ewolucja klimatu od stanu początkowego.
        
        Args:
            initial_state: [N_LAT, N_LON] — początkowa mapa temperatur
            planet_params: [8] — parametry planety
            t_span: (t_start, t_end) — przedział czasowy (znormalizowany)
            n_steps: Liczba kroków czasowych (frames animacji)
        
        Returns:
            evolution: [n_steps, N_LAT, N_LON] — sekwencja map temperatur
        """
        self.ode_func.eval()
        
        # Przygotowanie tensorów
        y0 = torch.FloatTensor(initial_state.flatten()).unsqueeze(0)
        params = torch.FloatTensor(planet_params).unsqueeze(0)
        t = torch.linspace(t_span[0], t_span[1], n_steps)
        
        self.ode_func.set_params(params)
        
        with torch.no_grad():
            trajectory = odeint(
                self.ode_func,
                y0,
                t,
                method='dopri5',  # Dormand-Prince (adaptive RK45)
                rtol=1e-4,
                atol=1e-6
            )
        
        # [n_steps, 1, state_dim] → [n_steps, N_LAT, N_LON]
        evolution = trajectory.squeeze(1).numpy()
        evolution = evolution.reshape(n_steps, self.N_LAT, self.N_LON)
        
        return evolution
    
    def train(self, trajectories: list, planet_params_list: list,
              epochs: int = 500, lr: float = 1e-3):
        """
        Trening Neural ODE na parach (parametry, trajektoria_klimatu).
        
        Args:
            trajectories: Lista [n_steps, N_LAT*N_LON] — znane ewolucje
            planet_params_list: Lista [8] — odpowiadające parametry planet
        """
        optimizer = torch.optim.Adam(self.ode_func.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0
            
            for traj, params in zip(trajectories, planet_params_list):
                traj_tensor = torch.FloatTensor(traj)
                params_tensor = torch.FloatTensor(params).unsqueeze(0)
                t = torch.linspace(0, 1, len(traj))
                
                y0 = traj_tensor[0:1]
                self.ode_func.set_params(params_tensor)
                
                pred = odeint(self.ode_func, y0, t, method='euler')
                loss = nn.MSELoss()(pred.squeeze(1), traj_tensor)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} — Loss: {total_loss/len(trajectories):.6f}")
    
    def save(self, path: str = "models/neural_ode_climate.pt"):
        torch.save(self.ode_func.state_dict(), path)
    
    def load(self, path: str = "models/neural_ode_climate.pt"):
        self.ode_func.load_state_dict(torch.load(path))
```

### 9.4 Wymagania Sprzętowe

| Zasób | Wartość |
|---|---|
| **VRAM** | ~500 MB – 1 GB (trening GPU) |
| **RAM** | ~200 MB |
| **Trening** | ~10–30 min (GPU), ~1–3h (CPU) |
| **Inference** | ~100–500 ms (integracja ODE) |
| **Biblioteka** | PyTorch + `torchdiffeq` |

### 9.5 Referencje

- Chen, R. T. Q., et al. (2018). *Neural Ordinary Differential Equations.* NeurIPS 2018.

### 9.6 Szacowany Effort: ~5–6h | Impact: ⭐⭐⭐⭐⭐ (animacja ewolucji)

---

## 10. Normalizing Flows (RealNVP) — Alternatywa Generatywna dla CTGAN

### 10.1 Teoria

Normalizing Flows (Rezende & Mohamed, 2015) transformują prosty rozkład bazowy $p_0(\mathbf{z})$ (np. Gaussian) w złożony rozkład danych $p(\mathbf{x})$ przez sekwencję odwracalnych transformacji:

$$\mathbf{x} = f_K \circ f_{K-1} \circ \cdots \circ f_1(\mathbf{z})$$

**Kluczowa przewaga:** Dają **dokładną gęstość prawdopodobieństwa** (w odróżnieniu od GAN, który nie daje $p(\mathbf{x})$):

$$\log p(\mathbf{x}) = \log p_0(\mathbf{z}) - \sum_{k=1}^{K} \log \left|\det \frac{\partial f_k}{\partial \mathbf{z}_{k-1}}\right|$$

**RealNVP** (Dinh et al., 2017): Affine coupling layers z efektywnym obliczaniem jakobianu (trójkątna macierz → determinant = iloczyn diagonali).

### 10.2 Rola w Projekcie

Porównanie dwóch paradygmatów generatywnych:

| Cecha | CTGAN | Normalizing Flow |
|---|---|---|
| Gęstość $p(\mathbf{x})$ | Nie daje | **Dokładna** |
| Próbkowanie | Szybkie | Szybkie |
| Stabilność treningu | ⚠️ Mode collapse | ✅ Stabilne (likelihood) |
| Dane tabelaryczne | ✅ Dedykowane | ⚠️ Wymaga adaptacji |

**Use case:** Flow ocenia "jak prawdopodobna fizycznie jest ta konfiguracja planetarna" → walidacja syntetetycznych planet CTGAN.

### 10.3 Implementacja

```python
# normalizing_flow.py
import torch
import torch.nn as nn
import numpy as np


class AffineCouplingLayer(nn.Module):
    """Warstwa affine coupling (RealNVP)."""
    
    def __init__(self, dim, mask, hidden_dim=64):
        super().__init__()
        self.mask = mask  # Binarna maska [dim] — które wymiary się zmieniają
        
        # Sieci s(·) i t(·) — scale i translation
        self.s_net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, dim), nn.Tanh()  # Tanh stabilizuje skalę
        )
        self.t_net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x):
        """Forward: z → x (generowanie)."""
        x_masked = x * self.mask
        s = self.s_net(x_masked) * (1 - self.mask)
        t = self.t_net(x_masked) * (1 - self.mask)
        y = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det = s.sum(dim=-1)
        return y, log_det
    
    def inverse(self, y):
        """Inverse: x → z (inference)."""
        y_masked = y * self.mask
        s = self.s_net(y_masked) * (1 - self.mask)
        t = self.t_net(y_masked) * (1 - self.mask)
        x = y_masked + (1 - self.mask) * ((y - t) * torch.exp(-s))
        return x


class PlanetaryNormalizingFlow(nn.Module):
    """
    Normalizing Flow do modelowania rozkładu parametrów planet.
    
    Zastosowania:
    1. Generowanie nowych konfiguracji: z ~ N(0,I) → x (parametry)
    2. Ocena prawdopodobieństwa: p(x) — czy konfiguracja jest realistyczna?
    3. Walidacja CTGAN: czy syntetyczne planety mają wysoki p(x)?
    """
    
    def __init__(self, dim=8, n_layers=8, hidden_dim=64):
        super().__init__()
        self.dim = dim
        
        # Alternujące maski
        masks = []
        for i in range(n_layers):
            mask = torch.zeros(dim)
            mask[:dim//2] = 1 if i % 2 == 0 else 0
            mask[dim//2:] = 0 if i % 2 == 0 else 1
            masks.append(mask)
        
        self.layers = nn.ModuleList([
            AffineCouplingLayer(dim, mask, hidden_dim)
            for mask in masks
        ])
        
        # Prior: standardowy Gaussian
        self.prior = torch.distributions.MultivariateNormal(
            torch.zeros(dim), torch.eye(dim)
        )
    
    def forward(self, z):
        """z → x (generowanie)."""
        log_det_total = 0
        x = z
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_total += log_det
        return x, log_det_total
    
    def inverse(self, x):
        """x → z (inference/encoding)."""
        z = x
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z
    
    def log_prob(self, x):
        """Oblicza log-prawdopodobieństwo danych."""
        z = self.inverse(x)
        log_pz = self.prior.log_prob(z)
        
        # Forward pass for log_det (potrzebny do zmiany zmiennych)
        _, log_det = self.forward(z)
        
        return log_pz + log_det
    
    def sample(self, n_samples: int) -> np.ndarray:
        """Generuje syntetyczne konfiguracje planetarne."""
        with torch.no_grad():
            z = self.prior.sample((n_samples,))
            x, _ = self.forward(z)
            return x.numpy()
    
    def evaluate_ctgan_samples(self, ctgan_samples: np.ndarray) -> np.ndarray:
        """
        Ocena prawdopodobieństwa próbek z CTGAN.
        Wysokie log_prob → realistyczna konfiguracja.
        Niskie log_prob → potencjalnie niefizyczna.
        """
        with torch.no_grad():
            x = torch.FloatTensor(ctgan_samples)
            return self.log_prob(x).numpy()
```

### 10.4 Wymagania Sprzętowe

| Zasób | Wartość |
|---|---|
| **VRAM** | ~300–800 MB (trening GPU) lub 0 (CPU) |
| **RAM** | ~100 MB |
| **Trening** | ~5–15 min (GPU), ~30–60 min (CPU) |
| **Inference** | < 5 ms per sample |
| **Biblioteka** | PyTorch (custom) lub `nflows` |

### 10.5 Referencje

- Rezende, D. J., Mohamed, S. (2015). *Variational Inference with Normalizing Flows.* ICML 2015.
- Dinh, L., Sohl-Dickstein, J., Bengio, S. (2017). *Density Estimation Using Real-NVP.* ICLR 2017.

### 10.6 Szacowany Effort: ~4–5h | Impact: ⭐⭐⭐

---

## 11. Autoencoder (AE) — Kompresja i Porównywanie Map Temperatur

### 11.1 Teoria

Autoencoder to sieć ucząca się kompresji danych do niskowymiarowej reprezentacji (bottleneck) i ich odtwarzania:

$$\mathbf{z} = f_\text{enc}(\mathbf{x}), \quad \hat{\mathbf{x}} = f_\text{dec}(\mathbf{z})$$

$$\mathcal{L} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2$$

Przestrzeń latentna $\mathbf{z}$ jest kompaktową reprezentacją, w której **podobne mapy temperatur są blisko siebie**.

### 11.2 Rola w Projekcie

1. **Porównywanie map:** Odległość euklidesowa w latent space zamiast porównywania 2048 pikseli bezpośrednio
2. **"Znajdź podobne planety":** Encode mapa wybranej planety → k-NN w latent space → planety o podobnym klimacie
3. **Detekcja anomalinych map:** Wysoki błąd rekonstrukcji = mapa, której AE nie widziało w treningu
4. **Kompresja storage:** 2048 → 32 floaty per mapa (64× kompresja)

### 11.3 Implementacja

```python
# map_autoencoder.py
import torch
import torch.nn as nn
import numpy as np
import pickle


class TemperatureMapAutoencoder(nn.Module):
    """
    Convolutional Autoencoder do kompresji map temperatur [32, 64].
    
    Encoder: [1, 32, 64] → Conv → Pool → Conv → Pool → Flatten → Latent [32]
    Decoder: Latent [32] → Unflatten → ConvT → Upsample → ConvT → [1, 32, 64]
    """
    
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder (konwolucyjny)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # [16, 32, 64]
            nn.ReLU(),
            nn.MaxPool2d(2),                                # [16, 16, 32]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),   # [32, 16, 32]
            nn.ReLU(),
            nn.MaxPool2d(2),                                # [32, 8, 16]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # [64, 8, 16]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)),                   # [64, 4, 8]
            nn.Flatten(),                                    # [2048]
            nn.Linear(64 * 4 * 8, latent_dim),              # [latent_dim]
        )
        
        # Decoder (dekonwolucyjny)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 4 * 8),
            nn.Unflatten(1, (64, 4, 8)),                    # [64, 4, 8]
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [32, 8, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # [16, 16, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # [1, 32, 64]
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


class MapSimilarityEngine:
    """
    Silnik porównywania map temperatur za pomocą Autoencodera.
    
    Workflow:
    1. Encode wszystkie mapy do latent space
    2. Dla nowej mapy → encode → k-NN w latent space
    3. Zwróć najppodobniejsze planety
    """
    
    def __init__(self, model: TemperatureMapAutoencoder):
        self.model = model
        self.latent_db = None      # [N, latent_dim]
        self.planet_names = None   # [N]
    
    def build_index(self, maps: np.ndarray, names: list):
        """
        Buduje indeks latent vectors dla wszystkich znanych map.
        
        Args:
            maps: [N, 32, 64] — mapy temperatur
            names: Lista nazw planet
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(maps).unsqueeze(1)  # [N, 1, 32, 64]
            self.latent_db = self.model.encode(x).numpy()
        self.planet_names = names
        print(f"Zbudowano indeks: {len(names)} planet w {self.latent_db.shape[1]}D space")
    
    def find_similar(self, query_map: np.ndarray, k: int = 5) -> list:
        """
        Znajduje k planet z najabardziej podobnym klimatem.
        
        Returns:
            Lista dict z kluczami: 'name', 'distance', 'rank'
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(query_map).unsqueeze(0).unsqueeze(0)
            z = self.model.encode(x).numpy()
        
        distances = np.linalg.norm(self.latent_db - z, axis=1)
        indices = np.argsort(distances)[:k]
        
        return [
            {
                "name": self.planet_names[i],
                "distance": round(float(distances[i]), 4),
                "rank": rank + 1
            }
            for rank, i in enumerate(indices)
        ]
    
    def reconstruction_anomaly(self, temp_map: np.ndarray) -> float:
        """
        Anomaly score = błąd rekonstrukcji.
        Wysokie → mapa niepodobna do niczego w zbiorze treningowym.
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(temp_map).unsqueeze(0).unsqueeze(0)
            recon, _ = self.model(x)
            error = torch.mean((x - recon) ** 2).item()
        return error
```

### 11.4 Wymagania Sprzętowe

| Zasób | Wartość |
|---|---|
| **VRAM** | ~50–200 MB (trening) lub 0 (CPU) |
| **RAM** | ~100 MB |
| **Trening** | ~2–5 min (GPU), ~15–30 min (CPU) |
| **Inference** | < 5 ms |
| **Biblioteka** | PyTorch |

### 11.5 Referencje

- Hinton, G. E., Salakhutdinov, R. R. (2006). *Reducing the Dimensionality of Data with Neural Networks.* Science 313.

### 11.6 Szacowany Effort: ~2–3h | Impact: ⭐⭐⭐

---

## 12. Conformal Prediction — Formalne Gwarancje Pokrycia

### 12.1 Teoria

Conformal Prediction (Vovk et al., 2005) to framework dający **statystycznie gwarantowane** przedziały ufności **bez założeń o rozkładzie** danych. Dla poziomu ufności $1 - \alpha$:

$$P(Y_{\text{new}} \in C(X_{\text{new}})) \geq 1 - \alpha$$

**Split Conformal Prediction:**
1. Podziel dane na training set i calibration set
2. Wytrenuj model na training set
3. Oblicz **nonconformity scores** na calibration set: $s_i = |y_i - \hat{y}_i|$
4. Wyznacz kwantyl: $\hat{q} = \text{Quantile}(s_1, \ldots, s_n; \lceil(1-\alpha)(n+1)\rceil / n)$
5. Prediction interval: $C(x) = [\hat{y}(x) - \hat{q}, \hat{y}(x) + \hat{q}]$

### 12.2 Rola w Projekcie

Wrapper na ELM Ensemble dający **formalne** przedziały ufności:
- "Model przewiduje T_eq = 234 K, z **95% gwarancją** prawdziwa wartość ∈ [218, 251] K"
- "Habitable Surface Fraction = 14.3% ± 4.2% (coverage 90%)"

W odróżnieniu od empirycznego std z ensemble'u — conformal daje **matematyczną gwarancję**.

### 12.3 Implementacja

```python
# conformal_prediction.py
import numpy as np
from typing import Tuple


class ConformalPredictor:
    """
    Split Conformal Prediction wrapper na dowolny regressor (np. ELM).
    Daje statystycznie gwarantowane prediction intervals.
    
    Gwarancja: P(y_true ∈ [ŷ - q̂, ŷ + q̂]) ≥ 1 - α
    Dla KAŻDEGO nowego punktu (distribution-free, finite-sample).
    """
    
    def __init__(self, base_model, alpha: float = 0.05):
        """
        Args:
            base_model: Wytrenowany model z metodą .predict(X)
            alpha: Poziom błędu (0.05 → 95% coverage)
        """
        self.model = base_model
        self.alpha = alpha
        self.q_hat = None  # Wyznaczony kwantyl
        self.cal_scores = None
    
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """
        Kalibracja na zbiorze kalibracyjnym.
        
        KRYTYCZNE: X_cal i y_cal NIE mogą być częścią zbioru treningowego!
        Dane muszą być "exchangeable" (zamienne) z danymi testowymi.
        
        Args:
            X_cal: [n_cal, n_features] — cechy kalibracyjne
            y_cal: [n_cal, ...] — prawdziwe wartości
        """
        y_pred = self.model.predict(X_cal)
        
        # Nonconformity scores: |y - ŷ|
        self.cal_scores = np.abs(y_cal - y_pred)
        
        # Kwantyl (z korekcją na finite-sample)
        n = len(self.cal_scores)
        
        if self.cal_scores.ndim == 1:
            # Scalar output (np. T_eq)
            level = np.ceil((1 - self.alpha) * (n + 1)) / n
            self.q_hat = np.quantile(self.cal_scores, min(level, 1.0))
        else:
            # Multi-output (np. mapa temperatur) — per-pixel kwantyl
            level = np.ceil((1 - self.alpha) * (n + 1)) / n
            self.q_hat = np.quantile(self.cal_scores, min(level, 1.0), axis=0)
        
        print(f"Kalibracja conformal (α={self.alpha}):")
        print(f"  N_cal = {n}")
        
        if isinstance(self.q_hat, np.ndarray):
            print(f"  q̂ (mean per pixel) = {self.q_hat.mean():.2f} K")
            print(f"  q̂ (max per pixel) = {self.q_hat.max():.2f} K")
        else:
            print(f"  q̂ = {self.q_hat:.2f} K")
    
    def predict_with_interval(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predykcja z gwarantowanym prediction interval.
        
        Returns:
            y_pred: Predykcja punktowa
            lower: Dolna granica (1-α coverage)
            upper: Górna granica (1-α coverage)
        """
        if self.q_hat is None:
            raise RuntimeError("Model nie jest skalibrowany! Użyj .calibrate() najpierw.")
        
        y_pred = self.model.predict(X)
        
        lower = y_pred - self.q_hat
        upper = y_pred + self.q_hat
        
        return y_pred, lower, upper
    
    def coverage_report(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Weryfikacja empirycznego coverage na zbiorze testowym.
        Coverage powinien być ≥ 1-α.
        """
        y_pred, lower, upper = self.predict_with_interval(X_test)
        covered = (y_test >= lower) & (y_test <= upper)
        
        if covered.ndim > 1:
            # Per-pixel coverage
            coverage = covered.mean()
            per_pixel_coverage = covered.mean(axis=0)
            return {
                "overall_coverage": round(float(coverage), 4),
                "target_coverage": round(1 - self.alpha, 4),
                "min_pixel_coverage": round(float(per_pixel_coverage.min()), 4),
                "max_pixel_coverage": round(float(per_pixel_coverage.max()), 4),
                "mean_interval_width_K": round(float((upper - lower).mean()), 2)
            }
        else:
            coverage = covered.mean()
            return {
                "empirical_coverage": round(float(coverage), 4),
                "target_coverage": round(1 - self.alpha, 4),
                "mean_interval_width_K": round(float((upper - lower).mean()), 2),
                "satisfied": bool(coverage >= 1 - self.alpha)
            }


# Użycie z ELM:
def create_conformal_elm(elm_surrogate, X_cal, y_cal, alpha=0.05):
    """
    Tworzy conformal wrapper na wytrenowany ELM.
    
    Typowy workflow:
    1. Podziel dane: 80% trening ELM, 20% kalibracja conformal
    2. Wytrenuj ELM na 80%
    3. Kalibruj conformal na 20%
    """
    cp = ConformalPredictor(elm_surrogate, alpha=alpha)
    cp.calibrate(X_cal, y_cal)
    return cp


# Integracja ze Streamlit:
def display_conformal_results(st, y_pred, lower, upper, alpha):
    """Wyświetlenie wyników z przedziałami ufności."""
    import streamlit as st_lib
    
    coverage_pct = round((1 - alpha) * 100)
    
    col1, col2, col3 = st_lib.columns(3)
    col1.metric("T_eq (predykcja)", f"{y_pred.mean():.0f} K")
    col2.metric(f"Dolna granica ({coverage_pct}%)", f"{lower.mean():.0f} K")
    col3.metric(f"Górna granica ({coverage_pct}%)", f"{upper.mean():.0f} K")
    
    st_lib.info(
        f"📊 **Conformal Prediction:** Z {coverage_pct}% gwarancją statystyczną, "
        f"prawdziwa temperatura znajduje się w przedziale "
        f"[{lower.mean():.0f} K, {upper.mean():.0f} K]."
    )
```

### 12.4 Wymagania Sprzętowe

| Zasób | Wartość |
|---|---|
| **VRAM** | 0 MB (pure NumPy) |
| **RAM** | ~20 MB |
| **Kalibracja** | < 1 sek |
| **Inference** | < 1 ms (dodane do czasu bazowego modelu) |
| **Biblioteka** | NumPy (custom) lub `mapie` |

### 12.5 Ograniczenia

- Wymaga **osobnego** zbioru kalibracyjnego (nie tego samego co treningowy)
- Zakłada "exchangeability" — dane kalibracyjne i testowe z tego samego rozkładu
- Prediction intervals mogą być szerokie, jeśli model bazowy jest słaby
- Marginal (nie per-pixel conditional) coverage w wersji split conformal

### 12.6 Referencje

- Vovk, V., Gammerman, A., Shafer, G. (2005). *Algorithmic Learning in a Random World.* Springer.
- Angelopoulos, A. N., Bates, S. (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* arXiv:2107.07511.
- Barber, R. F., et al. (2021). *Predictive Inference with the Jackknife+.* Annals of Statistics.

### 12.7 Szacowany Effort: ~2h | Impact: ⭐⭐⭐⭐⭐ (naukowy rygor, ~50 linii kodu)

---

## Podsumowanie — Priorytetyzacja

### Tabela Zbiorcza

| # | Model | Effort | VRAM | Rola | Impact |
|---|---|---|---|---|---|
| 1 | **XGBoost** | 2h | 0 | Klasyfikacja topologii klimatu | ⭐⭐⭐⭐ |
| 2 | **GPR** | 3–4h | 0 | Surogat z natywną uncertainty | ⭐⭐⭐⭐ |
| 3 | **VAE** | 4–5h | 200–500 MB | Latent space planet | ⭐⭐⭐⭐ |
| 4 | **FNO** | 6–8h | 1–3 GB | Operator PDE (Fourier) | ⭐⭐⭐⭐⭐ |
| 5 | **Isolation Forest** | 1–2h | 0 | Detekcja anomalii | ⭐⭐⭐ |
| 6 | **UMAP + HDBSCAN** | 2–3h | 0 | Klasteryzacja i wizualizacja | ⭐⭐⭐⭐ |
| 7 | **Sentence Transformers + RAG** | 2–3h | 0 | Agent cytuje artykuły | ⭐⭐⭐⭐⭐ |
| 8 | **KAN-PINN** | 3–4h | 200–500 MB | Alternatywna architektura PINN | ⭐⭐⭐⭐ |
| 9 | **Neural ODE** | 5–6h | 500 MB–1 GB | Ewolucja czasowa klimatu | ⭐⭐⭐⭐⭐ |
| 10 | **Normalizing Flow** | 4–5h | 300–800 MB | Alternatywa generatywna + ocena gęstości | ⭐⭐⭐ |
| 11 | **Autoencoder** | 2–3h | 50–200 MB | Kompresja i porównywanie map | ⭐⭐⭐ |
| 12 | **Conformal Prediction** | 2h | 0 | Formalna uncertainty | ⭐⭐⭐⭐⭐ |

### Top 5 Rekomendacji (Impact/Effort)

| Rank | Model | Effort | Powód |
|---|---|---|---|
| 🥇 | **Conformal Prediction** | 2h | ~50 linii, CPU, formalna gwarancja naukowa |
| 🥈 | **Sentence Transformers + RAG** | 2–3h | Agent cytuje artykuły — ogromny skok jakości |
| 🥉 | **XGBoost** | 2h | CPU, instant, etykieta klimatu "🔵 Eyeball" |
| 4 | **Isolation Forest** | 1–2h | CPU, 10 linii, nowy tool agenta |
| 5 | **UMAP + HDBSCAN** | 2–3h | CPU, piękna wizualizacja taksonomii |

Te 5 modeli razem: **~10h effort, 0 VRAM, 5 nowych paradygmatów AI** — łączna liczba modeli w systemie rośnie z 4 do 9.
