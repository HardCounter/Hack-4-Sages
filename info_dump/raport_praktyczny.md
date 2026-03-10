# Raport Praktyczny: Implementacja Cyfrowego Bliźniaka Egzoplanetarnego

## Spis Treści
1. [Konfiguracja Środowiska](#1-konfiguracja-środowiska)
2. [Streamlit — Interfejs Użytkownika](#2-streamlit--interfejs-użytkownika)
3. [NASA Exoplanet Archive — Pozyskiwanie Danych](#3-nasa-exoplanet-archive--pozyskiwanie-danych)
4. [Obliczenia Astrofizyczne (Pure Python)](#4-obliczenia-astrofizyczne-pure-python)
5. [Pydantic — Walidacja i Bariery Bezpieczeństwa](#5-pydantic--walidacja-i-bariery-bezpieczeństwa)
6. [CTGAN — Augmentacja Danych Tabelarycznych](#6-ctgan--augmentacja-danych-tabelarycznych)
7. [Extreme Learning Machines (ELM) — Surogat Klimatyczny](#7-extreme-learning-machines-elm--surogat-klimatyczny)
8. [Ollama — Lokalne Hostowanie LLM](#8-ollama--lokalne-hostowanie-llm)
9. [LangChain / smolagents — Orkiestracja Agentowa](#9-langchain--smolagents--orkiestracja-agentowa)
10. [Plotly — Wizualizacja 3D Sfery Planetarnej](#10-plotly--wizualizacja-3d-sfery-planetarnej)
11. [DeepXDE — PINN (Opcjonalnie)](#11-deepxde--pinn-opcjonalnie)
12. [Graceful Degradation — Implementacja Odporności](#12-graceful-degradation--implementacja-odporności)
13. [Struktura Projektu i Integracja Końcowa](#13-struktura-projektu-i-integracja-końcowa)

---

## 1. Konfiguracja Środowiska

### 1.1 Wymagania Systemowe
- Python 3.10+ (zalecane 3.11)
- RAM: minimum 8 GB (16 GB zalecane dla Ollama + modeli)
- Dysk: ~10 GB wolnego miejsca (modele LLM)
- GPU: opcjonalne (CUDA dla przyspieszenia CTGAN)

### 1.2 Inicjalizacja Projektu

```powershell
# Tworzenie katalogu projektu
mkdir HACK-4-SAGES
cd HACK-4-SAGES

# Tworzenie wirtualnego środowiska
python -m venv .venv

# Aktywacja (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Aktywacja (Linux/Mac)
# source .venv/bin/activate
```

### 1.3 Instalacja Zależności

```powershell
# Plik requirements.txt
pip install streamlit==1.41.0
pip install pandas==2.2.3
pip install numpy==1.26.4
pip install plotly==5.24.1
pip install requests==2.32.3
pip install pydantic==2.10.3
pip install scikit-learn==1.6.0
pip install scikit-elm==0.21a0
pip install ctgan==0.10.2
pip install langchain==0.3.13
pip install langchain-ollama==0.2.3
pip install ollama==0.4.4
pip install scipy==1.14.1
pip install netCDF4==1.7.2
pip install deepxde==1.12.0  # opcjonalne
```

Alternatywnie — jednolinijkowa instalacja:

```powershell
pip install streamlit pandas numpy plotly requests pydantic scikit-learn scikit-elm ctgan langchain langchain-ollama ollama scipy netCDF4
```

### 1.4 Generowanie `requirements.txt`

```powershell
pip freeze > requirements.txt
```

---

## 2. Streamlit — Interfejs Użytkownika

### 2.1 Czym jest Streamlit?
Streamlit to framework Pythona do budowy aplikacji webowych (data apps) bez znajomości HTML/CSS/JS. Każde wywołanie funkcji Streamlit tworzy element UI w przeglądarce. Plik `.py` jest jednocześnie backendem i frontendem.

**Kluczowa mechanika:** Streamlit re-uruchamia cały skrypt od góry do dołu przy każdej interakcji użytkownika (klik przycisku, zmiana suwaka). To wymaga prawidłowego zarządzania stanem.

### 2.2 Uruchomienie Aplikacji

```powershell
streamlit run app.py
# Otworzy się przeglądarka: http://localhost:8501
```

### 2.3 Podstawowa Struktura Aplikacji

```python
# app.py
import streamlit as st
import pandas as pd
import numpy as np

# ─── Konfiguracja strony ───
st.set_page_config(
    page_title="Exoplanetary Digital Twin",
    page_icon="🪐",
    layout="wide",           # Pełna szerokość
    initial_sidebar_state="expanded"
)

# ─── Tytuł i opis ───
st.title("🪐 Autonomiczny Cyfrowy Bliźniak Egzoplanetarny")
st.markdown("Symuluj klimaty obcych światów w czasie rzeczywistym.")

# ─── Sidebar z kontrolkami ───
with st.sidebar:
    st.header("Parametry Planety")
    
    planet_name = st.text_input(
        "Nazwa planety",
        value="Proxima Centauri b",
        help="Wpisz nazwę z katalogu NASA lub własną"
    )
    
    stellar_temp = st.slider(
        "Temperatura gwiazdy [K]",
        min_value=2500, max_value=7500, value=3042, step=50
    )
    
    planet_radius = st.slider(
        "Promień planety [R⊕]",
        min_value=0.5, max_value=2.5, value=1.07, step=0.01
    )
    
    semi_major_axis = st.slider(
        "Półoś wielka [AU]",
        min_value=0.01, max_value=2.0, value=0.0485, step=0.001,
        format="%.4f"
    )
    
    albedo = st.slider(
        "Albedo Bonda",
        min_value=0.0, max_value=1.0, value=0.3, step=0.01
    )
    
    tidal_lock = st.checkbox("Uwięzienie pływowe", value=True)
    
    run_button = st.button("🚀 Uruchom symulację", type="primary")

# ─── Główna sekcja ───
if run_button:
    with st.spinner("Symulacja w toku..."):
        # Tu uruchamiamy pipeline
        st.success("Symulacja zakończona!")

# ─── Layout wielokolumnowy ───
col1, col2 = st.columns(2)
with col1:
    st.subheader("Dane planetarne")
    st.metric("Temperatura równowagowa", "255 K", delta="-18 vs Ziemia")
with col2:
    st.subheader("Wizualizacja 3D")
    # st.plotly_chart(fig, use_container_width=True)
```

### 2.4 Kluczowe Widgety Streamlit

| Widget | Zastosowanie | Składnia |
|---|---|---|
| `st.text_input()` | Wejście tekstowe (zapytania do LLM) | `st.text_input("Pytanie")` |
| `st.slider()` | Suwak numeryczny | `st.slider("Temp", 0, 1000, 300)` |
| `st.selectbox()` | Dropdown (wybór planety) | `st.selectbox("Planeta", ["a","b"])` |
| `st.checkbox()` | Checkbox (tidal lock) | `st.checkbox("Locked")` |
| `st.button()` | Przycisk akcji | `st.button("Run")` |
| `st.columns()` | Layout kolumnowy | `col1, col2 = st.columns(2)` |
| `st.tabs()` | Zakładki | `tab1, tab2 = st.tabs(["A","B"])` |
| `st.expander()` | Panel rozwijalny | `with st.expander("Detale"):` |
| `st.plotly_chart()` | Embed wykresu Plotly | `st.plotly_chart(fig)` |
| `st.dataframe()` | Interaktywna tabela | `st.dataframe(df)` |
| `st.metric()` | Metryka (wartość + delta) | `st.metric("ESI", 0.87)` |
| `st.spinner()` | Loader | `with st.spinner("..."):` |
| `st.chat_input()` | Wejście chatowe | `prompt = st.chat_input()` |
| `st.chat_message()` | Bąbelek czatu | `with st.chat_message("ai"):` |

### 2.5 Caching — Krytyczny Mechanizm Wydajności

Bez cachingu Streamlit ładuje modele i dane od nowa przy **każdej** interakcji. To zabija wydajność.

```python
@st.cache_resource  # Cache dla obiektów dużych, mutowalnych (modele ML, połączenia)
def load_elm_model():
    """Ładuje wagi ELM — wykonuje się TYLKO RAZ."""
    import pickle
    with open("models/elm_ensemble.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data  # Cache dla danych (DataFrames, tablice numpy)  
def fetch_nasa_data(query: str) -> pd.DataFrame:
    """Pobiera dane z NASA — cache'uje wynik dla tego samego query."""
    import requests
    url = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={query}&format=csv"
    response = requests.get(url)
    df = pd.read_csv(io.StringIO(response.text))
    return df

# Użycie — te wywołania NIE blokują UI po pierwszym załadowaniu
elm_model = load_elm_model()
nasa_df = fetch_nasa_data("SELECT+*+FROM+pscomppars+WHERE+pl_radj+<+2.5")
```

**Różnica między dekoratorami:**

| Dekorator | Kiedy używać | Serializacja |
|---|---|---|
| `@st.cache_data` | Dane (DataFrame, numpy, str, int) | Kopia (każdy session ma kopię) |
| `@st.cache_resource` | Modele ML, połączenia DB, klienty API | Referencja (współdzielone między sesjami) |

### 2.6 Session State — Zarządzanie Stanem

```python
# Inicjalizacja stanu (wykonuje się raz per sesja użytkownika)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "simulation_results" not in st.session_state:
    st.session_state.simulation_results = None

# Zapis wyniku
st.session_state.simulation_results = {"t_eq": 255, "esi": 0.87}

# Odczyt w innej części UI
if st.session_state.simulation_results:
    st.metric("ESI", st.session_state.simulation_results["esi"])
```

### 2.7 Chat Interface — Komunikacja z LLM

```python
# ─── Chat z agentem ───
st.subheader("💬 Rozmowa z Agentem Astronomicznym")

# Wyświetlanie historii
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Wejście użytkownika
if prompt := st.chat_input("Zapytaj o egzoplanetę..."):
    # Dodaj wiadomość użytkownika
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Odpowiedź agenta
    with st.chat_message("assistant"):
        with st.spinner("Agent analizuje..."):
            response = run_agent(prompt)  # Wywołanie LLM agenta
            st.markdown(response)
    
    st.session_state.chat_history.append({"role": "assistant", "content": response})
```

---

## 3. NASA Exoplanet Archive — Pozyskiwanie Danych

### 3.1 Czym jest TAP?
Table Access Protocol (TAP) to standard IVOA pozwalający na wykonywanie zapytań SQL-like (w dialekcie ADQL) do zdalnych baz astronomicznych przez HTTP.

**Endpoint:** `https://exoplanetarchive.ipac.caltech.edu/TAP/sync`

### 3.2 Implementacja Klienta TAP

```python
# nasa_client.py
import requests
import pandas as pd
import io
from typing import Optional

NASA_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

def query_nasa_archive(adql_query: str, format: str = "csv") -> pd.DataFrame:
    """
    Wykonuje zapytanie ADQL do NASA Exoplanet Archive.
    
    Args:
        adql_query: Zapytanie w języku ADQL (Astronomical Data Query Language)
        format: Format odpowiedzi ('csv', 'votable', 'json')
    
    Returns:
        DataFrame z wynikami zapytania
    
    Raises:
        requests.HTTPError: Błąd komunikacji z API
        ValueError: Pusta odpowiedź lub nieprawidłowy format
    """
    params = {
        "query": adql_query,
        "format": format
    }
    
    response = requests.get(NASA_TAP_URL, params=params, timeout=30)
    response.raise_for_status()
    
    if not response.text.strip():
        raise ValueError("Pusta odpowiedź z NASA Archive")
    
    df = pd.read_csv(io.StringIO(response.text))
    return df


def get_planet_data(planet_name: str) -> Optional[pd.Series]:
    """Pobiera pełne dane konkretnej planety."""
    query = f"""
    SELECT pl_name, pl_radj, pl_bmassj, pl_orbsmax, pl_orbper,
           pl_insol, pl_eqt, pl_dens,
           st_teff, st_rad, st_lum, st_mass
    FROM pscomppars
    WHERE pl_name = '{planet_name}'
    """
    df = query_nasa_archive(query)
    if df.empty:
        return None
    return df.iloc[0]


def get_habitable_candidates(
    radius_min: float = 0.5,
    radius_max: float = 2.5,
    insol_min: float = 0.2,
    insol_max: float = 2.0,
    teff_min: int = 2500,
    teff_max: int = 7000
) -> pd.DataFrame:
    """Pobiera planety o ziemskich rozmiarach w strefie zamieszkiwalnej."""
    query = f"""
    SELECT pl_name, pl_radj, pl_bmassj, pl_orbsmax, pl_orbper,
           pl_insol, pl_eqt, pl_dens,
           st_teff, st_rad, st_lum, st_mass,
           disc_year, discoverymethod
    FROM pscomppars
    WHERE pl_radj BETWEEN {radius_min} AND {radius_max}
      AND pl_insol BETWEEN {insol_min} AND {insol_max}
      AND st_teff BETWEEN {teff_min} AND {teff_max}
      AND pl_radj IS NOT NULL
      AND st_teff IS NOT NULL
    ORDER BY pl_insol ASC
    """
    return query_nasa_archive(query)


def get_all_confirmed_planets() -> pd.DataFrame:
    """Pobiera pełny katalog potwierdzonych planet (dane treningowe dla CTGAN)."""
    query = """
    SELECT pl_name, pl_radj, pl_bmassj, pl_orbsmax, pl_orbper,
           pl_insol, pl_eqt, pl_dens, pl_orbeccen,
           st_teff, st_rad, st_lum, st_mass, st_dens,
           sy_dist
    FROM pscomppars
    WHERE pl_radj IS NOT NULL
      AND st_teff IS NOT NULL
      AND pl_orbsmax IS NOT NULL
    """
    return query_nasa_archive(query)
```

### 3.3 Przykład Użycia

```python
# Pobranie danych Proxima Centauri b
planet = get_planet_data("Proxima Cen b")
print(f"Promień: {planet['pl_radj']} R_J")
print(f"Temp gwiazdy: {planet['st_teff']} K")
print(f"Naświetlenie: {planet['pl_insol']} S_⊕")

# Pobranie kandydatów do habitabilności
candidates = get_habitable_candidates()
print(f"Znaleziono {len(candidates)} kandydatów")
```

### 3.4 Kluczowe Kolumny i Konwersje Jednostek

```python
# Konwersje jednostek — NASA podaje w jednostkach Jowisza, potrzebujemy ziemskich

def jupiter_to_earth_radius(r_jup: float) -> float:
    """R_Jupiter → R_Earth (1 R_J = 11.209 R_E)"""
    return r_jup * 11.209

def jupiter_to_earth_mass(m_jup: float) -> float:
    """M_Jupiter → M_Earth (1 M_J = 317.83 M_E)"""
    return m_jup * 317.83

def log_solar_lum_to_watts(log_lum: float) -> float:
    """log(L/L_☉) → Waty"""
    L_sun = 3.828e26  # Luminozja Słońca [W]
    return 10**log_lum * L_sun

def solar_to_meters_radius(r_solar: float) -> float:
    """R_☉ → metry"""
    R_sun = 6.957e8  # Promień Słońca [m]
    return r_solar * R_sun

def au_to_meters(a_au: float) -> float:
    """AU → metry"""
    AU = 1.496e11  # Jednostka astronomiczna [m]
    return a_au * AU
```

---

## 4. Obliczenia Astrofizyczne (Pure Python)

### 4.1 Temperatura Równowagowa

```python
# astro_physics.py
import numpy as np
from typing import Tuple

# ─── Stałe fizyczne ───
STEFAN_BOLTZMANN = 5.670374419e-8  # σ [W/(m²·K⁴)]
L_SUN = 3.828e26                   # Luminozja Słońca [W]
R_SUN = 6.957e8                    # Promień Słońca [m]
AU = 1.496e11                      # Jednostka astronomiczna [m]
S_EARTH = 1361.0                   # Stała słoneczna [W/m²]


def equilibrium_temperature(
    stellar_temp: float,     # T_* [K]
    stellar_radius: float,   # R_* [R_☉]
    semi_major_axis: float,  # a [AU]
    albedo: float = 0.3,     # A_B
    tidally_locked: bool = False
) -> float:
    """
    Oblicza temperaturę równowagi radiacyjnej planety.
    
    Formula:
        T_eq = T_* × √(R_* / (2a)) × (1 - A_B)^(1/4)
    
    Dla planety uwięzionej pływowo re-emisja zachodzi z jednej półkuli,
    co modyfikuje czynnik redystrybucji.
    """
    R_star_m = stellar_radius * R_SUN    # Konwersja do metrów
    a_m = semi_major_axis * AU           # Konwersja do metrów
    
    # Czynnik redystrybucji ciepła
    if tidally_locked:
        # Re-emisja z jednej półkuli → czynnik √2 zamiast 2
        redistribution = np.sqrt(2)
    else:
        # Pełna redystrybucja → uśrednienie po całej sferze
        redistribution = 2.0
    
    T_eq = stellar_temp * np.sqrt(R_star_m / (redistribution * a_m)) * (1 - albedo)**0.25
    
    return round(T_eq, 2)
```

### 4.2 Strumień Naświetlenia

```python
def stellar_flux(
    stellar_temp: float,      # K
    stellar_radius: float,    # R_☉
    semi_major_axis: float    # AU
) -> Tuple[float, float]:
    """
    Oblicza strumień naświetlenia i normalizację względem Ziemi.
    
    Returns:
        (S_abs [W/m²], S_norm [S_⊕])
    """
    R_star_m = stellar_radius * R_SUN
    a_m = semi_major_axis * AU
    
    # Luminozja gwiazdy (prawo Stefana-Boltzmanna)
    L_star = 4 * np.pi * R_star_m**2 * STEFAN_BOLTZMANN * stellar_temp**4
    
    # Strumień na orbicie planety
    S = L_star / (4 * np.pi * a_m**2)
    
    # Normalizacja względem Ziemi
    S_norm = S / S_EARTH
    
    return round(S, 2), round(S_norm, 4)
```

### 4.3 Earth Similarity Index (ESI)

```python
def compute_esi(
    radius: float,          # R_⊕
    density: float,         # g/cm³
    escape_vel: float,      # km/s
    surface_temp: float     # K
) -> float:
    """
    Oblicza wieloparametrowy Earth Similarity Index.
    
    ESI = ∏(1 - |x_i - x_ref| / (x_i + x_ref))^(w_i/n)
    
    Wartości referencyjne Ziemi:
        R = 1.0 R_⊕, ρ = 5.51 g/cm³, v_e = 11.19 km/s, T_s = 288 K
    """
    # Wartości referencyjne (Ziemia)
    ref = {
        "radius":      1.0,     # R_⊕
        "density":     5.51,    # g/cm³
        "escape_vel":  11.19,   # km/s
        "temperature": 288.0    # K
    }
    
    # Wagi (Schulze-Makuch et al. 2011)
    weights = {
        "radius":      0.57,
        "density":     1.07,
        "escape_vel":  0.70,
        "temperature": 5.58
    }
    
    values = {
        "radius": radius,
        "density": density,
        "escape_vel": escape_vel,
        "temperature": surface_temp
    }
    
    n = len(values)
    esi = 1.0
    
    for key in values:
        x = values[key]
        x_ref = ref[key]
        w = weights[key]
        
        similarity = 1.0 - abs(x - x_ref) / (x + x_ref)
        esi *= similarity ** (w / n)
    
    return round(esi, 4)


def estimate_escape_velocity(mass_earth: float, radius_earth: float) -> float:
    """
    Szacuje prędkość ucieczki planety.
    
    v_e = 11.19 × √(M/R) [km/s] (w jednostkach ziemskich)
    """
    return 11.19 * np.sqrt(mass_earth / radius_earth)


def estimate_density(mass_earth: float, radius_earth: float) -> float:
    """
    Szacuje gęstość planety.
    
    ρ = 5.51 × M / R³ [g/cm³] (w jednostkach ziemskich)
    """
    return 5.51 * mass_earth / radius_earth**3
```

### 4.4 Kompletny Pipeline Obliczeniowy

```python
def compute_full_analysis(
    stellar_temp: float,
    stellar_radius: float,
    planet_radius_jup: float,
    planet_mass_jup: float,
    semi_major_axis: float,
    albedo: float = 0.3,
    tidally_locked: bool = True
) -> dict:
    """
    Pełna analiza parametrów planetarnych — Pipeline od surowych danych do wyników.
    """
    # Konwersja jednostek
    R_earth = planet_radius_jup * 11.209
    M_earth = planet_mass_jup * 317.83
    
    # Temperatura równowagowa
    T_eq = equilibrium_temperature(
        stellar_temp, stellar_radius, semi_major_axis, albedo, tidally_locked
    )
    
    # Strumień naświetlenia
    S_abs, S_norm = stellar_flux(stellar_temp, stellar_radius, semi_major_axis)
    
    # Parametry pochodne
    density = estimate_density(M_earth, R_earth)
    v_escape = estimate_escape_velocity(M_earth, R_earth)
    
    # ESI
    esi = compute_esi(R_earth, density, v_escape, T_eq)
    
    # Ocena habitabilności
    in_hz = 0.2 <= S_norm <= 2.0
    has_liquid_water = 200 <= T_eq <= 380  # z uwzgl. efektu cieplarnianego
    
    return {
        "T_eq_K": T_eq,
        "flux_Wm2": S_abs,
        "flux_earth": S_norm,
        "radius_earth": round(R_earth, 3),
        "mass_earth": round(M_earth, 3),
        "density_gcc": round(density, 3),
        "escape_vel_kms": round(v_escape, 3),
        "ESI": esi,
        "in_habitable_zone": in_hz,
        "liquid_water_possible": has_liquid_water,
        "tidally_locked": tidally_locked
    }
```

---

## 5. Pydantic — Walidacja i Bariery Bezpieczeństwa

### 5.1 Czym jest Pydantic?
Pydantic to biblioteka walidacji danych w Pythonie oparta na type hints. Definiujesz schemat danych jako klasę, a Pydantic **automatycznie** waliduje typy, zakresy i zależności. Jeśli dane są niepoprawne — rzuca wyjątek.

W naszym systemie Pydantic chroni przed halucynacjami LLM i niefizycznymi wynikami.

### 5.2 Modele Walidacyjne

```python
# validators.py
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Literal
import numpy as np


class StellarParameters(BaseModel):
    """Walidator parametrów gwiazdy macierzystej."""
    
    name: str = Field(min_length=1, max_length=100)
    teff: float = Field(
        ge=2300, le=10000,
        description="Temperatura efektywna gwiazdy [K]"
    )
    radius: float = Field(
        ge=0.08, le=100.0,
        description="Promień gwiazdy [R_☉]"
    )
    mass: float = Field(
        ge=0.08, le=150.0,
        description="Masa gwiazdy [M_☉]"
    )
    luminosity: Optional[float] = Field(
        default=None, ge=-5.0, le=7.0,
        description="log(L/L_☉)"
    )
    
    @field_validator("teff")
    @classmethod
    def validate_teff_physical(cls, v: float) -> float:
        """Gwiazdy ciągu głównego: M0 ≈ 3800K, O5 ≈ 45000K."""
        if v < 2300:
            raise ValueError(
                f"T_eff={v}K poniżej limitu brązowego karła (~2300K)"
            )
        return v


class PlanetaryParameters(BaseModel):
    """Walidator parametrów planetarnych z ograniczeniami fizycznymi."""
    
    name: str = Field(min_length=1, max_length=100)
    radius_earth: float = Field(
        ge=0.3, le=25.0,
        description="Promień planety [R_⊕]"
    )
    mass_earth: Optional[float] = Field(
        default=None, ge=0.01, le=5000.0,
        description="Masa planety [M_⊕]"
    )
    semi_major_axis: float = Field(
        ge=0.001, le=1000.0,
        description="Półoś wielka orbity [AU]"
    )
    albedo: float = Field(
        ge=0.0, le=1.0,
        description="Albedo Bonda"
    )
    tidally_locked: bool = False
    orbital_period: Optional[float] = Field(
        default=None, ge=0.01,
        description="Okres orbitalny [dni]"
    )
    insol: Optional[float] = Field(
        default=None, ge=0.0, le=10000.0,
        description="Strumień naświetlenia [S_⊕]"
    )
    
    @field_validator("radius_earth")
    @classmethod
    def validate_not_star(cls, v: float) -> float:
        """Ponad ~25 R_⊕ to już gwiazda, nie planeta."""
        if v > 25.0:
            raise ValueError(
                f"Promień {v} R_⊕ wykracza poza definicję planety"
            )
        return v
    
    @model_validator(mode="after")
    def validate_mass_radius_consistency(self):
        """Sprawdza empiryczną relację masa-promień (Chen & Kipping 2017)."""
        if self.mass_earth is not None and self.radius_earth is not None:
            # Ziemskie planety: R ≈ M^0.27
            if self.radius_earth < 4.0:  # Typ ziemski
                expected_r = self.mass_earth ** 0.27
                ratio = self.radius_earth / expected_r if expected_r > 0 else float("inf")
                if ratio < 0.2 or ratio > 5.0:
                    raise ValueError(
                        f"Relacja masa-promień (M={self.mass_earth}, R={self.radius_earth}) "
                        f"drastycznie odbiega od empirycznej (oczekiwane R≈{expected_r:.2f})"
                    )
        return self


class SimulationOutput(BaseModel):
    """Walidator wyjść z silnika fizycznego."""
    
    T_eq_K: float = Field(ge=10.0, le=5000.0)
    ESI: float = Field(ge=0.0, le=1.0)
    flux_earth: float = Field(ge=0.0)
    
    temperature_map: Optional[list] = None
    
    @field_validator("T_eq_K")
    @classmethod
    def validate_temperature_thermodynamics(cls, v: float) -> float:
        """Temperatura musi być fizycznie sensowna."""
        if v < 2.7:  # Temperatura CMB (promieniowanie tła)
            raise ValueError(
                f"T={v}K poniżej temperatury promieniowania tła kosmicznego (2.7K)"
            )
        return v
    
    @field_validator("temperature_map")
    @classmethod
    def validate_temperature_map(cls, v: Optional[list]) -> Optional[list]:
        """Walidacja macierzy temperatur z ELM."""
        if v is not None:
            arr = np.array(v)
            if np.any(arr < 0):
                raise ValueError("Macierz temperatur zawiera wartości ujemne!")
            if np.any(arr > 5000):
                raise ValueError("Macierz temperatur zawiera niefizycznie wysokie wartości!")
            if np.any(np.isnan(arr)):
                raise ValueError("Macierz temperatur zawiera NaN!")
        return v
```

### 5.3 Użycie Walidatorów w Pipeline

```python
# Walidacja danych wejściowych od użytkownika / LLM
try:
    planet = PlanetaryParameters(
        name="Proxima Cen b",
        radius_earth=1.07,
        mass_earth=1.27,
        semi_major_axis=0.0485,
        albedo=0.3,
        tidally_locked=True
    )
    print(f"✅ Walidacja przeszła: {planet.name}")
except Exception as e:
    print(f"❌ Odrzucono: {e}")

# Walidacja wyjścia z ELM
try:
    result = SimulationOutput(
        T_eq_K=234.0,
        ESI=0.87,
        flux_earth=0.65,
        temperature_map=temperature_matrix.tolist()
    )
    print("✅ Wynik fizycznie poprawny")
except Exception as e:
    print(f"❌ ELM wygenerował niefizyczny wynik: {e}")
    # → Fallback do modelu algebraicznego
```

---

## 6. CTGAN — Augmentacja Danych Tabelarycznych

### 6.1 Czym jest CTGAN?
CTGAN (Conditional Tabular GAN) to sieć generatywna zaprojektowana specjalnie do syntezy danych tabelarycznych. W odróżnieniu od standardowych GANów (obrazy), CTGAN radzi sobie z:
- Mieszanymi typami kolumn (ciągłe + kategoryczne)
- Multimodalnymi rozkładami (np. bimodalna masa planet)
- Silnymi korelacjami między kolumnami

### 6.2 Instalacja i Import

```python
pip install ctgan

from ctgan import CTGAN
import pandas as pd
```

### 6.3 Pełna Implementacja Pipeline Augmentacji

```python
# data_augmentation.py
import pandas as pd
import numpy as np
from ctgan import CTGAN
from typing import List, Optional


class ExoplanetDataAugmenter:
    """
    Augmentacja danych egzoplanetarnych za pomocą CTGAN.
    Rozwiązuje problem ekstremalnego niezbalansowania klas.
    """
    
    def __init__(self, epochs: int = 300, batch_size: int = 500):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model: Optional[CTGAN] = None
        self.discrete_columns: List[str] = []
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Przygotowuje dane z NASA Archive do treningu CTGAN.
        - Usuwa NaN-y (CTGAN ich nie obsługuje)
        - Tworzy kolumnę 'habitable' (target do warunkowania)
        - Normalizuje jednostki
        """
        # Wybierz kluczowe kolumny
        columns = [
            "pl_radj", "pl_bmassj", "pl_orbsmax", "pl_orbper",
            "pl_insol", "pl_eqt", "st_teff", "st_rad", "st_mass"
        ]
        
        data = df[columns].copy()
        
        # Usuń wiersze z brakującymi danymi
        data = data.dropna()
        
        # Konwersja do jednostek ziemskich
        data["pl_radj"] = data["pl_radj"] * 11.209      # R_J → R_E
        data["pl_bmassj"] = data["pl_bmassj"] * 317.83   # M_J → M_E
        
        # Etykieta habitabilności (uproszczona)
        data["habitable"] = (
            (data["pl_radj"].between(0.5, 2.5)) &
            (data["pl_insol"].between(0.2, 2.0)) &
            (data["st_teff"].between(2500, 7000))
        ).astype(int)
        
        # Rename dla czytelności
        data.columns = [
            "radius_earth", "mass_earth", "semi_major_axis_au", "period_days",
            "insol_earth", "t_eq_K", "star_teff_K", "star_radius_solar",
            "star_mass_solar", "habitable"
        ]
        
        self.discrete_columns = ["habitable"]
        
        print(f"Dane po filtracji: {len(data)} wierszy")
        print(f"  Habitabilne: {data['habitable'].sum()}")
        print(f"  Niehabitabilne: {(~data['habitable'].astype(bool)).sum()}")
        
        return data
    
    def train(self, data: pd.DataFrame):
        """
        Trenuje CTGAN na danych planetarnych.
        
        Kluczowe parametry:
        - epochs: Liczba epok jak w standardowym GAN (300–1000)
        - batch_size: Rozmiar batcha (500 dla małych zbiorów)
        - generator_dim: Architektura generatora (256, 256)
        - discriminator_dim: Architektura dyskryminatora (256, 256)
        """
        self.model = CTGAN(
            epochs=self.epochs,
            batch_size=min(self.batch_size, len(data)),
            generator_dim=(256, 256),
            discriminator_dim=(256, 256),
            generator_lr=2e-4,
            discriminator_lr=2e-4,
            discriminator_steps=1,
            verbose=True
        )
        
        print(f"Trening CTGAN: {self.epochs} epok...")
        self.model.fit(data, discrete_columns=self.discrete_columns)
        print("Trening zakończony.")
    
    def generate_synthetic_planets(
        self,
        n_samples: int = 5000,
        condition_column: str = "habitable",
        condition_value: int = 1
    ) -> pd.DataFrame:
        """
        Generuje syntetyczne planety, warunkując na habitabilności.
        """
        if self.model is None:
            raise RuntimeError("Model CTGAN nie jest wytrenowany!")
        
        # Generowanie z warunkiem
        from ctgan import CTGAN
        
        synthetic = self.model.sample(
            n=n_samples,
            condition_column=condition_column,
            condition_value=condition_value
        )
        
        return synthetic
    
    def validate_synthetic_data(self, synthetic: pd.DataFrame) -> pd.DataFrame:
        """
        Post-hoc walidacja fizyczna syntetycznych planet.
        Odrzuca wiersze naruszające prawa fizyki.
        """
        original_len = len(synthetic)
        
        # Ograniczenia fizyczne
        synthetic = synthetic[
            (synthetic["radius_earth"] > 0.3) &
            (synthetic["radius_earth"] < 25.0) &
            (synthetic["mass_earth"] > 0.01) &
            (synthetic["mass_earth"] < 5000) &
            (synthetic["semi_major_axis_au"] > 0.001) &
            (synthetic["star_teff_K"] > 2300) &
            (synthetic["star_teff_K"] < 10000) &
            (synthetic["t_eq_K"] > 10) &
            (synthetic["t_eq_K"] < 3000) &
            (synthetic["period_days"] > 0.1)
        ]
        
        filtered_len = len(synthetic)
        print(f"Walidacja: {original_len} → {filtered_len} "
              f"({original_len - filtered_len} odrzuconych)")
        
        return synthetic
    
    def save_model(self, path: str = "models/ctgan_exoplanets.pkl"):
        """Zapisuje wytrenowany model."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
    
    def load_model(self, path: str = "models/ctgan_exoplanets.pkl"):
        """Ładuje wytrenowany model."""
        import pickle
        with open(path, "rb") as f:
            self.model = pickle.load(f)
```

### 6.4 Użycie Kompletne

```python
from nasa_client import get_all_confirmed_planets
from data_augmentation import ExoplanetDataAugmenter

# 1. Pobranie danych z NASA
raw_data = get_all_confirmed_planets()
print(f"Pobrano {len(raw_data)} planet z NASA Archive")

# 2. Inicjalizacja augmentera
augmenter = ExoplanetDataAugmenter(epochs=500, batch_size=500)

# 3. Przygotowanie danych
clean_data = augmenter.prepare_data(raw_data)

# 4. Trening CTGAN
augmenter.train(clean_data)

# 5. Generowanie 5000 syntetycznych planet habitabilnych
synthetic_habitable = augmenter.generate_synthetic_planets(
    n_samples=5000,
    condition_column="habitable",
    condition_value=1
)

# 6. Walidacja fizyczna
validated = augmenter.validate_synthetic_data(synthetic_habitable)
print(f"Gotowe: {len(validated)} fizycznie poprawnych syntetycznych planet")

# 7. Połączenie z oryginalnymi danymi
training_data = pd.concat([clean_data, validated], ignore_index=True)
print(f"Zbilansowany zbiór: {len(training_data)} wierszy")

# 8. Zapis modelu
augmenter.save_model("models/ctgan_exoplanets.pkl")
```

---

## 7. Extreme Learning Machines (ELM) — Surogat Klimatyczny

### 7.1 Czym jest ELM?
ELM to jednowarstwowa sieć neuronowa, w której losowe wagi wejściowe są zamrożone, a jedyne parametry do wyuczenia (wagi wyjściowe) oblicza się analitycznie przez pseudoodwrotność macierzy. Trening trwa **milisekundy** zamiast godzin.

### 7.2 Implementacja z `scikit-elm`

```python
pip install scikit-elm
```

```python
# elm_surrogate.py
import numpy as np
from skelm import ELMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from typing import Tuple, List


class ELMClimateSurrogate:
    """
    Zespół Maszyn Ekstremalnego Uczenia emulujący rozkłady temperatur.
    Zastępuje dni obliczeń GCM predykcją w milisekundach.
    """
    
    # Rozdzielczość siatki klimatycznej
    N_LAT = 32   # Szerokość geograficzna
    N_LON = 64   # Długość geograficzna
    
    def __init__(self, n_ensemble: int = 10, n_neurons: int = 500, alpha: float = 1e-4):
        """
        Args:
            n_ensemble: Liczba modeli w zespole (stabilizacja predykcji)
            n_neurons: Liczba neuronów w warstwie ukrytej
            alpha: Parametr regularyzacji (Tikhonov / Ridge)
        """
        self.n_ensemble = n_ensemble
        self.n_neurons = n_neurons
        self.alpha = alpha
        self.models: List[ELMRegressor] = []
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
    
    def _create_model(self) -> ELMRegressor:
        """Tworzy pojedynczy model ELM."""
        return ELMRegressor(
            n_neurons=self.n_neurons,
            alpha=self.alpha,
            ufunc="tanh"        # Funkcja aktywacji: tangens hiperboliczny
            # alternatywy: "sigm" (sigmoid), "relu", "rbf_l2" (radial basis)
        )
    
    def prepare_features(self, data: dict) -> np.ndarray:
        """
        Konwersja parametrów planetarnych na wektor cech.
        
        Wejścia:
            - radius_earth: promień [R_⊕]
            - mass_earth: masa [M_⊕]
            - semi_major_axis_au: półoś wielka [AU]
            - star_teff_K: temperatura gwiazdy [K]
            - star_radius_solar: promień gwiazdy [R_☉]
            - insol_earth: naświetlenie [S_⊕]
            - albedo: albedo Bonda
            - tidally_locked: uwięzienie pływowe (0/1)
        """
        features = np.array([
            data["radius_earth"],
            data["mass_earth"],
            data["semi_major_axis_au"],
            data["star_teff_K"],
            data["star_radius_solar"],
            data["insol_earth"],
            data.get("albedo", 0.3),
            float(data.get("tidally_locked", 1))
        ]).reshape(1, -1)
        
        return features
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Trenuje zespół ELM na danych (X → wektor cech, y → spłaszczona mapa temperatur).
        
        Args:
            X: Macierz cech [N_samples, N_features]
            y: Macierz docelowa [N_samples, N_LAT * N_LON]
        """
        print(f"Trening ELM Ensemble: {self.n_ensemble} modeli × {self.n_neurons} neuronów")
        
        # Normalizacja
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        self.models = []
        for i in range(self.n_ensemble):
            model = self._create_model()
            model.fit(X_scaled, y_scaled)
            self.models.append(model)
            print(f"  Model {i+1}/{self.n_ensemble} — wytrenowany")
        
        print("Zespół ELM gotowy.")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predykcja zespołowa — uśrednienie wyników K modeli.
        
        Returns:
            Macierz temperatur [N_LAT, N_LON] w Kelwinach
        """
        X_scaled = self.scaler_X.transform(X)
        
        predictions = []
        for model in self.models:
            pred = model.predict(X_scaled)
            predictions.append(pred)
        
        # Uśrednienie zespołowe
        ensemble_pred = np.mean(predictions, axis=0)
        
        # Odwrócenie normalizacji
        temp_flat = self.scaler_y.inverse_transform(ensemble_pred)
        
        # Reshape do siatki 2D
        temp_map = temp_flat.reshape(self.N_LAT, self.N_LON)
        
        return temp_map
    
    def predict_from_params(self, params: dict) -> np.ndarray:
        """Predykcja bezpośrednio z parametrów planetarnych."""
        X = self.prepare_features(params)
        return self.predict(X)
    
    def save(self, path: str = "models/elm_ensemble.pkl"):
        """Zapisuje cały zespół i skalery."""
        bundle = {
            "models": self.models,
            "scaler_X": self.scaler_X,
            "scaler_y": self.scaler_y,
            "n_ensemble": self.n_ensemble,
            "n_neurons": self.n_neurons,
            "n_lat": self.N_LAT,
            "n_lon": self.N_LON,
        }
        with open(path, "wb") as f:
            pickle.dump(bundle, f)
        print(f"Model zapisany: {path}")
    
    def load(self, path: str = "models/elm_ensemble.pkl"):
        """Ładuje zespół z pliku."""
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        self.models = bundle["models"]
        self.scaler_X = bundle["scaler_X"]
        self.scaler_y = bundle["scaler_y"]
        self.n_ensemble = bundle["n_ensemble"]
        self.n_neurons = bundle["n_neurons"]
        self.N_LAT = bundle["n_lat"]
        self.N_LON = bundle["n_lon"]
        print(f"Załadowano: {self.n_ensemble} modeli × {self.n_neurons} neuronów")
```

### 7.3 Implementacja ELM od Zera (bez biblioteki)

Dla pełnego zrozumienia — czysta implementacja NumPy:

```python
# elm_from_scratch.py
import numpy as np

class PureELM:
    """
    Implementacja ELM od zera w NumPy.
    Pokazuje mechanizm pseudoodwrotności Moore'a-Penrose'a.
    """
    
    def __init__(self, n_neurons: int = 500, activation: str = "tanh", C: float = 1e4):
        self.n_neurons = n_neurons
        self.activation = activation
        self.C = C          # Parametr regularyzacji (im większy → mniejsza regularyzacja)
        self.W_in = None    # Wagi wejściowe (zamrożone)
        self.bias = None    # Biasy (zamrożone)
        self.beta = None    # Wagi wyjściowe (wyuczone)
    
    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Funkcja aktywacji warstwy ukrytej."""
        if self.activation == "tanh":
            return np.tanh(x)
        elif self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation == "relu":
            return np.maximum(0, x)
        else:
            raise ValueError(f"Nieznana aktywacja: {self.activation}")
    
    def _hidden_layer_output(self, X: np.ndarray) -> np.ndarray:
        """
        Oblicza macierz H (hidden layer output matrix).
        H[i,j] = g(W_in[j] · X[i] + bias[j])
        """
        return self._activate(X @ self.W_in + self.bias)
    
    def fit(self, X: np.ndarray, T: np.ndarray):
        """
        Trening ELM — analityczne wyznaczenie wag wyjściowych.
        
        Krok 1: Losowa inicjalizacja W_in, bias (ZAMROŻONE)
        Krok 2: Obliczenie H = g(X @ W_in + bias)
        Krok 3: beta = (H^T @ H + I/C)^(-1) @ H^T @ T
        """
        n_features = X.shape[1]
        
        # Krok 1: Losowa inicjalizacja (te wagi NIE są trenowane!)
        np.random.seed(None)  # Różny seed dla każdego modelu w zespole
        self.W_in = np.random.randn(n_features, self.n_neurons) * 0.5
        self.bias = np.random.randn(1, self.n_neurons) * 0.5
        
        # Krok 2: Obliczenie macierzy ukrytej warstwy
        H = self._hidden_layer_output(X)
        
        # Krok 3: Rozwiązanie analityczne z regularyzacją Tikhonov
        # β = (H^T H + I/C)^{-1} H^T T
        I = np.eye(self.n_neurons)
        self.beta = np.linalg.solve(
            H.T @ H + I / self.C,   # Macierz kwadratowa (L × L)
            H.T @ T                  # Prawa strona
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predykcja: y = H @ beta"""
        H = self._hidden_layer_output(X)
        return H @ self.beta


# ─── Ensemble ───
class ELMEnsemble:
    """Zespół K niezależnych ELM — redukcja wariancji."""
    
    def __init__(self, K: int = 10, **elm_kwargs):
        self.K = K
        self.models = [PureELM(**elm_kwargs) for _ in range(K)]
    
    def fit(self, X, T):
        for i, model in enumerate(self.models):
            model.fit(X, T)
        return self
    
    def predict(self, X) -> np.ndarray:
        predictions = np.array([m.predict(X) for m in self.models])
        return predictions.mean(axis=0)  # Uśrednienie zespołowe
```

### 7.4 Generowanie Danych Treningowych (bez ROCKE-3D)

Jeśli nie mamy dostępu do zrzutów GCM, generujemy analityczne aproksymacje map temperatur:

```python
def generate_analytical_training_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generuje syntetyczne dane treningowe dla ELM na podstawie
    analitycznego modelu klimatu planety uwięzionej pływowo.
    """
    N_LAT, N_LON = 32, 64
    
    # Siatka geograficzna
    lat = np.linspace(-np.pi/2, np.pi/2, N_LAT)
    lon = np.linspace(0, 2*np.pi, N_LON)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    
    X_features = []
    y_maps = []
    
    for _ in range(n_samples):
        # Losowe parametry planetarne
        star_teff = np.random.uniform(2500, 7000)
        star_radius = np.random.uniform(0.1, 2.0)
        semi_major = np.random.uniform(0.01, 2.0)
        albedo = np.random.uniform(0.05, 0.7)
        radius_earth = np.random.uniform(0.5, 2.5)
        mass_earth = radius_earth ** (1/0.27)  # Relacja Chen & Kipping
        insol = (star_teff/5778)**4 * (star_radius)**2 / semi_major**2
        locked = np.random.choice([0, 1], p=[0.3, 0.7])
        
        # Temperatura globalna
        T_eq = star_teff * np.sqrt(star_radius * 6.957e8 / (2 * semi_major * 1.496e11)) \
               * (1 - albedo)**0.25
        
        if locked:
            # Profil planety uwięzionej pływowo
            cos_zenith = np.cos(LAT) * np.cos(LON - np.pi)  # Punkt substelarny w (0, π)
            cos_zenith = np.clip(cos_zenith, 0, 1)
            
            T_max = T_eq * 1.4  # Punkt substelarny gorętszy
            T_min = max(T_eq * 0.3, 40)  # Nocna strona — zimna, ale >0K
            
            temp_map = T_min + (T_max - T_min) * cos_zenith**0.25
        else:
            # Profil planety rotującej (gradient lat)
            temp_map = T_eq * (1 + 0.15 * np.cos(LAT))
            # Dodanie szumu konwekcyjnego
            temp_map += np.random.normal(0, T_eq * 0.02, temp_map.shape)
        
        # Zapis
        features = [radius_earth, mass_earth, semi_major, star_teff,
                     star_radius, insol, albedo, locked]
        X_features.append(features)
        y_maps.append(temp_map.flatten())
    
    return np.array(X_features), np.array(y_maps)
```

### 7.5 Kompletny Pipeline Treningowy

```python
# Generowanie danych
X_train, y_train = generate_analytical_training_data(n_samples=5000)

# Trening
surrogate = ELMClimateSurrogate(n_ensemble=10, n_neurons=500, alpha=1e-4)
surrogate.train(X_train, y_train)

# Predykcja dla Proxima Centauri b
params = {
    "radius_earth": 1.07,
    "mass_earth": 1.27,
    "semi_major_axis_au": 0.0485,
    "star_teff_K": 3042,
    "star_radius_solar": 0.141,
    "insol_earth": 0.65,
    "albedo": 0.3,
    "tidally_locked": 1
}

temp_map = surrogate.predict_from_params(params)
print(f"Rozmiar mapy: {temp_map.shape}")  # (32, 64)
print(f"Zakres temperatur: {temp_map.min():.1f} K – {temp_map.max():.1f} K")

# Zapis
surrogate.save("models/elm_ensemble.pkl")
```

---

## 8. Ollama — Lokalne Hostowanie LLM

### 8.1 Czym jest Ollama?
Ollama to narzędzie do uruchamiania dużych modeli językowych (LLM) **lokalnie** na własnym komputerze. Działa jak prywatny serwer API kompatybilny z formatem OpenAI. Eliminuje zależność od chmury i opłaty.

### 8.2 Instalacja

```powershell
# Windows: Pobranie instalatora z https://ollama.com/download
# Po instalacji dostępna jest komenda `ollama` w terminalu

# Sprawdzenie instalacji
ollama --version
```

### 8.3 Pobieranie Modeli

```powershell
# Pobranie modelu (jednorazowe — potem cache)
ollama pull llama3.1:8b         # LLaMA 3.1 8B (4.7 GB)
ollama pull qwen2.5:7b          # Qwen 2.5 7B (4.4 GB) — świetny function calling
ollama pull mistral:7b           # Mistral 7B (4.1 GB)

# Listowanie zainstalowanych modeli
ollama list

# Uruchomienie chatu (test)
ollama run qwen2.5:7b
```

### 8.4 API Ollama — Uruchomienie jako Serwer

```powershell
# Ollama automatycznie startuje serwer na http://localhost:11434
# Weryfikacja:
curl http://localhost:11434/api/tags
```

### 8.5 Użycie z Pythona (biblioteka `ollama`)

```python
# llm_client.py
import ollama

# ─── Prosty prompt ───
response = ollama.chat(
    model="qwen2.5:7b",
    messages=[
        {
            "role": "system",
            "content": "Jesteś ekspertem od astrofizyki egzoplanet. Odpowiadaj precyzyjnie."
        },
        {
            "role": "user",
            "content": "Jakie warunki panują na Proxima Centauri b?"
        }
    ]
)
print(response["message"]["content"])


# ─── Streaming (odpowiedź token po tokenie) ───
stream = ollama.chat(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": "Wyjaśnij uwięzienie pływowe."}],
    stream=True
)
for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)


# ─── Structured Output (JSON mode) ───
response = ollama.chat(
    model="qwen2.5:7b",
    messages=[
        {
            "role": "system",
            "content": "Odpowiedz JSON-em z polami: planet_name, t_eq_K, esi, habitable (bool)."
        },
        {
            "role": "user",
            "content": "Przeanalizuj planetę TRAPPIST-1e."
        }
    ],
    format="json"  # Wymusza odpowiedź JSON
)
import json
data = json.loads(response["message"]["content"])
print(data)
```

### 8.6 Tworzenie Własnego Modelu (Modelfile)

```dockerfile
# Modelfile — customowy agent astronomiczny
FROM qwen2.5:7b

# System prompt wbudowany w model
SYSTEM """
Jesteś AstroAgent — autonomicznym agentem do analizy egzoplanet.
Twoje zadania:
1. Analizować parametry planetarne
2. Obliczać wskaźniki habitabilności (ESI, T_eq)
3. Wywoływać narzędzia systemowe
4. Nigdy nie podawać niefizycznych wartości

Zasady fizyczne (BEZWZGLĘDNE):
- Temperatura > 0 K (nie istnieje ujemna)
- ESI ∈ [0, 1]
- Albedo ∈ [0, 1]
- Planety skaliste: R < 2.5 R_⊕
"""

# Parametry modelu
PARAMETER temperature 0.3    # Niższa temperatura = bardziej deterministyczny
PARAMETER top_p 0.9
PARAMETER num_ctx 8192       # Okno kontekstowe
```

```powershell
# Budowanie modelu
ollama create astro-agent -f Modelfile

# Uruchomienie
ollama run astro-agent
```

---

## 9. LangChain / smolagents — Orkiestracja Agentowa

### 9.1 LangChain z Ollama — Setup Agenta

```python
# agent_setup.py
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from typing import Annotated
import json


# ─── Krok 1: Inicjalizacja LLM ───
llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0.3,
    num_ctx=8192,
    base_url="http://localhost:11434"
)


# ─── Krok 2: Definicja Narzędzi (Tools) ───
@tool
def query_nasa_archive(planet_name: str) -> str:
    """
    Pobiera dane planety z NASA Exoplanet Archive.
    Użyj tego narzędzia gdy użytkownik pyta o konkretną egzoplanetę.
    
    Args:
        planet_name: Nazwa planety (np. 'Proxima Cen b', 'TRAPPIST-1e')
    """
    from nasa_client import get_planet_data
    
    data = get_planet_data(planet_name)
    if data is None:
        return f"Nie znaleziono planety: {planet_name}"
    return data.to_json()


@tool
def compute_habitability(
    stellar_temp: float,
    stellar_radius: float,
    planet_radius_jup: float,
    planet_mass_jup: float,
    semi_major_axis: float,
    albedo: float = 0.3,
    tidally_locked: bool = True
) -> str:
    """
    Oblicza wskaźniki habitabilności planety (T_eq, ESI, strumień naświetlenia).
    Użyj po pobraniu danych z NASA.
    
    Args:
        stellar_temp: Temperatura gwiazdy [K]
        stellar_radius: Promień gwiazdy [R_☉]
        planet_radius_jup: Promień planety [R_Jupiter]
        planet_mass_jup: Masa planety [M_Jupiter]
        semi_major_axis: Półoś wielka [AU]
        albedo: Albedo Bonda (domyślnie 0.3)
        tidally_locked: Czy uwięziona pływowo (domyślnie True)
    """
    from astro_physics import compute_full_analysis
    
    result = compute_full_analysis(
        stellar_temp, stellar_radius, planet_radius_jup,
        planet_mass_jup, semi_major_axis, albedo, tidally_locked
    )
    return json.dumps(result, indent=2)


@tool
def run_climate_simulation(
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
    Uruchamia symulację klimatu ELM i generuje mapę temperatur.
    Użyj do predykcji rozkładu temperatur na powierzchni planety.
    
    Returns:
        JSON z zakresem temperatur i statystykami
    """
    from elm_surrogate import ELMClimateSurrogate
    import streamlit as st
    
    model = st.cache_resource(lambda: ELMClimateSurrogate().load("models/elm_ensemble.pkl"))()
    
    params = {
        "radius_earth": radius_earth,
        "mass_earth": mass_earth,
        "semi_major_axis_au": semi_major_axis_au,
        "star_teff_K": star_teff_K,
        "star_radius_solar": star_radius_solar,
        "insol_earth": insol_earth,
        "albedo": albedo,
        "tidally_locked": tidally_locked
    }
    
    temp_map = model.predict_from_params(params)
    
    result = {
        "T_min_K": float(temp_map.min()),
        "T_max_K": float(temp_map.max()),
        "T_mean_K": float(temp_map.mean()),
        "T_std_K": float(temp_map.std()),
        "map_shape": list(temp_map.shape),
        "has_liquid_water": bool(273 <= temp_map.mean() <= 373)
    }
    
    return json.dumps(result, indent=2)


# ─── Krok 3: Lista narzędzi ───
tools = [query_nasa_archive, compute_habitability, run_climate_simulation]


# ─── Krok 4: Prompt agenta ───
system_prompt = """Jesteś AstroAgent — autonomicznym asystentem do analizy egzoplanet.

TWOJE MOŻLIWOŚCI:
1. Pobieranie danych planet z NASA Exoplanet Archive
2. Obliczanie wskaźników habitabilności (ESI, T_eq, SEPHI)
3. Uruchamianie symulacji klimatycznych (mapy temperatur)

PROCEDURA DZIAŁANIA:
1. Gdy użytkownik pyta o planetę → NAJPIERW pobierz dane z NASA (query_nasa_archive)
2. Na podstawie danych → oblicz habitabilność (compute_habitability)
3. Jeśli potrzeba → uruchom symulację klimatu (run_climate_simulation)
4. Podsumuj wyniki w przystępny sposób

ZASADY:
- Zawsze podawaj źródło danych (NASA Exoplanet Archive)
- Nigdy nie wymyślaj wartości — używaj narzędzi
- Temperatury zawsze w Kelwinach
- ESI ∈ [0, 1], 1.0 = identyczna z Ziemią
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


# ─── Krok 5: Utworzenie agenta ───
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,        # Pokazuje łańcuch rozumowania
    max_iterations=10,   # Limit pętli (zabezpieczenie)
    handle_parsing_errors=True
)
```

### 9.2 Użycie Agenta

```python
# Wywołanie agenta
response = agent_executor.invoke({
    "input": "Przeanalizuj planetę TRAPPIST-1e i oceń jej habitabilność",
    "chat_history": []
})
print(response["output"])

# Agent AUTONOMICZNIE:
# 1. Wywoła query_nasa_archive("TRAPPIST-1 e")
# 2. Z wynikami wywoła compute_habitability(...)
# 3. Opcjonalnie uruchomi run_climate_simulation(...)
# 4. Podsumuje wyniki w języku naturalnym
```

### 9.3 Alternatywa: smolagents (Hugging Face)

```python
# Lżejsza alternatywa — minimalistyczny framework agentowy
pip install smolagents

from smolagents import CodeAgent, Tool, LiteLLMModel

# Model (Ollama przez LiteLLM)
model = LiteLLMModel(model_id="ollama_chat/qwen2.5:7b", api_base="http://localhost:11434")

# Definicja narzędzi
class NASAQueryTool(Tool):
    name = "query_nasa"
    description = "Pobiera dane planety z NASA Exoplanet Archive"
    inputs = {"planet_name": {"type": "string", "description": "Nazwa planety"}}
    output_type = "string"
    
    def forward(self, planet_name: str) -> str:
        from nasa_client import get_planet_data
        data = get_planet_data(planet_name)
        return data.to_json() if data is not None else "Nie znaleziono"

# Agent
agent = CodeAgent(
    tools=[NASAQueryTool()],
    model=model,
    max_steps=5
)

# Wywołanie
result = agent.run("Jakie parametry ma planeta Kepler-442b?")
```

### 9.4 Integracja Agenta ze Streamlit

```python
# W app.py — sekcja czatu
def run_agent(user_input: str) -> str:
    """Uruchamia agenta i zwraca odpowiedź tekstową."""
    try:
        response = agent_executor.invoke({
            "input": user_input,
            "chat_history": st.session_state.get("langchain_history", [])
        })
        return response["output"]
    except Exception as e:
        # Graceful degradation — fallback na prosty prompt
        fallback = ollama.chat(
            model="qwen2.5:7b",
            messages=[
                {"role": "system", "content": "Jesteś astronomem. Odpowiadaj krótko."},
                {"role": "user", "content": user_input}
            ]
        )
        return f"⚠️ Agent w trybie uproszczonym:\n\n{fallback['message']['content']}"
```

---

## 10. Plotly — Wizualizacja 3D Sfery Planetarnej

### 10.1 Teoria Renderowania Sfery

Sfera jest generowana trygonometrycznie z dwuwymiarowej siatki kątów, a temperatura jest mapowana jako tekstura koloru.

### 10.2 Implementacja Kompletna

```python
# visualization.py
import numpy as np
import plotly.graph_objects as go
from typing import Optional, Tuple


def create_3d_globe(
    temperature_map: np.ndarray,
    planet_name: str = "Exoplanet",
    resolution: int = 64,
    colorscale: Optional[list] = None
) -> go.Figure:
    """
    Renderuje interaktywną sferę 3D z mapą temperatury.
    
    Args:
        temperature_map: Macierz temperatur [N_lat, N_lon] w Kelwinach
        planet_name: Nazwa planety (tytuł)
        resolution: Rozdzielczość siatki sferycznej
        colorscale: Niestandardowa skala kolorów (None = domyślna naukowa)
    
    Returns:
        go.Figure — interaktywny wykres 3D
    """
    
    # ─── Krok 1: Siatka sferyczna ───
    N_lat = temperature_map.shape[0]
    N_lon = temperature_map.shape[1]
    
    # Kąty
    theta = np.linspace(0, 2 * np.pi, N_lon)      # Długość geograficzna [0, 2π]
    phi = np.linspace(-np.pi / 2, np.pi / 2, N_lat)  # Szerokość [-π/2, π/2]
    
    THETA, PHI = np.meshgrid(theta, phi)
    
    # ─── Krok 2: Transformacja do współrzędnych kartezjańskich ───
    r = 1.0  # Promień sfery (znormalizowany)
    X = r * np.cos(PHI) * np.cos(THETA)
    Y = r * np.cos(PHI) * np.sin(THETA)
    Z = r * np.sin(PHI)
    
    # ─── Krok 3: Naukowo uzasadniona skala kolorów ───
    if colorscale is None:
        colorscale = [
            [0.00, "#1a0533"],    # Ultra zimno (< 100K) — głęboki fiolet
            [0.15, "#08306b"],    # Lód stały (~150K) — granat
            [0.30, "#2171b5"],    # Zamrożone (~200K) — błękit
            [0.45, "#4292c6"],    # Granica lodu (~240K) — jasny błękit
            [0.50, "#41ab5d"],    # Punkt zamarzania wody (273K) — zieleń
            [0.60, "#78c679"],    # Umiarkowane (~290K) — jasna zieleń
            [0.70, "#fee08b"],    # Ciepło (~310K) — żółty
            [0.80, "#fc8d59"],    # Gorąco (~340K) — pomarańcz
            [0.90, "#d73027"],    # Bardzo gorąco (~380K) — czerwień
            [1.00, "#67001f"],    # Ekstremalnie (~500K+) — ciemna czerwień
        ]
    
    # ─── Krok 4: Obiekt Surface ───
    T_min = temperature_map.min()
    T_max = temperature_map.max()
    
    surface = go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=temperature_map,   # Kluczowe: mapowanie temperatury na kolor
        colorscale=colorscale,
        cmin=T_min,
        cmax=T_max,
        colorbar=dict(
            title=dict(text="Temperatura [K]", font=dict(size=14)),
            ticksuffix=" K",
            len=0.75,
            thickness=20,
            x=1.02
        ),
        hovertemplate=(
            "<b>%{surfacecolor:.1f} K</b><br>"
            "(%{customdata[0]:.1f}°, %{customdata[1]:.1f}°)"
            "<extra></extra>"
        ),
        customdata=np.stack([
            np.degrees(PHI),   # Szerokość w stopniach
            np.degrees(THETA)  # Długość w stopniach
        ], axis=-1),
        lighting=dict(
            ambient=0.4,
            diffuse=0.6,
            specular=0.15,
            roughness=0.8,
            fresnel=0.1
        ),
        lightposition=dict(x=1000, y=0, z=0)  # Gwiazda po lewej
    )
    
    # ─── Krok 5: Layout ───
    fig = go.Figure(data=[surface])
    
    fig.update_layout(
        title=dict(
            text=f"🪐 {planet_name} — Rozkład Temperatury Powierzchniowej",
            font=dict(size=18)
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.5, y=0.5, z=0.5),
                up=dict(x=0, y=0, z=1)
            ),
            bgcolor="black"
        ),
        paper_bgcolor="black",
        font=dict(color="white"),
        width=800,
        height=700,
        margin=dict(l=0, r=80, t=60, b=0)
    )
    
    return fig


def create_2d_heatmap(
    temperature_map: np.ndarray,
    planet_name: str = "Exoplanet"
) -> go.Figure:
    """
    Fallback — płaska heatmapa 2D (gdy 3D jest za wolne).
    """
    import plotly.express as px
    
    fig = px.imshow(
        temperature_map,
        labels=dict(x="Długość geograficzna [°]", y="Szerokość geograficzna [°]", color="T [K]"),
        x=np.linspace(0, 360, temperature_map.shape[1]),
        y=np.linspace(-90, 90, temperature_map.shape[0]),
        color_continuous_scale="RdYlBu_r",
        origin="lower",
        aspect="auto"
    )
    
    fig.update_layout(
        title=f"{planet_name} — Mapa Temperatur (2D)",
        coloraxis_colorbar=dict(title="T [K]"),
        width=800, height=400
    )
    
    return fig
```

### 10.3 Generowanie Analitycznej Mapy (Bez ELM)

```python
def generate_eyeball_map(
    T_eq: float,
    tidally_locked: bool = True,
    n_lat: int = 64,
    n_lon: int = 128
) -> np.ndarray:
    """
    Generuje analityczną mapę temperatury typu 'Eyeball'.
    Fallback gdy ELM jest niedostępny.
    """
    lat = np.linspace(-np.pi/2, np.pi/2, n_lat)
    lon = np.linspace(0, 2*np.pi, n_lon)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    
    if tidally_locked:
        # Punkt substelarny w (0, π)
        cos_zenith = np.cos(LAT) * np.cos(LON - np.pi)
        cos_zenith = np.clip(cos_zenith, 0, 1)
        
        T_sub = T_eq * 1.4    # Punkt substelarny
        T_night = max(T_eq * 0.3, 40)  # Nocna strona
        
        temp_map = T_night + (T_sub - T_night) * cos_zenith**0.25
    else:
        # Planeta rotująca — gradient szerokościowy
        temp_map = T_eq * (1 + 0.15 * np.cos(LAT))
    
    return temp_map
```

### 10.4 Użycie w Streamlit

```python
# W app.py
import streamlit as st
from visualization import create_3d_globe, create_2d_heatmap, generate_eyeball_map

# Generowanie mapy
temp_map = generate_eyeball_map(T_eq=255, tidally_locked=True)

# Renderowanie 3D
fig_3d = create_3d_globe(temp_map, planet_name="Proxima Centauri b")

# Wyświetlenie w Streamlit
st.plotly_chart(fig_3d, use_container_width=True)

# Fallback 2D (przycisk)
if st.button("Przełącz na widok 2D"):
    fig_2d = create_2d_heatmap(temp_map, planet_name="Proxima Centauri b")
    st.plotly_chart(fig_2d, use_container_width=True)
```

---

## 11. DeepXDE — PINN (Opcjonalnie)

### 11.1 Czym jest PINN?
Physics-Informed Neural Network to sieć neuronowa, która w funkcji kosztu ma wbudowane równanie różniczkowe. Sieć uczy się nie tylko z danych, ale i z fizyki.

### 11.2 Implementacja 1D Równania Ciepła

```python
# pinn_heat.py
"""
Rozwiązanie 1D równania propagacji ciepła na linii terminatora
(granica dzień-noc planety uwięzionej pływowo).

PDE: ρ·c_p · ∂T/∂t = κ · ∂²T/∂x² + S_abs(x) - σ·T⁴

Uproszczenie: Stan stacjonarny (∂T/∂t = 0)
→ κ · ∂²T/∂x² + S_abs(x) - σ·T⁴ = 0
"""
import deepxde as dde
import numpy as np

# Stałe fizyczne
SIGMA = 5.670374419e-8   # Stefan-Boltzmann [W/(m²·K⁴)]
KAPPA = 0.025             # Przewodnictwo cieplne [W/(m·K)] (atmosfera)

# Strumień gwiazdowy jako funkcja pozycji
S_MAX = 900.0  # W/m² (punkt substelarny)

def source_term(x):
    """Strumień absorbowany — max w x=0 (substelarny), 0 w x=π (antystelarny)."""
    return S_MAX * np.maximum(0, np.cos(x))

def pde(x, T):
    """
    Rezydualne PDE: κ·T'' + S(x) - σ·T⁴ = 0
    DeepXDE automatycznie oblicza pochodne (autograd).
    """
    dT_xx = dde.grad.hessian(T, x)  # ∂²T/∂x²
    
    S = S_MAX * dde.backend.maximum(0, dde.backend.cos(x))
    radiation = SIGMA * T**4
    
    return KAPPA * dT_xx + S - radiation

# Domena: x ∈ [0, π] (od punktu substelarnego do antystelarnego)
geom = dde.geometry.Interval(0, np.pi)

# Warunki brzegowe: T(0) = T_sub, T(π) = T_night
bc_left = dde.icbc.DirichletBC(geom, lambda x: 320.0, lambda x, on: np.isclose(x[0], 0))
bc_right = dde.icbc.DirichletBC(geom, lambda x: 80.0, lambda x, on: np.isclose(x[0], np.pi))

# Problem
data = dde.data.PDE(geom, pde, [bc_left, bc_right], num_domain=256, num_boundary=2)

# Sieć
net = dde.nn.FNN(
    [1] + [64] * 3 + [1],  # 1 wejście, 3 warstwy po 64, 1 wyjście
    "tanh",
    "Glorot normal"
)

# Trening
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(epochs=10000, display_every=1000)

# Predykcja
x_test = np.linspace(0, np.pi, 200).reshape(-1, 1)
T_pred = model.predict(x_test)

print(f"T(substelarny) = {T_pred[0,0]:.1f} K")
print(f"T(terminator)  = {T_pred[100,0]:.1f} K")
print(f"T(nocna)       = {T_pred[-1,0]:.1f} K")
```

---

## 12. Graceful Degradation — Implementacja Odporności

### 12.1 Wzorzec Try-Except z Fallbackiem

```python
# degradation.py
import time
import numpy as np
from typing import Callable, Any
import streamlit as st


class GracefulDegradation:
    """
    Manager graceful degradation — automatycznie przełącza się
    na prostsze tryby w przypadku awarii.
    """
    
    @staticmethod
    def run_with_fallback(
        primary_fn: Callable,
        fallback_fn: Callable,
        timeout: float = 10.0,
        label: str = "moduł"
    ) -> Any:
        """
        Uruchamia primary_fn. Jeśli się nie powiedzie lub przekroczy timeout,
        automatycznie przełącza na fallback_fn.
        """
        try:
            start = time.time()
            result = primary_fn()
            elapsed = time.time() - start
            
            if elapsed > timeout:
                st.warning(f"⚠️ {label} przekroczył timeout ({elapsed:.1f}s). "
                           f"Przełączanie na tryb uproszczony.")
                return fallback_fn()
            
            return result
            
        except Exception as e:
            st.warning(f"⚠️ {label} — awaria: {e}. "
                       f"Przełączanie na tryb uproszczony.")
            return fallback_fn()
    
    @staticmethod
    def validate_temperature_map(temp_map: np.ndarray) -> bool:
        """Sprawdza fizyczną poprawność mapy temperatur."""
        if temp_map is None:
            return False
        if np.any(np.isnan(temp_map)):
            return False
        if np.any(temp_map < 0):
            return False
        if np.any(temp_map > 5000):
            return False
        return True


# ─── Użycie w pipeline ───
def run_simulation_pipeline(params: dict) -> dict:
    """Pipeline z wbudowaną degradacją."""
    
    gd = GracefulDegradation()
    
    # Poziom 1: Próba ELM
    def elm_prediction():
        model = load_elm_model()
        return model.predict_from_params(params)
    
    # Poziom 2: Fallback — analityczny model
    def analytical_fallback():
        from astro_physics import equilibrium_temperature
        T_eq = equilibrium_temperature(
            params["star_teff_K"],
            params["star_radius_solar"],
            params["semi_major_axis_au"],
            params.get("albedo", 0.3),
            bool(params.get("tidally_locked", True))
        )
        from visualization import generate_eyeball_map
        return generate_eyeball_map(T_eq, tidally_locked=bool(params.get("tidally_locked", 1)))
    
    # Uruchomienie z fallbackiem
    temp_map = gd.run_with_fallback(
        primary_fn=elm_prediction,
        fallback_fn=analytical_fallback,
        timeout=5.0,
        label="ELM Surrogate"
    )
    
    # Walidacja wyniku
    if not gd.validate_temperature_map(temp_map):
        st.error("❌ Mapa temperatur niefizyczna. Używam modelu algebraicznego.")
        temp_map = analytical_fallback()
    
    # Poziom 3: Wizualizacja z fallbackiem 3D → 2D
    def render_3d():
        from visualization import create_3d_globe
        return create_3d_globe(temp_map, planet_name=params.get("name", "Exoplanet"))
    
    def render_2d():
        from visualization import create_2d_heatmap
        return create_2d_heatmap(temp_map, planet_name=params.get("name", "Exoplanet"))
    
    fig = gd.run_with_fallback(
        primary_fn=render_3d,
        fallback_fn=render_2d,
        timeout=8.0,
        label="Renderer 3D"
    )
    
    return {
        "temperature_map": temp_map,
        "figure": fig,
        "T_min": float(temp_map.min()),
        "T_max": float(temp_map.max()),
        "T_mean": float(temp_map.mean())
    }
```

---

## 13. Struktura Projektu i Integracja Końcowa

### 13.1 Zalecana Struktura Katalogów

```
HACK-4-SAGES/
├── app.py                      # Główna aplikacja Streamlit
├── requirements.txt            # Zależności
├── Modelfile                   # Konfiguracja Ollama
│
├── modules/
│   ├── __init__.py
│   ├── nasa_client.py          # Klient TAP do NASA Archive
│   ├── astro_physics.py        # Obliczenia fizyczne (T_eq, ESI, SEPHI)
│   ├── validators.py           # Walidatory Pydantic
│   ├── data_augmentation.py    # CTGAN augmentacja
│   ├── elm_surrogate.py        # ELM surrogate model
│   ├── agent_setup.py          # LangChain agent
│   ├── visualization.py        # Plotly 3D/2D rendering
│   ├── degradation.py          # Graceful degradation
│   └── pinn_heat.py            # (opcjonalne) DeepXDE PINN
│
├── models/
│   ├── elm_ensemble.pkl        # Wytrenowany zespół ELM
│   └── ctgan_exoplanets.pkl    # Wytrenowany CTGAN
│
├── data/
│   ├── nasa_cache/             # Cache danych NASA
│   └── training/               # Dane treningowe
│
├── notebooks/
│   └── exploration.ipynb       # Jupyter do eksploracji danych
│
└── docs/
    ├── raport_teoretyczny.md
    └── raport_praktyczny.md
```

### 13.2 Kompletna Aplikacja Streamlit (Szkielet)

```python
# app.py — Kompletny szkielet aplikacji
import streamlit as st
import pandas as pd
import numpy as np
import json

# ─── Importy modułów ───
from modules.nasa_client import get_planet_data, get_habitable_candidates
from modules.astro_physics import compute_full_analysis, equilibrium_temperature
from modules.validators import PlanetaryParameters, SimulationOutput
from modules.visualization import create_3d_globe, create_2d_heatmap, generate_eyeball_map
from modules.degradation import GracefulDegradation, run_simulation_pipeline

# ─── Konfiguracja ───
st.set_page_config(page_title="Exoplanetary Digital Twin", page_icon="🪐", layout="wide")

# ─── Cache modeli ───
@st.cache_resource
def load_agent():
    from modules.agent_setup import agent_executor
    return agent_executor

# ─── Inicjalizacja stanu ───
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_planet_data" not in st.session_state:
    st.session_state.current_planet_data = None
if "temperature_map" not in st.session_state:
    st.session_state.temperature_map = None

# ─── Header ───
st.title("🪐 Autonomiczny Cyfrowy Bliźniak Egzoplanetarny")

# ─── Tabs ───
tab_agent, tab_manual, tab_catalog = st.tabs([
    "🤖 Agent AI", "🎛️ Tryb Ręczny", "📊 Katalog Planet"
])

# ═══════════════ TAB 1: Agent AI ═══════════════
with tab_agent:
    st.subheader("Rozmowa z Agentem Astronomicznym")
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("Zapytaj o egzoplanetę..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Agent analizuje..."):
                try:
                    agent = load_agent()
                    response = agent.invoke({
                        "input": prompt,
                        "chat_history": []
                    })
                    answer = response["output"]
                except Exception as e:
                    answer = f"⚠️ Agent niedostępny. Błąd: {e}"
                
                st.markdown(answer)
        
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

# ═══════════════ TAB 2: Tryb Ręczny ═══════════════
with tab_manual:
    col_params, col_viz = st.columns([1, 2])
    
    with col_params:
        st.subheader("🎛️ Parametry")
        
        star_teff = st.slider("Temperatura gwiazdy [K]", 2500, 7500, 3042, 50)
        star_radius = st.slider("Promień gwiazdy [R☉]", 0.08, 3.0, 0.141, 0.01)
        planet_radius = st.slider("Promień planety [R⊕]", 0.5, 2.5, 1.07, 0.01)
        planet_mass = st.slider("Masa planety [M⊕]", 0.1, 15.0, 1.27, 0.1)
        semi_major = st.slider("Półoś wielka [AU]", 0.01, 2.0, 0.0485, 0.001, format="%.4f")
        albedo = st.slider("Albedo", 0.0, 1.0, 0.3, 0.01)
        locked = st.checkbox("Uwięzienie pływowe", True)
        
        if st.button("🚀 Symuluj", type="primary", use_container_width=True):
            # Walidacja Pydantic
            try:
                params = PlanetaryParameters(
                    name="Custom Planet",
                    radius_earth=planet_radius,
                    mass_earth=planet_mass,
                    semi_major_axis=semi_major,
                    albedo=albedo,
                    tidally_locked=locked
                )
                
                # Obliczenia
                T_eq = equilibrium_temperature(star_teff, star_radius, semi_major, albedo, locked)
                temp_map = generate_eyeball_map(T_eq, tidally_locked=locked)
                
                st.session_state.temperature_map = temp_map
                st.session_state.current_planet_data = {
                    "T_eq": T_eq,
                    "T_min": float(temp_map.min()),
                    "T_max": float(temp_map.max()),
                }
                
            except Exception as e:
                st.error(f"❌ Walidacja nieudana: {e}")
    
    with col_viz:
        st.subheader("🌍 Wizualizacja")
        
        if st.session_state.temperature_map is not None:
            data = st.session_state.current_planet_data
            
            # Metryki
            m1, m2, m3 = st.columns(3)
            m1.metric("T_eq", f"{data['T_eq']:.0f} K")
            m2.metric("T_min", f"{data['T_min']:.0f} K")
            m3.metric("T_max", f"{data['T_max']:.0f} K")
            
            # Renderowanie 3D / 2D
            view_mode = st.radio("Widok", ["3D Globe", "2D Heatmap"], horizontal=True)
            
            if view_mode == "3D Globe":
                fig = create_3d_globe(st.session_state.temperature_map, "Custom Planet")
            else:
                fig = create_2d_heatmap(st.session_state.temperature_map, "Custom Planet")
            
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════ TAB 3: Katalog ═══════════════
with tab_catalog:
    st.subheader("📊 Kandydaci do Habitabilności z NASA Archive")
    
    if st.button("📥 Pobierz dane z NASA"):
        with st.spinner("Pobieranie z NASA Exoplanet Archive..."):
            try:
                candidates = get_habitable_candidates()
                st.dataframe(candidates, use_container_width=True)
                st.success(f"Znaleziono {len(candidates)} kandydatów")
            except Exception as e:
                st.error(f"Błąd komunikacji z NASA: {e}")
```

### 13.3 Uruchomienie

```powershell
# 1. Uruchom Ollama (jeśli nie jest uruchomione)
ollama serve

# 2. Pobierz model (jednorazowo)
ollama pull qwen2.5:7b

# 3. Uruchom aplikację
streamlit run app.py
```

### 13.4 Kolejność Implementacji (Roadmap Hackathonowy)

| Godzina | Zadanie | Priorytet |
|---|---|---|
| 0–4 | Setup środowiska, `requirements.txt`, Streamlit skeleton | 🔴 Krytyczny |
| 4–10 | `nasa_client.py` + `astro_physics.py` (T_eq, ESI) | 🔴 Krytyczny |
| 10–16 | `validators.py` (Pydantic) + suwaki Streamlit | 🔴 Krytyczny |
| 16–24 | `visualization.py` (Plotly 3D globe) | 🔴 Krytyczny |
| 24–34 | Ollama + `agent_setup.py` (LangChain agent) | 🟡 Wysoki |
| 34–44 | `elm_surrogate.py` (trening + predykcja) | 🟡 Wysoki |
| 44–52 | `data_augmentation.py` (CTGAN) | 🟢 Średni |
| 52–60 | Integracja, testy, debug | 🔴 Krytyczny |
| 60–66 | `degradation.py` + polish UI | 🟡 Wysoki |
| 66–72 | PINN (opcjonalne) + prezentacja | 🟢 Opcjonalny |

---

## Podsumowanie Narzędzi

| Narzędzie | Rola w Systemie | Kluczowe API |
|---|---|---|
| **Streamlit** | UI / Frontend | `st.slider()`, `st.chat_input()`, `st.plotly_chart()` |
| **Pandas** | Przetwarzanie danych | `pd.read_csv()`, `df.dropna()`, `df.to_json()` |
| **NumPy** | Obliczenia numeryczne | `np.meshgrid()`, `np.cos()`, `np.linalg.solve()` |
| **Requests** | HTTP (NASA API) | `requests.get(url, params)` |
| **Pydantic** | Walidacja danych | `BaseModel`, `Field()`, `@field_validator` |
| **CTGAN** | Augmentacja danych | `CTGAN().fit()`, `.sample()` |
| **scikit-elm** | Surogat klimatyczny | `ELMRegressor().fit()`, `.predict()` |
| **Ollama** | Serwer LLM | `ollama.chat()`, `ollama pull` |
| **LangChain** | Orkiestracja agenta | `@tool`, `AgentExecutor`, `create_tool_calling_agent()` |
| **Plotly** | Wizualizacja 3D | `go.Surface()`, `surfacecolor`, `go.Figure()` |
| **DeepXDE** | PINN (opcja) | `dde.data.PDE()`, `dde.Model()` |
