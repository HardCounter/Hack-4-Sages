# Raport Teoretyczny: Autonomiczny Cyfrowy Bliźniak Egzoplanetarny

## Spis Treści
1. [Wprowadzenie i Kontekst Naukowy](#1-wprowadzenie-i-kontekst-naukowy)
2. [Fundamenty Astrofizyczne](#2-fundamenty-astrofizyczne)
3. [Aparat Matematyczny](#3-aparat-matematyczny)
4. [Teoria Maszyn Ekstremalnego Uczenia (ELM)](#4-teoria-maszyn-ekstremalnego-uczenia-elm)
5. [Generatywne Sieci Przeciwstawne dla Danych Tabelarycznych (CTGAN)](#5-generatywne-sieci-przeciwstawne-dla-danych-tabelarycznych-ctgan)
6. [Agentowe Modele Językowe i Orkiestracja](#6-agentowe-modele-językowe-i-orkiestracja)
7. [Pozyskiwanie Danych – NASA Exoplanet Archive](#7-pozyskiwanie-danych--nasa-exoplanet-archive)
8. [Wizualizacja 3D – Teoria Rzutowania Sferycznego](#8-wizualizacja-3d--teoria-rzutowania-sferycznego)
9. [Architektura Walidacji i Bezpieczeństwa](#9-architektura-walidacji-i-bezpieczeństwa)
10. [Graceful Degradation – Teoria Odporności Systemowej](#10-graceful-degradation--teoria-odporności-systemowej)
11. [Podsumowanie Teoretyczne](#11-podsumowanie-teoretyczne)
12. [Bibliografia](#12-bibliografia)

---

## 1. Wprowadzenie i Kontekst Naukowy

### 1.1 Problem Badawczy

Odkrycie ponad 5 700 potwierdzonych egzoplanet (stan na 2026 r.) postawiło przed astrofizyką fundamentalne pytanie: **które z tych światów mogą podtrzymywać życie?** Odpowiedź wymaga nie tylko detekcji ciała niebieskiego, lecz pełnej rekonstrukcji jego profilu klimatycznego — rozkładu temperatur powierzchniowych, cyrkulacji atmosferycznej, obecności ciekłej wody i stabilności termodynamicznej.

### 1.2 Ograniczenia Klasycznych Modeli Klimatycznych (GCM)

Modele Ogólnej Cyrkulacji (General Circulation Models, GCM), takie jak **NASA ROCKE-3D** (Resolving Orbital and Climate Keys of Earth and Extraterrestrial Environments with Dynamics), stanowią złoty standard symulacji klimatów planetarnych. Działają one na zasadzie:

- Dyskretyzacji atmosfery na trójwymiarową siatkę komórek (typowo $64 \times 32 \times 40$ — długość × szerokość × warstwa atmosferyczna)
- Rozwiązywania równań Naviera-Stokesa dla przepływu atmosferycznego w każdej komórce
- Iteracyjnego kroczenia czasowego (time-stepping) z krokiem $\Delta t \approx 30$ min do osiągnięcia stanu równowagi ($\sim 10^4$ lat symulacyjnych)

**Koszty obliczeniowe:** Pojedyncza symulacja ROCKE-3D dla jednej konfiguracji planetarnej wymaga **2–7 dni** obliczeń na superkomputerze klasy HPC. Oznacza to, że systematyczne przeszukanie tysięcy znanych egzoplanet jest obliczeniowo nieosiągalne.

### 1.3 Propozycja Rozwiązania: Surogaty Fizyczne

System Cyfrowego Bliźniaka Egzoplanetarnego zastępuje kosztowne symulacje GCM **surogatami fizycznymi** — szybkimi modelami uczenia maszynowego wytrenowanymi na wyjściach GCM, zdolnymi do odtworzenia rozkładów klimatycznych w ułamku sekundy. Ta klasa podejścia jest znana w literaturze jako **emulacja klimatyczna** (climate emulation) lub **meta-modelowanie** (surrogate modeling).

---

## 2. Fundamenty Astrofizyczne

### 2.1 Strefa Zamieszkiwalna (Habitable Zone, HZ)

Strefa zamieszkiwalna (circumstellar habitable zone) definiowana jest jako region wokół gwiazdy, w którym planeta o odpowiedniej atmosferze może utrzymać ciekłą wodę na powierzchni. Granice HZ zależą od:

- **Luminozji gwiazdy** $L_*$ [W]
- **Typu widmowego gwiazdy** (M, K, G, F — determinujący rozkład spektralny promieniowania)
- **Składu atmosfery planety** (efekt cieplarniany, albedo)

Klasyczna definicja Kopparapu et al. (2013) wyznacza granice poprzez krytyczne strumienie gwiazdowe:

$$S_{\text{eff}} = S_{\text{eff},\odot} + aT_* + bT_*^2 + cT_*^3 + dT_*^4$$

gdzie $T_* = T_{\text{eff}} - 5780$ K jest odchyleniem temperatury efektywnej gwiazdy od Słońca.

### 2.2 Uwięzienie Pływowe (Tidal Locking)

Znaczna część egzoplanet w strefie zamieszkiwalnej gwiazd typu M i K podlega **synchronicznej rotacji pływowej** — jedna półkula stale zwrócona jest ku gwieździe (permanentny dzień), druga pozostaje w wiecznej ciemności.

Warunek uwięzienia pływowego (tempo dyssypacji energii pływowej):

$$t_{\text{lock}} \approx \frac{\omega a^6 I Q}{3 G M_*^2 k_2 R_p^5}$$

gdzie:
- $\omega$ — prędkość kątowa rotacji,
- $a$ — półoś wielka orbity,
- $I$ — moment bezwładności planety,
- $Q$ — współczynnik dyssypacji pływowej,
- $k_2$ — liczba Love'a drugiego rzędu,
- $M_*$ — masa gwiazdy,
- $R_p$ — promień planety.

### 2.3 Stany Klimatyczne Planet Uwięzionych Pływowo

Symulacje GCM przewidują trzy charakterystyczne topologie klimatyczne dla planet w synchronicznej rotacji:

| Stan Klimatyczny | Opis | Warunek |
|---|---|---|
| **Gałka Oczna** (Eyeball) | Okrągły ocean w punkcie substelarnym otoczony globalnym lodem | Niski strumień gwiazdowy, słaba cyrkulacja |
| **Homar** (Lobster) | Równikowy pas oceanu rozciągający się symetrycznie | Umiarkowany strumień, silniejsze prądy równikowe |
| **Cieplarnia** (Greenhouse) | Globalny ocean, brak lodu | Wysoki strumień gwiazdowy, silny efekt cieplarniany |

Te nietrywialne rozkłady temperatur stanowią kluczowe wyzwanie dla surogatów fizycznych.

---

## 3. Aparat Matematyczny

### 3.1 Temperatura Równowagowa ($T_{eq}$)

Fundamentalnym modelem zerowego przybliżenia jest **temperatura równowagi radiacyjnej**, wynikająca z prawa Stefana-Boltzmanna. Planeta absorbuje promieniowanie gwiazdowe i re-emituje je jako promieniowanie cieplne:

$$T_{eq} = T_* \cdot \sqrt{\frac{R_*}{2a}} \cdot (1 - A_B)^{1/4}$$

gdzie:
- $T_*$ — temperatura efektywna gwiazdy [K],
- $R_*$ — promień gwiazdy [m],
- $a$ — półoś wielka orbity [m],
- $A_B$ — albedo Bonda planety (ułamek odbitego promieniowania).

**Założenia modelu:**
- Izotropowa re-emisja promieniowania (czynnik $\frac{1}{2}$ dla szybkiej rotacji → uśrednienie po całej powierzchni; czynnik $\frac{1}{\sqrt{2}}$ dla uwięzienia pływowego → re-emisja tylko z oświetlonej półkuli).
- Brak efektu cieplarnianego (temperatura brzegowa, „bare rock").

### 3.2 Strumień Naświetlenia ($S$)

Strumień naświetlenia (stellar flux, instellation) na orbicie planety:

$$S = \frac{L_*}{4\pi a^2} \quad [\text{W/m}^2]$$

Normalizacja względem stałej słonecznej ($S_\oplus = 1361 \text{ W/m}^2$):

$$S_{\text{norm}} = \frac{S}{S_\oplus}$$

### 3.3 Indeks Podobieństwa do Ziemi (ESI)

Earth Similarity Index (Schulze-Makuch et al., 2011) kwantyfikuje podobieństwo planety do Ziemi w przestrzeni wieloparametrowej:

$$\text{ESI} = \prod_{i=1}^{n} \left(1 - \left|\frac{x_i - x_{i,\oplus}}{x_i + x_{i,\oplus}}\right|\right)^{w_i / n}$$

gdzie:
- $x_i$ — wartość $i$-tego parametru planety (promień, gęstość, temperatura, prędkość ucieczki),
- $x_{i,\oplus}$ — wartość referencyjna dla Ziemi,
- $w_i$ — waga parametru (typowo: $w_R = 0.57$, $w_\rho = 1.07$, $w_{T_{eq}} = 5.58$, $w_{v_e} = 0.70$),
- $n$ — liczba parametrów.

**Interpretacja:** $\text{ESI} \in [0, 1]$, gdzie $\text{ESI} = 1$ oznacza identyczność z Ziemią. Przyjmuje się, że planety z $\text{ESI} > 0.8$ kwalifikują się jako potencjalnie habitabilne.

### 3.4 Wskaźnik SEPHI

Surface Exoplanetary Planetary Habitability Index (SEPHI) (Rodríguez-Mozos & Moya, 2017) rozszerza ESI o dodatkowe warunki fizyczne:

- **Kryterium atmosferyczne:** Planeta musi mieć wystarczającą masę, by utrzymać atmosferę ($v_e > v_{\text{th}}$ dla kluczowych gazów).
- **Kryterium termiczne:** Temperatura powierzchniowa musi mieścić się w zakresie $273 \leq T_s \leq 373$ K (faza ciekła wody).
- **Kryterium magnetyczne:** Opcjonalna ocena zdolności do utrzymania magnetosfery ochronnej.

$$\text{SEPHI} = f(T_s, M_p, R_p, a, L_*, \text{skład atmosfery})$$

### 3.5 Równanie Propagacji Ciepła (1D)

Dla jednowymiarowej analizy profilu termicznego wzdłuż linii terminatora (granica dzień-noc na planecie uwięzionej pływowo):

$$\rho c_p \frac{\partial T}{\partial t} = \kappa \frac{\partial^2 T}{\partial x^2} + S_{\text{abs}}(x) - \sigma T^4$$

gdzie:
- $\rho$ — gęstość atmosfery [kg/m³],
- $c_p$ — ciepło właściwe przy stałym ciśnieniu [J/(kg·K)],
- $\kappa$ — współczynnik przewodnictwa cieplnego [W/(m·K)],
- $S_{\text{abs}}(x)$ — absorbowany strumień gwiazdowy jako funkcja pozycji $x$,
- $\sigma T^4$ — emisja cieplna zgodna z prawem Stefana-Boltzmanna.

To równanie stanowi naturalny kandydat do rozwiązania metodą Physics-Informed Neural Networks (PINN), co stanowi opcjonalny moduł systemu (DeepXDE).

---

## 4. Teoria Maszyn Ekstremalnego Uczenia (ELM)

### 4.1 Definicja Formalna

Maszyna Ekstremalnego Uczenia (Extreme Learning Machine, Huang et al., 2006) jest jednoukrytowarstwową siecią neuronową ze sprzężeniem w przód (Single-hidden Layer Feedforward Network, SLFN), w której:

- **Wagi wejściowe** $\mathbf{W}_{\text{in}} \in \mathbb{R}^{d \times L}$ są losowo inicjalizowane i **zamrażane** (nie podlegają optymalizacji).
- **Biasy** $\mathbf{b} \in \mathbb{R}^{L}$ są losowe i zamrożone.
- **Wagi wyjściowe** $\boldsymbol{\beta} \in \mathbb{R}^{L \times m}$ są jedynymi parametrami do wyznaczenia.

Dla zbioru treningowego $\{(\mathbf{x}_j, \mathbf{t}_j)\}_{j=1}^{N}$, odpowiedź sieci:

$$f(\mathbf{x}) = \sum_{i=1}^{L} \beta_i \, g(\mathbf{w}_i \cdot \mathbf{x} + b_i)$$

gdzie $g(\cdot)$ to nieliniowa funkcja aktywacji (np. sigmoid, RBF, tangens hiperboliczny).

### 4.2 Rozwiązanie Analityczne (Pseudoodwrotność Moore'a-Penrose'a)

Kluczową przewagą ELM jest fakt, że problem sprowadza się do **liniowego układu równań** w przestrzeni ukrytej:

$$\mathbf{H} \boldsymbol{\beta} = \mathbf{T}$$

gdzie $\mathbf{H}$ to macierz ukrytej warstwy (hidden layer output matrix):

$$\mathbf{H} = \begin{bmatrix} g(\mathbf{w}_1 \cdot \mathbf{x}_1 + b_1) & \cdots & g(\mathbf{w}_L \cdot \mathbf{x}_1 + b_L) \\ \vdots & \ddots & \vdots \\ g(\mathbf{w}_1 \cdot \mathbf{x}_N + b_1) & \cdots & g(\mathbf{w}_L \cdot \mathbf{x}_N + b_L) \end{bmatrix}_{N \times L}$$

Optymalne wagi wyjściowe wyznacza się analitycznie:

$$\hat{\boldsymbol{\beta}} = \mathbf{H}^{\dagger} \mathbf{T}$$

gdzie $\mathbf{H}^{\dagger}$ jest pseudoodwrotnością Moore'a-Penrose'a macierzy $\mathbf{H}$:

$$\mathbf{H}^{\dagger} = (\mathbf{H}^T \mathbf{H})^{-1} \mathbf{H}^T \quad \text{gdy } N \geq L$$

lub

$$\mathbf{H}^{\dagger} = \mathbf{H}^T (\mathbf{H} \mathbf{H}^T)^{-1} \quad \text{gdy } N < L$$

### 4.3 Regularyzacja (Tikhonov / Ridge)

W celu zapobiegania przeuczeniu (overfitting), stosuje się regularyzację $L_2$:

$$\hat{\boldsymbol{\beta}} = \left(\mathbf{H}^T \mathbf{H} + \frac{\mathbf{I}}{C}\right)^{-1} \mathbf{H}^T \mathbf{T}$$

gdzie $C > 0$ jest parametrem regularyzacji kontrolującym kompromis bias-wariancja.

### 4.4 Przewaga Nad Sieciami Gradientowymi

| Cecha | ELM | MLP (backpropagation) |
|---|---|---|
| Metoda treningu | Analityczna (pseudoodwrotność) | Iteracyjna (SGD/Adam) |
| Czas treningu | $\mathcal{O}(NL^2)$ — sekundy | $\mathcal{O}(E \cdot N \cdot L^2)$ — minuty/godziny |
| Hiperparametry | $L$ (liczba neuronów), $C$ | Learning rate, batch size, epochs, architektura |
| Lokalne minima | Brak (rozwiązanie globalne) | TAK (powierzchnia strat niekonweksna) |
| Determinizm rozwiązania | Tak (dla ustalonego $\mathbf{W}_{\text{in}}$) | Nie (zależność od inicjalizacji) |

### 4.5 Zespół ELM (Ensemble of ELMs)

Losowość inicjalizacji wag wejściowych ELM wprowadza wariancję predykcji. Dla stabilizacji stosuje się **zespół** $K$ niezależnych ELM:

$$\hat{f}_{\text{ens}}(\mathbf{x}) = \frac{1}{K} \sum_{k=1}^{K} f_k(\mathbf{x})$$

Zgodnie z prawem wielkich liczb, wariancja zespołu maleje jak $\frac{\sigma^2}{K}$, co zapewnia stabilność predykcji. Typowo $K \in [5, 20]$.

### 4.6 Zastosowanie w Projekcie

**Wejścia modelu ELM:**
- Masa gwiazdy $M_*$ [$M_\odot$]
- Promień planety $R_p$ [$R_\oplus$]
- Półoś wielka orbity $a$ [AU]
- Albedo $A_B$
- Parametr uwięzienia pływowego (binarny: 0/1)
- Strumień naświetlenia $S$ [$S_\oplus$]

**Wyjścia modelu ELM:**
- Dwuwymiarowa macierz temperatury powierzchniowej $\mathbf{T} \in \mathbb{R}^{n_{\text{lat}} \times n_{\text{lon}}}$

Dane treningowe pochodzą z prekalkulowanych zrzutów ROCKE-3D (format NetCDF) lub syntetycznych próbek CTGAN.

---

## 5. Generatywne Sieci Przeciwstawne dla Danych Tabelarycznych (CTGAN)

### 5.1 Problem Niezbalansowania Klas

W katalogu NASA Exoplanet Archive:
- ~5 700 potwierdzonych egzoplanet
- ~60 planet z ESI > 0.8 (potencjalnie habitabilne)
- ~15 planet z pełnymi parametrami klimatycznymi

Tak ekstremalne niezbalansowanie klas ($\approx 1:380$ ratio) prowadzi do:
- **Kolapsu statystycznego** — klasyfikatory uczą się trywialne reguły „zawsze przewiduj klasę większościową"
- **Niedouczenia w regionach habitabilnych** — model ELM ma zbyt mało próbek, by odtworzyć topologie klimatyczne typu „gałka oczna"

### 5.2 Architektura CTGAN

Conditional Tabular GAN (Xu et al., 2019) składa się z:

**Generator** $G: \mathbb{R}^z \times \mathbb{R}^c \to \mathbb{R}^d$:
- Pobiera wektor szumu $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$ i wektor warunkowania $\mathbf{c}$
- Produkuje syntetyczny wiersz danych $\hat{\mathbf{x}} = G(\mathbf{z}, \mathbf{c})$
- Architektura: wielowarstwowy perceptron z normalizacją wsadową (batch normalization)

**Dyskryminator** $D: \mathbb{R}^d \times \mathbb{R}^c \to [0, 1]$:
- Ocenia, czy wiersz danych jest rzeczywisty czy syntetyczny
- Architektura: MLP z warstwami dropout i leaky ReLU

**Funkcja kosztu (Wasserstein loss z karą gradientową):**

$$\min_G \max_D \; \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[D(\mathbf{x})] - \mathbb{E}_{\mathbf{z} \sim p_z}[D(G(\mathbf{z}))] + \lambda \, \mathbb{E}_{\hat{\mathbf{x}}}[(\|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2 - 1)^2]$$

### 5.3 Mechanizmy Specyficzne dla Danych Tabelarycznych

1. **Mode-Specific Normalization:** Zmienne ciągłe (np. masa planety) modelowane są jako mieszanki gaussowskie. Każda wartość jest normalizowana względem najbliższego modu, co pozwala GANowi uchwycić multimodalne rozkłady.

2. **Conditional Vector i Training-by-Sampling:** Dyskretne zmienne warunkujące (np. typ gwiazdy) są kodowane jako wektory one-hot. Próbkowanie treningowe wymusza równomierne reprezentowanie wszystkich kategorii.

3. **Data Transformer:** Kolumny ciągłe → Variational Gaussian Mixture; kolumny dyskretne → one-hot encoding.

### 5.4 Zastosowanie w Projekcie

```
Dane NASA (N ~ 5700, habitabilne ~ 60)
    ↓
CTGAN.fit(dane_oryginalne)
    ↓
syntetyczne_planety = CTGAN.sample(n=5000, condition={"habitabilna": True})
    ↓
Walidacja fizyczna (Pydantic):
  - T_eq ∈ [150, 500] K
  - R_p ∈ [0.5, 2.5] R_⊕
  - S_norm ∈ [0.2, 2.0]
    ↓
Zbilansowany zbiór treningowy → ELM
```

**Kluczowa korzyść:** CTGAN generuje **fizycznie spójne** konfiguracje planetarne, zachowując korelacje między parametrami (np. antykorelacja promień–gęstość, korelacja luminozja–strefa zamieszkiwalna).

---

## 6. Agentowe Modele Językowe i Orkiestracja

### 6.1 Paradygmat ReAct (Reason + Act)

System wykorzystuje agentowy model językowy operujący w pętli ReAct (Yao et al., 2023):

```
Observation → Thought → Action → Observation → Thought → Action → ... → Final Answer
```

Cykl ten formalizuje się jako:

1. **Obserwacja ($o_t$):** Agent otrzymuje dane wejściowe (zapytanie użytkownika, wyniki API, output ELM)
2. **Myśl ($\tau_t$):** Agent generuje łańcuch rozumowania w języku naturalnym
3. **Działanie ($a_t$):** Agent wybiera narzędzie z predefiniowanego zestawu i wywołuje je z parametrami

### 6.2 Function Calling (Wywoływanie Funkcji)

Agent ma dostęp do zestawu narzędzi (tools), z których każde jest opisane schematem JSON:

| Narzędzie | Opis | Parametry wejściowe |
|---|---|---|
| `query_nasa_archive` | Zapytanie SQL do TAP | nazwa planety / filtry |
| `compute_equilibrium_temp` | Obliczenie $T_{eq}$ | $T_*$, $R_*$, $a$, $A_B$ |
| `compute_esi` | Obliczenie ESI | parametry planetarne |
| `run_elm_prediction` | Predykcja ELM | wektor cech |
| `generate_3d_globe` | Renderowanie sfery 3D | macierz temperatur |

Agent **samodzielnie** decyduje o kolejności i parametrach wywołań, bez hardkodowanych skryptów.

### 6.3 Modele Fundacyjne

**AstroLLaMA** (Nguyen et al., 2023):
- Bazuje na architekturze LLaMA
- Dotrenowany (fine-tuned) na ~300 000 abstraktów astronomicznych z arXiv
- Minimalna perpleksja w terminologii astrofizycznej ($\text{PPL}_{\text{astro}} \approx 6.1$ vs. $\text{PPL}_{\text{LLaMA}} \approx 32.0$)

**Alternatywy:**
- Qwen2.5-7B — silne zdolności function calling
- LLaMA 3.1-8B — uniwersalny model z dobrą generalizacją

**Hostowanie lokalne** za pomocą Ollama eliminuje zależność od chmurowych API i gwarantuje prywatność danych badawczych.

### 6.4 Framework Agentowy

**LangChain** lub **smolagents** (Hugging Face):
- Zarządzanie pamięcią konwersacji (context window management)
- Rejestracja narzędzi (tool registry)
- Parsowanie odpowiedzi agenta na struktury danych
- Obsługa łańcuchów wywołań i fallbacków

---

## 7. Pozyskiwanie Danych – NASA Exoplanet Archive

### 7.1 Protokół TAP (Table Access Protocol)

TAP jest standardem IVOA (International Virtual Observatory Alliance) umożliwiającym wykonywanie zapytań SQL na zdalnych bazach danych astronomicznych:

```
Endpoint: https://exoplanetarchive.ipac.caltech.edu/TAP/sync
Metoda: HTTP GET/POST
Język zapytań: ADQL (Astronomical Data Query Language) ≈ SQL z rozszerzeniami geometrycznymi
Format odpowiedzi: VOTable (XML), CSV, JSON
```

### 7.2 Kluczowe Parametry z Archiwum

| Parametr | Symbol | Kolumna TAP | Jednostka |
|---|---|---|---|
| Masa planety | $M_p$ | `pl_bmassj` | $M_J$ (masy Jowisza) |
| Promień planety | $R_p$ | `pl_radj` | $R_J$ |
| Półoś wielka | $a$ | `pl_orbsmax` | AU |
| Okres orbitalny | $P$ | `pl_orbper` | dni |
| Temperatura gwiazdy | $T_*$ | `st_teff` | K |
| Promień gwiazdy | $R_*$ | `st_rad` | $R_\odot$ |
| Luminozja gwiazdy | $L_*$ | `st_lum` | $\log(L_\odot)$ |
| Strumień naświetlenia | $S$ | `pl_insol` | $S_\oplus$ |

### 7.3 Przykład Zapytania ADQL

```sql
SELECT pl_name, pl_radj, pl_bmassj, pl_orbsmax, pl_insol,
       st_teff, st_rad, st_lum
FROM pscomppars
WHERE pl_radj BETWEEN 0.5 AND 2.5
  AND pl_insol BETWEEN 0.2 AND 2.0
  AND st_teff BETWEEN 2500 AND 7000
ORDER BY pl_insol ASC
```

To zapytanie zwraca planety o rozmiarach ziemskich w potencjalnych strefach zamieszkiwalnych.

---

## 8. Wizualizacja 3D – Teoria Rzutowania Sferycznego

### 8.1 Transformacja Współrzędnych

Dwuwymiarowa macierz temperatur $\mathbf{T}[\phi, \lambda]$ (szerokość × długość geograficzna) musi zostać zmapowana na powierzchnię sfery w przestrzeni 3D.

Transformacja ze współrzędnych sferycznych $(\phi, \lambda, r)$ na kartezjańskie $(x, y, z)$:

$$x = r \cdot \cos(\phi) \cdot \cos(\lambda)$$
$$y = r \cdot \cos(\phi) \cdot \sin(\lambda)$$
$$z = r \cdot \sin(\phi)$$

gdzie:
- $\phi \in [-\frac{\pi}{2}, \frac{\pi}{2}]$ — szerokość geograficzna,
- $\lambda \in [0, 2\pi]$ — długość geograficzna,
- $r$ — promień sfery (normalizowany do 1).

### 8.2 Dyskretyzacja Siatki

Sfera jest aproksymowana regularną siatką:

$$\phi_i = -\frac{\pi}{2} + i \cdot \frac{\pi}{N_\phi - 1}, \quad i = 0, \ldots, N_\phi - 1$$
$$\lambda_j = j \cdot \frac{2\pi}{N_\lambda - 1}, \quad j = 0, \ldots, N_\lambda - 1$$

Typowa rozdzielczość: $N_\phi \times N_\lambda = 64 \times 128$ (8192 wierzchołki siatki).

### 8.3 Mapowanie Tekstury Temperatury

Właściwość `surfacecolor` w Plotly `go.Surface` pozwala na bezpośrednie mapowanie macierzy skalarnej na kolor powierzchni. Zastosowane podejście:

1. Macierz $\mathbf{T}$ generowana przez ELM jest interpolowana na siatkę sfery.
2. Naukowo uzasadniona paleta kolorów (colorscale):
   - **Błękit** ($< 200$ K): Lód stały — strefy nocne
   - **Granat** ($200–273$ K): Granica fazy — regiony przejściowe
   - **Zieleń** ($273–323$ K): Ciekła woda — potencjalna habitabilność
   - **Czerwień/Pomarańcz** ($> 323$ K): Strefy substellarne — parowanie wody

### 8.4 Niestandardowe Topologie Klimatyczne

Dla planety uwięzionej pływowo z centrum substelarnym w $(\phi_0, \lambda_0) = (0, 0)$, rozkład temperatury aproksymowany jest jako:

$$T(\phi, \lambda) \approx T_{\text{max}} \cdot \max\left(0, \cos(\theta_{*})\right)^{1/4} + T_{\text{min}} \cdot \left(1 - \max(0, \cos(\theta_{*}))\right)$$

gdzie $\theta_*$ jest kątem zenitalnym gwiazdy:

$$\cos(\theta_*) = \cos(\phi) \cdot \cos(\lambda)$$

Ten profil naturalnie generuje topologię „gałki ocznej" — okrągły ciepły region w punkcie substelarnym z symetrycznym spadkiem temperatury.

---

## 9. Architektura Walidacji i Bezpieczeństwa

### 9.1 Problem Halucynacji Fizycznych

Duże modele językowe (LLM) wykazują inherentną skłonność do **konfabulacji** — generowania treści brzmiących wiarygodnie, lecz sprzecznych z faktami. W kontekście symulacji fizycznych zagrożenia obejmują:

- Generowanie temperatur naruszających zakres fizyczny ($T < 0$ K, $T > 10^6$ K)
- Przypisywanie niemożliwych kombinacji parametrów (np. planeta o masie Jowisza z promieniem Merkurego)
- Ignorowanie praw zachowania energii

### 9.2 Walidatory Pydantic

System wdraża **programistyczne bariery bezpieczeństwa** używając biblioteki Pydantic:

```python
class PlanetaryParameters(BaseModel):
    """Walidator parametrów planetarnych z ograniczeniami fizycznymi."""
    
    mass: float = Field(ge=0.01, le=13.0)    # Masy Jowisza
    radius: float = Field(ge=0.3, le=2.5)    # Promienie Ziemi
    semi_major_axis: float = Field(ge=0.01, le=10.0)  # AU
    albedo: float = Field(ge=0.0, le=1.0)    # Bezwymiarowe
    stellar_temp: float = Field(ge=2300, le=7500)  # Kelwiny
    
    @validator('radius')
    def validate_mass_radius_relation(cls, v, values):
        """Sprawdza zgodność z empiryczną relacją masa-promień."""
        if 'mass' in values:
            expected_r = values['mass'] ** 0.27  # Chen & Kipping (2017)
            if abs(v - expected_r) / expected_r > 0.5:
                raise ValueError("Relacja masa-promień naruszona")
        return v
```

### 9.3 Strategia Walidacji Kaskadowej

```
Wejście użytkownika → [Walidator Wejścia] → Agent LLM → [Walidator Pośredni]
→ Moduł Fizyczny (ELM) → [Walidator Wyjścia] → Wizualizacja
```

Na każdym etapie walidator Pydantic sprawdza:
1. **Typy danych** — zapewnienie poprawnych typów numerycznych
2. **Zakresy fizyczne** — ograniczenia termodynamiczne
3. **Spójność relacyjna** — korelacje między parametrami (masa-promień, luminozja-temperatura)
4. **Zachowanie energii** — bilans energetyczny musi być zbieżny

---

## 10. Graceful Degradation – Teoria Odporności Systemowej

### 10.1 Definicja

Graceful Degradation (łagodna degradacja) to paradygmat projektowania systemów, w którym awaria komponentu nie powoduje katastrofalnej awarii całego systemu, lecz **kontrolowane przejście do trybu o ograniczonej funkcjonalności**.

### 10.2 Ścieżki Degradacji w Systemie

| Poziom | Stan Normalny | Awaria | Tryb Degradacji |
|---|---|---|---|
| **L1** | Agent LLM + Function Calling | LLM niedostępny / timeout | Suwaki Streamlit + obliczenia algebraiczne (ESI, $T_{eq}$) |
| **L2** | Ensemble ELM → mapa 3D | ELM generuje $T < 0$ K | Fallback na analityczny profil $\cos^{1/4}(\theta_*)$ |
| **L3** | Sfera 3D (Plotly go.Surface) | Timeout przeglądarki (> 10s render) | Zrzut do heatmapy 2D (`px.imshow`) |
| **L4** | CTGAN augmentacja | GAN nie konwerguje | Użycie wyłącznie danych NASA (bez augmentacji) |

### 10.3 Formalizacja Detektora Awarii

Każdy moduł implementuje interfejs z monitorem stanu:

```
if module.execution_time > TIMEOUT or module.output violates CONSTRAINTS:
    activate_fallback(module.degradation_level)
    log_degradation_event(module, timestamp)
```

---

## 11. Podsumowanie Teoretyczne

### 11.1 Innowacyjność Podejścia

System łączy **cztery paradygmaty** sztucznej inteligencji w jedną spójną architekturę:

1. **AI Generatywna** (CTGAN) — rozwiązanie problemu rzadkości danych
2. **AI Agentowa** (LLM + ReAct) — autonomiczna orkiestracja bez hardkodowanych reguł
3. **AI Bezgradientowa** (ELM) — ultraszybki surogat klimatyczny
4. **AI Weryfikowalna** (Pydantic) — gwarancja spójności fizycznej

### 11.2 Porównanie z Istniejącymi Rozwiązaniami

| Aspekt | ROCKE-3D (NASA) | ExoSim (klasyczny) | **Nasz System** |
|---|---|---|---|
| Czas symulacji | 2–7 dni | Godziny | **< 1 sekunda** |
| Wymagania sprzętowe | Supercomputer | Klaster GPU | **Laptop / przeglądarka** |
| Interakcja | Skrypty CLI | GUI desktop | **Język naturalny** |
| Augmentacja danych | Brak | Brak | **CTGAN** |
| Walidacja fizyczna | Ręczna ekspertyza | Testy jednostkowe | **Automatyczna (Pydantic)** |

### 11.3 Ograniczenia Teoretyczne

1. **Wierność surogatów:** ELM emuluje wynik GCM, nie fizykę GCM — błędy systematyczne modelu treningowego propagują się.
2. **Przestrzeń ekstrapolacji:** Planety drastycznie różne od zbioru treningowego (np. super-Ziemie z atmosferą wodorową) mogą generować niepewne predykcje.
3. **Halucynacje rezydualne:** Mimo walidatorów Pydantic, LLM może generować subtlne błędy w interpretacji tekstowej wyników.
4. **Jakość danych syntetycznych:** CTGAN zachowuje korelacje statystyczne, ale nie gwarantuje spełnienia wszystkich praw fizyki wyższego rzędu.

### 11.4 Potencjał Rozwojowy

- Integracja z danymi z teleskopu JWST (James Webb Space Telescope) do kalibracji atmosferycznej
- Rozszerzenie ELM o architektury PINN (Physics-Informed Neural Networks) dla ścisłego wymuszania równań różniczkowych
- Transfer learning z modeli GCM nowej generacji (ExoCAM, LMD Generic)
- Rozszerzenie o modelowanie biosygnatur (O₂, CH₄, O₃ w widmach transmisyjnych)

---

## 12. Bibliografia

1. **Huang, G.-B., Zhu, Q.-Y., Siew, C.-K.** (2006). *Extreme Learning Machine: Theory and Applications.* Neurocomputing, 70(1-3), 489–501.
2. **Xu, L., Skoularidou, M., Cuesta-Infante, A., Veeramachaneni, K.** (2019). *Modeling Tabular Data using Conditional GAN.* NeurIPS 2019.
3. **Yao, S., Zhao, J., Yu, D., et al.** (2023). *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR 2023.
4. **Nguyen, T. D., et al.** (2023). *AstroLLaMA: Towards Specialized Foundation Models in Astronomy.* arXiv:2309.06126.
5. **Schulze-Makuch, D., et al.** (2011). *A Two-Tiered Approach to Assessing the Habitability of Exoplanets.* Astrobiology, 11(10), 1041–1052.
6. **Kopparapu, R. K., et al.** (2013). *Habitable Zones around Main-Sequence Stars.* The Astrophysical Journal, 765(2), 131.
7. **Rodríguez-Mozos, J. M., Moya, A.** (2017). *Statistical-Likelihood Exo-Planetary Habitability Index (SEPHI).* MNRAS, 471(4), 4628–4636.
8. **Chen, J., Kipping, D.** (2017). *Probabilistic Forecasting of the Masses and Radii of Other Worlds.* The Astrophysical Journal, 834(1), 17.
9. **Way, M. J., et al.** (2017). *Resolving Orbital and Climate Keys of Earth and Extraterrestrial Environments with Dynamics (ROCKE-3D).* The Astrophysical Journal Supplement, 231(1), 12.
10. **Pierrehumbert, R. T.** (2010). *Principles of Planetary Climate.* Cambridge University Press.
11. **Lu, L., Meng, X., Mao, Z., Karniadakis, G. E.** (2021). *DeepXDE: A Deep Learning Library for Solving Differential Equations.* SIAM Review, 63(1), 208–228.
12. **Turbet, M., et al.** (2016). *The Habitability of Proxima Centauri b: Environmental States and Observational Discriminants.* Astronomy & Astrophysics, 596, A112.
