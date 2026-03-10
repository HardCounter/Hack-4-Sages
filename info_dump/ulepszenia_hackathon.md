# Katalog Ulepszeń — Cyfrowy Bliźniak Egzoplanetarny (HACK-4-SAGES)

> **Cel dokumentu:** Systematyczna lista ulepszeń, które podnoszą wartość merytoryczną, efekt wizualny i wrażenie na jury hackathonu. Każde ulepszenie zawiera opis, uzasadnienie, szacowany czas implementacji i wpływ na projekt.

---

## Podsumowanie Kategorii

| # | Kategoria | Liczba ulepszeń | Łączny wpływ |
|---|---|---|---|
| A | Efekt WOW — Wizualizacja i Prezentacja | 8 | 🔴 Krytyczny |
| B | AI i Inteligencja Agenta | 7 | 🔴 Krytyczny |
| C | Rygor Naukowy i Merytoryka | 6 | 🟡 Wysoki |
| D | UX / UI i Doświadczenie Użytkownika | 7 | 🟡 Wysoki |
| E | Architektura i Inżynieria | 5 | 🟢 Średni |
| F | Elementy Innowacyjne (Stretch Goals) | 5 | 🟢 Średni |

---

## A. Efekt WOW — Wizualizacja i Prezentacja

### A1. Animacja Rotacji Planety z Cyklem Dzień-Noc
**Opis:** Automatyczna, ciągła rotacja sfery 3D w Plotly z dynamicznym oświetleniem symulującym cykl dzień-noc. Dla planet uwięzionych pływowo — drobna oscylacja libracyjna (libration) zamiast pełnej rotacji.  
**Uzasadnienie:** Statyczna sfera wygląda jak screenshot — rotująca planeta to **żywy organizm**. Na prezentacji jury widzi ruch, co natychmiast przyciąga uwagę.  
**Implementacja:** Użycie `fig.update_layout(updatemenus=[...])` z `frame` animation + Plotly `Frames`. Dla libracji — sinusoidalna zmiana `camera.eye.x/y` z amplitudą ~10°.  
**Czas:** ~2–3h  
**Impact/Effort:** ⭐⭐⭐⭐⭐

---

### A2. Chmury Atmosferyczne jako Drugi Warstwa Sfery (Dual-Layer Globe)
**Opis:** Renderowanie **dwóch** nałożonych sfer w Plotly — wewnętrzna (r=1.0) z mapą temperatury, zewnętrzna (r=1.02) z półprzezroczystą teksturą chmur. Chmury generowane proceduralnie z szumu Perlina lub z mapy wilgotności ELM.  
**Uzasadnienie:** Pojedyncza sfera wygląda jak sucha skała. Dodanie mglistej atmosfery nadaje wrażenie **prawdziwego świata** — jury natychmiast rozumie, że to nie jest prosta heatmapa.  
**Implementacja:**
```python
# Druga sfera z opacity < 1
cloud_surface = go.Surface(
    x=X*1.02, y=Y*1.02, z=Z*1.02,
    surfacecolor=cloud_map,
    opacity=0.3,
    colorscale=[[0, 'rgba(255,255,255,0)'], [1, 'rgba(255,255,255,0.6)']],
    showscale=False
)
fig.add_trace(cloud_surface)
```
**Czas:** ~3–4h  
**Impact/Effort:** ⭐⭐⭐⭐⭐

---

### A3. Gwiazda Macierzysta jako Świecący Punkt Obok Planety
**Opis:** Dodanie obiektu `go.Scatter3d` symulującego gwiazdę macierzystą w scenie 3D — jasny punkt z kolorem odpowiadającym temperaturze gwiazdy (M → czerwony, G → żółty, F → biały). Opcjonalnie — linie strumienia radiacyjnego od gwiazdy do planety.  
**Uzasadnienie:** Kontekstualizuje wizualizację — planeta nie "wisi w próżni", ale jest częścią **systemu gwiazdowego**. Pomaga zrozumieć uwięzienie pływowe (gwiazda zawsze po tej samej stronie).  
**Implementacja:** Prosta — punkt 3D z kolorem `star_teff → blackbody_rgb()` + opcjonalne linie `go.Scatter3d(mode='lines')`.  
**Czas:** ~1–2h  
**Impact/Effort:** ⭐⭐⭐⭐⭐

---

### A4. Dashboard z Żywymi Wskaźnikami (Animated Metrics)
**Opis:** Zamiast statycznych `st.metric()` — animowane wskaźniki z gauge/dial charts (Plotly `go.Indicator`) dla ESI, T_eq, habitabilności. ESI jako półokrągły gauge z kolorową skalą (czerwony→żółty→zielony). Habitabilność jako "termometr" z fazami wody zaznaczonymi.  
**Uzasadnienie:** Metryki jako zwykłe liczby giną w UI. **Gauges wizualnie krzyczą** — jury widzi jeden rzut oka i rozumie "o, ta planeta jest podobna do Ziemi w 87%".  
**Implementacja:**
```python
gauge = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=esi_value,
    delta={'reference': 0.8, 'increasing': {'color': "green"}},
    gauge={
        'axis': {'range': [0, 1]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 0.4], 'color': '#d73027'},
            {'range': [0.4, 0.7], 'color': '#fee08b'},
            {'range': [0.7, 1.0], 'color': '#1a9850'}
        ],
        'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': 0.8}
    },
    title={'text': "Earth Similarity Index"}
))
```
**Czas:** ~2h  
**Impact/Effort:** ⭐⭐⭐⭐

---

### A5. Mapa Habitabilności (Habitable Zone Overlay na Globie)
**Opis:** Na sferze 3D nałożenie konturowych linii wyznaczających granicę fazy ciekłej wody (273K i 373K). Regiony habitabilne (273–373K) podświetlone zielonym obramowaniem. Wyświetlenie procentu powierzchni habitabilnej.  
**Uzasadnienie:** Jury od razu widzi **"ile planety jest zamieszkiwalnych"** bez analizowania skali kolorów. Procent habitabilnej powierzchni to unikalna metryka, której inne projekty nie oferują.  
**Implementacja:** `go.Surface` z `contours` parameter + obliczenie `(273 <= temp_map) & (temp_map <= 373)`.sum() / total.  
**Czas:** ~2h  
**Impact/Effort:** ⭐⭐⭐⭐⭐

---

### A6. Widok Porównawczy "Split-Screen" z Ziemią
**Opis:** Obok wizualizacji egzoplanety — zawsze widoczna sfera Ziemi (z uproszczoną mapą temperatury) jako punkt odniesienia. Identyczna skala kolorów. Przycisk "Nałóż Ziemię" superimposes transparent Earth onto exoplanet.  
**Uzasadnienie:** Ludzie myślą komparatywnie. Bez odniesienia "234K" jest abstrakcyjne. Obok Ziemi (288K) — natychmiast rozumieją, że planeta jest **chłodniejsza, ale w podobnym zakresie**.  
**Implementacja:** Dwie kolumny Streamlit, precomputed Earth temperature map (NOAA dane klimatyczne lub syntetyczna).  
**Czas:** ~3h  
**Impact/Effort:** ⭐⭐⭐⭐⭐

---

### A7. Eksport Wizualizacji jako Interaktywny HTML + Animowany GIF
**Opis:** Przycisk "Eksportuj" generujący: (1) samodzielny plik HTML z interaktywną sferą Plotly (do otwarcia w dowolnej przeglądarce), (2) animowany GIF z rotacją planety (do wklejenia w prezentację/poster).  
**Uzasadnienie:** Jury/mentorzy chcą pokazać projekt dalej. HTML to **przenośna demo**, GIF to **social media / poster material**. Dowód profesjonalizmu.  
**Implementacja:** `fig.write_html("planet.html")` + `kaleido` backend dla eksportu ramek + `imageio` do GIF.  
**Czas:** ~2h  
**Impact/Effort:** ⭐⭐⭐⭐

---

### A8. Dark Mode Kosmiczny z Particle Background
**Opis:** Tło aplikacji Streamlit stylizowane na kosmos — ciemne tło z animowanymi "gwiazdami" (CSS particles lub Streamlit custom HTML). Logo projektu jako SVG z animacją. Custom font (Space Grotesk / Orbitron).  
**Uzasadnienie:** Domyślny Streamlit wygląda jak "kolejna aplikacja szkolna". Kosmiczny theme sprawia, że projekt wygląda jak **profesjonalne narzędzie NASA**, nie studencki hack.  
**Implementacja:**
```python
st.markdown("""
<style>
    .stApp { background: radial-gradient(ellipse at center, #0a0e27 0%, #000000 100%); }
    .stMetricValue { color: #00d4ff !important; font-family: 'Orbitron', sans-serif; }
    [data-testid="stSidebar"] { background: rgba(10,14,39,0.95); border-right: 1px solid #1a237e; }
</style>
""", unsafe_allow_html=True)
```
**Czas:** ~2–3h  
**Impact/Effort:** ⭐⭐⭐⭐⭐

---

## B. AI i Inteligencja Agenta

### B1. Multi-Step Autonomous Discovery — "Odkryj Najciekawszą Planetę"
**Opis:** Nowe narzędzie agenta: `discover_most_interesting_planet()`. Agent autonomicznie: (1) odpytuje NASA o wszystkie planety w HZ, (2) oblicza ESI dla każdej, (3) sortuje po interesowności (ESI + unikalność typu klimatu), (4) zwraca top-3 z uzasadnieniem. User pisze: "Znajdź najciekawszą planetę" — agent sam przeprowadza pełne badanie.  
**Uzasadnienie:** Pokazuje agentowi **autonomiczność** — kluczowy buzzword hackathonu. Nie jest to prosty Q&A, ale wielokrokowe rozumowanie z planem działań.  
**Czas:** ~3–4h  
**Impact/Effort:** ⭐⭐⭐⭐⭐

---

### B2. Chain-of-Thought Visible Reasoning (Transparent Agent)
**Opis:** Panel boczny w UI pokazujący **w czasie rzeczywistym** łańcuch rozumowania agenta: `Thought → Action → Observation → Thought → ...`. Formatowany jako timeline z kolorowymi etykietami (🧠 Myśl, 🔧 Akcja, 👁️ Obserwacja). Zwijany/rozwijalny.  
**Uzasadnienie:** Jury chce widzieć **jak** agent myśli, nie tylko wynik. Transparent reasoning to USP projektu — pokazuje, że to nie "czarna skrzynka GPT", ale kontrolowany, obserwowalny system agenturowy.  
**Implementacja:** LangChain `AgentExecutor(verbose=True)` → przechwytywanie `intermediate_steps` → wyświetlanie w `st.expander()` w real-time ze `st.status()`.  
**Czas:** ~3h  
**Impact/Effort:** ⭐⭐⭐⭐⭐

---

### B3. Agent Memory — Pamięć Między Sesjami Konwersacji
**Opis:** Agent zapamiętuje planety analizowane w trakcie sesji i referencuje je w kolejnych pytaniach. "Porównaj ją z tą planetą, którą analizowaliśmy wcześniej" — agent wie, o którą chodzi.  
**Uzasadnienie:** Pokazuje, że system jest **konwersacyjny**, nie jednorazowy. Jury może prowadzić dialog, budując na wcześniejszych wynikach.  
**Implementacja:** LangChain `ConversationBufferMemory` + przetwarzanie `chat_history` w prompcie. Zapis wyników do `st.session_state.analysis_history`.  
**Czas:** ~2h  
**Impact/Effort:** ⭐⭐⭐⭐

---

### B4. Sugestie Kontekstowe (Agent Proactive Suggestions)
**Opis:** Po każdej analizie agent proaktywnie sugeruje następne kroki: "Na podstawie analizy TRAPPIST-1e, mogę: (1) Porównać z TRAPPIST-1f, (2) Uruchomić symulację 3D, (3) Wygenerować raport PDF, (4) Sprawdzić inne planety z podobnym ESI". Sugestie jako klikalne buttony Streamlit.  
**Uzasadnienie:** Eliminuje "a co teraz?" moment — użytkownik naturalnie kontynuuje eksplorację. Sprawia wrażenie **inteligentnego asystenta**, nie biernego narzędzia.  
**Implementacja:** Post-processing odpowiedzi agenta → ekstrakcja kontekstu → generowanie 3–4 relevantnych sugestii → `st.button()` per sugestia.  
**Czas:** ~3h  
**Impact/Effort:** ⭐⭐⭐⭐⭐

---

### B5. Narzędzie Agenta: "Explain Like I'm a Scientist / Student / Media"
**Opis:** Parametr głębokości wyjaśnienia: agent dostosowuje poziom języka do odbiorcy. "Explain TRAPPIST-1e habitability for a journalist" → uproszczone analogie. "...for an astrophysicist" → równania i referencje. Przełącznik w UI.  
**Uzasadnienie:** Hackathon ocenia **komunikowalność**. Pokazanie, że system umie mówić zarówno językiem nauki, jak i przystępnie — potężny argument.  
**Czas:** ~1–2h (głównie prompt engineering)  
**Impact/Effort:** ⭐⭐⭐⭐

---

### B6. Automatyczne Wykrywanie Anomalii w Danych NASA
**Opis:** Narzędzie agenta `detect_anomalies()` — po pobraniu danych planety system automatycznie flaguje parametry, które są nietypowe w kontekście populacji egzoplanet (np. "Promień tej planety jest w 97. percentylu — to ultra-puffy planet"). Wizualizacja jako histogram z zaznaczeniem pozycji planety.  
**Uzasadnienie:** Pokazuje nie tylko obliczenia, ale **interpretacyjną inteligencję** — system rozumie, co jest normalne, a co ciekawe. To zachowanie naukowca, nie kalkulatora.  
**Implementacja:** Precomputed percentyle z pełnego katalogu NASA → porównanie pojedynczej planety → flagowanie ekstremalnych percentyli.  
**Czas:** ~3h  
**Impact/Effort:** ⭐⭐⭐⭐

---

### B7. Głosowe Sterowanie Agentem (Speech-to-Text)
**Opis:** Przycisk mikrofonu w UI — użytkownik mówi zapytanie głosem, Whisper (lub Web Speech API przeglądarki) zamienia na tekst, agent odpowiada. Opcjonalnie: synteza mowy (TTS) odpowiedzi.  
**Uzasadnienie:** "Efekt Star Trek" — mówisz do komputera o kosmosie, komputer odpowiada. Na prezentacji live demo z głosem robi **ogromne wrażenie**.  
**Implementacja:** `streamlit-webrtc` lub `st.audio_input()` + Whisper API (Groq — darmowe) lub browser Web Speech API przez custom component.  
**Czas:** ~4–5h  
**Impact/Effort:** ⭐⭐⭐⭐

---

## C. Rygor Naukowy i Merytoryka

### C1. Strefa Zamieszkiwalna Dynamicznie Rysowana (HZ Diagram)
**Opis:** Interaktywny wykres Plotly pokazujący strefę zamieszkiwalną danej gwiazdy (inner/outer HZ boundary wg Kopparapu 2013) z zaznaczoną pozycją analizowanej planety. Strefy kolorowe: runaway greenhouse (wewnętrzna), HZ (zielona), maximum greenhouse (zewnętrzna), snowball (ultra-zewnętrzna).  
**Uzasadnienie:** To **standardowy diagram** w astrobiologii — jego brak sprawiłby, że astrofizyk w jury powiedziałby "brakuje podstawowego narzędzia". Jego obecność buduje wiarygodność naukową.  
**Implementacja:**
```python
def hz_boundaries(star_teff: float) -> dict:
    """Kopparapu et al. 2013 — granice HZ."""
    T = star_teff - 5780
    coeffs = {
        "recent_venus":   [1.7763, 1.4335e-4, 3.3954e-9, -7.6364e-12, -1.1950e-15],
        "runaway_gh":     [1.0385, 1.2456e-4, 1.4612e-8, -7.6345e-12, -1.7511e-15],
        "max_gh":         [0.3507, 5.9578e-5, 1.6707e-9, -3.0058e-12, -5.1925e-16],
        "early_mars":     [0.3207, 5.4471e-5, 1.5275e-9, -2.1709e-12, -3.8282e-16],
    }
    # S_eff = S_effsun + a*T + b*T^2 + c*T^3 + d*T^4
    # a_hz = sqrt(L_star / S_eff)
    ...
```
**Czas:** ~3h  
**Impact/Effort:** ⭐⭐⭐⭐⭐

---

### C2. Diagram Hertzsprung-Russell z Pozycją Gwiazdy
**Opis:** Interaktywny diagram H-R (luminozja vs temperatura) z zaznaczonymi ~5700 gwiazdami z katalogu NASA, na którym podświetlona jest gwiazda macierzysta analizowanej planety. Kolor punktu = typ widmowy. Ciąg główny jako linia referencyjna.  
**Uzasadnienie:** H-R diagram to **ikona astrofizyki**. Jego obecność pokazuje, że zespół rozumie kontekst gwiazdowy, nie tylko planetarny. Jury widzi natychmiast — "aha, to gwiazda typu M, stąd tidal locking".  
**Czas:** ~2–3h  
**Impact/Effort:** ⭐⭐⭐⭐

---

### C3. SEPHI Rozszerzony — Pełna Implementacja z Kryterium Atmosfery
**Opis:** Aktualnie ESI jest jedynym wskaźnikiem. Implementacja pełnego SEPHI (Rodríguez-Mozos & Moya 2017) z trzema kryteriami: (1) atmosferycznym (v_escape > v_thermal gazów), (2) termicznym (273–373K), (3) opcjonalnym magnetycznym. Wynik jako wielopoziomowy "traffic light" (✅/⚠️/❌).  
**Uzasadnienie:** SEPHI jest **cytowany w opisie projektu** ale nie zaimplementowany w kodzie. Jury to zauważy. Pełna implementacja wzmacnia spójność między dokumentacją a produktem.  
**Czas:** ~3–4h  
**Impact/Effort:** ⭐⭐⭐⭐

---

### C4. Wizualizacja Profilu Atmosferycznego (Transect / Cross-Section)
**Opis:** Wykres 2D przekroju planety wzdłuż terminatora (granica dzień-noc): oś X = pozycja (od punktu substelarnego do antystelarnego), oś Y = temperatura. Na wykresie zaznaczone: punkt zamarzania wody, punkt wrzenia, strefa habitabilna. Jeśli PINN jest zaimplementowany — profil z rozwiązania PDE nałożony na profil z ELM.  
**Uzasadnienie:** 3D glob jest piękny, ale **naukowiec chce widzieć dane ilościowo**. Przekrój daje precyzyjny wgląd w gradient termiczny. Porównanie PINN vs ELM na tym samym wykresie dowodzi, że oba podejścia są spójne.  
**Czas:** ~2h  
**Impact/Effort:** ⭐⭐⭐⭐

---

### C5. Metryka Powierzchni Habitabilnej (Habitable Surface Fraction)
**Opis:** Nowa metryka: **procent powierzchni planety z temperaturą w zakresie 273–373K** — "Habitable Surface Fraction" (HSF). Wyświetlana obok ESI. Np. "14.3% powierzchni TRAPPIST-1e może utrzymać ciekłą wodę". Na globie 3D regiony te podświetlone konturem.  
**Uzasadnienie:** ESI porównuje z Ziemią globalnie, HSF mówi **"ile planety jest zamieszkiwalne"** — bardziej intuicyjna i oryginalna metryka. Jury może zapytać: "ale czy cała planeta jest habitabilna?" — mamy odpowiedź.  
**Implementacja:**
```python
hsf = np.sum((temp_map >= 273) & (temp_map <= 373)) / temp_map.size
# Ważona powierzchnią (cos latitude correction):
lat_weights = np.cos(np.linspace(-np.pi/2, np.pi/2, N_LAT)).reshape(-1,1)
hsf_weighted = np.sum(((temp_map >= 273) & (temp_map <= 373)) * lat_weights) / np.sum(lat_weights * np.ones_like(temp_map))
```
**Czas:** ~1h  
**Impact/Effort:** ⭐⭐⭐⭐⭐

---

### C6. Tabela z Oceną Niepewności (Confidence Dashboard)
**Opis:** Dla każdego obliczonego parametru — wyświetlenie poziomu pewności i źródła ewentualnej niepewności. ESI ± 0.05 (niepewność pomiaru promienia), T_eq ± 15K (niepewność albedo). Prezentowane jako error bars na wykresach i jako kolorowe tagi (🟢 wysoka pewność, 🟡 średnia, 🔴 niska) przy metrykach.  
**Uzasadnienie:** Nauka bez uncertainty analysis jest pseudonauką. Jury z tłem naukowym **natychmiast to zauważy**. Ensemble ELM naturalnie daje std dev — wystarczy to pokazać.  
**Implementacja:** `std_map = np.std([m.predict(X) for m in ensemble], axis=0)` + propagacja błędów ESI via Monte Carlo.  
**Czas:** ~3h  
**Impact/Effort:** ⭐⭐⭐⭐⭐

---

## D. UX / UI i Doświadczenie Użytkownika

### D1. Onboarding Tutorial — "Guided Tour" Dla Nowego Użytkownika
**Opis:** Przy pierwszym uruchomieniu — interaktywny tutorial krok-po-kroku: (1) "Wpisz nazwę planety lub wybierz z listy", (2) "Agent pobierze dane z NASA", (3) "Kliknij aby zobaczyć wizualizację 3D", (4) "Zadaj dowolne pytanie agentowi". Każdy krok z animacją i podświetleniem elementu UI.  
**Uzasadnienie:** Na hackathonie jury ma **2–3 minuty** na interakcję. Jeśli nie wiedzą co kliknąć — projekt przegrywa. Tutorial eliminuje barierę wejścia.  
**Implementacja:** `st.session_state.onboarding_step` + warunkowe `st.info()` + `st.balloons()` na końcu.  
**Czas:** ~2h  
**Impact/Effort:** ⭐⭐⭐⭐

---

### D2. Quick-Select: "Famous Exoplanets" Gallery
**Opis:** Panel z predefiniowanymi, najpopularniejszymi egzoplanetami (TRAPPIST-1e, Proxima Cen b, Kepler-442b, LHS 1140b, TOI-700d, K2-18b) jako klikalne karty z miniaturkami i kluczowymi statystykami. Jeden klik = pełna analiza.  
**Uzasadnienie:** Większość użytkowników (i jury!) **nie zna nazw egzoplanet**. Gotowa galeria eliminuje "a jaką planetę wpisać?". Przyspiesza demo 10×.  
**Implementacja:**
```python
famous = [
    {"name": "TRAPPIST-1 e", "icon": "🔴", "desc": "Rocky, temperate, tidally locked"},
    {"name": "Proxima Cen b", "icon": "🟠", "desc": "Closest habitable candidate"},
    {"name": "K2-18 b", "icon": "🔵", "desc": "Water vapor detected (JWST)"},
    ...
]
cols = st.columns(len(famous))
for col, planet in zip(cols, famous):
    with col:
        if st.button(f"{planet['icon']} {planet['name']}", use_container_width=True):
            run_full_analysis(planet['name'])
```
**Czas:** ~2h  
**Impact/Effort:** ⭐⭐⭐⭐⭐

---

### D3. Progress Bar z Fazami Pipeline (Real-Time Status)
**Opis:** Podczas uruchamiania pełnej analizy — dynamiczny progress bar z opisem aktualnej fazy: "📡 Pobieranie z NASA... → 🧮 Obliczanie T_eq i ESI... → 🧠 ELM generuje mapę klimatu... → 🌍 Renderowanie sfery 3D...". Każda faza z szacowanym czasem.  
**Uzasadnienie:** Bez feedbacku 3-sekundowe oczekiwanie wydaje się wiecznością. Z progress barem jury widzi **że system pracuje, a nie zawiesił się**. Plus demonstruje złożoność pipeline'u.  
**Implementacja:** `st.progress(percent)` + `st.status("Running...", expanded=True)` z aktualizacjami per fase.  
**Czas:** ~1–2h  
**Impact/Effort:** ⭐⭐⭐⭐

---

### D4. "What If" Mode — Interaktywne Suwaki z Live Update Globu
**Opis:** Real-time update wizualizacji 3D podczas przesuwania suwaków (albedo, naświetlenie, temperatura gwiazdy). Bez klikania "Uruchom" — glob zmienia się na żywo. Efekt: przesuwasz suwak albedo i patrzysz jak planeta przechodzi z "eyeball" do "snowball".  
**Uzasadnienie:** Interaktywność jest **kluczowa na hackathonie** — demo musi być dynamiczne. "What If" scenarios naturalnie prowadzą do pytań ("co się stanie, jeśli gwiazda jest gorętsza?") — idealne na Q&A z jury.  
**Implementacja:** Użycie analitycznego modelu (szybki, bez ELM — generacja < 50ms) + `st.session_state` + Streamlit auto-rerun na zmianę suwaka. Opcjonalnie `st_autorefresh` z debounce.  
**Czas:** ~2–3h  
**Impact/Effort:** ⭐⭐⭐⭐⭐

---

### D5. Tabela Porównawcza N Planet (Multi-Planet Comparison)
**Opis:** Rozszerzenie porównania z 2 do N planet. Użytkownik dodaje planety do "koszyka porównawczego" → generowana jest tabela z heatmapem (komórki kolorowane wg wartości). Wykres radarowy (spider chart) nakładający profile planet na siebie (promień, masa, T_eq, ESI, naświetlenie).  
**Uzasadnienie:** Porównanie wielu planet jednocześnie to **core use case** dla astrobiologa. Spider chart nakładający kilka planet jest wizualnie efektowny i informacyjny.  
**Implementacja:** `go.Scatterpolar()` w Plotly + pandas styling z `background_gradient()`.  
**Czas:** ~3–4h  
**Impact/Effort:** ⭐⭐⭐⭐

---

### D6. Tooltip Encyclopedia — Hovery z Wyjaśnieniami Terminologii
**Opis:** Każdy termin naukowy w UI (ESI, Albedo, Tidal Locking, Habitable Zone, etc.) ma tooltip/hover z krótkim wyjaśnieniem (1–2 zdania + link do Wikipedia/artykułu). Zaimplementowane jako `st.markdown` z `<abbr title="...">` lub własny komponent.  
**Uzasadnienie:** Jury nie musi znać astrofizyki. **Samowyjaśniający się interfejs** eliminuje barierę wiedzy i pokazuje dbałość o UX.  
**Implementacja:**
```python
def tooltip(text: str, explanation: str) -> str:
    return f'<span title="{explanation}" style="border-bottom: 1px dotted #888; cursor:help;">{text}</span>'

st.markdown(tooltip("ESI", "Earth Similarity Index — skala 0-1 mierząca podobieństwo do Ziemi"), unsafe_allow_html=True)
```
**Czas:** ~1–2h  
**Impact/Effort:** ⭐⭐⭐⭐

---

### D7. Responsive Layout + Mobile-Friendly
**Opis:** Streamlit layout testowany i zoptymalizowany pod kątem wyświetlania na tabletach/telefonach (media queries w custom CSS). Kolumny zawijają się do jednokolumnowego widoku. Czcionki skalowane. Glob 3D responsive.  
**Uzasadnienie:** Jeśli juror otworzy link na telefonie i zobaczy działającą aplikację — **bonus points za profesjonalizm**. Większość projektów hackathonowych łamie się na mobile.  
**Czas:** ~2h  
**Impact/Effort:** ⭐⭐⭐

---

## E. Architektura i Inżynieria

### E1. Monitoring i Metryki Systemu (System Health Dashboard)
**Opis:** Ukryta zakładka "🔧 System" widoczna po kliknięciu — pokazuje: zużycie VRAM, szybkość inference LLM (tok/s), czas odpowiedzi ELM, status Ollama, wersje modeli, cache hit rate. Automatyczna detekcja degradacji.  
**Uzasadnienie:** Pokazuje **inżynierię produkcyjną** — system monitoruje sam siebie. Jury technicze to doceni. Plus pomaga w debugowaniu na prezentacji.  
**Implementacja:** `psutil` + `ollama.list()` + timing decorators + `st.expander("System Health")`.  
**Czas:** ~2–3h  
**Impact/Effort:** ⭐⭐⭐

---

### E2. Automatyczne Testy z Demonstracją (Test Suite as Feature)
**Opis:** Przycisk "🧪 Run Self-Diagnostics" w UI uruchamiający testowy pipeline: (1) zapytanie do NASA (test API), (2) obliczenie T_eq dla Ziemi (powinno dać ~255K), (3) walidacja Pydantic z celowo złymi danymi (pokaz odrzucenia), (4) predykcja ELM, (5) sprawdzenie Ollama status. Wynik: zielone checkmarki.  
**Uzasadnienie:** Live self-test na prezentacji to **demo confidence** — "nasz system sam weryfikuje, że działa poprawnie". Imponujące i praktyczne.  
**Czas:** ~2h  
**Impact/Effort:** ⭐⭐⭐⭐

---

### E3. Caching Inteligentny z TTL i Invalidation
**Opis:** Zamiast prostego `@st.cache_data` — cache z: (1) TTL (time-to-live) — dane NASA odświeżane co 24h, (2) invalidation — zmiana parametrów suwaków czyści odpowiedni cache, (3) cache stats — ile zapytań obsłużono z cache vs live. Widoczne w System Dashboard.  
**Uzasadnienie:** Demonstracja **świadomości inżynierskiej** — cache to nie "włącz i zapomnij". TTL zapobiega stale danymprzywięzeniu.  
**Czas:** ~1–2h  
**Impact/Effort:** ⭐⭐⭐

---

### E4. Docker Containerization + One-Click Deploy
**Opis:** `Dockerfile` + `docker-compose.yml` pozwalające na uruchomienie **całego systemu** jednym poleceniem: `docker compose up`. Kontener z Ollama + Streamlit + wszystkie modele.  
**Uzasadnienie:** "Jak to uruchomić?" — najczęstsze pytanie jury. Docker odpowiada: "jednym poleceniem". Profesjonalizm na poziomie produkcji.  
**Implementacja:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```
**Czas:** ~3h  
**Impact/Effort:** ⭐⭐⭐⭐

---

### E5. Logging Naukowy — Audit Trail Analiz
**Opis:** Każda analiza przeprowadzona w systemie jest logowana z timestampem, parametrami wejściowymi, wynikami i narzędziami użytymi przez agenta. Log eksportowalny jako JSON/CSV. Pozwala na reprodukowalność wyników.  
**Uzasadnienie:** Reprodukowalność to fundament nauki. Log dowodzi, że wyniki nie są generowane losowo. Jury z tłem naukowym to doceni.  
**Implementacja:** `logging` module + JSON append + `st.download_button("Download Audit Log")`.  
**Czas:** ~2h  
**Impact/Effort:** ⭐⭐⭐

---

## F. Elementy Innowacyjne (Stretch Goals)

### F1. Planetary Soundscape — Sonifikacja Danych Klimatycznych
**Opis:** Konwersja mapy temperatur na dźwięk: niskie temperatury → niskie tony, wysokie → wysokie. "Posłuchaj jak brzmi TRAPPIST-1e". Planeta uwięziona pływowo ma wyraźny kontrast (substelarny = wysoki ton, nocna strona = niski). Odtwarzacz audio w Streamlit.  
**Uzasadnienie:** **Absolutnie unikalna funkcja**, którą żaden inny projekt hackathonowy nie będzie miał. Sonifikacja danych to rosnący trend w astrofizyce (Chandra X-ray Sonification). WOW-factor na prezentacji jest ogromny.  
**Implementacja:**
```python
import numpy as np
from scipy.io import wavfile

def sonify_planet(temp_map: np.ndarray, duration_s: float = 5.0, sr: int = 22050):
    """Zamienia profil temperatury na dźwięk — 'skan' wzdłuż równika."""
    equator = temp_map[temp_map.shape[0]//2, :]  # Przekrój wzdłuż równika
    T_min, T_max = equator.min(), equator.max()
    
    # Mapowanie temperatury na częstotliwość (200 Hz – 2000 Hz)
    freqs = 200 + (equator - T_min) / (T_max - T_min) * 1800
    
    t = np.linspace(0, duration_s, sr * int(duration_s))
    samples_per_freq = len(t) // len(freqs)
    
    audio = np.array([])
    for f in freqs:
        segment = 0.3 * np.sin(2 * np.pi * f * np.linspace(0, samples_per_freq/sr, samples_per_freq))
        audio = np.concatenate([audio, segment])
    
    audio = (audio * 32767).astype(np.int16)
    wavfile.write("planet_sound.wav", sr, audio)
    return "planet_sound.wav"
```
**Czas:** ~3h  
**Impact/Effort:** ⭐⭐⭐⭐⭐

---

### F2. "What If Earth Was Here?" — Pozycjonowanie Ziemi w Innych Systemach
**Opis:** Interaktywne narzędzie: "Co by się stało, gdyby Ziemia orbitowała wokół gwiazdy TRAPPIST-1?" System umieszcza Ziemię (z jej albedo i masą) na orbitach różnych planet w systemie i oblicza T_eq, ESI, klimat. Wizualizacja orbit z zaznaczonymi HZ.  
**Uzasadnienie:** Najbardziej **intuicyjny sposób na zrozumienie egzoplanetarności** — ludzie rozumieją Ziemię, więc pozycjonowanie jej w obcym systemie daje natychmiastowe zrozumienie skali.  
**Czas:** ~4h  
**Impact/Effort:** ⭐⭐⭐⭐⭐

---

### F3. Timeline Odkryć — "History of Exoplanet Discovery"
**Opis:** Animowany wykres liniowy (Plotly animation) pokazujący skumulowaną liczbę odkrytych egzoplanet w czasie (1992–2026) z zaznaczonymi kamieniami milowymi (Kepler launch, TESS, JWST). Filtr po metodzie odkrycia (transit, radial velocity, imaging). Na hover — nazwa planety.  
**Uzasadnienie:** Kontekstualizuje projekt w **historii nauki**. Jury widzi, że zespół rozumie tło tematyki. Animacja jest efektowna i edukacyjna.  
**Implementacja:** Dane z NASA Archive (kolumna `disc_year`, `discoverymethod`) → `px.scatter` z `animation_frame='disc_year'`.  
**Czas:** ~2–3h  
**Impact/Effort:** ⭐⭐⭐⭐

---

### F4. Collaborative Mode — Multi-User Planet Exploration
**Opis:** Dwóch użytkowników może jednocześnie analizować różne planety i dzielić się wynikami przez link. "Ja analizuję TRAPPIST-1e, ty Proxima Cen b — porównajmy wyniki". Zaimplementowane jako shared `st.session_state` z identyfikatorem sesji (UUID) + synchronizacja przez pliki JSON.  
**Uzasadnienie:** Pokazuje myślenie o **real-world use case** — naukowcy pracują w zespołach. Na hackathonie — dwóch członków drużyny może równolegle eksplorować.  
**Czas:** ~5–6h  
**Impact/Effort:** ⭐⭐⭐

---

### F5. Auto-Generated Scientific Poster (LaTeX)
**Opis:** Przycisk generujący **naukowy poster A0** w LaTeX/PDF podsumowujący analizę: tytuł, parametry planety, wizualizacja 3D, metryki, profil termiczny, źródła naukowe. Template profesjonalnego posteru konferencyjnego.  
**Uzasadnienie:** Naukowiec dostaje gotowy **artifact do wydruku** — wartość użytkowa na poziomie profesjonalnym. Na hackathonie — wydrukowanie wygenerowanego posteru i powieszenie przy stanowisku = **instant credibility**.  
**Czas:** ~5–6h  
**Impact/Effort:** ⭐⭐⭐⭐

---

## Priorytetyzacja — Top 10 Ulepszeń do Zaimplementowania

Ranking na podstawie stosunku **Impact na jury / Effort implementacyjny**:

| Rank | ID | Nazwa | Czas | Impact |
|---|---|---|---|---|
| 1 | **A8** | Dark Mode Kosmiczny (CSS Theme) | 2–3h | 🔴 Ogromny |
| 2 | **D2** | Quick-Select Famous Exoplanets Gallery | 2h | 🔴 Ogromny |
| 3 | **B2** | Transparent Agent Reasoning (CoT visible) | 3h | 🔴 Ogromny |
| 4 | **A5** | Habitable Zone Overlay na Globie + HSF | 2h | 🔴 Ogromny |
| 5 | **C5** | Habitable Surface Fraction Metric | 1h | 🟡 Wysoki |
| 6 | **A3** | Gwiazda Macierzysta jako Punkt 3D | 1–2h | 🟡 Wysoki |
| 7 | **B4** | Proactive Agent Suggestions (clickable) | 3h | 🔴 Ogromny |
| 8 | **C1** | HZ Diagram z Pozycją Planety | 3h | 🟡 Wysoki |
| 9 | **D4** | "What If" Mode z Live Update | 2–3h | 🔴 Ogromny |
| 10 | **F1** | Planetary Soundscape (Sonifikacja) | 3h | 🔴 Ogromny (WOW) |

**Szacowany łączny czas Top-10:** ~22–25h (realistyczne na hackathon równolegle z core dev).

---

## Quick Wins (< 1.5h każdy, duży efekt)

| ID | Ulepszenie | Czas |
|---|---|---|
| C5 | Habitable Surface Fraction | ~1h |
| D6 | Tooltip Encyclopedia | ~1h |
| B5 | Explain for Scientist/Student/Media | ~1h |
| A3 | Gwiazda jako punkt 3D | ~1h |
| D3 | Pipeline Progress Bar | ~1h |

Te 5 quick wins daje się zrobić w **łączne 5 godzin** — mały koszt, duży efekt.
