# Architektura Systemu: Autonomiczny Cyfrowy Bliźniak Egzoplanetarny

## 1. Cel Projektu i Założenia Architektoniczne
Projekt ma na celu stworzenie zorkiestrowanej platformy asystującej pełniącej funkcję Cyfrowego Bliźniaka (Digital Twin) – wirtualnej repliki egzoplanety pozwalającej na badanie jej ewolucji klimatycznej. Przejście od prostej detekcji ciał niebieskich do ewaluacji ich potencjału biologicznego wymaga modeli zdolnych do symulowania wielowymiarowych układów termodynamicznych. Architektura zastępuje powolne, klasyczne Modele Ogólnej Cyrkulacji (GCM), takie jak rozwijany przez NASA system ROCKE-3D. Zastosowanie bezgradientowych surogatów fizycznych (ELM), agentowych modeli językowych (LLM) oraz sieci przeciwstawnych (CTGAN) umożliwia symulowanie nieliniowych układów w czasie zbliżonym do rzeczywistego.

---

## 2. Przepływ Danych (Data Pipeline)
System opiera się na kaskadowym potoku zdarzeń, w którym warstwy wnioskowania, fizyki i generatywnej sztucznej inteligencji komunikują się w izolowanych cyklach.

### Krok 1: Inicjacja i Orkiestracja (Interfejs ➡️ Agent LLM)
* **Interfejs Użytkownika:** Punktem wejścia jest aplikacja webowa zbudowana we frameworku Streamlit.
* **Rola Agenta LLM:** Centralnym punktem dystrybucyjnym jest dziedzinowy model językowy, taki jak AstroLLaMA lub specjalistyczne warianty LLaMA/Qwen. Model operuje w paradygmacie agentowym (ReAct - Reason and Act).
* **Mechanika:** Agent przyjmuje komendy w języku naturalnym, rozbija je na składowe planu działania i samodzielnie decyduje, które z predefiniowanych narzędzi systemowych aktywować (Function Calling).

### Krok 2: Akwizycja Danych i Filtracja (Agent LLM ➡️ Bazy NASA)
* **Pobieranie Obserwacji:** Agent asynchronicznie komunikuje się z archiwum NASA Exoplanet Archive. Wykorzystując ustrukturyzowane zapytania SQL przez protokół TAP (Table Access Protocol), system wyciąga parametry takie jak masa, promień, odległość od gwiazdy i strumień naświetlenia.
* **Aparat Matematyczny:** Zanim uruchomione zostaną sieci neuronowe, system ewaluuje pobrane dane za pomocą fundamentalnych praw fizyki. Obliczana jest teoretyczna temperatura równowagowa ($T_{eq}$) z prawa Stefana-Boltzmanna.
* **Wskaźniki Zdatności:** Następuje kalkulacja wieloparametrowych indeksów, takich jak Indeks Podobieństwa do Ziemi (ESI) oraz Wskaźnik SEPHI.

### Krok 3: Rekonstrukcja Przestrzeni Danych (Dane NASA ➡️ Sieć CTGAN)
* **Identyfikacja Problemu:** Architektury uczenia maszynowego w astrobiologii zmagają się z ekstremalnym niezbilansowaniem klas – światy habitabilne stanowią statystyczny margines w odczytach z teleskopów.
* **Augmentacja:** Do architektury wpięty jest moduł Warunkowych Generatywnych Sieci Przeciwstawnych ukierunkowanych na dane tabelaryczne (CTGAN).
* **Działanie:** Sieć rozwiązuje problem niekompletnych parametrów za pomocą warunkowania stochastycznego i uczenia przez próbkowanie. Wytwarza syntetyczne, ale fizycznie rygorystyczne warianty obcych systemów (w przestrzeni ukrytej), dostarczając zbilansowany zbiór danych dla klasyfikatorów i chroniąc je przed kolapsem statystycznym.

### Krok 4: Surogat Termodynamiczny (Dane ➡️ Maszyny ELM / PINNFormer)
* **Ominięcie Wąskiego Gardła GCM:** Agent przekazuje zebrane ramy początkowe do wczytanych w pamięć wag surogatu fizycznego.
* **Silnik ELM:** System wykorzystuje zespół Maszyn Ekstremalnego Uczenia (Ensemble of ELMs), które posiadają architekturę bezgradientową z zamrożonymi wagami warstwy ukrytej.
* **Szybkość i Emulacja:** Optymalizacja zachodzi analitycznie z wykorzystaniem pseudoodwrotności Moore'a-Penrose'a, co redukuje czas treningu o czynnik rzędu 100 000. Model na podstawie masy gwiazdy, parametru uwięzienia pływowego i albedo przewiduje ułamku sekundy dystrybucję temperatury powierzchniowej. Precyzyjnie emuluje stany klimatyczne takie jak "gałka oczna" (okrągły ocean otoczony lodem) oraz "homar" (ocean równikowy).

### Krok 5: Walidacja Logiczna i Synteza (Silnik Fizyczny ➡️ LLM)
* **Bariera Bezpieczeństwa (Pydantic):** Modele językowe wykazują inherentną skłonność do konfabulacji fizycznych (halucynacji). Aby chronić integralność symulacji, wdrożono restrykcyjne walidatory biblioteki Pydantic jako bariery wejścia i wyjścia.
* **Egzekucja:** Jeśli Agent wygeneruje ustrukturyzowany format wykraczający poza prawa termodynamiki, walidator natychmiastowo odrzuca paczkę (Raise Exception) i kaskaduje żądanie korekty.
* **Synteza:** Po zebraniu poprawnych macierzy, LLM dokonuje syntezy tekstowej, tłumacząc na język zrozumiały ewolucję prądów i warunków na wybranej planecie.

### Krok 6: Rzutowanie 3D i "Efekt WOW" (Wyniki ➡️ Interfejs)
* **Transformacja Tensora:** Dwuwymiarowe tensory ciepła wygenerowane przez zespół ELM są transformowane do trójwymiarowego środowiska sferycznego.
* **Renderowanie:** Za pomocą biblioteki Plotly i obiektu `go.Surface`, generowana jest idealna sfera w oparciu o równania trygonometryczne współrzędnych geograficznych. Właściwość `surfacecolor` mapuje nieliniową matrycę rozkładu temperatur bezpośrednio na teksturę sfery.
* **Kolorystyka Naukowa:** Stosowane są naukowo uzasadnione gradienty mapujące zjawiska fazowe (np. błękit dla stref lodu, granat dla akwenów ciekłych, czerwień dla stref substelarnych powyżej 323 Kelwinów).

---

## 3. Architektura Bezpieczeństwa (Graceful Degradation)
Zintegrowane projekty uczenia maszynowego są obarczone ryzykiem regresji i wycieków pamięci w warunkach 72-godzinnego sprintu. System opiera się na ukrytych mechanizmach płynnej degradacji (Graceful Degradation):
1. **Awaria Surogatu / Agenta:** Jeżeli moduł ELM wygeneruje niefizyczne temperatury lub LLM straci połączenie z pętlą narzędziową, architektura odcina fizyczny potok predykcyjny. Moduł obliczeniowy przełącza się wyłącznie na bazowy silnik czysto algebraiczny (ESI, SEPHI), a nawigacja wraca do klasycznych, ręcznie obsługiwanych suwaków Streamlit.
2. **Przeciążenie Pamięci Wizualizatora:** Mechanika rzutowania 3D tysięcy poligonów może spowodować spadek wydajności (running timeout) w przeglądarce. W takim przypadku detektor ramkowy zrzuca skomplikowaną sferyczną mapę 3D do zoptymalizowanej, ortogonalnej płaskiej Heatmapy 2D z wykorzystaniem algorytmów np. `px.imshow`, gwarantując natychmiastowe odzyskanie płynności animacji.