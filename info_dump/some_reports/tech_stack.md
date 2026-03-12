# Stos Technologiczny (Tech Stack): Cyfrowy Bliźniak Egzoplanetarny

Poniższe zestawienie technologii i bibliotek jest niezbędne do wdrożenia architektury systemu podczas hackathonu. Zostało podzielone na logiczne warstwy, aby ułatwić konfigurację środowiska.

## 1. Frontend i Interfejs Użytkownika (UI)
* **Język bazowy:** Python.
* **Framework Webowy:** Streamlit – zapewnia najwyższy stosunek jakości do czasu wdrożenia dla aplikacji analitycznych. Umożliwia tworzenie interaktywnych suwaków i okien tekstowych.
* **Zarządzanie Stanem i Pamięcią (Caching):** Wykorzystanie dekoratorów `@st.cache_resource` (do alokacji pamięci modeli lokalnych i wag maszyn ELM) oraz `@st.cache_data` (dla operacji na dużych pakietach numerycznych), co zapobiega dławieniu wydajności i przeładowywaniu pamięci.
* **Komunikacja:** Wydajna serializacja plików JSON między logiką w Pythonie a front-endem w przeglądarce.

## 2. Orkiestracja LLM i Frameworki Agentowe
* **Hostowanie LLM:** Platforma Ollama do uruchamiania lokalnych modeli, co gwarantuje prywatność i niezależność od zewnętrznych API chmurowych.
* **Modele Fundacyjne:** Wyspecjalizowane modele dziedzinowe takie jak AstroLLaMA-3-8B-Base, wykazujące minimalną perpleksję w żargonie astrofizycznym, lub klasyczne warianty z rodziny LLaMA i Qwen.
* **Agenty:** Zaawansowane frameworki agentowe LangChain lub minimalistyczny `smolagents` (zoptymalizowany pod kątem wykonywania kodu) do wdrożenia pętli wnioskowania i działania (ReAct) oraz wywoływania zdefiniowanych funkcji (Function Calling).
* **Walidacja Bezpieczeństwa:** Biblioteka Pydantic używana jako restrykcyjny walidator wejścia/wyjścia. Służy do nakładania "kagańców fizycznych" na Agenta – wymusza ścisłą kontrolę typowania i obwarowania graniczne (np. ograniczenia temperatury), odrzucając wyniki łamiące prawa termodynamiki.

## 3. Modele Fizyczne i Uczenie Maszynowe (ML)
* **Surogat Klimatyczny (ELM):** Maszyny Ekstremalnego Uczenia (Extreme Learning Machines) o architekturze bezgradientowej. Implementowane za pomocą biblioteki `scikit-elm`, integrującej się natywnie z ekosystemem Scikit-Learn.
* **Augmentacja Danych (CTGAN):** Warunkowe Generatywne Sieci Przeciwstawne (CTGAN) ukierunkowane na dane tabelaryczne. Służą do syntezy zmiennych wielowymiarowych i eliminacji problemu ekstremalnego niezbilansowania klas w danych obserwacyjnych.
* **Dowód Matematyczny (PINN):** Opcjonalne wykorzystanie dojrzałej biblioteki Python `DeepXDE` do rozwiązania jednowymiarowego równania propagacji ciepła na linii terminatora, dowodząc zrozumienia równań różniczkowych.

## 4. Inżynieria Danych i Wizualizacja 3D
* **Akwizycja Danych:** Bezpośrednie mapowanie złożonych zapytań SQL na serwery NASA poprzez protokół TAP (Table Access Protocol) i wysyłanie zapytań HTTP do endpointów API.
* **Przetwarzanie Danych:** Biblioteka `pandas` do natychmiastowej deserializacji danych wejściowych z formatu VOTable lub CSV do postaci ramek danych (DataFrames).
* **Obsługa Siatek Przestrzennych:** Narzędzia do asynchronicznego pobierania i odczytu próbek symulacyjnych w formacie NetCDF ze zrzutów modeli NASA ROCKE-3D.
* **Renderowanie 3D ("Efekt WOW"):** Zaawansowana biblioteka `plotly.graph_objects`. Wykorzystanie obiektu `go.Surface` oraz właściwości `surfacecolor` do mapowania nieliniowej matrycy dystrybucji temperatur bezpośrednio na strukturę gładkiej tekstury owijającej sferę wyliczoną z równań trygonometrycznych.