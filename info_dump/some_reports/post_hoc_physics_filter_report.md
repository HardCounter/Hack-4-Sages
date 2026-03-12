## Post‑Hoc Physics Filter for Synthetic Exoplanets

This report reviews each constraint in `validate_synthetic_data` in `modules/data_augmentation.py`, compares it to astrophysical expectations, and provides literature context.

The filter is:

```100:113:/home/lasotar/stud/hackathons/hack4sages/Hack-4-Sages/modules/data_augmentation.py
    def validate_synthetic_data(synthetic: pd.DataFrame) -> pd.DataFrame:
        mask = (
            (synthetic["radius_earth"] > 0.3)
            & (synthetic["radius_earth"] < 25.0)
            & (synthetic["mass_earth"] > 0.01)
            & (synthetic["mass_earth"] < 5000)
            & (synthetic["semi_major_axis_au"] > 0.001)
            & (synthetic["star_teff_K"] > 2300)
            & (synthetic["star_teff_K"] < 10000)
            & (synthetic["t_eq_K"] > 10)
            & (synthetic["t_eq_K"] < 3000)
            & (synthetic["period_days"] > 0.1)
        )
        return synthetic[mask].copy()
```

This post‑hoc filter is **not** the main habitability selector. Habitability is primarily encoded via the `habitable` label defined earlier in `prepare_data` using radius, insolation, and stellar effective temperature. The post‑hoc filter is a **broad physical sanity check** to remove obviously unphysical GAN samples.

---

## 1. Planet Radius (`radius_earth`)

- **Code constraint**:  
  $0.3 < R_p/R_\oplus < 25.0$

- **Physical interpretation**:
  - **Lower bound 0.3**: ~Mars-sized. Below $\sim 0.1–0.3\,R_\oplus$, bodies are asteroid‑like and struggle to retain atmospheres over Gyr timescales.
  - **Upper bound 25**: ~2.2 Jupiter radii. Most observed gas giants have radii $\lesssim 2\,R_J$; beyond that, objects approach brown‑dwarf‑like properties.

- **Habitability perspective**:
  - Catalogs of **potentially habitable exoplanets** typically restrict to **$R_p \lesssim 1.6–2.0\,R_\oplus$** to select likely rocky planets. Above this, planets tend to be mini‑Neptunes with thick volatile envelopes rather than Earth‑like rocky worlds.
  - This type of cut is used in compilations of habitable‑zone exoplanets and by the NASA Exoplanet Archive when flagging “small” planets.

- **Assessment**:
  - As a **sanity filter**, the range 0.3–25 $R_\oplus$ is reasonable: it spans Mars‑sized planets to super‑Jovians, excluding only obviously absurd radii.
  - As a **habitability filter**, it is too broad. A habitability‑oriented upper bound would be $\sim 2–2.5\,R_\oplus$, similar to what the pipeline already uses when defining `habitable`.

- **Key sources**:
  - General terrestrial vs mini‑Neptune discussion: *Planetary habitability*, Wikipedia (`https://en.wikipedia.org/wiki/Planetary_habitability`).
  - Habitable‑zone exoplanet catalogs: e.g. *A Catalog of Habitable Zone Exoplanets*, AJ, 2022 (`https://iopscience.iop.org/article/10.3847/1538-3881/aca1c0`).

---

## 2. Planet Mass (`mass_earth`)

- **Code constraint**:  
  $0.01 < M_p/M_\oplus < 5000$

- **Physical interpretation**:
  - **Lower bound 0.01**: 1% Earth mass—dwarf‑planet scale, well below Mars (0.107 $M_\oplus$).
  - **Upper bound 5000**: $\sim 15–16\,M_J$ (since $1\,M_J \approx 318\,M_\oplus$), entering the **brown dwarf** regime rather than normal planets.

- **Habitability perspective**:
  - Kopparapu et al. (2014) study habitable zones for **0.1–5 $M_\oplus$** planets. Their results and subsequent work suggest:
    - Below $\sim 0.1\,M_\oplus$, planets have difficulty retaining atmospheres and liquid water over long timescales.
    - Above $\sim 5–10\,M_\oplus$, planets are more likely to be volatile‑rich (super‑Earths / mini‑Neptunes) with deep envelopes rather than Earth‑like rocky surfaces.
  - Thus, $0.1–5\,M_\oplus$ is a commonly used mass range for “Earth‑like” habitability modeling.

- **Assessment**:
  - For a **general physical sanity check**, 0.01–5000 $M_\oplus$ is extremely broad and admits many clearly uninhabitable objects (gas giants, brown‑dwarf‑like companions).
  - For **CTGAN conditioned on habitability**, the post‑hoc filter should itself be restrictive: a range like $0.1 \lesssim M_p/M_\oplus \lesssim 5$ is much more appropriate and consistent with mass‑dependent habitable‑zone calculations (effectively assuming a rocky or super‑Earth regime).

- **Key sources**:
  - *Habitable Zones around Main‑Sequence Stars: Dependence on Planetary Mass* (Kopparapu et al. 2014), ApJ 787 L29 (`https://ui.adsabs.harvard.edu/abs/2014ApJ...787L..29K/abstract`, preprint at `https://hal.science/hal-01016703/`).

---

## 3. Semi‑Major Axis (`semi_major_axis_au`)

- **Code constraint**:  
  $a > 0.001$ AU

- **Physical interpretation**:
  - 0.001 AU $\approx 150{,}000$ km, about 0.4× the Earth–Moon distance, and well **inside** the Roche limit for any main‑sequence star. Orbits this close are not long‑term stable planets.
  - The bound simply forbids orbits at essentially zero distance.

- **Habitability perspective**:
  - The **habitable zone distance** depends on stellar luminosity and effective temperature:
    - For Sun‑like stars, conservative HZ: $\sim 0.95–1.7$ AU; optimistic: $\sim 0.75–1.8$ AU.
    - For cool M dwarfs, the HZ can be as close as $\sim 0.02–0.2$ AU.
  - Any planet in the classical HZ thus has $a$ values many orders of magnitude above 0.001 AU.

- **Assessment**:
  - As a **minimal sanity check**, $a > 0.001$ AU only removes nearly zero or negative semi‑major axes and is therefore too loose to say anything meaningful about habitability.
  - For **CTGAN focused on habitable planets**, the semi‑major‑axis filter should be tightened; for example, one can require $a \gtrsim 0.01$ AU (to avoid clearly unphysical ultra‑tight orbits) and $a \lesssim 5$ AU, while the actual “habitable zone” constraint is enforced more rigorously via the insolation cut ($0.2 \leq S_\mathrm{norm} \leq 2.0$) and stellar properties.
  - It does **not** encode the habitable zone itself; that role is played by the insolation‑based and `habitable` label logic.

- **Key sources**:
  - *Habitable zone*, Wikipedia (`https://en.wikipedia.org/wiki/Habitable_zone`).
  - *Calculation of Habitable Zones* – Virtual Planetary Laboratory (`https://vpl.uw.edu/calculation-of-habitable-zones/`).

---

## 4. Stellar Effective Temperature (`star_teff_K`)

- **Code constraint**:  
  $2300 < T_\mathrm{eff} < 10000$ K

- **Physical interpretation**:
  - **Lower bound 2300 K**: Very cool late M‑dwarfs or borderline substellar objects.
  - **Upper bound 10000 K**: Early A‑type stars: hot, luminous, and relatively short‑lived on the main sequence.

- **Habitability perspective**:
  - Kopparapu’s habitable zone models and many exoplanet habitability studies focus on **main‑sequence stars with $T_\mathrm{eff} \sim 2600–7200$ K**, roughly spanning M to F spectral types.
  - Above $\sim 7200$ K (early A‑type stars), stellar lifetimes (hundreds of Myr to $\lesssim 1$ Gyr) and intense UV environments make long‑term biological evolution less likely.

- **Assessment**:
  - As a **sanity filter**, 2300–10000 K keeps stars in a broad, physically plausible temperature range and excludes extremely unphysical values.
  - For **habitability**, a more literature‑aligned range is $T_\mathrm{eff} \sim 2600–7200$ K. The existing `habitable` label in `prepare_data` already uses 2500–7000 K, close to this.

- **Key sources**:
  - *Planetary habitability*, Wikipedia (`https://en.wikipedia.org/wiki/Planetary_habitability`).
  - *Habitable Zones around Main‑Sequence Stars: Dependence on Planetary Mass* (Kopparapu et al. 2014) as above.

---

## 5. Equilibrium Temperature (`t_eq_K`)

- **Code constraint**:  
  $10 < T_\mathrm{eq} < 3000$ K

- **Physical interpretation**:
  - **Lower bound 10 K**: Colder than any planet in the classical habitable zones of main‑sequence stars; more like distant or free‑floating bodies.
  - **Upper bound 3000 K**: Extremely hot, beyond typical ultra‑hot Jupiter equilibrium temperatures, and well above any plausible solid surface equilibrium.

- **Habitability perspective**:
  - Liquid water is stable at surface temperatures of roughly **273–373 K** (0–100 °C). Equilibrium temperature is generally **lower** than surface temperature because of greenhouse warming.
  - For Earth‑like albedo and greenhouse, habitable‑zone equilibrium temperatures in climate models are often **$\sim 180–300$ K**.
  - A “temperate” equilibrium‑temperature cut might be something like **150–350 K**, depending on assumptions about albedo and greenhouse effect.

- **Assessment**:
  - As a **sanity filter**, 10–3000 K is a wide but reasonable band for “not absurd” planets across a wide range of stellar types and orbital separations.
  - It is **not a habitability constraint**; temperateness is instead approximated by the insolation cut in the `habitable` label (`pl_insol` between 0.2 and 2.0 Earth).

- **Key sources**:
  - Qualitative relation between equilibrium and surface temperature: *Planetary habitability*, Wikipedia (`https://en.wikipedia.org/wiki/Planetary_habitability`).
  - Climate‑based HZ frameworks: *Calculation of Habitable Zones* – VPL (`https://vpl.uw.edu/calculation-of-habitable-zones/`).

---

## 6. Orbital Period (`period_days`)

- **Code constraint**:  
  $P > 0.1$ days

- **Physical interpretation**:
  - 0.1 d = 2.4 hours. This is essentially a lower bound preventing extremely fast, unphysical orbits around main‑sequence stars.
  - Observed ultra‑short‑period planets have periods down to about **0.8–1 days**; shorter orbits would be extremely close to the star and prone to tidal disruption.

- **Habitability perspective**:
  - HZ orbital periods:
    - For M dwarfs: typically a few days to several tens of days.
    - For Sun‑like stars: a few hundred days (Earth: 365 d).
  - Therefore, $P > 0.1$ days has no practical effect on habitable‑zone planets; it simply filters out non‑physical edge cases.

- **Assessment**:
  - As a **sanity check**, the lower bound is fine: it removes orbits that are obviously incompatible with long‑term planet stability.
  - It does **not** contribute to selecting habitable planets; the true HZ is defined by insolation and stellar properties, with period implied via Kepler’s third law.

- **Key sources**:
  - HZ distance and period ranges inferred from *Habitable zone*, Wikipedia (`https://en.wikipedia.org/wiki/Habitable_zone`) and the exoplanet catalogs in *A Catalog of Habitable Zone Exoplanets* (`https://iopscience.iop.org/article/10.3847/1538-3881/aca1c0`).

---

## 7. Relationship to the `habitable` Label in `prepare_data`

In `prepare_data`, the pipeline defines a **binary `habitable` label**:

```46:51:/home/lasotar/stud/hackathons/hack4sages/Hack-4-Sages/modules/data_augmentation.py
        data["habitable"] = (
            data["pl_radj"].between(0.5, 2.5)
            & data["pl_insol"].between(0.2, 2.0)
            & data["st_teff"].between(2500, 7000)
        ).astype(int)
```

This label already encodes a **literature‑inspired habitability proxy**:

- **Planet radius** 0.5–2.5 $R_\oplus$: consistent with focusing on rocky or small sub‑Neptune regimes.
- **Insolation** 0.2–2.0 $S_\oplus$: approximates optimistic habitable zone limits (between “Recent Venus” and “Early Mars” equivalents).
- **Stellar effective temperature** 2500–7000 K: close to the 2600–7200 K range used for HZ modeling around main‑sequence stars.

The CTGAN is then trained and **conditioned on this `habitable` label**, so the main habitability physics is already present at the **labeling and conditioning** stage. The post‑hoc `validate_synthetic_data` function is intentionally broader: it acts as a **sanity layer** to discard egregiously unphysical synthetic samples that slip through the generator.

---

## 8. Overall Conclusions

- **Scope of the current filters**:
  - Each numerical range in `validate_synthetic_data` is a **broad physical plausibility bound**, intended to prevent obviously nonsensical planet and star parameters produced by the CTGAN.
  - They are **not meant to be strict habitability criteria**; those are primarily enforced via the `habitable` label (radius, insolation, stellar \(T_\mathrm{eff}\)).

- **Compatibility with literature**:
  - Planetary radius and mass ranges encompass known exoplanet classes from sub‑Earth to super‑Jovian and low‑mass brown dwarfs.
  - Stellar effective temperature bounds include most main‑sequence stars, though they extend somewhat beyond the typical 2600–7200 K range used in habitable‑zone calculations.
  - Semi‑major axis, equilibrium temperature, and period constraints are deliberately loose lower/upper bounds that only remove grossly unphysical values.

- **If stricter “habitability” filters are desired at this stage**:
  - Radius: tighten to $\lesssim 2–2.5\,R_\oplus$.
  - Mass: tighten to $\sim 0.1–5\,M_\oplus$.
  - Stellar $T_\mathrm{eff}$: narrow to $\sim 2600–7200$ K.
  - Equilibrium temperature or insolation: restrict to a band consistent with temperate conditions (e.g., $T_\mathrm{eq} \sim 150–350$ K or similar insolation‑based constraints).

The current design is therefore consistent with the project goal of generating **synthetic but physically plausible** planets, while leaving habitability selection mostly to the earlier labeling and conditioning steps.

