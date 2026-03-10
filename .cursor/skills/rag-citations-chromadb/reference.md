# Paper Catalog — Full Reference

Canonical list of the 15 peer-reviewed papers indexed in the ChromaDB
`astro_papers` collection (`modules/rag_citations.py`).

---

## 1. kopparapu2013

- **Title:** Habitable Zones Around Main-Sequence Stars: New Estimates
- **Authors:** Kopparapu, R.K. et al.
- **Year:** 2013
- **Journal:** ApJ, 765, 131
- **Abstract:** Updated habitable zone boundaries using 1-D climate models with revised H₂O and CO₂ absorption coefficients. Conservative HZ boundaries: 0.99–1.70 AU for the Sun.
- **Usage:** HZ boundary calculations (`modules/astro_physics.py :: hz_boundaries`), ESI context.

## 2. schulze-makuch2011

- **Title:** A Two-Tiered Approach to Assessing the Habitability of Exoplanets
- **Authors:** Schulze-Makuch, D. et al.
- **Year:** 2011
- **Journal:** Astrobiology, 11(10), 1041-1052
- **Abstract:** Proposes ESI and PHI as complementary habitability metrics. ESI measures physical similarity to Earth via radius, density, escape velocity, surface temperature.
- **Usage:** ESI computation (`modules/astro_physics.py :: earth_similarity_index`).

## 3. turbet2016

- **Title:** The habitability of Proxima Centauri b
- **Authors:** Turbet, M. et al.
- **Year:** 2016
- **Journal:** A&A, 596, A112
- **Abstract:** 3-D GCM simulations of Proxima Centauri b. Synchronous rotation produces eyeball climate state. Surface liquid water maintainable with 1 bar N₂ and variable CO₂.
- **Usage:** Climate model validation, tidally locked planet references.

## 4. shields2016

- **Title:** The habitability of planets orbiting M-dwarf stars
- **Authors:** Shields, A.L. et al.
- **Year:** 2016
- **Journal:** Physics Reports, 663, 1-38
- **Abstract:** M-dwarf habitability depends on tidal locking, stellar activity, atmospheric erosion. Atmospheric heat transport enables habitable conditions on tidally locked planets.
- **Usage:** Tidal locking and M-dwarf context.

## 5. kasting1993

- **Title:** Habitable Zones Around Main Sequence Stars
- **Authors:** Kasting, J.F., Whitmire, D.P., Reynolds, R.T.
- **Year:** 1993
- **Journal:** Icarus, 101, 108-128
- **Abstract:** Seminal HZ calculation. Inner edge: water loss. Outer edge: maximum CO₂ greenhouse. Framework later refined by Kopparapu et al. (2013).
- **Usage:** Historical HZ context, foundational reference.

## 6. meadows2018

- **Title:** Exoplanet Biosignatures: Understanding Oxygen as a Biosignature in the Context of Its Environment
- **Authors:** Meadows, V.S. et al.
- **Year:** 2018
- **Journal:** Astrobiology, 18(6), 630-662
- **Abstract:** Abiotic O₂ production via photolysis of CO₂ and H₂O around M-dwarfs. False positive identification requires understanding UV environment, atmospheric composition, geological context.
- **Usage:** Photochemical false-positive mitigation logic.

## 7. petkowski2020

- **Title:** On the Potential of Silicon as a Building Block for Life
- **Authors:** Petkowski, J.J., Bains, W., Seager, S.
- **Year:** 2020
- **Journal:** Life, 10(6), 84
- **Abstract:** Alternative biochemistries beyond carbon. Silicon's chemical limitations make carbon-based life far more probable.
- **Usage:** Alternative biochemistry context.

## 8. chen_kipping2017

- **Title:** Probabilistic Forecasting of the Masses and Radii of Other Worlds
- **Authors:** Chen, J. & Kipping, D.M.
- **Year:** 2017
- **Journal:** ApJ, 834, 17
- **Abstract:** Probabilistic mass-radius relation. Rocky planets: R ~ M^0.27. Neptunian worlds: R ~ M^0.59.
- **Usage:** Mass-radius estimation when one parameter is missing.

## 9. rodriguez-mozos2017

- **Title:** SEPHI: A Scoring System for Exoplanet Habitability
- **Authors:** Rodríguez-Mozos, J.M. & Moya, A.
- **Year:** 2017
- **Journal:** MNRAS, 471(4), 4628-4636
- **Abstract:** SEPHI evaluates habitability via thermal, atmospheric retention, and magnetic field criteria.
- **Usage:** SEPHI computation (`modules/astro_physics.py :: sephi_score`).

## 10. leconte2013

- **Title:** 3D climate modeling of close-in land planets
- **Authors:** Leconte, J. et al.
- **Year:** 2013
- **Journal:** A&A, 554, A69
- **Abstract:** Tidally locked planets: strong day-night temperature contrasts modulated by atmospheric circulation. Terminator region critical for habitability.
- **Usage:** Climate surrogate validation context.

## 11. pierrehumbert2011

- **Title:** A Palette of Climates for Gliese 581g
- **Authors:** Pierrehumbert, R.T.
- **Year:** 2011
- **Journal:** ApJ Letters, 726, L8
- **Abstract:** Climate state taxonomy: eyeball, lobster, snowball. Climate topology depends on atmospheric composition and heat transport.
- **Usage:** Climate state classification terminology.

## 12. wordsworth2015

- **Title:** Atmospheric Heat Redistribution and Collapse on Tidally Locked Rocky Planets
- **Authors:** Wordsworth, R.
- **Year:** 2015
- **Journal:** ApJ, 806, 180
- **Abstract:** Atmospheric collapse when nightside T drops below condensation point. Sets minimum atmospheric mass for habitability.
- **Usage:** Atmospheric collapse threshold calculations.

## 13. kite2009

- **Title:** Geodynamics and Rate of Volcanism on Massive Earth-like Planets
- **Authors:** Kite, E.S. et al.
- **Year:** 2009
- **Journal:** ApJ, 700, 1732
- **Abstract:** Volcanic outgassing on super-Earths. Non-monotonic relationship between gravity and outgassing rate due to competing lithospheric strength effects.
- **Usage:** Outgassing model (`modules/astro_physics.py :: outgassing_rate`), ISA interactions.

## 14. seager2016

- **Title:** Toward a List of Molecules as Potential Biosignature Gases
- **Authors:** Seager, S. et al.
- **Year:** 2016
- **Journal:** Astrobiology, 16(6), 465-485
- **Abstract:** Comprehensive biosignature gas catalog. Key biosignatures: O₂, O₃, CH₄, N₂O, DMS. Each evaluated for biological vs. abiotic production likelihood.
- **Usage:** Biosignature assessment context.

## 15. luger2015

- **Title:** Extreme Water Loss and Abiotic O₂ Buildup on Planets Throughout the Habitable Zones of M Dwarfs
- **Authors:** Luger, R. & Barnes, R.
- **Year:** 2015
- **Journal:** Astrobiology, 15(2), 119-143
- **Abstract:** M-dwarf planets lose up to several Earth-oceans of water during pre-main-sequence. Resulting O₂ buildup produces detectable O₂/O₃ without biology — major false-positive concern.
- **Usage:** O₂ false-positive flag, water loss context.
