# Paper Catalog — Full Reference

Canonical list of the 40 peer-reviewed papers indexed in the ChromaDB
`astro_papers` collection (`modules/rag_citations.py`).

Papers are organised by scientific domain. Each entry lists the paper ID,
bibliographic metadata, a summary of the abstract, and the usage context
within the project.

---

## Habitable Zone & Habitability Metrics

### 1. kopparapu2013

- **Title:** Habitable Zones Around Main-Sequence Stars: New Estimates
- **Authors:** Kopparapu, R.K. et al.
- **Year:** 2013
- **Journal:** ApJ, 765, 131
- **Topics:** `habitable_zone`, `hz_boundaries`, `climate_modeling`
- **Abstract:** Updated HZ boundaries using 1-D climate models with revised H₂O and CO₂ absorption coefficients. Conservative HZ: 0.99–1.70 AU for the Sun. Parameterised as 4th-degree polynomial in T_eff.
- **Usage:** HZ boundary calculations (`modules/astro_physics.py :: hz_boundaries`).

### 2. schulze-makuch2011

- **Title:** A Two-Tiered Approach to Assessing the Habitability of Exoplanets
- **Authors:** Schulze-Makuch, D. et al.
- **Year:** 2011
- **Journal:** Astrobiology, 11(10), 1041-1052
- **Topics:** `habitability_metrics`, `esi`
- **Abstract:** Proposes ESI and PHI as complementary habitability metrics. ESI: geometric mean of 4 weighted similarity ratios (radius, density, v_esc, T_surf). ESI measures Earth-likeness, not habitability.
- **Usage:** ESI computation (`modules/astro_physics.py :: compute_esi`).

### 3. kasting1993

- **Title:** Habitable Zones Around Main Sequence Stars
- **Authors:** Kasting, J.F., Whitmire, D.P., Reynolds, R.T.
- **Year:** 1993
- **Journal:** Icarus, 101, 108-128
- **Topics:** `habitable_zone`, `hz_boundaries`, `climate_modeling`
- **Abstract:** Seminal HZ calculation. Inner edge: water loss. Outer edge: maximum CO₂ greenhouse. Conservative HZ ~0.95–1.37 AU for present Sun.
- **Usage:** Historical HZ context, foundational reference.

### 4. rodriguez-mozos2017

- **Title:** SEPHI: A Scoring System for Exoplanet Habitability
- **Authors:** Rodríguez-Mozos, J.M. & Moya, A.
- **Year:** 2017
- **Journal:** MNRAS, 471(4), 4628-4636
- **Topics:** `habitability_metrics`, `sephi`
- **Abstract:** SEPHI: 3 binary criteria (thermal, atmospheric retention, magnetic field). Score = criteria met / 3.
- **Usage:** SEPHI computation (`modules/astro_physics.py :: compute_sephi`).

---

## Tidal Locking & Climate States

### 5. turbet2016

- **Title:** The habitability of Proxima Centauri b
- **Authors:** Turbet, M. et al.
- **Year:** 2016
- **Journal:** A&A, 596, A112
- **Topics:** `gcm`, `tidal_locking`, `m_dwarf`, `climate_modeling`, `proxima_centauri`
- **Abstract:** 3-D GCM of Proxima Cen b. Eyeball climate with substellar T ~280–310 K. Surface liquid water for wide atmospheric range.
- **Usage:** Climate model validation, tidally locked planet references.

### 6. shields2016

- **Title:** The habitability of planets orbiting M-dwarf stars
- **Authors:** Shields, A.L. et al.
- **Year:** 2016
- **Journal:** Physics Reports, 663, 1-38
- **Topics:** `m_dwarf`, `tidal_locking`, `habitability`, `stellar_activity`, `atmospheric_escape`
- **Abstract:** Comprehensive M-dwarf habitability review. Tidal locking, stellar activity, atmospheric erosion, pre-MS desiccation. GCMs show viable heat redistribution.
- **Usage:** Tidal locking and M-dwarf context.

### 7. leconte2013

- **Title:** 3D climate modeling of close-in land planets
- **Authors:** Leconte, J. et al.
- **Year:** 2013
- **Journal:** A&A, 554, A69
- **Topics:** `gcm`, `tidal_locking`, `climate_modeling`, `atmospheric_dynamics`
- **Abstract:** Thick (≥1 bar) atmospheres reduce day-night ΔT by 50–80%. Minimum ~0.1 bar to prevent nightside collapse.
- **Usage:** Climate surrogate validation context.

### 8. pierrehumbert2011

- **Title:** A Palette of Climates for Gliese 581g
- **Authors:** Pierrehumbert, R.T.
- **Year:** 2011
- **Journal:** ApJ Letters, 726, L8
- **Topics:** `tidal_locking`, `climate_modeling`, `climate_states`
- **Abstract:** Climate state taxonomy: eyeball, lobster, snowball. Climate topology depends on greenhouse strength and heat transport.
- **Usage:** Climate state classification terminology.

### 9. wordsworth2015

- **Title:** Atmospheric Heat Redistribution and Collapse on Tidally Locked Rocky Planets
- **Authors:** Wordsworth, R.
- **Year:** 2015
- **Journal:** ApJ, 806, 180
- **Topics:** `atmospheric_collapse`, `tidal_locking`, `atmospheric_dynamics`
- **Abstract:** Atmospheric collapse when nightside T drops below condensation point. Minimum ~0.1 bar N₂ to avoid collapse. Analytical scaling laws.
- **Usage:** Atmospheric collapse threshold calculations.

---

## Biosignatures & False Positives

### 10. meadows2018

- **Title:** Exoplanet Biosignatures: Understanding Oxygen as a Biosignature in the Context of Its Environment
- **Authors:** Meadows, V.S. et al.
- **Year:** 2018
- **Journal:** Astrobiology, 18(6), 630-662
- **Topics:** `biosignatures`, `false_positives`, `uv_environment`, `m_dwarf`, `atmospheric_chemistry`
- **Abstract:** Abiotic O₂ via CO₂ photolysis, H₂O photolysis, radiolysis. Decision tree for biological vs abiotic O₂. Disequilibrium pairs (O₂+CH₄).
- **Usage:** Photochemical false-positive mitigation logic.

### 11. seager2016

- **Title:** Toward a List of Molecules as Potential Biosignature Gases
- **Authors:** Seager, S. et al.
- **Year:** 2016
- **Journal:** Astrobiology, 16(6), 465-485
- **Topics:** `biosignatures`, `atmospheric_chemistry`, `spectroscopy`
- **Abstract:** Systematic evaluation of biosignature gases. Key: O₂, O₃, CH₄, N₂O, DMS. Candidates: PH₃, isoprene.
- **Usage:** Biosignature assessment context.

### 12. luger2015

- **Title:** Extreme Water Loss and Abiotic O₂ Buildup on Planets Throughout the Habitable Zones of M Dwarfs
- **Authors:** Luger, R. & Barnes, R.
- **Year:** 2015
- **Journal:** Astrobiology, 15(2), 119-143
- **Topics:** `false_positives`, `m_dwarf`, `atmospheric_escape`, `water_loss`, `pre_main_sequence`
- **Abstract:** Pre-MS water loss: up to several Earth-oceans. Residual O₂: tens to hundreds of bars. Late M dwarfs highest risk.
- **Usage:** O₂ false-positive flag, water loss context.

### 13. schwieterman2018

- **Title:** Exoplanet Biosignatures: A Review of Remotely Detectable Signs of Life
- **Authors:** Schwieterman, E.W. et al.
- **Year:** 2018
- **Journal:** Astrobiology, 18(6), 663-708
- **Topics:** `biosignatures`, `atmospheric_chemistry`, `remote_detection`, `spectroscopy`
- **Abstract:** Comprehensive review. O₂+CH₄ disequilibrium, vegetation red edge at ~750 nm, anti-biosignatures.
- **Usage:** Biosignature reference for agent narratives.

### 14. catling2018

- **Title:** Exoplanet Biosignatures: A Framework for Their Assessment
- **Authors:** Catling, D.C. et al.
- **Year:** 2018
- **Journal:** Astrobiology, 18(6), 709-738
- **Topics:** `biosignatures`, `framework`, `bayesian_assessment`, `false_positives`
- **Abstract:** Bayesian framework: P(life|data). Contextual information shifts posterior by orders of magnitude.
- **Usage:** Biosignature assessment methodology.

---

## Mass-Radius & Planetary Interiors

### 15. chen_kipping2017

- **Title:** Probabilistic Forecasting of the Masses and Radii of Other Worlds
- **Authors:** Chen, J. & Kipping, D.M.
- **Year:** 2017
- **Journal:** ApJ, 834, 17
- **Topics:** `mass_radius`, `planetary_interior`, `exoplanet_characterization`
- **Abstract:** Piecewise M-R relation: Terran R~M^0.28, Neptunian R~M^0.59, Jovian plateau. Break at ~2 M_Earth.
- **Usage:** Mass-radius estimation (`modules/validators.py`).

### 16. kite2009

- **Title:** Geodynamics and Rate of Volcanism on Massive Earth-like Planets
- **Authors:** Kite, E.S. et al.
- **Year:** 2009
- **Journal:** ApJ, 700, 1732
- **Topics:** `volcanism`, `planetary_interior`, `plate_tectonics`, `super_earth`, `outgassing`
- **Abstract:** Outgassing: V̇ ~ (g/g_E)^0.75 × (age/4.5 Gyr)^-1.5. Non-monotonic: 2–5 M_E peak, >5 M_E suppressed.
- **Usage:** Outgassing model, ISA interactions (`modules/astro_physics.py`).

### 17. zeng2019

- **Title:** Growth model interpretation of planet size distribution
- **Authors:** Zeng, L. et al.
- **Year:** 2019
- **Journal:** PNAS, 116(20), 9723-9728
- **Topics:** `mass_radius`, `planetary_interior`, `composition`, `exoplanet_characterization`
- **Abstract:** PREM-based M-R-composition curves. Rocky: R~(M/M_E)^0.27. Water worlds: R~1.39×(M/M_E)^0.27. Gap at 1.5–2.0 R_E.
- **Usage:** Compositional interpretation of measured mass-radius pairs.

### 18. stamenkovic2012

- **Title:** The Influence of Pressure-dependent Viscosity on the Thermal Evolution of Super-Earths
- **Authors:** Stamenković, V. et al.
- **Year:** 2012
- **Journal:** ApJ, 748, 41
- **Topics:** `plate_tectonics`, `planetary_interior`, `super_earth`, `mantle_dynamics`
- **Abstract:** Pressure-dependent viscosity may suppress plate tectonics above ~5–8 M_Earth. Stagnant-lid vs mobile-lid depends on composition.
- **Usage:** Plate tectonics feasibility in ISA assessment.

### 19. walker1981

- **Title:** A negative feedback mechanism for the long-term stabilization of Earth's surface temperature
- **Authors:** Walker, J.C.G., Hays, P.B., Kasting, J.F.
- **Year:** 1981
- **Journal:** JGR, 86, 9776-9782
- **Topics:** `carbonate_silicate_cycle`, `climate_feedback`, `habitability`, `plate_tectonics`
- **Abstract:** Carbonate-silicate cycle as negative climate feedback. Requires plate tectonics + liquid water. Explains faint young Sun.
- **Usage:** Long-term climate stability context for ISA.

### 20. driscoll_barnes2015

- **Title:** Tidal Heating of Earth-like Exoplanets around M Stars
- **Authors:** Driscoll, P.E. & Barnes, R.
- **Year:** 2015
- **Journal:** Astrobiology, 15(9), 739-760
- **Topics:** `tidal_heating`, `magnetic_field`, `m_dwarf`, `planetary_interior`, `orbital_dynamics`
- **Abstract:** Tidal heat flux 0.1–100 W/m² for e=0.1–0.3. Optimal 0.04–2 W/m² for habitability. Extends magnetic dynamo lifetime.
- **Usage:** Tidal heating and magnetic field context for M-dwarf planets.

---

## Alternative Biochemistry

### 21. petkowski2020

- **Title:** On the Potential of Silicon as a Building Block for Life
- **Authors:** Petkowski, J.J., Bains, W., Seager, S.
- **Year:** 2020
- **Journal:** Life, 10(6), 84
- **Topics:** `alternative_biochemistry`, `astrobiology`
- **Abstract:** Si-Si bonds weaker than C-C (226 vs 346 kJ/mol). Carbon-based life far more probable. Silicon viable only in reducing >500 K.
- **Usage:** Alternative biochemistry context.

---

## Atmospheric Science

### 22. owen_wu2013

- **Title:** Kepler Planets: A Tale of Evaporation
- **Authors:** Owen, J.E. & Wu, Y.
- **Year:** 2013
- **Journal:** ApJ, 775, 105
- **Topics:** `atmospheric_escape`, `radius_valley`, `exoplanet_evolution`, `xuv_flux`
- **Abstract:** Radius valley at 1.5–2.0 R_E sculpted by XUV photoevaporation. Critical core mass ~1.5 M_E at 0.1 AU.
- **Usage:** Atmospheric escape context for close-in planets.

### 23. goldblatt2013

- **Title:** Low simulated radiation limit for runaway greenhouse climates
- **Authors:** Goldblatt, C. et al.
- **Year:** 2013
- **Journal:** Nature Geoscience, 6, 661-667
- **Topics:** `runaway_greenhouse`, `climate_modeling`, `habitable_zone`, `atmospheric_physics`
- **Abstract:** Runaway greenhouse OLR threshold ~282 W/m². Earth margin: ~42 W/m². Irreversible transition.
- **Usage:** Inner HZ edge determination, runaway greenhouse context.

### 24. zahnle_catling2017

- **Title:** The Cosmic Shoreline
- **Authors:** Zahnle, K.J. & Catling, D.C.
- **Year:** 2017
- **Journal:** ApJ, 843, 122
- **Topics:** `atmospheric_escape`, `atmospheric_retention`, `cosmic_shoreline`
- **Abstract:** Cosmic shoreline: v_esc threshold vs stellar flux separates atmosphere vs airless. Proxima b needs ≥1.5 M_E.
- **Usage:** Rapid atmospheric retention assessment.

### 25. tian2015

- **Title:** Water Loss from Young Planets
- **Authors:** Tian, F. et al.
- **Year:** 2015
- **Journal:** Earth and Planetary Science Letters, 432, 126-132
- **Topics:** `atmospheric_escape`, `water_loss`, `xuv_flux`, `pre_main_sequence`
- **Abstract:** Energy-limited escape: dM/dt ~ ε×F_XUV×π×R_p³/(G×M_p). Earth loses ~0.5 oceans in 500 Myr. M-dwarf planets lose more.
- **Usage:** Water loss estimates for young planets.

### 26. wolf_toon2015

- **Title:** The evolution of habitable climates under the brightening Sun
- **Authors:** Wolf, E.T. & Toon, O.B.
- **Year:** 2015
- **Journal:** JGR Atmospheres, 120, 5775-5794
- **Topics:** `cloud_feedback`, `climate_modeling`, `habitable_zone`, `gcm`
- **Abstract:** Low-altitude clouds provide negative feedback. Breaks at ~8–10% solar flux increase. Tidally locked substellar clouds strongest.
- **Usage:** Cloud feedback uncertainty context.

---

## Stellar Context

### 27. lammer2009

- **Title:** What makes a planet habitable?
- **Authors:** Lammer, H. et al.
- **Year:** 2009
- **Journal:** A&A Review, 17, 181-249
- **Topics:** `stellar_activity`, `habitability`, `atmospheric_escape`, `magnetic_field`
- **Abstract:** Comprehensive stellar/planetary habitability review. Mass range 0.5–10 M_E. Non-thermal escape dominates unmagnetised M-dwarf planets.
- **Usage:** Habitability factor overview.

### 28. ramirez_kaltenegger2014

- **Title:** The Habitable Zones of Pre-Main-Sequence Stars
- **Authors:** Ramirez, R.M. & Kaltenegger, L.
- **Year:** 2014
- **Journal:** ApJ Letters, 797, L25
- **Topics:** `habitable_zone`, `pre_main_sequence`, `hz_boundaries`, `stellar_evolution`
- **Abstract:** M-dwarf pre-MS HZ starts at 0.2–0.5 AU, contracts to 0.03–0.1 AU. Pre-MS duration: ~50 Myr (F) to ~1 Gyr (late M).
- **Usage:** Pre-MS volatile loss context.

### 29. segura2010

- **Title:** The Effect of a Strong Stellar Flare on the Atmospheric Chemistry of an Earth-like Planet Orbiting an M Dwarf
- **Authors:** Segura, A. et al.
- **Year:** 2010
- **Journal:** Astrobiology, 10(7), 751-771
- **Topics:** `stellar_flares`, `m_dwarf`, `atmospheric_chemistry`, `uv_environment`, `ozone`
- **Abstract:** Large M-dwarf flare destroys 94% O₃ column, UV-B increases ~50×. Recovery ~50 years. >2 bar atmosphere or >0.5 Gauss field mitigates.
- **Usage:** Stellar flare impact assessment.

### 30. france2013

- **Title:** The Ultraviolet Radiation Environment around M dwarf Exoplanet Host Stars
- **Authors:** France, K. et al.
- **Year:** 2013
- **Journal:** ApJ, 763, 149
- **Topics:** `uv_environment`, `m_dwarf`, `stellar_activity`, `spectral_energy_distribution`
- **Abstract:** M-dwarf FUV/NUV ratio 10–1000× solar. Ly-α carries 37–75% of total UV. PHOENIX underpredicts by 1–2 orders.
- **Usage:** UV flux estimation and false-positive context.

---

## Observational / JWST

### 31. madhusudhan2023

- **Title:** Carbon-bearing Molecules in a Possible Hycean Atmosphere of an Exoplanet
- **Authors:** Madhusudhan, N. et al.
- **Year:** 2023
- **Journal:** ApJ Letters, 956, L13
- **Topics:** `jwst`, `transit_spectroscopy`, `atmospheric_characterization`, `sub_neptune`, `biosignatures`
- **Abstract:** JWST NIRSpec K2-18 b: CH₄ (~1%, 3.4σ), CO₂ (~1%, 2.9σ), low CO, tentative DMS. First HZ exoplanet with detected molecules.
- **Usage:** JWST atmospheric characterisation reference.

### 32. lustig-yaeger2023

- **Title:** A JWST transmission spectrum of the nearby Earth-sized exoplanet LHS 475 b
- **Authors:** Lustig-Yaeger, J. et al.
- **Year:** 2023
- **Journal:** Nature Astronomy, 7, 1317-1328
- **Topics:** `jwst`, `transit_spectroscopy`, `rocky_planet`, `atmospheric_characterization`
- **Abstract:** Flat spectrum rules out H₂-dominated atmosphere. Consistent with bare rock or high-MMW atmosphere (CO₂/O₂/N₂).
- **Usage:** Rocky planet atmospheric constraints context.

### 33. greene2023

- **Title:** Thermal emission from the Earth-sized exoplanet TRAPPIST-1 b using JWST
- **Authors:** Greene, T.P. et al.
- **Year:** 2023
- **Journal:** Nature, 618, 39-42
- **Topics:** `jwst`, `thermal_emission`, `trappist1`, `rocky_planet`
- **Abstract:** Dayside brightness T = 503±26 K. Consistent with bare rock, no atmosphere. Rules out ≥10 bar CO₂.
- **Usage:** TRAPPIST-1 system observational context.

### 34. benneke2019

- **Title:** Water Vapor and Clouds on the Habitable-Zone Sub-Neptune Exoplanet K2-18b
- **Authors:** Benneke, B. et al.
- **Year:** 2019
- **Journal:** ApJ Letters, 887, L14
- **Topics:** `transit_spectroscopy`, `atmospheric_characterization`, `sub_neptune`, `water_detection`
- **Abstract:** H₂O detected at 3.6σ via HST WFC3. First water vapour detection in HZ exoplanet. Sub-Neptune size leaves surface uncertain.
- **Usage:** Water detection reference for K2-18 b discussion.

---

## Climate Modeling

### 35. yang2013

- **Title:** Stabilizing Cloud Feedback Dramatically Expands the Habitable Zone of Tidally Locked Planets
- **Authors:** Yang, J., Cowan, N.B., Abbot, D.S.
- **Year:** 2013
- **Journal:** ApJ Letters, 771, L45
- **Topics:** `cloud_feedback`, `tidal_locking`, `habitable_zone`, `gcm`, `climate_modeling`
- **Abstract:** Substellar convective clouds (albedo 0.6–0.8) expand inner HZ by ~30% for tidally locked planets. Unique to synchronous rotators.
- **Usage:** Cloud feedback correction to HZ inner edge.

### 36. hu_yang2014

- **Title:** Role of ocean heat transport in climates of tidally locked exoplanets around M dwarf stars
- **Authors:** Hu, Y. & Yang, J.
- **Year:** 2014
- **Journal:** PNAS, 111(2), 629-634
- **Topics:** `ocean_heat_transport`, `tidal_locking`, `climate_modeling`, `gcm`
- **Abstract:** OHT increases open ocean area by 10–20%. Substellar T decreases 10–30 K; nightside T increases 20–40 K.
- **Usage:** Ocean heat transport context for climate simulations.

### 37. joshi1997

- **Title:** Simulations of the Atmospheres of Synchronously Rotating Terrestrial Planets Orbiting M Dwarfs
- **Authors:** Joshi, M.M., Haberle, R.M., Reynolds, R.T.
- **Year:** 1997
- **Journal:** Icarus, 129, 450-465
- **Topics:** `gcm`, `tidal_locking`, `m_dwarf`, `atmospheric_dynamics`, `habitability`
- **Abstract:** First GCM showing tidally locked planets habitable with ≥0.1 bar atmosphere. Equatorial superrotation prevents collapse.
- **Usage:** Foundational tidal locking habitability reference.

### 38. del_genio2019

- **Title:** Habitable Climate Scenarios for Proxima Centauri b with a Dynamic Ocean
- **Authors:** Del Genio, A.D. et al.
- **Year:** 2019
- **Journal:** Astrobiology, 19(1), 99-125
- **Topics:** `gcm`, `climate_modeling`, `ocean_dynamics`, `proxima_centauri`, `tidal_locking`
- **Abstract:** ROCKE-3D GCM. Habitable surface fractions 2–50%. Dynamic ocean adds 5–15% vs slab ocean. Eyeball or equatorial ocean.
- **Usage:** GCM validation benchmark for Proxima Cen b.

---

## Astrobiology

### 39. cockell2016

- **Title:** Habitability: A Review
- **Authors:** Cockell, C.S. et al.
- **Year:** 2016
- **Journal:** Astrobiology, 16(1), 89-117
- **Topics:** `habitability`, `astrobiology`, `definition`, `limits_of_life`
- **Abstract:** Four minimum requirements: liquid solvent, energy source, CHNOPS, survivable conditions. Known limits: -20 to 122°C. Habitability is a spectrum.
- **Usage:** Habitability definition and limits context.

### 40. raven_cockell2006

- **Title:** Influence on Photosynthesis of Starlight, Moonlight, Planetlight, and Light Pollution
- **Authors:** Raven, J.A. & Cockell, C.S.
- **Year:** 2006
- **Journal:** Astrobiology, 6(4), 668-675
- **Topics:** `photosynthesis`, `m_dwarf`, `astrobiology`, `spectral_energy_distribution`
- **Abstract:** Minimum PAR: ~1–5 µmol photons/m²/s. M-dwarf NIR emission poor for Earth chlorophylls. Bacteriochlorophylls (800–1050 nm) could enable M-dwarf photosynthesis.
- **Usage:** Photosynthesis feasibility under non-solar spectra.
