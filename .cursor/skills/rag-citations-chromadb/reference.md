# RAG Corpus — 40 Peer-Reviewed Papers

Canonical catalog for the `_PAPERS` list in `modules/rag_citations.py`.
Every entry here must have a corresponding dict in `_PAPERS` with all eight
required fields: `id`, `title`, `authors`, `year`, `journal`, `abstract`,
`topics`, `key_findings`.

## Habitable Zone & Habitability Metrics

| ID | Authors | Year | Title | Journal | Topics |
|----|---------|------|-------|---------|--------|
| `kopparapu2013` | Kopparapu, R.K. et al. | 2013 | Habitable Zones Around Main-Sequence Stars: New Estimates | ApJ, 765, 131 | habitable_zone, hz_boundaries, climate_modeling |
| `schulze-makuch2011` | Schulze-Makuch, D. et al. | 2011 | A Two-Tiered Approach to Assessing the Habitability of Exoplanets | Astrobiology, 11(10), 1041-1052 | habitability_metrics, esi |
| `kasting1993` | Kasting, J.F., Whitmire, D.P., Reynolds, R.T. | 1993 | Habitable Zones Around Main Sequence Stars | Icarus, 101, 108-128 | habitable_zone, hz_boundaries, climate_modeling |
| `rodriguez-mozos2017` | Rodríguez-Mozos, J.M. & Moya, A. | 2017 | SEPHI: A Scoring System for Exoplanet Habitability | MNRAS, 471(4), 4628-4636 | habitability_metrics, sephi |

## Tidal Locking & Climate States

| ID | Authors | Year | Title | Journal | Topics |
|----|---------|------|-------|---------|--------|
| `turbet2016` | Turbet, M. et al. | 2016 | The habitability of Proxima Centauri b | A&A, 596, A112 | gcm, tidal_locking, m_dwarf, climate_modeling, proxima_centauri |
| `shields2016` | Shields, A.L. et al. | 2016 | The habitability of planets orbiting M-dwarf stars | Physics Reports, 663, 1-38 | m_dwarf, tidal_locking, habitability, stellar_activity, atmospheric_escape |
| `leconte2013` | Leconte, J. et al. | 2013 | 3D climate modeling of close-in land planets | A&A, 554, A69 | gcm, tidal_locking, climate_modeling, atmospheric_dynamics |
| `pierrehumbert2011` | Pierrehumbert, R.T. | 2011 | A Palette of Climates for Gliese 581g | ApJ Letters, 726, L8 | tidal_locking, climate_modeling, climate_states |
| `wordsworth2015` | Wordsworth, R. | 2015 | Atmospheric Heat Redistribution and Collapse on Tidally Locked Rocky Planets | ApJ, 806, 180 | atmospheric_collapse, tidal_locking, atmospheric_dynamics |

## Biosignatures & False Positives

| ID | Authors | Year | Title | Journal | Topics |
|----|---------|------|-------|---------|--------|
| `meadows2018` | Meadows, V.S. et al. | 2018 | Exoplanet Biosignatures: Understanding Oxygen as a Biosignature in the Context of Its Environment | Astrobiology, 18(6), 630-662 | biosignatures, false_positives, uv_environment, m_dwarf, atmospheric_chemistry |
| `seager2016` | Seager, S. et al. | 2016 | Toward a List of Molecules as Potential Biosignature Gases | Astrobiology, 16(6), 465-485 | biosignatures, atmospheric_chemistry, spectroscopy |
| `luger2015` | Luger, R. & Barnes, R. | 2015 | Extreme Water Loss and Abiotic O2 Buildup on Planets Throughout the Habitable Zones of M Dwarfs | Astrobiology, 15(2), 119-143 | false_positives, m_dwarf, atmospheric_escape, water_loss, pre_main_sequence |
| `schwieterman2018` | Schwieterman, E.W. et al. | 2018 | Exoplanet Biosignatures: A Review of Remotely Detectable Signs of Life | Astrobiology, 18(6), 663-708 | biosignatures, atmospheric_chemistry, remote_detection, spectroscopy |
| `catling2018` | Catling, D.C. et al. | 2018 | Exoplanet Biosignatures: A Framework for Their Assessment | Astrobiology, 18(6), 709-738 | biosignatures, framework, bayesian_assessment, false_positives |

## Mass-Radius & Planetary Interiors

| ID | Authors | Year | Title | Journal | Topics |
|----|---------|------|-------|---------|--------|
| `chen_kipping2017` | Chen, J. & Kipping, D.M. | 2017 | Probabilistic Forecasting of the Masses and Radii of Other Worlds | ApJ, 834, 17 | mass_radius, planetary_interior, exoplanet_characterization |
| `kite2009` | Kite, E.S. et al. | 2009 | Geodynamics and Rate of Volcanism on Massive Earth-like Planets | ApJ, 700, 1732 | volcanism, planetary_interior, plate_tectonics, super_earth, outgassing |
| `zeng2019` | Zeng, L. et al. | 2019 | Growth model interpretation of planet size distribution | PNAS, 116(20), 9723-9728 | mass_radius, planetary_interior, composition, exoplanet_characterization |
| `stamenkovic2012` | Stamenković, V. et al. | 2012 | The Influence of Pressure-dependent Viscosity on the Thermal Evolution of Super-Earths | ApJ, 748, 41 | plate_tectonics, planetary_interior, super_earth, mantle_dynamics |
| `walker1981` | Walker, J.C.G., Hays, P.B., Kasting, J.F. | 1981 | A negative feedback mechanism for the long-term stabilization of Earth's surface temperature | JGR, 86, 9776-9782 | carbonate_silicate_cycle, climate_feedback, habitability, plate_tectonics |
| `driscoll_barnes2015` | Driscoll, P.E. & Barnes, R. | 2015 | Tidal Heating of Earth-like Exoplanets around M Stars: Thermal, Magnetic, and Orbital Evolutions | Astrobiology, 15(9), 739-760 | tidal_heating, magnetic_field, m_dwarf, planetary_interior, orbital_dynamics |

## Atmospheric Science

| ID | Authors | Year | Title | Journal | Topics |
|----|---------|------|-------|---------|--------|
| `owen_wu2013` | Owen, J.E. & Wu, Y. | 2013 | Kepler Planets: A Tale of Evaporation | ApJ, 775, 105 | atmospheric_escape, radius_valley, exoplanet_evolution, xuv_flux |
| `goldblatt2013` | Goldblatt, C. et al. | 2013 | Low simulated radiation limit for runaway greenhouse climates | Nature Geoscience, 6, 661-667 | runaway_greenhouse, climate_modeling, habitable_zone, atmospheric_physics |
| `zahnle_catling2017` | Zahnle, K.J. & Catling, D.C. | 2017 | The Cosmic Shoreline: The Evidence that Escape Determines which Planets Have Atmospheres | ApJ, 843, 122 | atmospheric_escape, atmospheric_retention, cosmic_shoreline |
| `tian2015` | Tian, F. et al. | 2015 | Water Loss from Young Planets | Earth and Planetary Science Letters, 432, 126-132 | atmospheric_escape, water_loss, xuv_flux, pre_main_sequence |

## Alternative Biochemistry

| ID | Authors | Year | Title | Journal | Topics |
|----|---------|------|-------|---------|--------|
| `petkowski2020` | Petkowski, J.J., Bains, W., Seager, S. | 2020 | On the Potential of Silicon as a Building Block for Life | Life, 10(6), 84 | alternative_biochemistry, astrobiology |

## Stellar Context

| ID | Authors | Year | Title | Journal | Topics |
|----|---------|------|-------|---------|--------|
| `lammer2009` | Lammer, H. et al. | 2009 | What makes a planet habitable? | A&A Review, 17, 181-249 | stellar_activity, habitability, atmospheric_escape, magnetic_field |
| `ramirez_kaltenegger2014` | Ramirez, R.M. & Kaltenegger, L. | 2014 | The Habitable Zones of Pre-Main-Sequence Stars | ApJ Letters, 797, L25 | habitable_zone, pre_main_sequence, hz_boundaries, stellar_evolution |
| `segura2010` | Segura, A. et al. | 2010 | The Effect of a Strong Stellar Flare on the Atmospheric Chemistry of an Earth-like Planet Orbiting an M Dwarf | Astrobiology, 10(7), 751-771 | stellar_flares, m_dwarf, atmospheric_chemistry, uv_environment, ozone |
| `france2013` | France, K. et al. | 2013 | The Ultraviolet Radiation Environment around M dwarf Exoplanet Host Stars | ApJ, 763, 149 | uv_environment, m_dwarf, stellar_activity, spectral_energy_distribution |

## Observational / JWST

| ID | Authors | Year | Title | Journal | Topics |
|----|---------|------|-------|---------|--------|
| `madhusudhan2023` | Madhusudhan, N. et al. | 2023 | Carbon-bearing Molecules in a Possible Hycean Atmosphere of an Exoplanet | ApJ Letters, 956, L13 | jwst, transit_spectroscopy, atmospheric_characterization, sub_neptune, biosignatures |
| `lustig-yaeger2023` | Lustig-Yaeger, J. et al. | 2023 | A JWST transmission spectrum of the nearby Earth-sized exoplanet LHS 475 b | Nature Astronomy, 7, 1317-1328 | jwst, transit_spectroscopy, rocky_planet, atmospheric_characterization |
| `greene2023` | Greene, T.P. et al. | 2023 | Thermal emission from the Earth-sized exoplanet TRAPPIST-1 b using JWST | Nature, 618, 39-42 | jwst, thermal_emission, trappist1, rocky_planet |
| `benneke2019` | Benneke, B. et al. | 2019 | Water Vapor and Clouds on the Habitable-Zone Sub-Neptune Exoplanet K2-18b | ApJ Letters, 887, L14 | transit_spectroscopy, atmospheric_characterization, sub_neptune, water_detection |

## Climate Modeling

| ID | Authors | Year | Title | Journal | Topics |
|----|---------|------|-------|---------|--------|
| `wolf_toon2015` | Wolf, E.T. & Toon, O.B. | 2015 | The evolution of habitable climates under the brightening Sun | JGR Atmospheres, 120, 5775-5794 | cloud_feedback, climate_modeling, habitable_zone, gcm |
| `yang2013` | Yang, J., Cowan, N.B., Abbot, D.S. | 2013 | Stabilizing Cloud Feedback Dramatically Expands the Habitable Zone of Tidally Locked Planets | ApJ Letters, 771, L45 | cloud_feedback, tidal_locking, habitable_zone, gcm, climate_modeling |
| `hu_yang2014` | Hu, Y. & Yang, J. | 2014 | Role of ocean heat transport in climates of tidally locked exoplanets around M dwarf stars | PNAS, 111(2), 629-634 | ocean_heat_transport, tidal_locking, climate_modeling, gcm |
| `joshi1997` | Joshi, M.M., Haberle, R.M., Reynolds, R.T. | 1997 | Simulations of the Atmospheres of Synchronously Rotating Terrestrial Planets Orbiting M Dwarfs | Icarus, 129, 450-465 | gcm, tidal_locking, m_dwarf, atmospheric_dynamics, habitability |
| `del_genio2019` | Del Genio, A.D. et al. | 2019 | Habitable Climate Scenarios for Proxima Centauri b with a Dynamic Ocean | Astrobiology, 19(1), 99-125 | gcm, climate_modeling, ocean_dynamics, proxima_centauri, tidal_locking |

## Astrobiology

| ID | Authors | Year | Title | Journal | Topics |
|----|---------|------|-------|---------|--------|
| `cockell2016` | Cockell, C.S. et al. | 2016 | Habitability: A Review | Astrobiology, 16(1), 89-117 | habitability, astrobiology, definition, limits_of_life |
| `raven_cockell2006` | Raven, J.A. & Cockell, C.S. | 2006 | Influence on Photosynthesis of Starlight, Moonlight, Planetlight, and Light Pollution | Astrobiology, 6(4), 668-675 | photosynthesis, m_dwarf, astrobiology, spectral_energy_distribution |

## Topic Taxonomy

| Domain | Tags |
|--------|------|
| Habitable Zone | `habitable_zone`, `hz_boundaries` |
| Habitability Metrics | `habitability_metrics`, `esi`, `sephi` |
| Climate Modeling | `climate_modeling`, `gcm`, `climate_states`, `cloud_feedback`, `ocean_heat_transport` |
| Tidal Locking | `tidal_locking`, `atmospheric_collapse`, `atmospheric_dynamics` |
| Biosignatures | `biosignatures`, `false_positives`, `spectroscopy`, `remote_detection` |
| Atmospheric Science | `atmospheric_escape`, `atmospheric_chemistry`, `runaway_greenhouse`, `atmospheric_retention` |
| Stellar Context | `m_dwarf`, `stellar_activity`, `stellar_flares`, `uv_environment`, `pre_main_sequence` |
| Planetary Interiors | `planetary_interior`, `plate_tectonics`, `volcanism`, `tidal_heating`, `magnetic_field`, `mass_radius`, `composition` |
| Observational | `jwst`, `transit_spectroscopy`, `thermal_emission`, `water_detection` |
| Astrobiology | `astrobiology`, `habitability`, `photosynthesis`, `alternative_biochemistry`, `limits_of_life` |
