"""
Retrieval-Augmented Generation (RAG) for scientific literature citations.

Maintains a persistent ChromaDB vector store of ~40 peer-reviewed exoplanet
and astrobiology paper abstracts with key findings. Provides hybrid search
(semantic via sentence-transformers + TF-IDF keyword scoring fused via
Reciprocal Rank Fusion) for high-precision retrieval.

Supports topic-based metadata filtering across six scientific domains:
atmospheric science, planetary interiors, stellar context, observational /
JWST, climate modeling, and astrobiology.

Gracefully degrades: if ChromaDB or sentence-transformers are not installed,
all public functions fall back to TF-IDF keyword search.
"""

import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional

# ═══════════════════════════════════════════════════════════════════════════════
#  Paper Corpus — 40 peer-reviewed papers
# ═══════════════════════════════════════════════════════════════════════════════

_PAPERS: List[Dict[str, Any]] = [
    # ── Habitable Zone & Habitability Metrics ─────────────────────────────────
    {
        "id": "kopparapu2013",
        "title": "Habitable Zones Around Main-Sequence Stars: New Estimates",
        "authors": "Kopparapu, R.K. et al.",
        "year": "2013",
        "journal": "ApJ, 765, 131",
        "abstract": (
            "We present updated habitable zone (HZ) boundaries using 1-D radiative-convective "
            "climate models with revised H2O and CO2 absorption coefficients derived from the "
            "HITRAN 2008 and HITEMP 2010 databases. The inner edge of the HZ is defined by the "
            "moist greenhouse limit where stratospheric water vapor mixing ratios exceed 3e-3, "
            "leading to rapid hydrogen escape. The outer edge is defined by the maximum greenhouse "
            "effect achievable by a CO2 atmosphere before condensation begins. For the Sun, our "
            "conservative HZ boundaries shift inward compared to Kasting et al. (1993): the moist "
            "greenhouse limit moves from 0.95 AU to 0.99 AU, and the maximum greenhouse moves from "
            "1.67 AU to 1.70 AU. We parameterise the effective stellar flux boundaries as 4th-degree "
            "polynomials in T_eff, enabling rapid computation for any main-sequence star. The "
            "optimistic HZ (recent Venus to early Mars) spans 0.75-1.77 AU for the Sun."
        ),
        "topics": ["habitable_zone", "hz_boundaries", "climate_modeling"],
        "key_findings": [
            "Conservative HZ for Sun: 0.99-1.70 AU (moist greenhouse to maximum greenhouse)",
            "Optimistic HZ for Sun: 0.75-1.77 AU (recent Venus to early Mars)",
            "S_eff parameterised as 4th-degree polynomial in T_eff - 5780 K",
            "HZ widens for cooler stars due to reduced Rayleigh scattering and shifted peak emission",
        ],
    },
    {
        "id": "schulze-makuch2011",
        "title": "A Two-Tiered Approach to Assessing the Habitability of Exoplanets",
        "authors": "Schulze-Makuch, D. et al.",
        "year": "2011",
        "journal": "Astrobiology, 11(10), 1041-1052",
        "abstract": (
            "We propose two complementary indices for evaluating exoplanet habitability: the "
            "Earth Similarity Index (ESI) and the Planetary Habitability Index (PHI). ESI "
            "quantifies how physically similar a planet is to Earth using a geometric mean "
            "of weighted similarity ratios across four parameters: radius (w=0.57), bulk "
            "density (w=1.07), escape velocity (w=0.70), and surface temperature (w=5.58). "
            "ESI ranges from 0 (no similarity) to 1 (identical to Earth). The PHI assesses "
            "habitability potential independent of Earth-likeness, incorporating substrate, "
            "energy availability, chemistry, and solvent criteria. Mars has ESI~0.70 but low "
            "PHI due to thin atmosphere, while Titan has low ESI but moderate PHI due to "
            "complex organic chemistry. We apply both indices to known exoplanets and Solar "
            "System bodies, demonstrating that high ESI does not guarantee habitability and "
            "that non-Earth-like worlds may still harbour life."
        ),
        "topics": ["habitability_metrics", "esi"],
        "key_findings": [
            "ESI = geometric mean of 4 weighted similarity ratios (radius, density, v_esc, T_surf)",
            "Weight distribution: T_surf dominates with w=5.58; radius contributes w=0.57",
            "Mars ESI ~ 0.70; Venus ESI ~ 0.44; Gliese 581g ESI ~ 0.89",
            "ESI measures Earth-likeness, not habitability — non-Earth-like worlds may still be habitable",
        ],
    },
    {
        "id": "kasting1993",
        "title": "Habitable Zones Around Main Sequence Stars",
        "authors": "Kasting, J.F., Whitmire, D.P., Reynolds, R.T.",
        "year": "1993",
        "journal": "Icarus, 101, 108-128",
        "abstract": (
            "The seminal calculation defining the circumstellar habitable zone based on "
            "radiative-convective climate modeling. The inner edge is set by water loss: "
            "as stellar flux increases, the stratosphere becomes water-saturated, enabling "
            "photodissociation and hydrogen escape to space. The outer edge is set by CO2 "
            "condensation: a dense CO2 atmosphere provides maximum greenhouse warming, but "
            "beyond a critical distance CO2 condenses, reducing the greenhouse effect. For "
            "the present Sun, the conservative HZ spans approximately 0.95-1.37 AU. We also "
            "examine continuously habitable zones (CHZ) accounting for stellar luminosity "
            "evolution over geological time. The framework establishes that HZ boundaries "
            "depend primarily on stellar luminosity and effective temperature. This work "
            "provides the foundation later refined by Kopparapu et al. (2013) with updated "
            "absorption databases."
        ),
        "topics": ["habitable_zone", "hz_boundaries", "climate_modeling"],
        "key_findings": [
            "Conservative HZ for present Sun: approximately 0.95-1.37 AU",
            "Inner edge: water loss via stratospheric H2O saturation and H escape",
            "Outer edge: maximum CO2 greenhouse before CO2 condensation",
            "Continuously habitable zone narrows when accounting for stellar evolution",
        ],
    },
    {
        "id": "rodriguez-mozos2017",
        "title": "SEPHI: A Scoring System for Exoplanet Habitability",
        "authors": "Rodríguez-Mozos, J.M. & Moya, A.",
        "year": "2017",
        "journal": "MNRAS, 471(4), 4628-4636",
        "abstract": (
            "We introduce SEPHI (Statistical-likelihood Exo-Planetary Habitability Index), "
            "a scoring system that evaluates habitability through three physically motivated "
            "binary criteria. The thermal criterion requires equilibrium temperature in the "
            "liquid water range (273-373 K). The atmospheric retention criterion requires "
            "escape velocity >= 5 km/s, sufficient to retain N2, O2, and CO2 against Jeans "
            "escape over geological timescales. The magnetic field criterion uses mass and "
            "radius proxies (M >= 0.5 M_Earth, R <= 2.5 R_Earth) to estimate whether a "
            "planet can sustain an internal dynamo. SEPHI = (criteria met) / 3, giving a "
            "score of 0, 1/3, 2/3, or 1. Applied to known rocky exoplanets, TRAPPIST-1 e "
            "scores 2/3 (thermal + atmospheric but uncertain magnetic) while Proxima Cen b "
            "scores 1/3-2/3 depending on assumed mass. The simplicity of SEPHI makes it "
            "suitable for rapid screening of large catalogs."
        ),
        "topics": ["habitability_metrics", "sephi"],
        "key_findings": [
            "Three binary criteria: thermal (273-373 K), atmospheric (v_esc >= 5 km/s), magnetic (M >= 0.5 M_E, R <= 2.5 R_E)",
            "SEPHI score = (criteria met) / 3, ranging from 0 to 1",
            "TRAPPIST-1 e scores 2/3; Proxima Cen b scores 1/3-2/3",
            "Designed for rapid catalog screening rather than detailed habitability assessment",
        ],
    },
    # ── Tidal Locking & Climate States ────────────────────────────────────────
    {
        "id": "turbet2016",
        "title": "The habitability of Proxima Centauri b",
        "authors": "Turbet, M. et al.",
        "year": "2016",
        "journal": "A&A, 596, A112",
        "abstract": (
            "We simulate Proxima Centauri b using the LMD Generic Global Climate Model under "
            "various atmospheric compositions (N2-dominated with 0.01-10 bar CO2) and both "
            "synchronous and 3:2 spin-orbit resonance configurations. For synchronous rotation, "
            "an eyeball climate state naturally emerges: a warm substellar region surrounded by "
            "frozen terrain, with open ocean possible on the dayside. With 1 bar N2 and >= 0.1 bar "
            "CO2, surface temperatures near the substellar point can reach 280-310 K. Even with "
            "minimal greenhouse forcing, atmospheric heat redistribution prevents complete nightside "
            "freeze-out. For the 3:2 resonance, equatorial regions experience two thermal maxima "
            "per orbit, broadening the habitable area. We find that water ice trapped on the "
            "nightside or in cold traps can be mobilised by episodic CO2 cycle variations."
        ),
        "topics": ["gcm", "tidal_locking", "m_dwarf", "climate_modeling", "proxima_centauri"],
        "key_findings": [
            "Synchronous rotation produces eyeball climate with substellar T ~ 280-310 K (1 bar N2 + CO2)",
            "Atmospheric heat transport prevents complete nightside atmospheric collapse",
            "3:2 spin-orbit resonance broadens habitable area via dual thermal maxima",
            "Surface liquid water maintainable for a wide range of atmospheric compositions",
        ],
    },
    {
        "id": "shields2016",
        "title": "The habitability of planets orbiting M-dwarf stars",
        "authors": "Shields, A.L. et al.",
        "year": "2016",
        "journal": "Physics Reports, 663, 1-38",
        "abstract": (
            "A comprehensive review of factors governing M-dwarf planet habitability. M dwarfs "
            "constitute ~70% of all stars and host the majority of known rocky exoplanets. Key "
            "challenges include: (1) tidal locking within the HZ, creating extreme day-night "
            "temperature contrasts that atmospheric circulation must mitigate; (2) intense stellar "
            "activity (flares, coronal mass ejections) that can strip atmospheres over Gyr timescales; "
            "(3) prolonged pre-main-sequence luminosity that subjects HZ planets to a runaway "
            "greenhouse before the star reaches the main sequence. However, GCM studies demonstrate "
            "that tidally locked planets can maintain habitable conditions via atmospheric heat "
            "redistribution and stabilising cloud feedbacks. The eyeball Earth model shows liquid "
            "water at the substellar point surrounded by global ice coverage. Magnetic field "
            "retention and atmospheric replenishment through volcanism are critical for long-term "
            "habitability around M dwarfs."
        ),
        "topics": ["m_dwarf", "tidal_locking", "habitability", "stellar_activity", "atmospheric_escape"],
        "key_findings": [
            "M dwarfs are ~70% of all stars; most known rocky exoplanets orbit M dwarfs",
            "Tidal locking within HZ is near-universal for M-dwarf planets",
            "GCMs show atmospheric heat transport can maintain habitable conditions despite synchronous rotation",
            "Pre-main-sequence super-luminous phase can desiccate HZ planets before the star settles",
        ],
    },
    {
        "id": "leconte2013",
        "title": "3D climate modeling of close-in land planets",
        "authors": "Leconte, J. et al.",
        "year": "2013",
        "journal": "A&A, 554, A69",
        "abstract": (
            "We use the LMD 3-D GCM to simulate tidally locked terrestrial planets with varying "
            "atmospheric masses and compositions. Day-night temperature contrasts are strongly "
            "modulated by atmospheric circulation: a thick (>= 1 bar) atmosphere transports enough "
            "heat via equatorial superrotation to reduce the substellar-antistellar temperature "
            "difference by 50-80%. The terminator region is the critical zone for habitability "
            "assessment, as it experiences moderate temperatures even when substellar and antistellar "
            "points are at extremes. For thin atmospheres (< 0.1 bar), heat transport is insufficient "
            "and atmospheric collapse on the nightside becomes possible, as gaseous species condense "
            "out. We identify a minimum atmospheric pressure threshold of ~0.1 bar for N2-dominated "
            "atmospheres to prevent collapse. Results are sensitive to surface friction, topography, "
            "and the presence of an ocean, which enhances thermal inertia."
        ),
        "topics": ["gcm", "tidal_locking", "climate_modeling", "atmospheric_dynamics"],
        "key_findings": [
            "Thick atmospheres (>= 1 bar) reduce day-night T contrast by 50-80% via equatorial superrotation",
            "Minimum ~0.1 bar atmospheric pressure needed to prevent nightside collapse for N2 atmospheres",
            "Terminator region is critical for habitability — moderate T even at extremes",
            "Ocean presence enhances thermal inertia and improves heat redistribution",
        ],
    },
    {
        "id": "pierrehumbert2011",
        "title": "A Palette of Climates for Gliese 581g",
        "authors": "Pierrehumbert, R.T.",
        "year": "2011",
        "journal": "ApJ Letters, 726, L8",
        "abstract": (
            "We develop an analytical and numerical framework for classifying climate states of "
            "tidally locked rocky exoplanets. Three principal climate topologies emerge depending on "
            "atmospheric composition and heat transport efficiency: (1) the eyeball state — a hot "
            "substellar point with liquid water surrounded by globally frozen surface, applicable "
            "when atmospheric heat transport is moderate; (2) the lobster state — a warm dayside "
            "with two temperate lobes extending toward the terminator, occurring with stronger "
            "atmospheric circulation; (3) the snowball state — complete surface freeze when stellar "
            "forcing is too weak or albedo too high. The transition between states depends critically "
            "on the greenhouse effect strength, atmospheric mass, and the efficiency of day-to-night "
            "heat transport. For Gliese 581g, multiple climate states are physically plausible, "
            "demonstrating fundamental degeneracy in predicting exoplanet climates from bulk parameters."
        ),
        "topics": ["tidal_locking", "climate_modeling", "climate_states"],
        "key_findings": [
            "Three climate topologies: eyeball (hot substellar), lobster (warm dayside lobes), snowball",
            "Climate state depends on greenhouse strength, atmospheric mass, and heat transport efficiency",
            "Multiple climate states are physically plausible for the same bulk planetary parameters",
            "Fundamental degeneracy: bulk observables alone cannot uniquely determine climate state",
        ],
    },
    {
        "id": "wordsworth2015",
        "title": "Atmospheric Heat Redistribution and Collapse on Tidally Locked Rocky Planets",
        "authors": "Wordsworth, R.",
        "year": "2015",
        "journal": "ApJ, 806, 180",
        "abstract": (
            "We develop an analytical theory for atmospheric heat redistribution on tidally locked "
            "planets, deriving scaling laws for the day-night temperature contrast as a function of "
            "atmospheric mass, composition, and stellar forcing. Atmospheric collapse occurs when the "
            "nightside temperature drops below the condensation point of major atmospheric constituents "
            "(e.g., N2 at ~63 K, CO2 at ~195 K at 1 bar). This sets a minimum atmospheric mass "
            "required for habitability: for an N2-dominated atmosphere, collapse is avoided when "
            "surface pressure exceeds ~0.1 bar for Earth-like instellation. The collapse threshold "
            "depends on stellar luminosity, planetary rotation rate, and atmospheric opacity. Planets "
            "near the outer edge of the HZ are more susceptible to collapse due to lower stellar "
            "forcing. We provide simple formulae for the critical pressure as a function of substellar "
            "temperature, enabling rapid assessment without full GCM simulations."
        ),
        "topics": ["atmospheric_collapse", "tidal_locking", "atmospheric_dynamics"],
        "key_findings": [
            "Atmospheric collapse occurs when nightside T drops below condensation point of major species",
            "N2 collapse threshold: ~63 K; CO2 collapse threshold: ~195 K at 1 bar",
            "Minimum ~0.1 bar N2 surface pressure to avoid collapse at Earth-like instellation",
            "Analytical scaling laws for day-night T contrast as function of atmospheric mass",
        ],
    },
    # ── Biosignatures & False Positives ───────────────────────────────────────
    {
        "id": "meadows2018",
        "title": "Exoplanet Biosignatures: Understanding Oxygen as a Biosignature in the Context of Its Environment",
        "authors": "Meadows, V.S. et al.",
        "year": "2018",
        "journal": "Astrobiology, 18(6), 630-662",
        "abstract": (
            "Molecular oxygen (O2) is considered a strong biosignature because Earth's atmospheric "
            "O2 is overwhelmingly biological in origin. However, O2 can be produced abiotically "
            "through several mechanisms: (1) photolysis of CO2 by stellar UV radiation, particularly "
            "around M dwarfs with high far-UV to near-UV ratios; (2) photolysis of H2O followed by "
            "hydrogen escape, especially during the pre-main-sequence super-luminous phase; "
            "(3) radiolysis of surface water ice. False positive identification requires contextual "
            "information: the stellar UV environment, atmospheric CO and CO2 abundances (which "
            "indicate photolysis origin), geological activity (volcanism consuming O2), and ocean "
            "presence (as a water photolysis sink). We provide a decision tree for discriminating "
            "biological from abiotic O2, emphasising that O2 should not be evaluated in isolation "
            "but as part of a disequilibrium biosignature pair (e.g., O2+CH4, O2+N2O)."
        ),
        "topics": ["biosignatures", "false_positives", "uv_environment", "m_dwarf", "atmospheric_chemistry"],
        "key_findings": [
            "Abiotic O2 sources: CO2 photolysis, H2O photolysis + H escape, water ice radiolysis",
            "M dwarfs with high FUV/NUV ratio are highest false-positive risk for O2",
            "Contextual discrimination: CO/CO2 abundance, geological activity, ocean presence",
            "O2 should be evaluated as disequilibrium pair (O2+CH4) rather than in isolation",
        ],
    },
    {
        "id": "seager2016",
        "title": "Toward a List of Molecules as Potential Biosignature Gases",
        "authors": "Seager, S. et al.",
        "year": "2016",
        "journal": "Astrobiology, 16(6), 465-485",
        "abstract": (
            "We develop a comprehensive and systematic framework for identifying molecules that "
            "could serve as remotely detectable biosignature gases in exoplanet atmospheres. "
            "Starting from the full set of stable small molecules (up to 6 non-hydrogen atoms), "
            "we evaluate each for: (1) thermodynamic stability in planetary atmospheric conditions, "
            "(2) spectroscopic detectability via absorption features in the UV/visible/IR, "
            "(3) likelihood of biological versus abiotic production. Key established biosignatures "
            "include O2, O3, CH4, N2O, and dimethyl sulfide (DMS). We identify additional candidates "
            "such as phosphine (PH3), isoprene, and methyl chloride. The biogenic production rate "
            "must exceed the abiotic production rate by a factor sufficient to build up a detectable "
            "atmospheric concentration. Context-dependent assessment is essential: a gas that is a "
            "strong biosignature in one planetary environment may be an unreliable indicator in another."
        ),
        "topics": ["biosignatures", "atmospheric_chemistry", "spectroscopy"],
        "key_findings": [
            "Key biosignatures: O2, O3, CH4, N2O, DMS; candidates: PH3, isoprene, methyl chloride",
            "Biogenic production rate must exceed abiotic rate sufficiently for atmospheric accumulation",
            "Context-dependent assessment essential — same gas can be biosignature or false positive",
            "Systematic evaluation of all stable small molecules (up to 6 non-H atoms)",
        ],
    },
    {
        "id": "luger2015",
        "title": "Extreme Water Loss and Abiotic O2 Buildup on Planets Throughout the Habitable Zones of M Dwarfs",
        "authors": "Luger, R. & Barnes, R.",
        "year": "2015",
        "journal": "Astrobiology, 15(2), 119-143",
        "abstract": (
            "M-dwarf stars have extended pre-main-sequence phases lasting 100 Myr to 1 Gyr during "
            "which their luminosity can be 10-100x higher than main-sequence values. Planets that "
            "will eventually reside in the HZ are subjected to intense irradiation during this period, "
            "driving a runaway greenhouse and massive water loss via XUV-powered hydrodynamic escape. "
            "We model this process for planets in the HZ of M0-M8 dwarfs, finding that they can lose "
            "up to several Earth-oceans of water. The resulting hydrogen escape leaves behind oxygen, "
            "which can accumulate to detectable levels (tens to hundreds of bars of O2) if geological "
            "sinks are insufficient. Late M dwarfs (M6-M8) pose the greatest risk: their prolonged "
            "pre-MS phase means HZ planets lose water for ~1 Gyr. This represents a major false-positive "
            "concern for O2 and O3 biosignature detection around M dwarfs."
        ),
        "topics": ["false_positives", "m_dwarf", "atmospheric_escape", "water_loss", "pre_main_sequence"],
        "key_findings": [
            "M-dwarf pre-MS phase: 100 Myr to 1 Gyr with luminosity 10-100x main-sequence values",
            "HZ planets can lose up to several Earth-oceans of water during pre-MS",
            "Residual O2 buildup: tens to hundreds of bars if geological sinks insufficient",
            "Late M dwarfs (M6-M8) are highest risk due to ~1 Gyr pre-MS duration",
        ],
    },
    {
        "id": "petkowski2020",
        "title": "On the Potential of Silicon as a Building Block for Life",
        "authors": "Petkowski, J.J., Bains, W., Seager, S.",
        "year": "2020",
        "journal": "Life, 10(6), 84",
        "abstract": (
            "Silicon, being in the same group as carbon, has long been proposed as an alternative "
            "backbone for biochemistry. We systematically evaluate silicon's potential by examining "
            "its chemical properties: Si-Si bonds are weaker than C-C bonds (226 vs 346 kJ/mol), "
            "Si-O bonds are extremely strong (452 kJ/mol) making silicones thermodynamically "
            "favoured over silanes in oxidising environments, and silicon lacks carbon's ability "
            "to form stable double bonds under standard conditions. While silicon polymers (silicones, "
            "silicates) are stable, they lack the functional-group diversity essential for encoding "
            "biological information. In reducing, high-temperature environments (>500 K), silicon "
            "chemistry becomes more viable, but such conditions are incompatible with liquid water "
            "and most known biochemical processes. We conclude that carbon-based life is far more "
            "probable in known planetary conditions, though silicon may play auxiliary biochemical "
            "roles (e.g., in biomineralization)."
        ),
        "topics": ["alternative_biochemistry", "astrobiology"],
        "key_findings": [
            "Si-Si bonds (226 kJ/mol) significantly weaker than C-C (346 kJ/mol)",
            "Si-O bonds (452 kJ/mol) make silicones thermodynamically preferred in oxidising environments",
            "Silicon lacks stable double bonds needed for functional-group diversity",
            "Carbon-based life far more probable; silicon viable only in reducing environments >500 K",
        ],
    },
    # ── Mass-Radius & Planetary Interiors ─────────────────────────────────────
    {
        "id": "chen_kipping2017",
        "title": "Probabilistic Forecasting of the Masses and Radii of Other Worlds",
        "authors": "Chen, J. & Kipping, D.M.",
        "year": "2017",
        "journal": "ApJ, 834, 17",
        "abstract": (
            "We present a probabilistic, piecewise mass-radius relation calibrated on a sample of "
            "316 well-characterised Solar System and exoplanetary bodies spanning nine orders of "
            "magnitude in mass. Using hierarchical Bayesian modelling, we identify natural break "
            "points in the M-R relation corresponding to physical transitions: Terran worlds "
            "(M < 2 M_Earth) follow R ~ M^0.28, Neptunian worlds (2-130 M_Earth) follow R ~ M^0.59, "
            "Jovian worlds flatten with R ~ M^-0.04 (radius plateau), and stellar objects resume "
            "positive scaling. The Terran-Neptunian transition at ~2 M_Earth reflects the onset of "
            "significant volatile/H-He envelope accretion. Our forecaster provides prediction "
            "intervals that correctly propagate uncertainties from both measurement error and intrinsic "
            "scatter. When only mass or radius is known, the complementary parameter can be estimated "
            "with typical uncertainties of 30-50%."
        ),
        "topics": ["mass_radius", "planetary_interior", "exoplanet_characterization"],
        "key_findings": [
            "Terran regime: R ~ M^0.28 for M < 2 M_Earth",
            "Neptunian regime: R ~ M^0.59 for 2-130 M_Earth",
            "Jovian radius plateau: R ~ M^-0.04 (nearly constant radius)",
            "Terran-Neptunian break at ~2 M_Earth marks onset of volatile envelope accretion",
        ],
    },
    {
        "id": "kite2009",
        "title": "Geodynamics and Rate of Volcanism on Massive Earth-like Planets",
        "authors": "Kite, E.S. et al.",
        "year": "2009",
        "journal": "ApJ, 700, 1732",
        "abstract": (
            "We model volcanic outgassing rates on super-Earths by coupling parameterised mantle "
            "convection with melt generation and eruption models. The outgassing rate scales as "
            "V_dot_rel = (g/g_Earth)^0.75 * (age/4.5 Gyr)^-1.5, reflecting the balance between "
            "higher gravity (which increases convective vigour and melt production) and stronger "
            "lithospheric resistance (which inhibits magma transport). This creates a non-monotonic "
            "relationship: moderately massive super-Earths (2-5 M_Earth) may have higher outgassing "
            "rates than Earth, while very massive planets (>5 M_Earth) have thicker stagnant lids "
            "that suppress volcanism. Outgassing is critical for the carbonate-silicate cycle and "
            "long-term climate regulation. Young planets outgas more vigorously due to higher "
            "radiogenic heating. We find that plate tectonics feasibility depends on mass, age, "
            "and internal composition."
        ),
        "topics": ["volcanism", "planetary_interior", "plate_tectonics", "super_earth", "outgassing"],
        "key_findings": [
            "Outgassing scaling: V_dot ~ (g/g_E)^0.75 * (age/4.5 Gyr)^-1.5",
            "Non-monotonic relationship: moderate super-Earths (2-5 M_E) outgas more than Earth",
            "Very massive planets (>5 M_E) develop stagnant lids suppressing volcanism",
            "Young planets have higher outgassing due to greater radiogenic heating",
        ],
    },
    # ── Atmospheric Science (new) ─────────────────────────────────────────────
    {
        "id": "owen_wu2013",
        "title": "Kepler Planets: A Tale of Evaporation",
        "authors": "Owen, J.E. & Wu, Y.",
        "year": "2013",
        "journal": "ApJ, 775, 105",
        "abstract": (
            "We propose that the observed distribution of Kepler planet radii — particularly the "
            "dearth of planets between 1.5-2.0 R_Earth (the radius valley or evaporation valley) — "
            "is sculpted by XUV-driven atmospheric photoevaporation. Close-in planets with H/He "
            "envelopes are irradiated by stellar high-energy photons (X-ray and extreme UV), which "
            "heat the upper atmosphere and drive hydrodynamic escape. Planets below a critical core "
            "mass (~1.5 M_Earth at 0.1 AU) lose their entire envelope within 100 Myr, becoming "
            "bare rocky cores. More massive cores retain some or all of their envelopes. This "
            "naturally produces the bimodal radius distribution observed by Kepler: super-Earths "
            "(stripped cores, R < 1.5 R_Earth) and sub-Neptunes (retained envelopes, R > 2 R_Earth). "
            "The evaporation timescale depends strongly on orbital distance, stellar XUV luminosity, "
            "and core mass."
        ),
        "topics": ["atmospheric_escape", "radius_valley", "exoplanet_evolution", "xuv_flux"],
        "key_findings": [
            "Radius valley at 1.5-2.0 R_Earth sculpted by XUV photoevaporation",
            "Critical core mass ~1.5 M_Earth at 0.1 AU — below this, full envelope loss within 100 Myr",
            "Bimodal distribution: super-Earths (stripped cores) vs sub-Neptunes (retained envelopes)",
            "Evaporation timescale depends on orbital distance, stellar XUV, and core mass",
        ],
    },
    {
        "id": "goldblatt2013",
        "title": "Low simulated radiation limit for runaway greenhouse climates",
        "authors": "Goldblatt, C. et al.",
        "year": "2013",
        "journal": "Nature Geoscience, 6, 661-667",
        "abstract": (
            "We use a line-by-line radiative transfer model to determine the thermal radiation "
            "limit for a runaway greenhouse on Earth-like planets. The Simpson-Nakajima limit — "
            "the maximum outgoing longwave radiation (OLR) a moist atmosphere can emit — sets the "
            "threshold for runaway greenhouse initiation. We find this limit is approximately "
            "282 W/m2 for an Earth-like planet, lower than previous estimates of ~300 W/m2, due "
            "to improved treatment of water vapour continuum absorption. Earth's current absorbed "
            "solar radiation (~240 W/m2) provides a margin of ~42 W/m2 before runaway. For "
            "exoplanets, this threshold determines the inner edge of the habitable zone: planets "
            "receiving stellar flux above the critical OLR threshold cannot maintain stable surface "
            "liquid water. The runaway greenhouse is an irreversible transition — once initiated, "
            "all surface water evaporates and is eventually lost to space via photodissociation "
            "and hydrogen escape."
        ),
        "topics": ["runaway_greenhouse", "climate_modeling", "habitable_zone", "atmospheric_physics"],
        "key_findings": [
            "Runaway greenhouse OLR threshold: ~282 W/m2 for Earth-like planets (lower than previous ~300 W/m2)",
            "Earth's current absorbed solar radiation ~240 W/m2 provides ~42 W/m2 margin",
            "Critical threshold determines HZ inner edge",
            "Runaway greenhouse is irreversible — all surface water eventually lost to space",
        ],
    },
    {
        "id": "zahnle_catling2017",
        "title": "The Cosmic Shoreline: The Evidence that Escape Determines which Planets Have Atmospheres, and what this May Mean for Proxima Centauri b",
        "authors": "Zahnle, K.J. & Catling, D.C.",
        "year": "2017",
        "journal": "ApJ, 843, 122",
        "abstract": (
            "We identify a 'cosmic shoreline' in the space of escape velocity versus incident "
            "stellar flux that separates bodies with atmospheres from those without. Solar System "
            "bodies and exoplanets with v_esc > ~(few km/s) * (S/S_Earth)^(1/4) retain substantial "
            "atmospheres, while those below this threshold are airless. This empirical boundary "
            "reflects the competition between atmospheric supply (volcanism, outgassing, cometary "
            "delivery) and loss (thermal escape, photochemical escape, sputtering, impact erosion). "
            "For Proxima Centauri b, the cosmic shoreline analysis suggests that atmospheric "
            "retention depends critically on the planet's mass and the star's XUV history. If "
            "Proxima b is >= 1.5 M_Earth with v_esc >= 12 km/s, it likely retains a substantial "
            "atmosphere despite the star's activity. The cosmic shoreline framework provides a "
            "rapid first-order assessment of atmospheric retention potential for newly discovered "
            "exoplanets."
        ),
        "topics": ["atmospheric_escape", "atmospheric_retention", "cosmic_shoreline"],
        "key_findings": [
            "Cosmic shoreline: v_esc threshold ~ (few km/s) * (S/S_Earth)^(1/4) separates atmosphere vs airless",
            "Boundary reflects supply (volcanism, outgassing) vs loss (thermal escape, sputtering, impacts)",
            "Proxima b needs >= 1.5 M_Earth (v_esc >= 12 km/s) for likely atmospheric retention",
            "Provides rapid first-order atmospheric retention assessment for exoplanets",
        ],
    },
    {
        "id": "tian2015",
        "title": "Water Loss from Young Planets",
        "authors": "Tian, F. et al.",
        "year": "2015",
        "journal": "Earth and Planetary Science Letters, 432, 126-132",
        "abstract": (
            "We model XUV-driven hydrodynamic escape of water from terrestrial planets during the "
            "first few hundred Myr after formation, when stellar XUV luminosity is 10-100 times "
            "higher than present-day values. The energy-limited escape rate scales as: "
            "dM/dt ~ (epsilon * F_XUV * pi * R_p^3) / (G * M_p), where epsilon is the heating "
            "efficiency (0.1-0.3). Earth-mass planets at 1 AU can lose up to 0.5 Earth-oceans of "
            "water in 500 Myr. Smaller planets (0.5 M_Earth) at the same distance lose significantly "
            "more due to lower gravitational binding energy. Planets in the HZ of M dwarfs are "
            "particularly vulnerable because M-dwarf XUV remains elevated for much longer (>1 Gyr). "
            "We quantify the initial water inventory required for a planet to emerge from the early "
            "intense bombardment / high-XUV phase with enough water for surface habitability."
        ),
        "topics": ["atmospheric_escape", "water_loss", "xuv_flux", "pre_main_sequence"],
        "key_findings": [
            "Energy-limited escape: dM/dt ~ epsilon * F_XUV * pi * R_p^3 / (G * M_p), epsilon ~ 0.1-0.3",
            "Earth-mass planets at 1 AU can lose ~0.5 Earth-oceans in 500 Myr",
            "Smaller planets (0.5 M_E) lose significantly more due to lower gravitational binding",
            "M-dwarf HZ planets lose water for >1 Gyr due to prolonged high-XUV phase",
        ],
    },
    {
        "id": "wolf_toon2015",
        "title": "The evolution of habitable climates under the brightening Sun",
        "authors": "Wolf, E.T. & Toon, O.B.",
        "year": "2015",
        "journal": "JGR Atmospheres, 120, 5775-5794",
        "abstract": (
            "We use the Community Atmosphere Model (CAM) GCM to simulate Earth's climate evolution "
            "from 2.5 Gya to the future, tracking how cloud feedbacks respond to increasing solar "
            "luminosity. Low-altitude clouds provide strong negative feedback: as surface temperature "
            "rises, enhanced evaporation increases low cloud cover and albedo, partially offsetting "
            "the warming. However, this feedback has limits — beyond a critical solar flux increase "
            "of ~8-10%, positive water vapour feedback overwhelms cloud cooling, and the climate "
            "transitions toward a moist greenhouse. Cloud feedbacks are the largest source of "
            "uncertainty in exoplanet climate projections: models with strong cloud feedback predict "
            "wider habitable zones, while models without clouds predict narrower zones. For tidally "
            "locked planets, substellar convective clouds provide uniquely strong stabilisation "
            "absent in fast rotators."
        ),
        "topics": ["cloud_feedback", "climate_modeling", "habitable_zone", "gcm"],
        "key_findings": [
            "Low-altitude clouds provide negative feedback via increased albedo under warming",
            "Cloud feedback breaks down at ~8-10% solar flux increase; moist greenhouse onset",
            "Cloud feedbacks are the largest uncertainty source in exoplanet climate projections",
            "Tidally locked substellar clouds provide stronger stabilisation than fast-rotator clouds",
        ],
    },
    # ── Planetary Interiors (new) ─────────────────────────────────────────────
    {
        "id": "driscoll_barnes2015",
        "title": "Tidal Heating of Earth-like Exoplanets around M Stars: Thermal, Magnetic, and Orbital Evolutions",
        "authors": "Driscoll, P.E. & Barnes, R.",
        "year": "2015",
        "journal": "Astrobiology, 15(9), 739-760",
        "abstract": (
            "We couple tidal dissipation, thermal evolution, and magnetic dynamo models to study "
            "Earth-like planets in the HZ of M dwarfs. Tidal heating from non-zero eccentricity "
            "provides an additional internal heat source that can exceed radiogenic heating: for "
            "eccentricities of 0.1-0.3, tidal heat fluxes range from 0.1 to 100 W/m2 (compared "
            "to Earth's ~0.09 W/m2 geothermal flux). This enhanced heating maintains a liquid iron "
            "core convection and thus a protective magnetic dynamo for longer than radiogenic heating "
            "alone. However, excessive tidal heating (>10 W/m2) can trigger a runaway tidal-volcanic "
            "state (analogous to Io), sterilising the surface. The optimal range for habitability "
            "is 0.04-2 W/m2 tidal heat flux, sufficient to maintain a dynamo without inducing "
            "excessive volcanism. Orbital circularisation timescales depend on the tidal quality "
            "factor Q; for Q ~ 100, e-folding time is ~1 Gyr for Earth-mass planets."
        ),
        "topics": ["tidal_heating", "magnetic_field", "m_dwarf", "planetary_interior", "orbital_dynamics"],
        "key_findings": [
            "Tidal heat flux 0.1-100 W/m2 for e=0.1-0.3 (Earth geothermal: ~0.09 W/m2)",
            "Optimal habitability range: 0.04-2 W/m2 tidal heating — dynamo without excessive volcanism",
            "Excessive tidal heating (>10 W/m2) triggers runaway tidal-volcanic state (Io analogue)",
            "Tidal heating can extend magnetic dynamo lifetime beyond radiogenic-only scenarios",
        ],
    },
    {
        "id": "walker1981",
        "title": "A negative feedback mechanism for the long-term stabilization of Earth's surface temperature",
        "authors": "Walker, J.C.G., Hays, P.B., Kasting, J.F.",
        "year": "1981",
        "journal": "JGR, 86, 9776-9782",
        "abstract": (
            "We describe the carbonate-silicate geochemical cycle as a negative feedback mechanism "
            "that stabilises Earth's surface temperature over geological timescales (10-100 Myr). "
            "Higher surface temperature increases atmospheric CO2 weathering rates via silicate "
            "rock dissolution: CO2 + CaSiO3 -> CaCO3 + SiO2. This removes CO2 from the atmosphere, "
            "reducing the greenhouse effect and cooling the planet. Conversely, lower temperatures "
            "reduce weathering, allowing volcanic CO2 to accumulate and warm the surface. The feedback "
            "operates on a timescale set by the weathering rate, which depends exponentially on "
            "temperature. This mechanism explains how Earth has maintained liquid water for ~4 Gyr "
            "despite a ~30% increase in solar luminosity (the faint young Sun problem). For "
            "exoplanets, an active carbonate-silicate cycle requires plate tectonics (to recycle "
            "carbonates) and liquid water (to dissolve CO2)."
        ),
        "topics": ["carbonate_silicate_cycle", "climate_feedback", "habitability", "plate_tectonics"],
        "key_findings": [
            "Carbonate-silicate cycle: CO2 + CaSiO3 -> CaCO3 + SiO2 (temperature-dependent weathering)",
            "Negative feedback stabilises surface T over 10-100 Myr timescales",
            "Explains faint young Sun problem: liquid water maintained despite 30% lower solar luminosity",
            "Requires plate tectonics (carbonate recycling) and liquid water (CO2 dissolution)",
        ],
    },
    {
        "id": "zeng2019",
        "title": "Growth model interpretation of planet size distribution",
        "authors": "Zeng, L. et al.",
        "year": "2019",
        "journal": "PNAS, 116(20), 9723-9728",
        "abstract": (
            "We present mass-radius-composition curves derived from Preliminary Reference Earth "
            "Model (PREM)-based interior structure models, covering pure iron, Earth-like rocky "
            "(32.5% Fe + 67.5% MgSiO3), pure silicate, and water-world compositions. Key "
            "relationships: pure iron planets follow R ~ 0.75 * (M/M_E)^0.27 R_Earth; rocky "
            "planets follow R ~ (M/M_E)^0.27 R_Earth; water worlds follow R ~ 1.39 * (M/M_E)^0.27 "
            "R_Earth. We interpret the Kepler radius distribution using a growth model where planets "
            "first accrete rocky cores, then accumulate water/ice shells from beyond the snow line, "
            "and finally capture H/He envelopes. The observed gap at 1.5-2.0 R_Earth separates "
            "rocky super-Earths from water-rich sub-Neptunes. Planets above 2 R_Earth likely contain "
            ">= 25% water by mass or have significant H/He envelopes."
        ),
        "topics": ["mass_radius", "planetary_interior", "composition", "exoplanet_characterization"],
        "key_findings": [
            "Rocky planets: R ~ (M/M_E)^0.27 R_Earth; iron: R ~ 0.75*(M/M_E)^0.27",
            "Water worlds: R ~ 1.39*(M/M_E)^0.27 R_Earth",
            "Radius gap at 1.5-2.0 R_Earth separates rocky from water-rich compositions",
            "Planets >2 R_Earth likely have >= 25% water by mass or significant H/He envelopes",
        ],
    },
    {
        "id": "stamenkovic2012",
        "title": "The Influence of Pressure-dependent Viscosity on the Thermal Evolution of Super-Earths",
        "authors": "Stamenković, V. et al.",
        "year": "2012",
        "journal": "ApJ, 748, 41",
        "abstract": (
            "We investigate whether super-Earths can sustain plate tectonics — a prerequisite for "
            "long-term climate regulation via the carbonate-silicate cycle. At the high pressures "
            "found in super-Earth mantles (> 100 GPa), silicate viscosity increases dramatically "
            "due to pressure-dependent activation volume effects. This increased viscosity can "
            "suppress convective vigour and potentially prevent plate tectonics, trapping the "
            "planet in a stagnant-lid regime. We find a critical mass threshold: planets above "
            "~5-8 M_Earth may not sustain plate tectonics due to the pressure viscosity barrier. "
            "However, tidal heating and higher radiogenic element concentrations can partially "
            "counteract this effect. The transition between mobile lid (plate tectonics) and "
            "stagnant lid depends on mantle composition (Mg/Si ratio), water content, and thermal "
            "state. Stagnant-lid planets can still outgas through volcanic hotspots but at "
            "reduced rates."
        ),
        "topics": ["plate_tectonics", "planetary_interior", "super_earth", "mantle_dynamics"],
        "key_findings": [
            "Pressure-dependent viscosity increases dramatically above 100 GPa in super-Earth mantles",
            "Critical mass ~5-8 M_Earth: above this, plate tectonics may be suppressed",
            "Stagnant-lid regime reduces but does not eliminate outgassing (volcanic hotspots)",
            "Mobile-lid vs stagnant-lid depends on composition, water content, and thermal state",
        ],
    },
    # ── Stellar Context (new) ─────────────────────────────────────────────────
    {
        "id": "lammer2009",
        "title": "What makes a planet habitable?",
        "authors": "Lammer, H. et al.",
        "year": "2009",
        "journal": "A&A Review, 17, 181-249",
        "abstract": (
            "A comprehensive review of stellar, planetary, and environmental factors required for "
            "habitability. Key stellar requirements include: stable main-sequence lifetime > 2 Gyr "
            "(excluding O, B, early A stars), moderate UV flux (sufficient for prebiotic chemistry "
            "but not sterilising), and low flare frequency. Planetary requirements include: mass "
            "range 0.5-10 M_Earth (sufficient gravity for atmosphere retention, not so massive as "
            "to accrete H/He), active interior (plate tectonics or equivalent recycling), and "
            "protective magnetic field. We review atmospheric escape mechanisms: thermal (Jeans) "
            "escape, hydrodynamic escape, ion pickup by stellar wind, and sputtering. Non-thermal "
            "escape dominates for planets without magnetic fields around active M dwarfs. The "
            "interplay between stellar activity history, planetary mass, and atmospheric composition "
            "determines whether a planet can maintain a habitable atmosphere over Gyr timescales."
        ),
        "topics": ["stellar_activity", "habitability", "atmospheric_escape", "magnetic_field"],
        "key_findings": [
            "Stellar requirement: main-sequence lifetime > 2 Gyr (excludes O, B, early A stars)",
            "Planetary mass range for habitability: 0.5-10 M_Earth",
            "Non-thermal escape (ion pickup, sputtering) dominates for unmagnetised M-dwarf planets",
            "Habitability requires interplay of stellar activity, planetary mass, and magnetic protection",
        ],
    },
    {
        "id": "ramirez_kaltenegger2014",
        "title": "The Habitable Zones of Pre-Main-Sequence Stars",
        "authors": "Ramirez, R.M. & Kaltenegger, L.",
        "year": "2014",
        "journal": "ApJ Letters, 797, L25",
        "abstract": (
            "We calculate habitable zone boundaries during the pre-main-sequence (pre-MS) phase "
            "when stellar luminosity is significantly higher and decreasing. For solar-type stars, "
            "the pre-MS HZ extends much farther out than the main-sequence HZ, then contracts as "
            "the star settles. For M dwarfs (0.1 M_Sun), the pre-MS lasts ~1 Gyr and the HZ starts "
            "at 0.2-0.5 AU before contracting to 0.03-0.1 AU. Planets that end up in the main-"
            "sequence HZ experience a period of intense irradiation during pre-MS, potentially "
            "losing volatiles. Conversely, planets initially beyond the HZ may experience a brief "
            "habitable window during pre-MS. The pre-MS habitable zone duration varies from ~50 Myr "
            "(F stars) to ~1 Gyr (late M dwarfs), setting constraints on early habitability and "
            "volatile delivery timing."
        ),
        "topics": ["habitable_zone", "pre_main_sequence", "hz_boundaries", "stellar_evolution"],
        "key_findings": [
            "M-dwarf pre-MS HZ starts at 0.2-0.5 AU, contracts to 0.03-0.1 AU on main sequence",
            "Pre-MS duration: ~50 Myr (F stars) to ~1 Gyr (late M dwarfs)",
            "Main-sequence HZ planets experience intense irradiation during pre-MS phase",
            "Brief habitable window possible for planets initially beyond the main-sequence HZ",
        ],
    },
    {
        "id": "segura2010",
        "title": "The Effect of a Strong Stellar Flare on the Atmospheric Chemistry of an Earth-like Planet Orbiting an M Dwarf",
        "authors": "Segura, A. et al.",
        "year": "2010",
        "journal": "Astrobiology, 10(7), 751-771",
        "abstract": (
            "We model the photochemical response of an Earth-like atmosphere to a large stellar "
            "flare from the M dwarf AD Leonis. The flare (UV enhancement factor ~100x for ~1 hour "
            "followed by elevated proton flux for days) drives dramatic ozone destruction: O3 column "
            "decreases by 94% within 2 years due to catalytic NOx cycles initiated by energetic "
            "particle precipitation. Surface UV-B flux increases by a factor of ~50 at peak ozone "
            "depletion, potentially lethal for surface life. Recovery to pre-flare O3 levels takes "
            "approximately 50 years. However, with a modest increase in atmospheric depth or "
            "planetary magnetic field strength, the impact is substantially reduced. Planets with "
            "surface pressures > 2 bar or strong dipole fields (> 0.5 Gauss) retain > 50% of their "
            "O3 shield even during extreme flares."
        ),
        "topics": ["stellar_flares", "m_dwarf", "atmospheric_chemistry", "uv_environment", "ozone"],
        "key_findings": [
            "Large M-dwarf flare destroys 94% of ozone column within 2 years via NOx catalysis",
            "Surface UV-B increases ~50x at peak O3 depletion — potentially lethal for surface life",
            "O3 recovery time: approximately 50 years post-flare",
            "Planets with >2 bar atmosphere or >0.5 Gauss dipole field retain >50% O3 shield",
        ],
    },
    {
        "id": "france2013",
        "title": "The Ultraviolet Radiation Environment around M dwarf Exoplanet Host Stars",
        "authors": "France, K. et al.",
        "year": "2013",
        "journal": "ApJ, 763, 149",
        "abstract": (
            "We present HST observations of the UV radiation fields of 6 M dwarf exoplanet hosts "
            "(GJ 581, GJ 876, GJ 436, GJ 832, GJ 667C, GJ 176). M dwarfs have very different UV "
            "spectral energy distributions compared to the Sun: the far-UV (912-1700 A) to near-UV "
            "(1700-3200 A) ratio (FUV/NUV) is 10-1000x higher than solar. This elevated FUV/NUV "
            "ratio drives photochemistry that can produce abiotic O2 and O3 via CO2 photolysis. "
            "The Lyman-alpha emission line alone can carry 37-75% of the total 1150-3100 A UV flux "
            "for quiet M dwarfs. Chromospheric and transition region emission dominates the FUV, "
            "while photospheric emission dominates the NUV. Accurate UV characterisation is essential "
            "for photochemical modeling of exoplanet atmospheres, as standard stellar models "
            "(PHOENIX) underpredict M-dwarf UV emission by 1-2 orders of magnitude."
        ),
        "topics": ["uv_environment", "m_dwarf", "stellar_activity", "spectral_energy_distribution"],
        "key_findings": [
            "M-dwarf FUV/NUV ratio is 10-1000x higher than solar",
            "Lyman-alpha carries 37-75% of total UV flux (1150-3100 A) for quiet M dwarfs",
            "Elevated FUV/NUV drives photochemistry producing abiotic O2/O3 via CO2 photolysis",
            "PHOENIX stellar models underpredict M-dwarf UV emission by 1-2 orders of magnitude",
        ],
    },
    # ── Observational / JWST (new) ────────────────────────────────────────────
    {
        "id": "madhusudhan2023",
        "title": "Carbon-bearing Molecules in a Possible Hycean Atmosphere of an Exoplanet",
        "authors": "Madhusudhan, N. et al.",
        "year": "2023",
        "journal": "ApJ Letters, 956, L13",
        "abstract": (
            "We present JWST NIRSpec transmission spectroscopy of K2-18 b, a habitable-zone "
            "sub-Neptune (R = 2.61 R_Earth, M = 8.63 M_Earth) orbiting an M2.5 dwarf at 0.14 AU. "
            "The spectrum reveals detections of CH4 (3.4 sigma) and CO2 (2.9 sigma) at abundances "
            "of ~1% each, with a non-detection of CO (< 0.01%) and tentative evidence for DMS "
            "(dimethyl sulfide). The CH4/CO2 combination with low CO is difficult to explain "
            "abiotically and is consistent with a hydrogen-rich atmosphere overlying a liquid water "
            "ocean — the so-called Hycean scenario. However, mini-Neptune (H/He envelope over "
            "magma ocean) interpretations cannot be ruled out. K2-18 b represents the first "
            "habitable-zone exoplanet with detected atmospheric molecules, demonstrating JWST's "
            "capability for characterising temperate sub-Neptune atmospheres."
        ),
        "topics": ["jwst", "transit_spectroscopy", "atmospheric_characterization", "sub_neptune", "biosignatures"],
        "key_findings": [
            "CH4 detected at ~1% (3.4 sigma) and CO2 at ~1% (2.9 sigma) in K2-18 b atmosphere",
            "Low CO (<0.01%) — CH4+CO2 without CO is difficult to explain abiotically",
            "Tentative DMS detection — if confirmed, would be a potential biosignature",
            "First habitable-zone exoplanet with detected atmospheric molecules via JWST",
        ],
    },
    {
        "id": "lustig-yaeger2023",
        "title": "A JWST transmission spectrum of the nearby Earth-sized exoplanet LHS 475 b",
        "authors": "Lustig-Yaeger, J. et al.",
        "year": "2023",
        "journal": "Nature Astronomy, 7, 1317-1328",
        "abstract": (
            "We present JWST NIRSpec G395H transmission spectroscopy of LHS 475 b, a 0.99 R_Earth "
            "planet orbiting an M3 dwarf at 0.02 AU (T_eq ~ 586 K). The flat transmission spectrum "
            "rules out a cloud-free hydrogen-dominated atmosphere at > 3 sigma and is consistent "
            "with either: (1) a bare rock with no atmosphere, (2) a high-mean-molecular-weight "
            "atmosphere (pure CO2, O2, or N2), or (3) a tenuous atmosphere below detection "
            "threshold. A pure CO2 atmosphere shows a marginal ~1 sigma feature at 4.3 micron. "
            "LHS 475 b demonstrates JWST's ability to characterise Earth-sized rocky planet "
            "atmospheres and constrain atmospheric composition. The result highlights that proximity "
            "to the host star makes H/He retention unlikely, consistent with atmospheric escape "
            "models for close-in planets around M dwarfs."
        ),
        "topics": ["jwst", "transit_spectroscopy", "rocky_planet", "atmospheric_characterization"],
        "key_findings": [
            "Flat spectrum rules out cloud-free H2-dominated atmosphere at >3 sigma",
            "Consistent with bare rock, high-MMW atmosphere (CO2/O2/N2), or tenuous atmosphere",
            "Marginal ~1 sigma CO2 feature at 4.3 micron",
            "First JWST characterisation of an Earth-sized rocky exoplanet atmosphere",
        ],
    },
    {
        "id": "greene2023",
        "title": "Thermal emission from the Earth-sized exoplanet TRAPPIST-1 b using JWST",
        "authors": "Greene, T.P. et al.",
        "year": "2023",
        "journal": "Nature, 618, 39-42",
        "abstract": (
            "We report JWST MIRI secondary eclipse photometry of TRAPPIST-1 b at 15 micron, "
            "measuring a dayside brightness temperature of 503 +/- 26 K. This is consistent with "
            "the predicted equilibrium temperature for a tidally locked planet with no atmosphere "
            "and low Bond albedo (A ~ 0.1), assuming zero heat redistribution to the nightside. "
            "The measurement effectively rules out a thick CO2-dominated atmosphere (>= 10 bar), "
            "which would redistribute heat and lower the dayside temperature. A bare rock or "
            "tenuous atmosphere (<< 1 bar) remains consistent with the data. This is the first "
            "thermal emission measurement of a TRAPPIST-1 planet and demonstrates that JWST can "
            "distinguish between atmospheric and bare-rock scenarios for nearby Earth-sized planets. "
            "The result has implications for the habitability of the outer TRAPPIST-1 planets "
            "(e, f, g) which may also lack thick atmospheres."
        ),
        "topics": ["jwst", "thermal_emission", "trappist1", "rocky_planet"],
        "key_findings": [
            "TRAPPIST-1 b dayside brightness temperature: 503 +/- 26 K at 15 micron",
            "Consistent with bare rock, low albedo (A~0.1), zero heat redistribution",
            "Rules out thick (>= 10 bar) CO2 atmosphere at high confidence",
            "First thermal emission measurement of a TRAPPIST-1 planet via JWST MIRI",
        ],
    },
    {
        "id": "benneke2019",
        "title": "Water Vapor and Clouds on the Habitable-Zone Sub-Neptune Exoplanet K2-18b",
        "authors": "Benneke, B. et al.",
        "year": "2019",
        "journal": "ApJ Letters, 887, L14",
        "abstract": (
            "We report the detection of water vapour in the atmosphere of K2-18 b using HST WFC3 "
            "transit spectroscopy in the 1.1-1.7 micron range. K2-18 b (R = 2.61 R_Earth, "
            "M = 8.63 M_Earth) orbits in the habitable zone of an M2.5 dwarf (T_eq ~ 255 K). "
            "The H2O feature is detected at 3.6 sigma with a volume mixing ratio of 0.01-12.5% "
            "depending on cloud assumptions. The spectrum is consistent with a hydrogen-rich "
            "atmosphere containing water vapour and high-altitude clouds or hazes that mute "
            "spectral features at shorter wavelengths. This marks the first detection of water "
            "vapour in a habitable-zone exoplanet atmosphere. However, whether K2-18 b has a "
            "rocky surface beneath its H/He envelope — necessary for a liquid water ocean — "
            "remains uncertain given its sub-Neptune size."
        ),
        "topics": ["transit_spectroscopy", "atmospheric_characterization", "sub_neptune", "water_detection"],
        "key_findings": [
            "H2O detected at 3.6 sigma in K2-18 b atmosphere via HST WFC3 (1.1-1.7 micron)",
            "H2O mixing ratio: 0.01-12.5% depending on cloud model assumptions",
            "First water vapour detection in a habitable-zone exoplanet",
            "Sub-Neptune size (2.61 R_E) leaves surface conditions uncertain",
        ],
    },
    # ── Climate Modeling (new) ────────────────────────────────────────────────
    {
        "id": "yang2013",
        "title": "Stabilizing Cloud Feedback Dramatically Expands the Habitable Zone of Tidally Locked Planets",
        "authors": "Yang, J., Cowan, N.B., Abbot, D.S.",
        "year": "2013",
        "journal": "ApJ Letters, 771, L45",
        "abstract": (
            "Using the Community Atmosphere Model (CAM3) GCM, we demonstrate that thick convective "
            "clouds forming at the substellar point of tidally locked planets provide a strong "
            "negative feedback that dramatically expands the inner edge of the habitable zone. "
            "Intense stellar heating at the substellar point drives vigorous convection and produces "
            "optically thick high-altitude clouds with albedo 0.6-0.8. These clouds reflect a "
            "large fraction of incoming stellar radiation, preventing the onset of a runaway "
            "greenhouse. For slowly rotating tidally locked planets, the inner HZ edge moves "
            "inward by approximately 0.05-0.1 AU (a ~30% increase in tolerable instellation) "
            "compared to the non-cloud Kopparapu et al. estimate. This effect is unique to "
            "synchronous rotators because fast-rotating planets distribute clouds more uniformly, "
            "preventing the concentrated substellar cloud feedback."
        ),
        "topics": ["cloud_feedback", "tidal_locking", "habitable_zone", "gcm", "climate_modeling"],
        "key_findings": [
            "Substellar convective clouds have albedo 0.6-0.8, providing strong negative feedback",
            "Inner HZ edge moves inward ~30% for tidally locked planets compared to 1-D estimates",
            "Effect is unique to synchronous rotators — fast rotators lack concentrated cloud feedback",
            "Prevents runaway greenhouse at instellation levels that would trigger it without clouds",
        ],
    },
    {
        "id": "hu_yang2014",
        "title": "Role of ocean heat transport in climates of tidally locked exoplanets around M dwarf stars",
        "authors": "Hu, Y. & Yang, J.",
        "year": "2014",
        "journal": "PNAS, 111(2), 629-634",
        "abstract": (
            "We investigate ocean heat transport (OHT) on tidally locked exoplanets using a fully "
            "coupled atmosphere-ocean GCM (CCSM3). Ocean circulation on synchronous rotators differs "
            "fundamentally from Earth: the substellar heating creates a radially divergent surface "
            "current carrying warm water toward the nightside, with cold deep return flow. OHT "
            "increases the open ocean area by 10-20% compared to atmosphere-only models, expanding "
            "the habitable surface fraction. The ocean moderates temperature extremes: substellar "
            "surface temperatures decrease by 10-30 K while nightside temperatures increase by "
            "20-40 K relative to land-planet simulations. Deep ocean circulation has timescales of "
            "~1000 years, creating significant thermal inertia that buffers against stellar variability. "
            "For planets with fractional ocean coverage, OHT effects are proportional to ocean "
            "fraction and strongest when the substellar point is over ocean."
        ),
        "topics": ["ocean_heat_transport", "tidal_locking", "climate_modeling", "gcm"],
        "key_findings": [
            "OHT increases open ocean area by 10-20% compared to atmosphere-only models",
            "Substellar T decreases 10-30 K; nightside T increases 20-40 K with ocean coupling",
            "Deep ocean circulation timescale ~1000 yr provides thermal inertia buffer",
            "Radially divergent surface current carries warm water substellar-to-nightside",
        ],
    },
    {
        "id": "joshi1997",
        "title": "Simulations of the Atmospheres of Synchronously Rotating Terrestrial Planets Orbiting M Dwarfs",
        "authors": "Joshi, M.M., Haberle, R.M., Reynolds, R.T.",
        "year": "1997",
        "journal": "Icarus, 129, 450-465",
        "abstract": (
            "We present some of the earliest 3D GCM simulations of tidally locked terrestrial "
            "planets in the habitable zones of M dwarfs, addressing the long-standing concern that "
            "synchronous rotation would render such planets uninhabitable due to atmospheric collapse "
            "on the nightside. Using an adapted Earth GCM with simplified physics, we find that for "
            "surface pressures >= 0.1 bar, atmospheric circulation is sufficient to transport heat "
            "from the dayside to the nightside, preventing atmospheric freeze-out. The key mechanism "
            "is the development of a strong equatorial jet (superrotation) driven by the day-night "
            "heating contrast. Even with 0.1 bar of CO2-N2 atmosphere, nightside temperatures "
            "remain above 200 K. This foundational result overturned decades of pessimism about "
            "M-dwarf planet habitability and motivated subsequent detailed GCM studies of tidal "
            "locking."
        ),
        "topics": ["gcm", "tidal_locking", "m_dwarf", "atmospheric_dynamics", "habitability"],
        "key_findings": [
            "Atmospheric collapse prevented for surface pressures >= 0.1 bar on tidally locked planets",
            "Equatorial superrotation jet is the primary day-to-night heat transport mechanism",
            "Nightside temperatures remain above 200 K even with minimal (0.1 bar) atmosphere",
            "First GCM demonstration that M-dwarf HZ planets can be habitable despite synchronous rotation",
        ],
    },
    {
        "id": "del_genio2019",
        "title": "Habitable Climate Scenarios for Proxima Centauri b with a Dynamic Ocean",
        "authors": "Del Genio, A.D. et al.",
        "year": "2019",
        "journal": "Astrobiology, 19(1), 99-125",
        "abstract": (
            "We use ROCKE-3D (NASA GISS coupled ocean-atmosphere GCM) to simulate Proxima Centauri b "
            "under multiple atmospheric compositions and rotation states. Key scenarios include "
            "N2-dominated atmospheres with 0.0004-1 bar CO2, for both synchronous (1:1) and 3:2 "
            "spin-orbit resonances. With a dynamic ocean coupled to the atmosphere, surface "
            "liquid water is maintained for a wide range of conditions: habitable surface fractions "
            "range from 2-50% depending on atmospheric composition and rotation state. The synchronous "
            "case produces an eyeball ocean at the substellar point with sea ice coverage of 50-85%. "
            "The 3:2 resonance case distributes warming more uniformly, potentially allowing "
            "equatorial open ocean. We find that the dynamic ocean adds ~5-15% to the habitable "
            "surface fraction compared to a slab ocean model, and significantly modifies atmospheric "
            "circulation patterns through wind-driven upwelling."
        ),
        "topics": ["gcm", "climate_modeling", "ocean_dynamics", "proxima_centauri", "tidal_locking"],
        "key_findings": [
            "Habitable surface fractions: 2-50% depending on atmosphere and rotation state",
            "Dynamic ocean adds ~5-15% habitable surface fraction vs slab ocean models",
            "Synchronous case: eyeball ocean with 50-85% sea ice coverage",
            "3:2 resonance distributes warming more uniformly, allowing equatorial open ocean",
        ],
    },
    # ── Astrobiology (new) ────────────────────────────────────────────────────
    {
        "id": "schwieterman2018",
        "title": "Exoplanet Biosignatures: A Review of Remotely Detectable Signs of Life",
        "authors": "Schwieterman, E.W. et al.",
        "year": "2018",
        "journal": "Astrobiology, 18(6), 663-708",
        "abstract": (
            "A comprehensive review of remotely detectable biosignatures in exoplanet atmospheres "
            "and surfaces. Gaseous biosignatures include O2/O3 (photosynthetic origin), CH4 "
            "(methanogenic origin), N2O (denitrification), and DMS (marine phytoplankton). "
            "Thermodynamic disequilibrium — the simultaneous presence of oxidising (O2) and "
            "reducing (CH4) species — is a robust indicator because it requires a sustained "
            "biological flux to maintain against photochemical destruction. Surface biosignatures "
            "include the vegetation red edge (VRE) at ~750 nm, caused by chlorophyll reflectance, "
            "and pigment absorption features from non-oxygenic phototrophs. Anti-biosignatures "
            "(e.g., high CO, absence of expected species) help constrain abiotic scenarios. "
            "Detection feasibility depends on planet size, host star, and telescope aperture: "
            "JWST can characterise sub-Neptune atmospheres; ELT-class telescopes needed for "
            "Earth-twin O2 detection."
        ),
        "topics": ["biosignatures", "atmospheric_chemistry", "remote_detection", "spectroscopy"],
        "key_findings": [
            "Thermodynamic disequilibrium (O2+CH4 coexistence) is most robust biosignature indicator",
            "Vegetation red edge at ~750 nm detectable for Earth-like planets around nearby stars",
            "Anti-biosignatures (high CO, missing species) help constrain abiotic scenarios",
            "JWST: sub-Neptune atmospheres; ELT-class: Earth-twin O2 detection",
        ],
    },
    {
        "id": "catling2018",
        "title": "Exoplanet Biosignatures: A Framework for Their Assessment",
        "authors": "Catling, D.C. et al.",
        "year": "2018",
        "journal": "Astrobiology, 18(6), 709-738",
        "abstract": (
            "We develop a Bayesian framework for assessing the probability that an observed "
            "atmospheric feature is a genuine biosignature. The framework requires evaluating "
            "P(life | data) via: (1) prior probability of life given planetary context "
            "(habitability, age, stellar environment), (2) likelihood of the observed spectral "
            "features given biological and abiotic models, (3) exhaustive enumeration of abiotic "
            "false-positive scenarios. A robust biosignature detection requires: the spectral "
            "feature is reliably detected, all known abiotic sources are quantitatively "
            "insufficient to explain the observed abundance, and the planetary context is consistent "
            "with habitability. We apply the framework to hypothetical detections of O2+CH4 "
            "disequilibrium, showing that contextual information (stellar type, planet mass, "
            "atmospheric composition) can shift the posterior by orders of magnitude. No single "
            "molecule is a definitive biosignature — assessment must be holistic."
        ),
        "topics": ["biosignatures", "framework", "bayesian_assessment", "false_positives"],
        "key_findings": [
            "Bayesian framework: P(life|data) requires prior, likelihood, and false-positive enumeration",
            "Contextual information shifts posterior probability by orders of magnitude",
            "No single molecule is definitive — holistic multi-species assessment required",
            "Reliable detection + insufficient abiotic sources + habitable context = robust biosignature",
        ],
    },
    {
        "id": "cockell2016",
        "title": "Habitability: A Review",
        "authors": "Cockell, C.S. et al.",
        "year": "2016",
        "journal": "Astrobiology, 16(1), 89-117",
        "abstract": (
            "A definitive review of the concept of habitability, distinguishing between 'habitable' "
            "(conditions permitting life) and 'inhabited' (life actually present). We identify "
            "minimum requirements for habitability: (1) liquid solvent (water, or potentially other "
            "polar solvents), (2) bioavailable energy source (chemical redox gradients, photons), "
            "(3) essential elements (CHNOPS + trace metals), (4) conditions within known limits of "
            "life (temperature: -20 to 122 C for Earth life; pH 0-12.5; salinity up to saturation). "
            "Habitability is not binary but a spectrum: environments can be transiently habitable, "
            "marginally habitable, or robustly habitable depending on the persistence and reliability "
            "of these conditions. We argue that habitability assessment must move beyond the liquid "
            "water HZ to consider energy availability, nutrient cycling, and protection from "
            "sterilising radiation."
        ),
        "topics": ["habitability", "astrobiology", "definition", "limits_of_life"],
        "key_findings": [
            "Four minimum requirements: liquid solvent, energy source, CHNOPS elements, survivable conditions",
            "Known limits of Earth life: -20 to 122 C, pH 0-12.5, up to saturated salinity",
            "Habitability is a spectrum (transient, marginal, robust), not binary",
            "Assessment must extend beyond liquid water HZ to energy, nutrients, and radiation",
        ],
    },
    {
        "id": "raven_cockell2006",
        "title": "Influence on Photosynthesis of Starlight, Moonlight, Planetlight, and Light Pollution",
        "authors": "Raven, J.A. & Cockell, C.S.",
        "year": "2006",
        "journal": "Astrobiology, 6(4), 668-675",
        "abstract": (
            "We analyse the minimum photon flux required for oxygenic photosynthesis and how it "
            "varies with stellar spectral type. On Earth, the minimum photosynthetically active "
            "radiation (PAR) for net carbon fixation is approximately 1-5 micromol photons/m2/s "
            "(0.2-1 W/m2 in the 400-700 nm range). For M dwarfs, the peak emission shifts to "
            "near-IR (700-1100 nm), where Earth chlorophylls absorb poorly. However, alternative "
            "photopigments absorbing at longer wavelengths (e.g., bacteriochlorophylls at 800-1050 "
            "nm) could enable photosynthesis under M-dwarf spectra. The habitable zone for "
            "photosynthesis is narrower than the liquid water HZ because the minimum PAR flux "
            "constrains the viable orbital distance range. Tidally locked planets may have "
            "sufficient photon flux only on the dayside, restricting photosynthetic biospheres "
            "to the substellar hemisphere."
        ),
        "topics": ["photosynthesis", "m_dwarf", "astrobiology", "spectral_energy_distribution"],
        "key_findings": [
            "Minimum PAR for oxygenic photosynthesis: ~1-5 micromol photons/m2/s (0.2-1 W/m2)",
            "M-dwarf peak emission in near-IR (700-1100 nm) — poor match for Earth chlorophylls",
            "Alternative pigments (bacteriochlorophylls, 800-1050 nm) could enable M-dwarf photosynthesis",
            "Photosynthetic HZ is narrower than liquid water HZ due to minimum PAR flux requirement",
        ],
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
#  TF-IDF Infrastructure
# ═══════════════════════════════════════════════════════════════════════════════

_STOP_WORDS: frozenset = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "we", "our", "they", "their", "which",
    "who", "whom", "what", "where", "when", "how", "not", "no", "than",
    "as", "if", "also", "into", "about", "between", "through", "during",
    "each", "all", "both", "such", "more", "most", "other", "some", "only",
    "very", "just", "over", "under", "after", "before", "then", "so",
})

_idf_cache: Optional[Dict[str, float]] = None


def _tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumeric, remove stop words and short tokens."""
    tokens = re.findall(r"[a-z0-9_]{2,}", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS]


def _build_composite_document(paper: Dict[str, Any]) -> str:
    """Combine abstract and key findings into a single searchable document."""
    parts = [paper["abstract"]]
    if paper.get("key_findings"):
        parts.append("\n\nKey findings:")
        for f in paper["key_findings"]:
            parts.append(f"- {f}")
    return "\n".join(parts)


def _get_idf_cache() -> Dict[str, float]:
    """Lazily build inverse-document-frequency cache over all paper composites."""
    global _idf_cache
    if _idf_cache is not None:
        return _idf_cache

    N = len(_PAPERS)
    df: Counter = Counter()
    for p in _PAPERS:
        doc_tokens = set(_tokenize(_build_composite_document(p)))
        for t in doc_tokens:
            df[t] += 1

    _idf_cache = {
        term: math.log((N + 1) / (count + 1)) + 1.0
        for term, count in df.items()
    }
    return _idf_cache


def _tfidf_score(query_tokens: List[str], doc_text: str) -> float:
    """Compute TF-IDF similarity between pre-tokenised query and a document."""
    idf = _get_idf_cache()
    doc_tokens = _tokenize(doc_text)
    if not doc_tokens or not query_tokens:
        return 0.0

    doc_tf: Counter = Counter(doc_tokens)
    doc_len = len(doc_tokens)
    score = 0.0
    for qt in query_tokens:
        tf = doc_tf.get(qt, 0) / doc_len
        score += tf * idf.get(qt, 1.0)
    return score


# ═══════════════════════════════════════════════════════════════════════════════
#  ChromaDB Vector Store (Persistent)
# ═══════════════════════════════════════════════════════════════════════════════

_collection = None
_CHROMA_PATH = os.path.join("data", "chroma_db")


def _get_collection():
    """Lazy-init persistent ChromaDB collection, seeding all papers on first call.

    Re-seeds if the stored paper count does not match ``len(_PAPERS)`` (handles
    corpus expansions without manual cache invalidation).
    """
    global _collection
    if _collection is not None:
        return _collection
    try:
        import chromadb
        from chromadb.utils import embedding_functions

        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        os.makedirs(_CHROMA_PATH, exist_ok=True)
        client = chromadb.PersistentClient(path=_CHROMA_PATH)

        _collection = client.get_or_create_collection(
            name="astro_papers", embedding_function=ef,
        )

        if _collection.count() != len(_PAPERS):
            client.delete_collection("astro_papers")
            _collection = client.create_collection(
                name="astro_papers", embedding_function=ef,
            )
            _collection.add(
                ids=[p["id"] for p in _PAPERS],
                documents=[_build_composite_document(p) for p in _PAPERS],
                metadatas=[
                    {
                        "title": p["title"],
                        "authors": p["authors"],
                        "year": p["year"],
                        "journal": p["journal"],
                    }
                    for p in _PAPERS
                ],
            )
        return _collection
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Search Functions
# ═══════════════════════════════════════════════════════════════════════════════

_PAPERS_BY_ID: Dict[str, Dict[str, Any]] = {p["id"]: p for p in _PAPERS}

_RRF_K = 60  # Reciprocal Rank Fusion constant (Cormack et al. 2009)


def _filter_by_topics(
    papers: List[Dict[str, Any]], topics: List[str],
) -> List[Dict[str, Any]]:
    """Keep papers matching at least one requested topic."""
    topic_set = set(topics)
    return [p for p in papers if topic_set & set(p.get("topics", []))]


def _paper_to_citation(paper: Dict[str, Any], score: float) -> Dict[str, str]:
    """Convert an internal paper dict to a public citation dict."""
    result: Dict[str, str] = {
        "title": paper["title"],
        "authors": paper["authors"],
        "year": paper["year"],
        "journal": paper["journal"],
        "abstract": paper["abstract"],
        "relevance_score": str(round(score, 3)),
    }
    if paper.get("key_findings"):
        result["key_findings"] = paper["key_findings"]
    return result


def _fallback_keyword_search(
    query: str,
    n_results: int = 5,
    topics: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """TF-IDF weighted keyword search — used when ChromaDB is unavailable."""
    query_tokens = _tokenize(query)
    candidates = _PAPERS
    if topics:
        candidates = _filter_by_topics(candidates, topics)

    scored = []
    for p in candidates:
        doc_text = _build_composite_document(p)
        score = _tfidf_score(query_tokens, doc_text)
        scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [_paper_to_citation(p, s) for s, p in scored[:n_results]]


def _hybrid_search(
    query: str,
    n_results: int = 5,
    topics: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Reciprocal Rank Fusion of ChromaDB semantic search and TF-IDF keywords.

    Fetches up to 3*n_results candidates from each source, merges rankings
    via RRF (Cormack et al., 2009), applies topic filter, returns top-N.
    """
    coll = _get_collection()
    if coll is None:
        return _fallback_keyword_search(query, n_results, topics)

    try:
        fetch_n = min(len(_PAPERS), n_results * 3)
        sem_results = coll.query(query_texts=[query], n_results=fetch_n)
    except Exception:
        return _fallback_keyword_search(query, n_results, topics)

    sem_ranking: Dict[str, int] = {}
    for rank, pid in enumerate(sem_results["ids"][0], start=1):
        sem_ranking[pid] = rank

    query_tokens = _tokenize(query)
    kw_scored = []
    for p in _PAPERS:
        doc_text = _build_composite_document(p)
        score = _tfidf_score(query_tokens, doc_text)
        kw_scored.append((score, p["id"]))
    kw_scored.sort(key=lambda x: x[0], reverse=True)

    kw_ranking: Dict[str, int] = {}
    for rank, (_, pid) in enumerate(kw_scored, start=1):
        kw_ranking[pid] = rank

    all_ids = set(sem_ranking.keys()) | set(kw_ranking.keys())
    rrf_scores: Dict[str, float] = {}
    for pid in all_ids:
        score = 0.0
        if pid in sem_ranking:
            score += 1.0 / (_RRF_K + sem_ranking[pid])
        if pid in kw_ranking:
            score += 1.0 / (_RRF_K + kw_ranking[pid])
        rrf_scores[pid] = score

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for pid, rrf_score in ranked:
        paper = _PAPERS_BY_ID.get(pid)
        if paper is None:
            continue
        if topics and not (set(topics) & set(paper.get("topics", []))):
            continue
        results.append(_paper_to_citation(paper, rrf_score))
        if len(results) >= n_results:
            break

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════

def cite_literature(
    query: str,
    n_results: int = 5,
    topics: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Retrieve the most relevant papers for a scientific query.

    Uses hybrid search (semantic + TF-IDF keyword with Reciprocal Rank Fusion)
    when ChromaDB is available, otherwise falls back to TF-IDF keyword search.

    Parameters
    ----------
    query : str
        Natural-language scientific query.
    n_results : int
        Maximum number of citations to return (default 5).
    topics : list of str, optional
        Topic tags to filter by (e.g. ``["m_dwarf", "biosignatures"]``).
        Papers matching *any* of the listed topics are included.

    Returns
    -------
    list of dict
        Each dict has keys: title, authors, year, journal, abstract,
        relevance_score, and optionally key_findings.
    """
    return _hybrid_search(query, n_results, topics)


def format_citations_markdown(citations: List[Dict[str, str]]) -> str:
    """Format a list of citations as a markdown reference block."""
    if not citations:
        return ""
    lines = ["**References:**"]
    for c in citations:
        lines.append(
            f"- {c['authors']} ({c['year']}). *{c['title']}*. {c['journal']}."
        )
    return "\n".join(lines)
