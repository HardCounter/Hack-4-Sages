# Scientific Methodology

This document details the mathematical formulations, physical assumptions, data sources, and known limitations of the Autonomous Exoplanetary Digital Twin.

## 1. Equilibrium Temperature

The radiative equilibrium temperature assumes a balance between absorbed stellar flux and re-emitted thermal radiation:

$$T_{eq} = T_\star \sqrt{\frac{R_\star}{f \cdot a}} \cdot (1 - A_B)^{1/4}$$

where:
- $T_\star$ = stellar effective temperature [K]
- $R_\star$ = stellar radius [m]
- $a$ = orbital semi-major axis [m]
- $A_B$ = Bond albedo
- $f$ = redistribution factor: $\sqrt{2}$ for tidally locked (single-hemisphere re-emission), $2$ for fast rotator

**Assumptions:** No greenhouse effect, no internal heat, uniform albedo, zero eccentricity.

**Source:** Standard radiative balance (e.g., Kasting et al. 1993).

## 2. Earth Similarity Index (ESI)

Schulze-Makuch et al. (2011) define ESI as a geometric mean of weighted similarity ratios:

$$ESI = \prod_{i} \left(1 - \left|\frac{x_i - x_{i,\oplus}}{x_i + x_{i,\oplus}}\right|\right)^{w_i / n}$$

| Parameter | Earth value | Weight $w_i$ |
|-----------|------------|---------------|
| Radius [R⊕] | 1.0 | 0.57 |
| Density [g/cm³] | 5.51 | 1.07 |
| Escape velocity [km/s] | 11.19 | 0.70 |
| Surface temperature [K] | 288 | 5.58 |

**Limitations:** ESI only measures similarity to Earth, not habitability per se. A planet can be habitable without resembling Earth.

**Source:** Schulze-Makuch, D. et al. (2011). Astrobiology, 11(10), 1041-1052.

## 3. SEPHI (Surface Exoplanetary Planetary Habitability Index)

Rodríguez-Mozos & Moya (2017). Three binary criteria:

1. **Thermal:** 273 K ≤ T_eq ≤ 373 K (liquid water range)
2. **Atmospheric retention:** $v_{esc} \geq 5$ km/s (retains N₂, O₂, CO₂)
3. **Magnetic field:** $M \geq 0.5 M_\oplus$ and $R \leq 2.5 R_\oplus$ (dynamo heuristic)

$$SEPHI = \frac{\text{criteria met}}{3}$$

**Source:** Rodríguez-Mozos, J.M. & Moya, A. (2017). MNRAS, 471(4), 4628-4636.

## 4. Habitable Zone Boundaries

Kopparapu et al. (2013) parameterisation of HZ inner/outer edges:

$$S_{eff} = S_0 + aT + bT^2 + cT^3 + dT^4$$

where $T = T_{eff,\star} - 5780$ K. The distance is:

$$d_{HZ} = \sqrt{L_\star / S_{eff}}$$

Four boundaries are computed: recent Venus, runaway greenhouse, maximum greenhouse, early Mars.

**Source:** Kopparapu, R.K. et al. (2013). ApJ, 765, 131.

## 5. Habitable Surface Fraction (HSF)

Fraction of the planetary surface with temperature in [273, 373] K, weighted by cosine of latitude for correct spherical area integration:

$$HSF = \frac{\sum_{i,j} \mathbb{1}_{[273,373]}(T_{i,j}) \cdot \cos(\phi_i)}{\sum_{i,j} \cos(\phi_i)}$$

## 6. Interior-Surface-Atmosphere (ISA) Interactions

Simplified outgassing model based on Kite et al. (2009):

$$\dot{V}_{rel} = \left(\frac{g}{g_\oplus}\right)^{0.75} \cdot \left(\frac{\tau}{4.5\text{ Gyr}}\right)^{-1.5}$$

Assessment criteria:
- **Plate tectonics:** $0.5 \leq M/M_\oplus \leq 5.0$ and $R \leq 2.0 R_\oplus$
- **Carbonate-silicate cycle:** plate tectonics active + 200 ≤ T_eq ≤ 400 K + sufficient outgassing
- **Water cycling:** 273 ≤ T_eq ≤ 373 K
- **Volatile retention:** $v_{esc} \geq 5$ km/s

**Source:** Kite, E.S. et al. (2009). ApJ, 700, 1732.

## 7. Photochemical False-Positive Assessment

UV flux estimation:

$$F_{UV} = f_{UV}(T_{eff}) \cdot \frac{L_{bol}}{4\pi a^2}$$

where $f_{UV}$ is the UV fraction, linearly interpolated from ~0.1% (M-dwarfs) to ~30% (early-type stars).

False-positive risks:
- **O₂:** High UV + cold trap → H₂O photolysis → abiotic oxygen (Luger & Barnes 2015)
- **CH₄:** High volcanic outgassing → abiotic methane
- **O₃:** Abiotic O₂ photochemistry under high UV

**Sources:** Meadows, V.S. et al. (2018). Astrobiology, 18(6), 630-662; Luger, R. & Barnes, R. (2015). Astrobiology, 15(2), 119-143.

## 8. ELM Climate Surrogate

Extreme Learning Machine (Huang et al. 2006) with:
- Single hidden layer, frozen random input weights
- Closed-form solution via Moore-Penrose pseudoinverse: $\beta = (H^TH + \lambda I)^{-1} H^T T$
- Ensemble of K=10 independent ELMs for variance reduction
- Training data: analytically generated temperature maps using simplified radiative model

**Uncertainty quantification:** Conformal prediction intervals using ensemble spread as nonconformity score. At significance level α, intervals are $[\hat{y} - z_{\alpha/2} \cdot \sigma, \hat{y} + z_{\alpha/2} \cdot \sigma]$ where σ is the ensemble standard deviation.

**Limitations:** No convective dynamics, no cloud feedback, no ocean heat transport. Should be validated against GCM output (e.g., ExoCAM, ROCKE-3D).

## 9. Anomaly Detection

Isolation Forest (Liu et al. 2008) applied to the multi-dimensional exoplanet parameter space. Features: radius, mass, orbital distance, period, instellation, equilibrium temperature, stellar temperature, stellar radius.

Anomalous planets may represent:
- Rare potentially habitable candidates in the tail of the distribution
- Data quality issues requiring manual inspection
- Genuinely unusual planetary systems

UMAP (McInnes et al. 2018) provides 2-D visualization of the planet population.

## 10. Data Sources

| Source | Access | Usage |
|--------|--------|-------|
| NASA Exoplanet Archive | TAP/ADQL via `pscomppars` table | Observational planet parameters |
| Kopparapu et al. (2013) coefficients | Hardcoded | HZ boundary computation |
| Chen & Kipping (2017) | Mass-radius relation in validators | Physics guardrails |

## 11. Known Limitations

1. **No GCM validation** — ELM trained on analytical data only; needs calibration against 3-D climate models.
2. **Static atmosphere** — no photochemistry, cloud feedback, or atmospheric evolution.
3. **Fixed albedo** — user-specified rather than self-consistently computed.
4. **No eccentricity** — circular orbits only; eccentric orbits can significantly affect habitability.
5. **ISA heuristic** — outgassing model is highly simplified; real mantle convection depends on rheology, composition, and thermal history.
6. **UV fraction linear** — real stellar spectra are complex; should use model atmospheres (PHOENIX, BT-Settl).
7. **No atmospheric escape modeling** — volatile retention criterion is a simple threshold, not time-integrated escape.

## 12. References

1. Chen, J. & Kipping, D.M. (2017). ApJ, 834, 17.
2. Huang, G.-B. et al. (2006). Neurocomputing, 70, 489-501.
3. Kasting, J.F. et al. (1993). Icarus, 101, 108-128.
4. Kite, E.S. et al. (2009). ApJ, 700, 1732.
5. Kopparapu, R.K. et al. (2013). ApJ, 765, 131.
6. Leconte, J. et al. (2013). A&A, 554, A69.
7. Liu, F.T. et al. (2008). ICDM 2008.
8. Luger, R. & Barnes, R. (2015). Astrobiology, 15(2), 119-143.
9. McInnes, L. et al. (2018). JOSS, 3(29), 861.
10. Meadows, V.S. et al. (2018). Astrobiology, 18(6), 630-662.
11. Pierrehumbert, R.T. (2011). ApJ Letters, 726, L8.
12. Rodríguez-Mozos, J.M. & Moya, A. (2017). MNRAS, 471(4), 4628-4636.
13. Schulze-Makuch, D. et al. (2011). Astrobiology, 11(10), 1041-1052.
14. Seager, S. et al. (2016). Astrobiology, 16(6), 465-485.
15. Shields, A.L. et al. (2016). Physics Reports, 663, 1-38.
16. Turbet, M. et al. (2016). A&A, 596, A112.
17. Wordsworth, R. (2015). ApJ, 806, 180.
