# CTGAN Tasks ⭐
**File to edit:** `modules/data_augmentation.py`

CTGAN generates synthetic training rows for the ELM surrogate. Right now it only knows about the original NASA archive columns. This task extends it to include the three new features so any future model training is chemistry-aware.

**Priority:** Lower than Math Engine and LLM Agent — do this after those are done.

---

## What needs to change

The current `prepare_data()` strips the DataFrame down to 9 columns and only adds a binary `habitable` label.  
We need to add three new columns before CTGAN trains on the data:

| New column | Type | Source |
|---|---|---|
| `co_ratio` | continuous | Set from stellar catalog if available, otherwise sample from U(0.3, 1.1) |
| `surface_pressure_bar` | continuous | Estimate from planet mass/radius or sample from log-uniform distribution |
| `radius_gap_class` | discrete (categorical) | Derived from `radius_earth` using `classify_radius_gap()` |

---

## Task 1 — Extend `prepare_data()` in `data_augmentation.py`

Find the `prepare_data` method.  After the existing `data.columns = [...]` rename block, add:

```python
# ── New chemistry-aware columns ──────────────────────────────────────────────
from modules.astro_physics import classify_radius_gap

# C/O ratio — sample from a physically plausible distribution if not in source
# Solar neighbourhood peaks near 0.55 with spread ~0.15
rng = np.random.default_rng(seed=42)
data["co_ratio"] = rng.normal(loc=0.55, scale=0.15, size=len(data)).clip(0.1, 1.4)

# Surface pressure — log-uniform between 0.001 and 100 bar
log_p = rng.uniform(np.log10(0.001), np.log10(100), size=len(data))
data["surface_pressure_bar"] = np.power(10, log_p).round(4)

# Radius gap classification — derived deterministically from radius
data["radius_gap_class"] = data["radius_earth"].apply(
    lambda r: classify_radius_gap(r)["classification"]
)

# Register new discrete column so CTGAN treats it correctly
self.discrete_columns.append("radius_gap_class")
```

---

## Task 2 — Extend `sample_habitable()` conditioning

Find the `sample_habitable` or equivalent sampling method. Make sure the returned synthetic rows include the three new columns.  
If CTGAN samples don't include them automatically (because the model was trained with them), no extra work is needed — they'll be part of the model.  
If the method post-filters the output DataFrame, make sure it doesn't drop them:

```python
# Keep all columns including new chemistry ones
required_cols = [
    "radius_earth", "mass_earth", "semi_major_axis_au",
    "period_days", "insol_earth", "t_eq_K",
    "star_teff_K", "star_radius_solar", "star_mass_solar",
    "habitable", "co_ratio", "surface_pressure_bar", "radius_gap_class",
]
return synthetic[required_cols]
```

---

## Task 3 — Update `train_models.py` if it calls `prepare_data`

Check `train_models.py` — if it passes a fixed list of column names anywhere after calling `prepare_data`, add the three new columns to that list too.

---

## Task 4 — Update ELM feature vector (optional, do last)

In `modules/elm_surrogate.py`, the `predict_from_params` method uses a fixed dict of input features.  
If you eventually want the ELM to take chemistry into account, add the new fields:

```python
params = {
    "radius_earth": planet_radius,
    "mass_earth": planet_mass,
    "semi_major_axis_au": semi_major,
    "star_teff_K": star_teff,
    "star_radius_solar": star_radius,
    "insol_earth": S_norm,
    "albedo": albedo,
    "tidally_locked": int(locked),
    "co_ratio": co_ratio,              # NEW
    "surface_pressure_bar": pressure,  # NEW
}
```

**Note:** Only do Task 4 if you retrain the ELM. Passing new columns to an already-trained ELM will break it.

---

## Checklist
- [ ] `co_ratio` column added in `prepare_data()`
- [ ] `surface_pressure_bar` column added in `prepare_data()`
- [ ] `radius_gap_class` column derived and added as discrete in `prepare_data()`
- [ ] `sample_habitable()` returns the new columns
- [ ] `train_models.py` updated if needed
- [ ] (Optional) ELM feature vector extended + model retrained
