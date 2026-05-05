# Lightning Data Integration Plan — 2026-05-04

## Goal

Add lightning as a model input channel. Lightning causes ~60% of boreal
Canadian wildfires; our current 9/13/16ch sets have **zero lightning
information**, leaving a known causal driver unmodelled.

---

## What we already have

```
data/lightning_raw/glm_raw_YYYYMMDD.tif   (1 230 files, 57 MB total)
  - Source: NOAA GOES-16/17 Geostationary Lightning Mapper (GLM)
  - Coverage: 2018-05-01 onwards (GLM launched late 2017)
  - Native projection: GOES-East geostationary (NOT EPSG:3978)
  - Format: per-day raster, value = strike count (or detection rate?)

data/lightning/lightning_20230701.tif  (1 file — sample reprojected?)
data/lightning_raw is unprocessed.
```

`train_v3.py` CHANNEL_REGISTRY already has `"lightning": {"type": "daily"}`
wired in (line 135), so the channel infra is ready — **only the data
preparation is missing**.

---

## Two paths (mutually compatible)

### Path A: **Lightning climatology** (static channel) — RECOMMENDED FIRST

A single per-pixel raster `lightning_climatology.tif` holding the mean
annual strike count, computed from all 1 230 days of GLM data. Static
across years, treated like `population` or `slope`.

**Pros**:
- Single 2 km TIF, ~50 MB
- Works for ANY training range (12y, 22y, 4y) — climatology is time-invariant
- Pre-2018 is NOT a problem (lightning hot-spots are stable over decades)
- One-time compute (~1 h CPU) + add to channel set + retrain

**Cons**:
- Doesn't capture day-to-day or year-to-year lightning variability
- Same input value for every fire-season day

**Argument for paper**:
> "Lightning detection requires geostationary satellites (GLM, 2018+) that
> do not cover our pre-2018 training range, so we use a 2018–2024
> lightning climatology as a static spatial prior. Lightning hot-spots are
> driven by orographic + frontal patterns that are stable on multi-decadal
> scales, making climatology a reasonable first-order approximation."

### Path B: **Lightning daily** (dynamic channel)

Each day's strike count, ingested at training time like FWI.

**Pros**:
- Real dynamic signal — anomalous lightning days → fire risk spike
- Direct mechanistic link

**Cons**:
- ONLY 2018+ data → can't use for 22y or 12y training (would need backfill)
- Requires per-day reprojection of 1 230 raw TIFs (~30 min CPU)
- Needs cache rebuild
- For 4y range (2018-2021) it would work, but 4y is not our SOTA range

→ **Defer Path B until we get a model trained on 2018+ range with
  dynamic lightning. Or backfill pre-2018 from another source (CLDN /
  ENSEMBLE).**

---

## Implementation steps for Path A (climatology)

### Step 1 — Build `lightning_climatology.tif`

```python
# scripts/build_lightning_climatology.py
import glob, rasterio, numpy as np
from rasterio.warp import reproject, Resampling

# Read all glm_raw files in parallel
files = sorted(glob.glob("data/lightning_raw/glm_raw_*.tif"))

# Open reference EPSG:3978 grid from existing FWI tif
with rasterio.open("data/fwi_data/fwi_20250615.tif") as ref:
    ref_profile = ref.profile
    H, W = ref.height, ref.width

# Sum per-pixel strikes across all days, on EPSG:3978 grid
strike_sum = np.zeros((H, W), dtype=np.float64)
for f in files:
    with rasterio.open(f) as src:
        src_data = src.read(1).astype(np.float32)
        # NaN → 0 (no detections)
        src_data = np.nan_to_num(src_data, nan=0.0)
    dst = np.zeros((H, W), dtype=np.float32)
    reproject(src_data, dst,
              src_transform=src.transform, src_crs=src.crs,
              dst_transform=ref_profile["transform"],
              dst_crs=ref_profile["crs"],
              resampling=Resampling.bilinear)
    strike_sum += dst

# Convert to per-year mean (1230 days ≈ 3.37 years if continuous, but
# actually fire-season-only — count distinct years × seasons)
n_unique_years = len(set(int(f[-12:-8]) for f in files))
strike_per_year = strike_sum / max(n_unique_years, 1)

# Log-transform for skew (lightning is heavy-tailed)
out = np.log1p(strike_per_year).astype(np.float32)

with rasterio.open("data/lightning_climatology.tif", "w",
                   **{**ref_profile, "dtype": "float32", "count": 1}) as dst:
    dst.write(out, 1)
```

**Cost**: ~30 min CPU (reproject 1 230 daily rasters + sum)
**Output**: `data/lightning_climatology.tif` (~50 MB)

### Step 2 — Add to channel registry

```python
# In train_v3.py CHANNEL_REGISTRY:
"lightning_climatology": {"type": "static", "required": False},
```

In static_arrays loading section:
```python
if "lightning_climatology" in CHANNEL_NAMES:
    static_arrays["lightning_climatology"] = _load_static_channel(
        "data/lightning_climatology.tif", H, W, "lightning_climatology")
```

### Step 3 — Train

`12ch = 11ch + lightning_climatology`:
```
CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,population,slope,burn_age,elevation,aspect,lightning_climatology"
```

12y enc21 training — ~6h GPU. Expected gain: +5–10% Lift@5000 (lightning is
the dominant ignition driver for 60% of boreal fires).

### Step 4 — Validate

Run save_scores → unified metrics → compare 12ch vs 11ch vs 9ch.

---

## Total work estimate (Path A only)

| Step | Time | Type |
|---|---|---|
| Write `build_lightning_climatology.py` | 30 min | code |
| Run climatology build | 30 min | CPU job |
| Update train_v3 channel registry (3 lines) | 5 min | code |
| Write `train_v3_12ch_12y_narval.sh` | 10 min | code |
| Train 12y enc21 + 12ch | 6 h + queue | GPU |
| save_scores + unified eval | 6 h + queue | GPU |
| **Total elapsed** | **~24 h** | |
| **Active engineer time** | **~1 h** | |

---

## Open questions to resolve before kicking off

1. **What does `glm_raw_*.tif` actually represent?**
   - Strike COUNT per pixel per day, or detection RATE [0–1], or energy?
   - Need to inspect one file: `gdalinfo -stats data/lightning_raw/glm_raw_20180601.tif`
2. **Native projection of glm_raw**: GOES-East geostationary. Resampling
   to 2 km EPSG:3978 may artefact at cluster boundaries. Check overlap
   visually.
3. **Coverage gaps**: 1 230 days but probably not contiguous. Need to
   know the exact set of days covered to compute correct
   "mean annual strikes" denominator.

---

## Path B (deferred — daily lightning) — what'd be needed if we go there

1. Reproject 1 230 raw TIFs to EPSG:3978 → `data/lightning/lightning_*.tif`
2. Add to a 4y range training (2018-2021) since GLM only post-2018
3. Eventually: backfill pre-2018 from CLDN (Environment Canada Canadian
   Lightning Detection Network) — point data, requires a partnership.
4. Cache rebuild needed.
