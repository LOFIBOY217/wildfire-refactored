# Data Conventions

**Last updated:** 2026-04-18

This document is the **single source of truth** for all data file conventions in the project. Any new data source or processing script MUST follow these conventions.

---

## 1. Spatial Reference (all rasters)

| Property | Value |
|---|---|
| CRS | **EPSG:3978** (Canada Lambert Conformal Conic) |
| Shape | **2281 × 2709** (H × W) |
| Resolution | ~2 km |
| Bounds | `(-2378164.0, -707617.0, 3039835.0, 3854382.0)` |
| dtype | `float32` (exceptions noted below) |

**Reference transform:**
```python
from rasterio.transform import from_bounds
FWI_BOUNDS = (-2378164.0, -707617.0, 3039835.0, 3854382.0)
dst_tf = from_bounds(*FWI_BOUNDS, 2709, 2281)
```

---

## 2. File Naming

### Daily (weather/observation channels)
**Pattern:** `{var}_{YYYYMMDD}.tif`

| Channel | Directory | Pattern |
|---|---|---|
| FWI | `data/fwi_data` | `fwi_YYYYMMDD.tif` |
| FFMC | `data/ffmc_data` | `ffmc_YYYYMMDD.tif` |
| DMC | `data/dmc_data` | `dmc_YYYYMMDD.tif` |
| DC | `data/dc_data` | `dc_YYYYMMDD.tif` |
| BUI | `data/bui_data` | `bui_YYYYMMDD.tif` |
| 2t | `data/ecmwf_observation/2t` | `2t_YYYYMMDD.tif` |
| 2d | `data/ecmwf_observation/2d` | `2d_YYYYMMDD.tif` |
| tcw | `data/ecmwf_observation/tcw` | `tcw_YYYYMMDD.tif` |
| sm20 | `data/ecmwf_observation/sm20` | `sm20_YYYYMMDD.tif` |
| st20 | `data/ecmwf_observation/st20` | `st20_YYYYMMDD.tif` |
| u10 | `data/era5_u10` | `u10_YYYYMMDD.tif` |
| v10 | `data/era5_v10` | `v10_YYYYMMDD.tif` |
| CAPE | `data/era5_cape` | `cape_YYYYMMDD.tif` |
| tp | `data/era5_precip` | `tp_YYYYMMDD.tif` |
| swvl2 | `data/era5_deep_soil` | `swvl2_YYYYMMDD.tif` |
| NDVI | `data/ndvi_data` | `ndvi_YYYYMMDD.tif` *(16-day representative date)* |

### Annual (rolling-window or per-year-snapshot)
**Two semantic variants — BOTH use the same naming:**

#### Variant A: "upto" (strictly before year Y)
**Pattern:** `{var}_upto_{YYYY}.tif`
- `fire_clim_upto_2020.tif` contains climatology from hotspots in years `[data_start, 2019]` — `2020` is NOT included. Use for target year 2020 **directly, no offset**.

#### Variant B: "atyear" (data in year Y)
**Pattern:** `{var}_{YYYY}.tif`
- `years_since_burn_2020.tif` contains burn info **including 2020 fires**.
- `burn_count_2020.tif` same.
- **Training MUST use year-1** (`cur_date.year - 1`) to avoid temporal leakage.

⚠️ **Leakage risk:** Variant B is fragile — easy to forget the year-1 offset. Prefer Variant A for new channels. Existing Variant B channels must stay to avoid breaking caches.

### Static (location-dependent, time-independent)
**Pattern:** `{name}.tif`
- `data/population_density.tif`
- `data/terrain/slope.tif`

---

## 3. nodata Convention

### Target standard: **`nan`**
IEEE 754 NaN, float32. Propagates loudly through math — missing data failures are visible, not silent.

### Current state (2026-04-18 audit)
| Convention | Count | Channels |
|---|---|---|
| `nan` ✅ | 16 | FWI/FFMC/DMC/DC/BUI/u10/v10/CAPE/tp/swvl2/NDVI/population/slope + 2000-2008 ERA5 obs |
| `-9999.0` ❌ | 5 | 2018+ `ecmwf_observation/{2t,2d,tcw,sm20,st20}` |
| `9999` uint16 sentinel ❌ | 1 | `years_since_burn_*.tif` (**historical bug source**) |
| `0.0` ❌ | 1 | `burn_count_*.tif` (0 is also a real value; ambiguous) |
| `None` (no metadata) ❌ | 1 | `fire_clim_upto_*.tif` (all values are valid, no missing) |

### Loader contract (defensive)
Every loader must handle both:
```python
arr[~np.isfinite(arr)] = 0.0    # handles nan/inf
if src.nodata is not None:
    arr[arr == src.nodata] = 0.0  # handles -9999, 9999 sentinels
```
Example: `src/training/train_v3.py:_load_static_channel`.

### Migration plan
- **Do not migrate** existing 2018+ ecmwf_observation/* from -9999 to nan (risk of breaking production caches).
- **New data** (like our 2000-2017 extension) uses `nan`. Mixed time series accepted (loader handles per-file).
- **burn_age 9999 sentinel**: fixed at loader level (`src.nodata` is correctly set to 9999 in metadata).

---

## 4. Compression & Tiling

| Property | Value |
|---|---|
| Compression | `lzw` (all channels) |
| Tiled | No (striped — default) |
| Driver | `GTiff` |

---

## 5. Unit Conventions

| Variable | Unit | Notes |
|---|---|---|
| 2t, 2d, st20 | **°C** | Converted from Kelvin at extraction time |
| tcw | kg/m² | As-is |
| sm20, swvl2 | volumetric (0-1) | As-is |
| u10, v10 | m/s | As-is |
| CAPE | J/kg | As-is |
| tp | **m/day** (storage) / **mm/day** (loader) | Stored as-is from ERA5; train_v3.py applies ×1000 at load time for precip_def channel |
| NDVI | [-1, 1] | Dimensionless |
| FWI/FFMC/DMC/DC/BUI | index | Van Wagner system (unitless) |
| fire_clim | log1p(density) | Log-transformed hotspot density |
| burn_age (encoded) | log1p(years) | Log-transformed years since burn |
| population | log-scaled | WorldPop derived |
| slope | degrees | 0-90 |

---

## 6. Temporal Coverage Audit (2026-04-18)

| Channel | 2000-2008 | 2009-2017 | 2018-2025 | Notes |
|---|---|---|---|---|
| FWI/FFMC/DMC/DC/BUI | ✅ | ✅ | ✅ | Full coverage |
| 2t/2d/tcw/sm20/st20 | ✅ (nan) | ✅ (nan, 2026-04-18) | ✅ (-9999) | mixed nodata within var |
| v10/CAPE | ✅ (nan) | ✅ (nan) | ✅ (nan) | Full |
| u10 | ⚠️ partial (311 files in 2000) | ❌ 2001-2008 MISSING | ✅ (2009+) | **Blocks 16ch extension** |
| tp | ❌ 2000-2017 missing | ❌ | ✅ | Blocks 13ch extension |
| swvl2 (deep_soil) | ⏳ processing now (2026-04-18 job 59565812) | ⏳ | ✅ | Will be full after job completes |
| NDVI | ✅ | ✅ | ✅ | Full 2000-2025 |
| fire_clim | ✅ 2000-2025 | ✅ | ✅ | 26 annual files, rebuilt 2026-04-17 |
| burn_age/count | ✅ 2000-2024 | ✅ | ✅ (to 2024) | 25 years |

### What's needed for each channel set
- **9ch** (FWI/2t/fire_clim/2d/tcw/sm20/pop/slope/burn_age): ✅ **2000-2025 complete**, ready
- **13ch** (+deep_soil/precip_def/NDVI/burn_count): ⚠️ needs swvl2 2000-2017 (processing now) + tp 2000-2017
- **16ch** (+u10/v10/CAPE): ❌ needs u10 2001-2008 (**missing**)

---

## 7a. 🚨 KNOWN DATA BUGS (2026-04-18 audit)

### Bug #1: reproject nan-edge (5 channels × multiple years)

Channels with 24% nan pixels concentrated in Canada Lambert edges
(N/E Canada beyond WGS84 source bbox [-141..-52, 41..83]):

| Channel | 2000-2008 | 2009-2017 | 2018-2025 |
|---|---|---|---|
| 2t, 2d, tcw, sm20, st20 | ✅ 100% | ✅ 100% (reprocessed 2026-04-18) | ✅ 100% |
| **u10, v10, cape** | ✅ 100% | ⏳ reprocessing | **❌ 76.3%** (legacy bug) |
| **swvl2** | ? | ⏳ reprocessing | **❌ 76.3%** (legacy bug) |
| **tp** | ❌ never processed | ❌ never processed | **❌ 76.3%** (legacy bug) |

Root cause: reproject() called with src_nodata=nan/dst_nodata=nan
prevents bilinear edge extension. Fixed in unified
slurm/process_era5_narval.sh (invariant: NO src_nodata/dst_nodata).

Impact on past experiments:
- **9ch training** (FWI/2t/fire_clim/2d/tcw/sm20/pop/slope/burn_age):
  ✅ NOT affected (only uses fully-valid channels)
- **13ch experiments** (+deep_soil/precip_def/NDVI/burn_count):
  ❌ deep_soil and precip_def used 24%-nan data → Lift 4.81x suspect
- **16ch experiments** (+u10/v10/CAPE):
  ❌ all three used 24%-nan data → Lift 4.98x suspect

Remediation plan (when needed for 13/16ch retraining):
```bash
# Delete bad 2018-2025 files
rm data/era5_{u10,v10,cape,deep_soil,precip}/*_{2018..2025}*.tif
# Re-resample from existing WGS84 source
START_YEAR=2018 END_YEAR=2025 VARS=u10,v10,cape \
  SKIP_STAGE1=1 sbatch slurm/process_era5_narval.sh
START_YEAR=2018 END_YEAR=2025 VARS=swvl2 \
  GRIB_DIR=data/era5_deep_soil SKIP_STAGE1=1 sbatch slurm/process_era5_narval.sh
# Etc. for tp (after downloading missing 2000-2017 tp data)
```

---

## 7. Historical Convention Violations (Do Not Repeat)

These past mistakes are preserved for reference:

1. **burn_age 9999 sentinel bug (fixed c2dd2c2)** — Loader didn't mask sentinel → 99.8% of pixels became sentinel value. Lesson: always read `src.nodata` and mask.
2. **fire_climatology.tif leak (V2)** — Static file included val-year fires → Lift inflated 2.5×. Lesson: use rolling `upto_Y.tif` convention.
3. **FFMC/DMC/DC/BUI stats dict missing** — Code built per-channel stats dict but forgot these 4. Stats defaulted to (0, 1) → model saw raw values. Lesson: `VARIABLES` constant must be single source of truth.
4. **NDVI `_YYYY_MM` vs `_YYYYMMDD`** — Earlier drafts used month-level name; now standardized to `YYYYMMDD` (representative date within 16-day period).

---

## 8. Adding a New Data Source — Checklist

Before committing new data files:

- [ ] Output CRS is EPSG:3978
- [ ] Output shape is 2281 × 2709
- [ ] Output dtype is `float32` (unless categorical)
- [ ] nodata is `nan` (project standard)
- [ ] Compression is `lzw`
- [ ] Filename follows `{var}_YYYYMMDD.tif` (daily), `{var}_upto_YYYY.tif` (annual-upto), or `{name}.tif` (static)
- [ ] Unit documented in this file (section 5)
- [ ] Temporal coverage added to section 6
- [ ] Loader has defensive nodata handling
- [ ] Smoke-tested with `scripts/audit_data_complete.py` (the canonical
      data validator — old `validate_all_channels.py` is deprecated)
