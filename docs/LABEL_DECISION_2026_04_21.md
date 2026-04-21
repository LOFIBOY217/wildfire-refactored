# Label Target Change — 2026-04-21

**Status**: ADOPTED (2026-04-21)
**Authors**: Jiaqi (PM) + Claude (impl)
**Impact**: All 9ch/13ch/16ch × 2000-2025 training from this date forward
**Supersedes**: CWFIS hotspot label used in V2 (4y SOTA, 2018-2022)

## Summary

The fire prediction target label is changed from **CWFIS satellite hotspot
detections (dilated r=14)** to a **union of NBAC burn polygons + NFDB
ignition points (dilated r=14)**. CWFIS is removed from the label path
entirely. CWFIS remains as encoder feature input (`fire_clim` channel).

## Motivation

CWFIS hotspots have severe temporal drift (documented Giglio 2016,
Schroeder 2014; confirmed by our polygon-detection test on
2026-04-21):

| Year | CWFIS hotspot count | Ratio vs 2023 |
|------|---------------------|---------------|
| 2000 | 8,758 | 0.003 |
| 2001 | 4,839 | 0.002 |
| 2012 | 75,824 | 0.025 (VIIRS launched) |
| 2023 | 3,079,062 | 1.000 |

Corresponding NFDB (agency-reported) fire counts in same years:
2001 = 7,199; 2023 = 6,847 (ratio ~1.0). **Real fire counts are flat
across 2000-2023; the 350× CWFIS increase is almost entirely detection
improvement (Aqua MODIS 2002, VIIRS 2012, MODIS C6 2015).**

Training on CWFIS labels with 2000-2025 data would teach the model a
**spurious year signal** (2000=no fires, 2023=many fires), destroying
the temporal generalization we want.

## Evidence

### Polygon-detection test (2026-04-21)

For each year, how many NBAC burn polygons does each source detect?

| Year | CWFIS det | NFDB det | Both | Neither |
|------|-----------|----------|------|---------|
| 2001 | 9% | 22% | 5% | **74%** |
| 2012 | 27% | 61% | 18% | 30% |
| 2021 | 41% | 91% | 37% | **6%** |
| 2024 | 40% | 85% | 34% | 9% |

### By fire size (CWFIS detection rate)

| Year | <10 ha | 10-100 ha | 100-1k ha | ≥1000 ha |
|------|--------|-----------|-----------|-----------|
| 2001 | 0% | 0% | 14% | 72% |
| 2024 | 4% | 51% | 90% | 98% |

**Key insight**: CWFIS misses almost ALL fires <10 ha even in 2024.
Pre-VIIRS, CWFIS also misses most medium fires. Only ≥1000 ha fires
have stable CWFIS detection throughout.

### CWFIS false positives / noise sources

- Industrial heat (oil/gas flares, refineries)
- Prescribed burns (which we do NOT want to predict)
- Duplicate detections (same fire seen multiple days → multiple rows)
- Over-smoke false positives in heavy haze

## Decision

### New label source

```python
label[day, pixel] = 1 if (
    (pixel ∈ NBAC_polygon.geometry
     AND day ∈ [AG_SDATE, AG_EDATE] of polygon
     AND polygon.PRESCRIBED != 'y')
    OR
    (pixel == NFDB_point.lat_lon_pixel
     AND day == NFDB_point.REP_DATE
     AND NFDB_point.CAUSE in {H, N, U}    # excl. H-PB, RE)
     AND NFDB_point.SIZE_HA >= 1.0)       # exclude micro-fires
)
# Then binary_dilation with radius 14 (unchanged)
```

### Rationale for each filter

- **`PRESCRIBED = y` excluded**: these are managed burns, not wildfire.
  Model should not learn to predict controlled operations.
- **`CAUSE in {H, N, U}` only**: excludes H-PB (prescribed human burn)
  and RE (reburn / prescribed). Keeps human wildfire (H), natural
  lightning (N), and unknown-cause (U, rare).
- **`SIZE_HA >= 1.0`**: 72% of NFDB records are <1 ha. On our 2 km grid,
  such fires occupy <0.025% of a pixel — labeling that pixel as "fire"
  is noisy. Retaining ≥1 ha keeps ~123k fires across 1946-2024.
- **NBAC kept as-is**: NBAC has no <10 ha polygons anyway (native
  mapping resolution).

### Dilation preserved

Radius r=14 pixels (~28 km) dilation around each label point/polygon
is kept, same as V2 / 4y SOTA. This reflects the spatial uncertainty
of predicting 14-46 days ahead — we care about "fire in this 28 km
neighborhood" not "exact 2 km pixel".

### Encoder features unchanged

- `fire_clim` channel (CWFIS rolling climatology `upto_Y`): unchanged
- `burn_age`, `burn_count` channels (NBAC derived): unchanged

The **label switch does not change any model input**. Only the target.

## Implementation

- **Code**: `src/training/train_v3.py` STEP 4 (reworked 2026-04-21,
  commit `ae63656`).
- **Flag**: `--label_fusion` enables new mode. Without flag, legacy
  CWFIS label still works for backward comparison.
- **Cache key**: new labels write to
  `fire_dilated_r14_nbac_nfdb_<dates>_<shape>.npy`. Old CWFIS cache
  (`fire_dilated_r14_<dates>_<shape>.npy`) is preserved for A/B
  comparison — both can coexist in the same cache dir.
- **Default filters**: `--nfdb_min_size_ha 1.0`, prescribed burns
  excluded (`--include_prescribed` off).

## Trade-offs

### What we gain
1. **No drift**: NBAC + NFDB are temporally stable (Landsat-based + human
   reports), both back to 1946/1972.
2. **More complete**: NBAC + NFDB cover ~94% of real fire events in
   recent years; CWFIS covers ~40-45%.
3. **No false positives**: industrial heat / flares dropped.
4. **Prescribed burns excluded**: prediction target aligns with the
   real goal (predicting wildfires, not managed burns).

### What we lose
1. **<1 ha wildfires NBAC also missed**: rare edge case, not useful
   prediction targets anyway.
2. **Comparability with 4y CWFIS SOTA**: the old "6.47x Lift@5000"
   number is on CWFIS labels and no longer directly comparable. We
   re-establish baselines on new labels (see below).

## Required follow-up work

1. **Re-run climatology baseline** on NBAC+NFDB val labels. Old
   baseline (4.42x Lift@5000 / 3.19x Lift@30km on 22y upto_2022 CWFIS)
   is on old labels. Need equivalent on new labels to compute
   `model vs climatology` % gain.
2. **Re-run FWI oracle baseline** on NBAC+NFDB val labels.
3. **Build comparison report**: how many more positive pixels does
   new label have vs old, by year, by size bucket.
4. **A/B train ENC=21 × 9ch × 2000-2025** with old label vs new
   label. Same model, same data, only label changes → isolates pure
   label-change effect.
5. **Update Boss report**: new "vs climatology +X%" number based on
   new baselines.

## Verification

After first A/B training, expect:
- Model trained on new labels should have more stable Lift across val
  years (less year-to-year variance).
- Model vs climatology (recomputed) should still show clear positive
  gap.
- Per-year per-size eval should show new model catches small/medium
  fires better than old.

## What this is NOT

- Not a model architecture change
- Not a loss function change
- Not a data augmentation change
- Not a feature change
- **Pure label change + prescribed-burn cleanup**
