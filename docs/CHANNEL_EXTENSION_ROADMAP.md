# Channel Extension Roadmap

*Drafted 2026-04-29. Based on direct comparison vs BCWildfire (AAAI 2026, arXiv:2511.17597) which uses 38 channels vs our 9-13-16.*

This document captures candidate channels we don't yet use, their data
sources, implementation cost, and expected ROI. Decision criteria for
activation are at the bottom.

---

## Current channel set

| Channel | Source | 9ch | 13ch | 16ch |
|---|---|---|---|---|
| FWI | CDS cems-fire-historical | ✓ | ✓ | ✓ |
| 2t (2m temperature) | ERA5 | ✓ | ✓ | ✓ |
| 2d (2m dewpoint) | ERA5 | ✓ | ✓ | ✓ |
| tcw (total column water) | ERA5 | ✓ | ✓ | ✓ |
| sm20 (soil moisture 0-7cm) | ERA5 swvl1 | ✓ | ✓ | ✓ |
| deep_soil (7-28cm) | ERA5 swvl2 | | ✓ | ✓ |
| precip_def (deficit) | computed | | ✓ | ✓ |
| u10 / v10 wind | ERA5 | | | ✓ |
| CAPE | ERA5 | | | ✓ |
| NDVI | MODIS MOD13A2 | | ✓ | ✓ |
| population | WorldPop | ✓ | ✓ | ✓ |
| slope | SRTM | ✓ | ✓ | ✓ |
| fire_clim | computed | ✓ | ✓ | ✓ |
| burn_age | NBAC | ✓ | ✓ | ✓ |
| burn_count | NBAC | | ✓ | ✓ |

---

## Tier 1 — high ROI, low effort (recommended if channel expansion shows uplift)

| Channel | Source | Effort | Expected uplift |
|---|---|---|---|
| **snow_cover** | ERA5 `sd` (snow depth) or `sde` | Low (CDS workflow exists) | ⭐⭐⭐ Critical for spring fire season — snow = non-flammable; current model has no snow signal |
| **landcover** (MCD12Q1) | MODIS annual | Low (MODIS workflow exists for NDVI) | ⭐⭐⭐ Distinguishes boreal/grassland/tundra — fire behavior differs by 10x across types |
| **dist_to_roads** | OSM Canada | Medium (geopandas + scipy.ndimage.distance_transform) | ⭐⭐⭐ Direct human-ignition proxy; static |
| **aspect** | SRTM (existing DEM) | Low (`gdal_aspect` one-liner) | ⭐⭐ Sun exposure → fuel drying; static |
| **MODIS LST day/night** | MOD11A1 | Medium (parallel to NDVI) | ⭐⭐ Direct fuel temperature (vs 2m air temp) |

### Tier 1 totals
- 5 new channels
- ~3 weeks one-time data prep
- 1 cache rebuild
- 8 training jobs (4y + 22y × 4 enc)

---

## Tier 2 — medium ROI, medium effort

| Channel | Source | Effort | Expected uplift |
|---|---|---|---|
| **LAI** (Leaf Area Index) | MCD15A3H | Medium | ⭐⭐ Direct fuel-load proxy |
| **FPAR** (photo activity) | MCD15A3H (same product) | Medium | ⭐⭐ Vegetation vigor / photosynthesis |
| **EVI** (vs NDVI) | MOD13A2 (NDVI tooling exists) | Low | ⭐ Marginal (overlapping with NDVI) |
| **MODIS bands 1-3 & 7** | MOD09CMG | Medium | ⭐ Overlaps with LAI/FPAR/NDVI |
| **Latent heat flux** | ERA5 `slhf` | Low | ⭐⭐ Surface drying signal |
| **dist_to_water** | OSM lakes/rivers | Medium | ⭐⭐ Moisture availability proxy |
| **Surface pressure** | ERA5 `sp` | Low | ⭐ Implicit in FWI; marginal |

---

## NOT to add (justified rejections)

| Channel | Why not |
|---|---|
| **MOD/MYD14A1 active fire** | We deliberately rejected CWFIS hotspots (350× detection drift across satellite swap-ins 2002/2012/2015). MODIS active fire has the same temporal-instability problem. Including it would re-introduce the problem we fixed by switching to NBAC+NFDB. |
| **VIIRS active fire** | Same reason; hot-spot products have known cross-platform calibration drift. |

---

## Activation criteria

Add Tier 1 channels IF AND ONLY IF the upcoming 4y/12y/22y × 9ch/13ch/16ch
sweep shows monotonic increase from 9ch → 13ch → 16ch on Lift@30km.

```
9ch_baseline →  +5% → 13ch  →  +5% → 16ch   ⇒  ADD Tier 1 (likely +5-10% more)
9ch_baseline →  +1% → 13ch  →  +1% → 16ch   ⇒  PLATEAU; Tier 1 unlikely to help
9ch_baseline →  -2% → 13ch                   ⇒  CHANNELS HURT; STOP, fix recipe first
```

---

## Decision date

After 12-model sweep completes (~D+5 from 2026-04-29). Re-evaluate this
roadmap based on 9/13/16 ch ranking.

---

## BCWildfire 38-channel reference (their list, for completeness)

From arXiv:2511.17597 Table 1:

**Fuel state (8)**:
- LAI, FPAR (MCD15A3H)
- MODIS reflectance bands 1, 2, 3, 7 (MOD09CMG)
- NDVI, EVI (MOD13A2)

**Meteorology (15+)**:
- 2m T, 2m dewpoint, surface pressure
- u10, v10 wind
- precipitation
- latent heat flux
- snow cover
- soil moisture × 4 layers (0-7, 7-28, 28-100, 100-289 cm)
- MODIS LST day, MODIS LST night
- MOD09CMG reflectance composite

**Topography (4)**:
- slope, aspect, hillshade (ASTER DEM derived)
- distance-to-water

**Anthropogenic (2)**:
- land use class (MCD12Q1)
- distance-to-infrastructure (OSM)

**Fire history (1+)**:
- MODIS active fire detections (MOD/MYD14A1)

**Total: 38 features**

Note: their target label is also MOD/MYD14A1 active fire (binary
"actively burning today"). Their setup is intra-MODIS — fuel signal,
target signal, and many features all from same MODIS family. Ours is
cross-source (NBAC perimeters + ERA5 reanalysis + MODIS NDVI).
