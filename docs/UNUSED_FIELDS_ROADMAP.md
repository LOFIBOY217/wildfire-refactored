# NBAC + NFDB Unused Fields Roadmap

**Updated**: 2026-04-21 — after adding `--label_fusion`.

Documents all NBAC / NFDB attributes beyond what the current label-fusion
path uses, and proposes concrete next-iteration uses with ROI estimate.

## Currently used (by `--label_fusion`)

| Source | Field | Used as |
|---|---|---|
| NBAC | `AG_SDATE` / `AG_EDATE` | fire_stack active date window |
| NBAC | `geometry` polygon | fire_stack spatial mask |
| NFDB | `REP_DATE` | ignition date |
| NFDB | `LATITUDE` / `LONGITUDE` (or geometry) | ignition pixel |
| NFDB | `SIZE_HA` (optional) | size threshold filter |

## NBAC — unused fields

| Field | Values | Potential use | ROI | Complexity |
|---|---|---|---|---|
| `FIRECAUS` | Natural / Human / Unknown / Prescribed | New 1-hot encoder channel → model learns that lightning-caused fires cluster with CAPE/thunderstorm weather, human-caused cluster near population | **High** (cause → weather linkage is physical) | Medium (need year-by-year rasterize with cause label) |
| `POLY_HA` | 0.1 – millions | **Sample weight**: larger polygons = more reliable label; weight loss by log(POLY_HA+1) to downweight noisy small fires | **High** (per Giglio 2016, small fires have high false-negative) | Low |
| `FIREMAPS` | Landsat / MODIS / VIIRS / Aerial / Digitized | Quality flag: Landsat-mapped fires (dNBR) are most trustworthy; aerial-digitized may have location error. Drop low-quality fires from training. | Medium | Low |
| `FIREMAPM` | Processed imagery / Vector overlay / other | Similar quality proxy | Low | Low |
| `HS_SDATE` – `AG_SDATE` (diff) | 0 – 30+ days | Detection lag feature: larger diff = fire more undetected by satellites. Could define per-year "detection quality" weight for loss. | Medium (novel) | Medium |
| `ADMIN_AREA` | Province code (BC/AB/ON/…) | Stratified per-province eval (Lift@5000 per province). Would expose regional biases. | Medium | Low |
| `NATPARK` | National park name or null | Identify protected-area fires (fire mgmt differs). Not directly useful for prediction. | Low | Low |
| `PRESCRIBED` | y/n | Exclude prescribed (managed) burns from training — these are NOT the target distribution we want to predict. **Remove from labels**. | **High** (data-quality fix) | Trivial |

## NFDB — unused fields

| Field | Values | Potential use | ROI | Complexity |
|---|---|---|---|---|
| `CAUSE` | H / N / U / H-PB / RE | Same use as NBAC FIRECAUS → cause-of-fire channel | High | Medium |
| `FIRE_TYPE` | Wildfire / Prescribed / etc. | Filter out `Prescribed` from training labels | High | Trivial |
| `ATTK_DATE` | Date agency began suppression | Measures fire management response; could weight loss (fires attacked later = more damage potential) | Low | Medium |
| `OUT_DATE` | Date fire declared out | With REP_DATE, gives fire duration. Longer fires = more significant. Could stratify eval. | Low | Low |
| `SIZE_HA` | 0.01 – 1M+ | Already optional filter. Could also: (a) weight loss by size, (b) stratified eval by size bucket. | **High** (size-bucket eval is informative for paper) | Low |
| `RESPONSE` | Full / Modified / Monitored / None | Management type; fires in "no-response" zones have different distribution | Low | Low |
| `PROTZONE` | FIMS protection zone category | Geographic management zone. Similar to ADMIN_AREA. | Low | Low |
| `SRC_AGENCY` | Province code | Per-agency analysis / eval stratification | Low | Low |
| `CAUSE2` | Secondary cause | Rare, mostly duplicates CAUSE | Very low | N/A |

## Proposed roadmap (by priority / ROI)

### Phase 1 (immediate, low-risk)
1. **Filter out prescribed burns** from label fusion. Trivial — just skip
   `FIRECAUS in ("Prescribed", "H-PB", "RE")` in rasterize. Fixes a small
   but genuine label noise.
2. **Per-province stratified eval** using NFDB `SRC_AGENCY`. Add
   `--eval_by_province` flag that reports Lift@5000 per province. No model
   change needed; just post-hoc eval.

### Phase 2 (if Phase 1 helps, medium-risk)
3. **Fire-size stratified eval**: break Lift@5000 into buckets
   (<10 ha, 10-100, 100-1000, 1000+). Makes paper compelling: "model best
   at detecting large fires, struggles with micro-fires".
4. **Sample weighting by log(POLY_HA+1)**: downweight tiny, noisy early
   fires. Requires modifying loss computation.

### Phase 3 (potentially transformative)
5. **Cause channel**: NBAC `FIRECAUS` → rasterize per year as an extra
   feature channel (1 = lightning zone, 0 = human-dominant / unknown).
   Natural fires correlate with CAPE / thunderstorm weather; human with
   road density + population. This makes the model physics-richer.
6. **Detection-lag quality weighting**: compute per-fire `AG_SDATE -
   HS_SDATE`, aggregate per year, use as inverse weight in loss to
   downweight years where CWFIS is noisy (early 2000s).

## Known pitfalls

- `FIRECAUS` in NBAC is *per-polygon* constant; model would see it as a
  static map per year, not per-day. This could cause label leakage if
  used incorrectly (seeing cause of current fire as input).
- NFDB `SIZE_HA` is FINAL fire size, known only after the fact. Using it
  as a feature at prediction time is leakage. Only safe uses:
  - Filter out tiny fires from TRAINING labels (not test)
  - Weight loss at training time
  - Stratify EVAL windows post-hoc

## Not included here

- CIFFC daily situation reports (not accessible historically; skipped)
- FIRMS MODIS 2000-2011 backfill (deferred; script exists for VIIRS 2012+,
  needs MODIS variant)
