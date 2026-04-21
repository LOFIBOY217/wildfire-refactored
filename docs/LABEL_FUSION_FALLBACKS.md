# Label Fusion Fallback Plans

If Phase 2 A/B shows `X2 (NBAC+NFDB training) ≤ X1 (old CWFIS baseline)`
— i.e., new label training does NOT improve over old — these are
the escalation paths, ranked by cost to try.

## Decision matrix (after Phase 2)

| Condition | Interpretation | Next action |
|---|---|---|
| X2 > X1 + 5% | Fusion clearly helps | Proceed Phase 3 (22y extension) |
| X2 ≈ X1 (±5%) | No clear effect | Try Fallback A first |
| X2 < X1 - 5% | Fusion HURTS | Try Fallback B or C |

## Fallback A: Narrow NBAC date window

**Hypothesis**: NBAC AG_SDATE..AG_EDATE is often 4-8 weeks wide.
Labeling every day of that window with label=1 may over-smooth the
time signal. Model sees "fire here for 6 weeks straight" and can't
learn when exactly to predict.

**Fix**: Change NBAC rasterize to only label `HS_SDATE ± 3 days`
(the narrow satellite-observation window, ~7-10 days typically).

**Implementation**:
```bash
python3 scripts/build_fire_labels.py --scheme nbac_nfdb \
    --nbac_date_source HS \          # instead of AG
    --nfdb_min_size_ha 1.0 \
    --start 2000-05-01 --end 2025-12-21 \
    --output_dir data/fire_labels
```

**Cost**: ~5h relabel + 3 days retrain.

## Fallback B: NFDB-only labels (drop NBAC)

**Hypothesis**: NBAC polygons cover tens of km² and introduce spatial
noise (the fire actually burned a fraction of polygon area at each
moment). NFDB points are single-pixel + single-day, precise.

**Fix**: Use ONLY NFDB ignition points (dilated r=14). Drop NBAC
from the union.

**Implementation**: need new flag `--label_source nfdb_only`.

**Cost**: ~1h relabel (NFDB is fast) + 3 days retrain.

**Trade-off**: Loses spatial extent information — model predicts "fire
near this ignition point" but misses the full burn area. May hurt
recall.

## Fallback C: Per-year sample weighting (keep CWFIS)

**Hypothesis**: Maybe drift is real but model handles it fine because
encoder features (fire_clim, burn_age) are also drifty — they cancel
out. Fusion may be unnecessary.

**Fix**: Go back to CWFIS labels but add per-year sample weighting
so early years contribute equally.

**Implementation**: add `--year_sample_weight linear` flag to train_v3.py
that upweights 2000-2011 samples by 1 / CWFIS_count[year].

**Cost**: 1 day code + 3 days retrain.

**Trade-off**: Fights symptom not cause — model still sees wrong
targets in early years.

## Fallback D: Only post-VIIRS era (2012+)

**Hypothesis**: Most of the drift is pre-VIIRS (2000-2011). If we
restrict training to 2012-2022, CWFIS labels are "good enough".

**Fix**: Change `--data_start 2012-05-01` in training script, use
CWFIS labels.

**Cost**: Minutes (just flag change).

**Trade-off**: Gives up half of 22y data extension. Effectively falls
back to 10y training, which is still 2.5× the 4y SOTA.

## Fallback E: Different dilation radius

**Hypothesis**: r=14 (28km) was tuned for CWFIS. Different labels may
need different r.

**Try**: r=20 (40km) for more positive signal density, r=7 (14km) for
sharper localization.

**Implementation**:
```bash
python3 scripts/build_fire_labels.py --scheme nbac_nfdb \
    --dilate_radius 20 --start ... --end ... \
    --output_dir data/fire_labels
```
Cache key includes `r{N}`, so multiple dilations coexist.

**Cost**: ~5h relabel per variant + 3 days retrain per variant.

**Risk**: Lift@30km is set up for r=14; changing dilation needs
re-validating the metric alignment.

## Fallback F: Abandon label change, focus elsewhere

If NONE of A-E help, the label change simply doesn't improve the
task. Revert to CWFIS labels, 4y SOTA, and pivot to:

1. **Architecture changes**: try different encoder sizes, attention
   variants, bigger model.
2. **Data augmentation**: rotations, masking, synthetic fires.
3. **Multi-task learning**: predict fire + burn area + ignition cause
   jointly.
4. **Ensemble**: train multiple models with different seeds/configs,
   average predictions.

**Cost**: Weeks of research.

## Recovery plan if worse than original

If Phase 2 shows X2 << X1, we don't lose anything:
- Old v3_9ch_enc21 CWFIS checkpoint is preserved
- Can continue reporting 4y SOTA (6.47x on CWFIS labels)
- Just rollback the `--label_fusion` flag default

## Current commitment

**Phase 2 = safety test** (only 3 days of compute). If it fails, we
back off. No permanent damage to the project.
