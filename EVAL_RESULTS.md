# Evaluation Results — Wildfire Prediction

All 20-window results unless marked (full) = 811 windows.
K=5000 pixels (~2万km²). Goal: improve Lift@5000 metric.

Last updated: **2026-04-17**

---

## Best Results Summary (Current Leader: enc35)

| Model | Best Lift@5000 | Config | Eval |
|-------|---------------|--------|------|
| **V3 9ch enc35 (ep1)** | **6.56x ★** | focal, bs=4096, drop=0.2, in_days=35 | 20-win |
| V3 9ch enc28 (ep1) | 6.43x | same, in_days=28 | 20-win |
| V3 9ch enc21 (ep1) | 5.80x | same, in_days=21 | 20-win |
| V3 9ch enc14 (ep3) | 5.37x | same, in_days=14 | 20-win |
| V3 9ch reg (ep1) | 5.43x | same, in_days=7 | 20-win |
| V3 9s2s (ep2) | 5.06x | OLD (bs=1024, drop=0.1), 7 day | 20-win |
| V3 9s2s (full eval) | **4.69x** | same | **full 625-win** |
| V3 full 16ch (ep1) | 4.98x | bs=1024 | 20-win |
| V3 13ch (ep3) | 4.81x | bs=1024 | 20-win |
| **V3 8ch BCE leak-free (ep7)** | **4.79x** | V2 channels + BCE, no focal | 20-win |
| Oracle V2 (ep2) | 10.01x | perfect future weather | partial |
| **Climatology** | **4.20x** | static map | full |
| FWI Oracle | 1.62x | same-day FWI | full |

⚠️ V2 S2S Legacy (7.35x) had **fire_clim data leakage** — real V2 is ~4.79x (confirmed by V3 8ch BCE).

---

## Key Findings (Ranked by Effect Size)

### 1. Encoder Length = Best Lever (diminishing returns past enc28)
```
enc7:  4.30-5.43x  (depends on regularization)
enc14: 5.37x
enc21: 5.80x
enc28: 6.43x
enc35: 6.56x ★  (+0.13x over enc28 — marginal)
```
Longer encoder captures multi-week drought/warming trends. enc35 is current SOTA
but the enc28→enc35 delta is much smaller than enc21→enc28 (+0.13x vs +0.63x).
Pending: enc42, enc56 (running 2026-04-17) to confirm ceiling.

### 2. Anti-Overfit Config = Necessary
- bs=4096 (was 1024) — 4× fewer gradient updates
- dropout=0.2 (was 0.1)
- epochs=4 (was 8 — best is always ep1-3)
- ROC-AUC degradation: 0.86→0.83 vs 0.86→0.80

### 3. More Channels (9→13→16) = No Help
- Adding deep_soil, precip_def, NDVI, burn_count, u10/v10/CAPE → 0 improvement
- Signal bottleneck is in S2S forecast quality + encoder history, not input richness

### 4. V2 Advantage = Fire_clim Leakage
- V2 used static fire_climatology.tif (included val-period fires)
- V3 fair comparison (same V2 channels, leak-free fire_clim) = 4.79x
- V2's reported 7.35x had ~2.5x leak inflation

### 5. Full-patch Decoder (2048 dim) = Impractical
- 4.9TB cache on Lustre → 5 days for 1 epoch
- Compressed 128-dim caches (309GB) viable alternative (experiments pending)

---

## V3 Anti-Overfit: Full Results (4 epochs each)

### enc35 (current best — added 2026-04-17)
| Ep | Loss | Lift@5000 | Prec | ROC-AUC |
|----|------|-----------|------|---------|
| **1** | 0.008869 | **6.56x ★** | 0.535 | 0.860 |
| 2 | 0.006114 | 6.49x | 0.541 | 0.847 |
| 3 | 0.005388 | 5.72x | 0.467 | 0.816 |
| 4 | 0.005039 | 5.75x | 0.477 | 0.796 |

### enc28
| Ep | Loss | Lift@5000 | Prec | ROC-AUC |
|----|------|-----------|------|---------|
| **1** | 0.008833 | **6.43x** | 0.528 | 0.858 |
| 2 | 0.006211 | 5.57x | 0.464 | 0.841 |
| 3 | 0.005495 | 5.63x | 0.465 | 0.811 |
| 4 | 0.005138 | 4.96x | 0.413 | 0.788 |

### enc21
| Ep | Lift@5000 | Prec | ROC-AUC |
|----|-----------|------|---------|
| **1** | **5.80x** | 0.470 | 0.856 |
| 2 | 4.86x | 0.399 | 0.842 |
| 3 | 5.29x | 0.433 | 0.810 |
| 4 | 4.98x | 0.402 | 0.805 |

### enc7 (anti-overfit)
| Ep | Lift@5000 | ROC-AUC |
|----|-----------|---------|
| **1** | **5.43x** | 0.868 |
| 2 | 4.80x | 0.841 |
| 3 | 4.70x | 0.836 |
| 4 | 4.72x | 0.830 |

---

## V3 8ch BCE (Fair V2 Comparison — Leak-free)

Same 8 channels as V2, BCE loss (not focal), no hard neg mining, annual fire_clim.

| Ep | Lift@5000 | Prec | ROC-AUC |
|----|-----------|------|---------|
| 1 | 3.90x | 0.310 | 0.799 |
| 3 | 4.21x | 0.339 | 0.806 |
| 5 | 4.67x | 0.376 | 0.776 |
| **7** | **4.79x ★** | 0.380 | 0.774 |

**Conclusion**: V2 architecture with leak-free fire_clim ≈ 4.79x. V2's reported 7.35x was **fire_clim leakage**.

---

## Pending Experiments

### In GPU queue (2026-04-14)
- **enc35** (59290999) — continue encoder sweep
- **enc42** (59291000)
- **enc56** (59291001)
- **enc28 + subpatch4x4 decoder (128 dim)** (59321609)
- **enc28 + PCA-128 decoder** (59321611)

### Not yet configured
- Ultimate experiment: best encoder + 128-dim decoder + new fire_clim + 2000-2025 training data

---

## Data Expansion (2026-04-14)

### Channels extended to 2000-2025 ✅
| Channel | Files | Status |
|---------|-------|--------|
| FWI, FFMC, DMC, DC, BUI, u10 | 3200+ each | Pre-existing |
| 2t, 2d, tcw, sm20, st20, v10, cape | 6292 each | Just resampled |
| fire_clim_annual | 8 TIFs | Rebuilt with 22-year history (2.5x more nonzero) |
| hotspot CSV | 10.8M rows | Merged 2000-2017 + 2018-2025 |

### Still downloading
- deep_soil (swvl2): CDS request queued
- precip (tp): CDS request queued
- NDVI: NASA Earthdata downloading

### Skipped
- NBAC burn_scars 2000-2017: URLs broken
- S2S forecast (decoder input): physically limited, no benefit from extending

---

## Disk Cleanup Log

| Date | Action | Freed |
|------|--------|-------|
| 2026-04-07 | Deleted ERA5 raw GRIBs (12K files) | 161G |
| 2026-04-07 | Deleted V2 memmap + fire caches | 302G |
| 2026-04-10 | Deleted s2s_processed/ | 10.7T |
| **Total** | | **~11.2T freed** |

Scratch: ~5TB used / 20TB quota.

---

## What's NOT Worth Doing (Based on Experiments)

- More channels — doesn't help
- Focal loss / hard neg mining — ~zero vs BCE
- Decoder_ctx dual-path — ~zero
- Full-patch decoder 4.9TB — too slow (5d/epoch)
- Lightning (GOES GLM <52°N limit, CAPE is proxy)

## What TO Do

1. **Longer encoder** (enc35-56 in progress)
2. **128-dim compressed S2S decoder** (subpatch4x4, PCA — in progress)
3. **Extend train set to 2000-2025** (data prep in progress)
4. **Combine all three** (ultimate experiment)
5. **Consider task reformulation** if above hits ceiling — coarsen to 30km grid matching S2S resolution
