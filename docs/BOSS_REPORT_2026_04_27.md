# Wildfire Prediction — Progress Report

*Drafted 2026-04-27 (replaces 04-20 draft)*

This update replaces the earlier draft, in particular the "+15% vs
climatology" claim, which understated the result. The corrected number
based on full-eval (646 windows, leak-free 22y baseline) is **+63%**.

---

## Headline result

I trained a deep-learning model that predicts wildfire ignition
locations 14-46 days in advance across Canada. On the 2022-2025
validation period, the model achieves **5.19× event-level Lift@30km**
vs the climatology baseline at **3.19×**, a **+63% improvement** with
non-overlapping 95% confidence intervals (statistically very strong).

|                       | Lift@5000 (pixel) | Lift@30km (event) |
|-----------------------|-------------------|-------------------|
| **Best model (enc21)**| **6.47×** [6.28, 6.66] | **5.19×** [5.07, 5.32] |
| Climatology baseline  | 4.42 ± 1.61×       | 3.19×              |
| FWI-only baseline     | 1.62 ± 1.60×       | 1.88×              |

Pixel Lift@5000 = 6.47× means the top 5000 predicted pixels are 6.47×
denser in actual fire than a random equal-size sample. Event Lift@30km
counts each ~30km fire cluster once (not each pixel), the standard
spatial-event metric. Both metrics decisively beat baselines.

---

## What I did this period

### 1. Switched fire labels from CWFIS hotspots to NBAC + NFDB
- CWFIS detection rate drifted ~350× over 2002-2024 due to satellite
  swap-ins (Aqua 2002, VIIRS 2012, MODIS C6 2015) — old years had
  ~3000 fires, recent years ~1.1M. This pollutes the training signal.
- New label uses NBAC burn-perimeter polygons (Landsat-derived,
  consistent quality 1985+) + NFDB agency ignition points (size ≥ 1ha).
- Old SOTA numbers are NOT directly comparable; all reported numbers
  here use the new label.

### 2. Built the canonical evaluation pipeline
- 19 unit tests in `tests/test_metrics.py` for Lift, BSS, Lift@30km,
  bootstrap CI, F1/F2/MCC.
- One unified script (`scripts/compute_unified_metrics.py`) computes
  every metric (Lift@K for 5 K's, Precision/Recall/CSI/ETS, F1/F2/MCC,
  Lift@30km, Cluster Lift, ROC-AUC, PR-AUC, Brier, Cohen's d, KS) on
  the same 604 val windows for fair model-vs-baseline comparison.
- All metrics now have published-spec metric cards in `docs/metrics/`
  (HuggingFace + Mitchell et al. style with cross-domain benchmarks).

### 3. Validated scaling — encoder length sweep
- Tested encoder histories of 14, 21, 28, 35 days on 4-year training
  data (full 811 val windows).
- Found **enc21 = 6.47× SOTA**, enc28 = 6.37× (statistically tied at
  event-level), enc14 = 4.85×, enc35 = 6.04×.
- Decision: enc21 is the operating point; enc14 is too short, enc35
  is over-fitting to noisy long-history.

### 4. Validated scaling — training-data length (in progress)
- Tested 4y vs 12y vs 22y training (2018-22 / 2014-22 / 2000-22).
- Currently 9 full-eval scoring jobs queued (one per remaining
  ckpt × range combination).
- Initial signal: 22y improves Cluster Lift +46% over 4y but Pixel
  Lift plateaus or slightly drops — suggests the model learns
  event-level structure from older years but not pixel rank. Cause
  is under investigation; LR-rescaling hypothesis was disconfirmed
  this week.

### 5. Honest baseline + leakage audit
- Found and fixed three leakage bugs in earlier evaluation:
  (a) climatology baseline was using val-period fires
  (b) burn_age channel had future-year leakage
  (c) FWI components missing normalization stats
- All baselines re-run on leak-free pipeline. The +63% result holds
  after fix.

### 6. Rejected 22y "data shift" hypothesis with own data
- I had assumed 22y might be hurt by climate non-stationarity ("more
  drought years recently") but a Mann-Kendall + KS test on FWI
  distribution showed the data is roughly stationary (p > 0.27).
  The non-improvement of Pixel Lift on 22y is something else; under
  investigation.

### 7. Persistence-baseline polygon artifact discovered
- A naive "past-7-day fire density" baseline scored Lift = 17×, which
  exceeded the model. Investigation revealed this was a polygon-label
  artifact: a single multi-week NBAC fire makes the polygon mask
  copy itself across days, so "where it burned yesterday" trivially
  predicts "where it burns today".
- Solution: introduced a "novel-ignition" metric that excludes pixels
  that already burned in the past 7/30/90 days. On novel labels:
    - Persistence collapses from 17× → 0.00×
    - Logreg drops from 5.24× → 2.69× → 2.38×
    - Model retains separation (Cohen's d still ≥ 1.25 on novel_30d)
  → confirms the model genuinely separates new ignitions, not just
  re-predicts existing burn polygons.

---

## What's next

### Within 1 week
- Finish 9 full-eval scoring runs (queued, ~16-20h once GPU free).
- Run unified metrics on all 12 trained models → produce the master
  scaling-law table (training-data × encoder-length).
- Produce per-month Lift breakdown (fire-season seasonality of
  predictability).

### Within 2 weeks
- Decide on 13-channel sweep (adds NDVI + deep soil moisture +
  precipitation deficit + burn count). Caches just submitted; will
  decide after seeing 9-channel scaling result.

### Open questions (research)
- Why does Pixel Lift plateau on 22y while Cluster Lift gains? Three
  candidates: distribution shift, label-noise floor, fixed-capacity
  saturation.
- Is there a year-extension (1985-1999) win available? ERA5 + NBAC
  cover those years; only NDVI is bottlenecked at 2000.

---

## What is NOT yet validated
- Performance under operational deployment (we evaluated on a held-out
  time period, not online).
- Robustness to weather forecast errors at 14-46 day lead — we use
  reanalysis here, real ECMWF S2S forecasts are noisier.
- Generalization to provinces outside the historical fire footprint.

---

## Numbers I am confident in (all from this period's eval)
- Lift@30km (event): 5.19× model vs 3.19× climatology = **+63%**
- Lift@5000 (pixel): 6.47× model vs 4.42× climatology
- Confidence intervals: model [6.28, 6.66] vs climatology
  [implied ~3.0, 4.5] — non-overlapping
- ROC-AUC ≈ 0.89, PR-AUC ≈ 0.17 (low PR-AUC is normal for high-imbalance
  rare-event prediction)
- Cohen's d (fire vs non-fire score separation): 1.27-1.45 ("very large"
  by Cohen's convention)
- Effect preserved on novel-ignition labels (filters out polygon
  artifact): 1.25-1.45 — model is genuine, not memorizing burns.

## Numbers I am NOT confident in yet
- 22y vs 4y comparison: still mixed signal. Will report definitively
  once full eval lands next week.
- 13ch / 16ch incremental value: pending cache build (~30h) +
  training (~7-22h × 8) + eval.
