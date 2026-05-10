# SOTA Attack Log — Phase 5 (2026-04-25 → 2026-05-10)

Single-cource log of every experiment we ran to push beyond
single-ckpt SOTA = 8.07× Lift@5000 (v3_9ch_enc21_12y_2014, NBAC+NFDB labels).

For older work (encoder-length sweep, 22y revival start, anomaly loss),
see `docs/SCALING_LAW_LOG_2026_05_02.md`.

---

## Outcome at a glance

| Direction | Best result | Δ vs SOTA | Verdict |
|---|---|---|---|
| Single-ckpt SOTA (baseline) | 8.07× | — | **the bar** |
| 10-ckpt prob-mean ensemble | **9.57×** [9.10, 10.02] | **+18.6%** | ✅ new ensemble SOTA |
| 10-ckpt logit-mean ensemble | running (job 60734864) | TBD | hypothesis: recovers Lift@30km |
| 22y revival (data ↑) | 4.91× | −39% | ❌ data quantity hurts |
| 22y + climsim (ENSO weighting) | 4.31× | −47% | ❌ |
| 22y + dm sweep 384/512 | 4.73 / 4.63× | −41% / −43% | ❌ |
| 12y dm sweep 128/384 | 7.32 / 7.95× | −9% / −1.5% | ❌ dm256 saturated |
| Anomaly-aware loss (1/clim^pow) | 5.12–5.87× | −27% to −37% | ❌ broke boreal |
| sub4x4 decoder (128-dim ctx) | crashed (collate bug) | — | ⏳ untested, only axis left |
| 11ch +terrain (DEM, aspect) | running (60669183) | TBD | static channel test |
| 12ch +terrain +lightning_clim | running (60669182) | TBD | static channel test |

---

## 1. Ensemble (the only thing that worked)

### 1a. 10-ckpt prob-mean ensemble — DONE
Average `prob_agg` arithmetically across 10 ckpts' saved npz, then top-K.

**Ckpts used** (all on 12y range, NBAC+NFDB labels):
```
v3_9ch_enc21_12y_2014                   (default 8.07× SOTA)
v3_9ch_enc21_12y_2014_climsim           (ENSO ONI weight)
v3_9ch_enc21_12y_2014_climblend_a0.3    (post-hoc α-blend with climatology)
v3_9ch_enc21_12y_2014_climblend_a0.5
v3_9ch_enc28_12y_2014                   (longer encoder)
v3_9ch_enc35_12y_2014
v3_13ch_enc14_12y_2014                  (more channels)
v3_13ch_enc21_12y_2014
v3_13ch_enc28_12y_2014
v3_13ch_enc35_12y_2014
```

**Coverage**: 583 windows per ckpt, 536 common across all 10, 402 valid (non-empty label).

**Result**:
- Lift@5000 = **9.57×** [9.10, 10.02], +18.6% over single SOTA 8.07×
- Lift@30km = **4.37×** [4.34, 4.40], DROPPED below single SOTA 7.26×

**Smoothing problem**: prob-mean averages out per-ckpt high-confidence peaks → sharpness loss matters most when 30km max-pool eats only the tallest pixel. Hypothesis for fix: logit-mean (geometric mean of odds) preserves peaks.

### 1b. 10-ckpt logit-mean ensemble — RUNNING
Job 60734864 (R, 2h45m left at session end). Uses `mean(logit(p))` instead.
Output: `outputs/ensemble_logit_10ckpt.json`. Should resolve the Lift@30km drop.

---

## 2. 22y revival — FOUR attempts, ALL failed

The 22y range (data_start=2000) gives 5× more training data than 12y (2014).
Hypothesis: more data should yield more skill. Reality: every variant lost.

| Run | Lift@5000 |
|---|---|
| 22y default | 4.91× |
| 22y + climsim ONI weighting | 4.31× |
| 22y + dm384 | 4.73× |
| 22y + dm512 | 4.63× |

**Diagnosis**: not a data-quantity issue — it's optimization. Cosine-LR + focal
loss on 5× more samples means each epoch reaches a different point on the
loss landscape, calibration drifts year-to-year. We could try:
- ReduceLROnPlateau instead of cosine
- BCE instead of focal (or focal_alpha=0.5)
- Per-year sample reweighting (climsim was an attempt, made it worse)

But **lower priority than ensemble + sub4x4 decoder**. Verdict: 22y is a
distraction at this scale; **decision: stop** until calibration is solved.

---

## 3. d_model sweep — saturation confirmed

12y range, fixed enc21, vary d_model:

| d_model | enc/dec layers | Best Lift@5000 | comment |
|---|---|---|---|
| 128 | 4/4 | 7.32× (ep3) | underfit |
| **256** | 4/4 | **8.07×** | sweet spot |
| 384 | 6/6 | 7.95× (ep1) | very mild improvement, then overfits ep2-4 |

22y range × dm sweep, see section 2. Both 384/512 worse than 22y dm256.

**Verdict**: 8.5M-param model is already at capacity for this label/data scale.
Going bigger requires more (good) data — and we don't have any.

---

## 4. Anomaly-aware loss — failed (kept for record)

Hypothesis: model overweights high-fire-density boreal regions; reweighting
each sample's loss by `1/clim^pow` would force it to learn rare-fire areas.

| pow | climsim? | Lift@5000 |
|---|---|---|
| 0.5 | no | 5.87× |
| 1.0 | no | 5.32× |
| 1.0 | yes | 5.12× |

**Diagnosis**: the boreal IS where the fires are. Downweighting it = telling
the model to ignore the signal. The "high-fire-density" pixels deserve their
weight; sparse-fire areas of southern Canada have very few samples to learn
from at any weight.

**Verdict**: idea is wrong, not the implementation. Don't revisit.

---

## 5. Static channel additions — RUNNING

Hypothesis: physiographic priors (terrain) + lightning ignition climatology
add information our model doesn't have.

| Run | Channels added | Status |
|---|---|---|
| 11ch | + elevation (DEM) + aspect | running (60669183, 18h left) |
| 12ch-static | + lightning climatology + terrain | running (60669182, 18h left) |

These are FREE (rasters already on disk). If they don't help, we know the
model has saturated its information channels at 9ch — and the only path is
better DECODER context (sub4x4) or ensemble.

---

## 6. sub4x4 decoder — UNTESTED (crashed)

Only architectural axis we haven't actually trained. s2s_legacy decoder
context is 9-12 dim; sub4x4 cache is 128-dim — **14× more information density**.

Job 60617228 (12y_sub4x4_climsim) crashed during dataloader collate:
```
RuntimeError: stack expects each tensor to be equal size,
  but got [32, 128] at entry 0 and [32, 136] at entry 6
```

**Cause**: variable feature dim per sample. Likely climsim sample-weight
column being concatenated unevenly. **Not yet fixed** — needs dataset code
inspection.

This is the **highest-priority untested axis** for SOTA improvement.

---

## 7. Scaling sweep — IN QUEUE

Goal: plot "training years vs Lift" curve to determine optimal data range.
Uses `slurm/train_v3_9ch_range_master_narval.sh` to time-slice the 22y
master cache (avoids 36h of cache rebuild per range).

| RANGE_TAG | DATA_START | Train years | Job ID | Status |
|---|---|---|---|---|
| 6y_2016  | 2016-05-01 | 6  | 60734978 | PD |
| 8y_2014  | 2014-05-01 | 8  | 60735097 | PD (sanity check vs SOTA) |
| 10y_2012 | 2012-05-01 | 10 | 60735096 | PD |
| 14y_2008 | 2008-05-01 | 14 | 60734977 | PD |
| 16y_2006 | 2006-05-01 | 16 | 60735095 | PD |
| 18y_2004 | 2004-05-01 | 18 | 60734979 | PD |
| 22y_2000 (already done) | 2000-05-01 | 22 | — | 4.91× / 4.31× climsim |
| 8y_2014 non-master (SOTA) | 2014-05-01 | 8 | — | **8.07×** |

Expected shapes:
- **Monotonic dec with years** → calibration-drift diagnosis confirmed (older data hurts)
- **U-shape with minimum near 22y** → noise overwhelms signal at extreme ranges
- **U-shape with maximum near 10-14y** → there's an optimum we missed

---

## Currently pending paper infrastructure

- 11 metric_card jobs (60734980–990) waiting to compute the full per-paper
  metric panel for every model + 3 baselines. Output: `outputs/metric_card_*.json`
- Need to update RESULTS_TABLE_2026_05_03.md once results land

---

## What NOT to retry (decisions)

- ❌ **Decoder-only architecture** — wrong inductive bias for regression. SOTA
  weather models all use encoder-decoder.
- ❌ **22y in any form** — see section 2. Calibration not data quantity.
- ❌ **Anomaly-aware loss** — see section 4. Reweighting kills the signal.
- ❌ **More 9ch encoder layers / heads beyond default** — implicit from dm sweep.

## What to retry next

1. Fix sub4x4 collate bug (section 6) — only untested major axis
2. Add the new ckpts (sub4x4 if works, 11ch, 12ch-static) into ensemble
3. After paper sign-off, try BCE-loss + plateau-LR on 22y to test calibration
   hypothesis (low priority, high effort)
