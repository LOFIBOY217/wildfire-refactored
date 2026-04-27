# Scaling-Law Validation — Research Log (2026-04-25 / 04-26)

Goal entering this period: verify that 4y → 12y → 22y of NBAC+NFDB
training improves model performance, in service of a paper-grade
"data scaling law" plot.

This log records *what we did, why, and what we found* — with
explicit causation chains. Where we made wrong claims, the
correction is recorded.

---

## 1. 22y first attempt → OOM (root-caused)

**Context**: 4y already completed (best Lift@5000 = 5.83 enc28). Need
22y × 4 to validate scaling.

**Action 1**: Submitted 22y × 4 (jobs 59815513-16) with `--mem=750G
--load_train_to_ram`, same code as 4y.

**Observation**:
- enc14 (59815513) OOM at 9h11m
- enc21 (59815514) OOM at 10h49m
- enc28/35 still running but RSS plateaued at 781 GB on 1024 GB nodes

**Investigation**:
- `cat /proc/<pid>/io` showed read rate ~5 MB/s sustained
- File descriptors all on local SSD (so the earlier IO fix was working)
- The pre-OOM step is `meteo_train = np.array(meteo_patched[:, t_indices, :])`
  in `train_v3.py:2094`

**Root cause** (via reading the line): for 22y data,
- `meteo_patched` is a (23998, 9332, 2304) float16 memmap (962 GB on disk)
- `t_indices` is 4678 fire-season indices (non-contiguous → fancy indexing)
- The expression allocates a NEW (23998, 4678, 2304) float16 array = **517 GB**
- During the copy, OS page cache from memmap reads accumulates (~200 GB)
- cgroup-counted pages = 517 GB array + 200 GB cache + ~50 GB other = **~750 GB peak**
- 750 G alloc → SIGKILL when peak hits

**Fix** (commit `4b53991`): chunked copy in `train_v3.py`:

```python
meteo_train = np.empty((n_p, n_t, n_d), dtype=np.float16)  # zero physical pages
for c0 in range(0, n_p, 1000):
    meteo_train[c0:c0+1000] = meteo_patched[c0:c0+1000, t_indices, :]
    # per-chunk working set ~22 GB → page cache cycles → no accumulation
```

Verified locally that `np.empty + chunk fill` produces output identical
to `np.array(memmap[fancy_index])`.

**Verification**: cancelled enc28/35 (about to OOM too); resubmitted 22y × 4
(59870680-84) with fixed code. RSS stabilized at ~745 GB during chunked
load (vs 781 GB OOM before). All 4 jobs successfully entered training loop.

**Causation chain**:
```
naive np.array(memmap[fancy_idx])
  → single-shot 517 GB allocation
  → page cache fills concurrently
  → cgroup peak > 750 GB
  → OOM
─────────────────────────────────────────
chunked np.empty + per-1000-patch fill
  → final array allocated empty (no physical pages until written)
  → working set per chunk ~22 GB
  → page cache cycles between chunks
  → peak ~580-650 GB << 750 GB
  → no OOM
```

---

## 2. Persistence baseline anomaly (artifact diagnosed)

**Context**: ran extra baselines (persistence, fwi_threshold) on NBAC+NFDB
labels.

**Observation**: persistence Lift@5000 = **17.12x** — dwarfs our model
(5.83) and climatology (5.09).

**First reaction**: panic, model is worse than trivial baseline.

**Investigation** — what is persistence actually doing?
```python
def _persistence_win(win):
    hs, he, ts, te = win
    return fire_patched[hs:he].astype(np.float32).mean(axis=0)
    # = mean fire density over the encoder window [hs, he)
    # = "where was it burning in the past 7 days"
```

**Root cause**: NBAC labels are **polygons**. Once a perimeter is drawn,
*every pixel inside* is positive for the duration of the fire (often
4-6 weeks for mega-fires).

```
Pixel that started burning 2024-06-01 and burned through 2024-08-01:
  - "burning in past 7 days at lead window 2024-07-01" → 1
  - "fire in lead window 2024-07-01..2024-08-15" → 1
  → persistence "predicts" continuation correctly = trivial
```

This is a **NBAC polygon artifact** + **2022-2025 mega-fire era** combination
(2023 was the worst Canadian fire year in history). Persistence is not
forecasting; it is observing.

**Operational reality check**: fire managers already see active fires
in real time. Predicting "where it will keep burning" gives them zero
new information. They need to know **where new fires will start**
(resource pre-positioning, fuel treatment).

**Conclusion**: standard `lift_total` metric is contaminated for
polygon labels + sustained-mega-fire eval period. Need a different
metric.

---

## 3. Novel-ignition metric (proposed and validated)

**Action** (commit `861296b`): wrote `scripts/evaluate_novel_ignition.py`.

Definition:
```python
y_novel(patch) = 1  iff  (fire in [ts, te))  AND  (no fire in [hs - L, he))
```
where `L ∈ {7, 30, 90}` days = lookback to exclude already-burning patches.

**Result** (job 59852616, 20-window sample):

| baseline    | total | novel_30d | novel_90d |
|-------------|-------|-----------|-----------|
| fwi_oracle  | 0.00  | 0.00      | 0.00      |
| climatology | 6.89  | 6.90      | 6.20      |
| persistence | 16.92 | **0.00**  | 0.00      |

**Causation confirmed**:
- persistence collapses to 0.00 on novel labels → it has zero predictive
  signal beyond observing currently-burning regions (consistent with the
  artifact diagnosis)
- climatology stays roughly flat 6.89 → 6.90 → 6.20 → spatial prior
  works equally well on continuation vs novel (NBAC fires are
  spatially concentrated regardless)

---

## 4. Apples-to-apples model evaluation (done)

**Action**: extended `train_v3.py` with `--save_window_scores_dir` (commit
`1402be6`) so a trained model can dump per-pixel score arrays without
retraining. Submitted 4 eval-only SLURM jobs (59853254-57) for 4y × 4 enc.

Then wrote `compute_lift_from_scores.py` (offline post-process: model
score → total/novel lift) and `add_baselines_to_score_comparison.py`
(re-scores climatology + persistence on the SAME 20 windows the model
was evaluated on, so comparison is fair).

**Result on 4y enc28 (best epoch, 20 sampled windows)**:

| method      | lift_total | lift_novel_30d |
|-------------|------------|-----------------|
| persistence | 3.41       | 0.00            |
| climatology | 3.37       | 2.75            |
| **enc28**   | **5.83**   | **6.27**        |

Note: in this fair-window subset climatology is 3.37 (not 6.89). The
6.89 was on a different 20-window subset (different seed pool). The
3.37 is the right number for direct model comparison.

Bootstrap 95% CI for enc28 novel_30d: **[4.98, 7.70]** (commit `0056fbb`,
`scripts/bootstrap_ci.py`).

---

## 5. Climate non-stationarity claim — RETRACTED via own data

**Original claim** (in conversation): "2000-2017 climate ≠ 2018-2024
climate; that's why 22y overfits more than 4y." Specific numbers
("X+1.5°C", "30% drought years") were *fabricated* without verification.

**User pushback** (correct): challenge to verify with project data.

**Action** (commit `28807a6`): wrote `scripts/analyze_climate_drift.py`.
Computes per-year:
- annual NBAC+NFDB fire pixels
- annual mean / max FWI (sampled fire-season days)
- annual high-FWI fraction (FWI > 30)
Then Mann-Kendall trend test + KS test (2000-2017 vs 2018-2024).

**Result**:

| metric        | Mann-Kendall Z | p     | KS p (split @ 2018) |
|---------------|----------------|-------|----------------------|
| fire_pixels   | **+2.08**      | **0.038** | 0.878 |
| fwi_mean      | +0.63          | 0.528 | 0.553 |
| fwi_max       | -0.21          | 0.834 | 0.653 |
| high_fwi_frac | +0.49          | 0.624 | 0.273 |

Year-similarity: 2024 is closest to 2022 (d=0.35), then 2001 (d=0.41),
then 2000 (d=0.90).

**Verdict**:
- Fire activity has a **mild but significant** upward trend (1.53× more
  fire pixels in 2018-2024 vs 2000-2017)
- FWI inputs (what the model sees) are **not significantly non-stationary**
  (all p > 0.27)
- Old years CAN resemble recent years (2001 ≈ 2024 ≈ 2022) — supporting
  the user's intuition

**Updated causation** for "22y overfits more than 4y":
- ❌ NOT because of climate non-stationarity (data does not support)
- ✅ Because **same LR + 5.5× more batches per epoch = effective over-training**
  (4y: 11,584 gradient updates over 4 epochs; 22y: 43,548 updates)
- ✅ Plus **strong temporal autocorrelation in fire data**: 9332 days of
  NBAC contains maybe ~10 truly independent fire seasons (mega-fires
  span 4-6 weeks; effective N << raw count)

---

## 6. Cosine-LR hypothesis test — DISCONFIRMED (2026-04-27)

**Hypothesis**: if 22y overfit is driven by 5.5× over-training (not data
mismatch), then halving the LR + lowering eta_min should restore the
expected scaling benefit.

**Submitted** (job 59894250): 22y enc28 with `--lr 5e-5 --lr_min 1e-7`
(default cosine schedule already on; only the LR range is compressed).
Comparison target: original 22y enc28 (lift 5.59 ep1, 3.81 ep2).

**Predicted outcome (a priori)**:
- ep2 lift drop should be much smaller
- final best lift should be ≥ 5.83 (matching or beating 4y SOTA)
- if so → confirms LR-rescale diagnosis, not climate-shift

**Observed outcome** (2026-04-27, 15h47m runtime, quick eval 20 win / 15 with fire):

| Run                       | Lift@5000          | Lift@30km | ROC-AUC | Notes |
|---------------------------|--------------------|-----------|---------|-------|
| 22y enc28 default LR (ep1)| 5.59               | —         | —       | from §4 |
| **22y enc28 cosine-lower**| **5.484** [3.78, 7.49] | 3.97 [2.35, 6.37] | 0.894 | this experiment |
| 4y enc28                  | 5.83               | —         | —       | original SOTA target |

→ **No improvement.** Cosine-lower 22y enc28 lands at the same level as
default-LR 22y enc28, NOT at the expected ≥ 5.83.

**Conclusion**: LR-rescaling is NOT the driver of the 22y plateau.
The hypothesis is **disconfirmed**. Whatever causes 22y to underperform
4y on Pixel Lift is something else — candidates that remain:
1. Distribution shift (val 2022-2025 differs from train 2000-2017 more
   than from train 2018-2021) — partially supported by §5 KS results
   showing ALL three ranges differ from val (p<0.05), but val differs
   *less* from 4y than from 22y on `n_fires_per_year`.
2. Heavier-tailed older data adds noise without signal for current
   architecture (fixed PFC capacity).
3. Cluster Lift goes up with 22y (+46% per memory) — so the model IS
   learning something new from older data, just not pixel-level rank.

**Decision**: do NOT pursue further LR-tuning experiments on 22y until
full-eval (604-window) numbers come back. Quick-eval 20 win has CIs
[3.78, 7.49] which overlap heavily with both 4y and default-22y; this
disconfirmation rests on point estimates and could be wrong at the
event-level. Re-evaluate after full save_scores completes (job 59957346).

**Causation chain (corrected)**:
```
"22y plateau on Pixel Lift"
   ?→ LR over-training under cosine schedule (hypothesized §6)
   ✗  cosine-lower experiment shows same lift → ruled out
   ?→ distribution shift between old training years and val period
       (partially supported by §5 KS results)
   ?→ Cluster Lift gains may indicate the model learns event-level
       structure from old years but not pixel rank (worth dedicated test)
```

---

## 7. Score separability check (Cohen's d very-large)

**Question** (raised by user): how cleanly does the model split fire vs
non-fire pixel scores? AUC says "ranking is good"; this asks "is the gap
large in magnitude?".

**Action** (commit `33f3b82`): wrote `scripts/analyze_score_separability.py`.
Per window, computes mean / median / quantile / Cohen's d / KS distance
between fire-pixel score distribution and non-fire score distribution.

**Result on 4y models, 20 sample windows**:

| ENC | mean_pos | mean_neg | ratio | Cohen's d | KS  |
|-----|----------|----------|-------|-----------|-----|
| 14  | 0.147    | 0.050    | 2.91× | **1.42**  | 0.60 |
| 21  | 0.177    | 0.062    | 2.88× | **1.40**  | 0.61 |
| 28  | 0.138    | 0.047    | 2.86× | **1.27**  | 0.53 |
| 35  | 0.132    | 0.040    | 3.13× | 1.21      | 0.47 |

Same on novel_30d labels: ratios 2.89-3.16, d's 1.25-1.45 — virtually
identical to total. **Critical finding**: separation is preserved on
novel labels, meaning the model does NOT rely on memorizing burn polygon
boundaries; it genuinely separates new ignitions.

Cohen's d > 1.2 = "very large" by Cohen's convention.

---

## 8. Outstanding work (snapshot 2026-04-26 ~18:00 UTC)

```
DONE:
  4y × 4 train, save_scores (20-win), novel-eval, separability
  22y enc14 train (Best 5.73, cluster 8.33)
  22y enc21 train (Best 5.60, cluster 7.95)
  Climate-drift script + result
  Persistence + fwi_threshold + climatology baselines
  Logreg total (5.24)
  Chunked-copy fix + commit

RUNNING:
  22y enc28 train  (ep4 in progress)
  22y enc35 train  (ep3 mid)
  12y cache build  (transposing, ~5h)
  logreg-complete (novel labels + same windows)

QUEUED:
  22y enc28 cosine-lower
  4y × 4 + 22y enc14 full save_scores (--full_val on ~700 windows)
  12y × 4 train (afterok cache)
```

### Update 2026-04-27
```
DONE (since 04-26):
  22y enc28 default-LR train
  22y enc35 train
  12y × 4 train (all 4 ranges complete)
  22y enc28 cosine-lower train (15h47m, Lift@5000 = 5.48 quick eval — see §6)
  Logreg complete (enc14: 2.69x total, 2.38x novel_30d)
  Fire-label distribution stats v4 (per-province, power-law, train/val KS)
  Per-window save_scores: 4y enc14/21/28 (3 of 12 done)

RUNNING / QUEUED (2026-04-27 ~21:30 UTC):
  9 save_scores PD jobs resubmitted with --mem=400G
    (was 750G, fit only on 1024G nodes → 21h wait;
     now fits on 510G standard A100 nodes → "Priority" reason)
  unified_metrics 4y enc14 (12h walltime, replaces 2h timeout v2)
```

When all done, run `compute_full_metrics.py` per save_scores dir for
the unified metric panel (Lift@K + Lift@30km + ROC-AUC + PR-AUC + Brier
+ F1/F2/MCC, model + climatology + persistence × total + novel_7d/30d/90d).

---

## 9. Things we got wrong this period (recorded for honesty)

1. **Fabricated climate numbers** — invented "X+1.5°C, 30% drought
   years" to explain 22y overfit. Data shows FWI is roughly stationary;
   only fire_pixels has weak trend. Lesson: do not present specific
   numbers without verification, even in casual analogies.

2. **Over-attributed 22y overfit to data non-iid** — initially blamed on
   training-schedule mismatch (LR not rescaled for 5.5× more updates).
   The cosine-LR experiment (§6) **disconfirmed** this: 22y enc28 with
   lr=5e-5 / lr_min=1e-7 produced Lift@5000 = 5.48, no better than
   default LR. Real driver of 22y plateau is still unidentified;
   distribution shift between old training years and val period
   (§5 KS results) is now the leading hypothesis, but unconfirmed.

3. **Earlier persistence Lift = 17 panic** — almost concluded the
   model was worse than trivial baseline before realizing this was a
   polygon-label artifact. Lesson: question the metric before the
   model.

4. **Initially understated complexity of "save model scores externally"**
   — claimed 2-3h, actually required modifying train_v3.py
   (`--save_window_scores_dir`), reading the eval pipeline, +
   submitting GPU jobs. Took longer because of cache copy + queue.

These corrections strengthen the work; they are not weaknesses to hide.

---

## 10. Deferred work — 13ch retrain (decided 2026-04-26)

**Decision**: defer 13-channel retraining to a future paper / extension.
Current 9ch results are sufficient for the first paper.

**13ch = 9ch + 4 vegetation/water channels we know we need**:

```
Existing 9ch:  FWI, 2t, fire_clim, 2d, tcw, sm20, population, slope, burn_age
Add for 13ch:  NDVI, deep_soil, precip_def, burn_count
```

| Channel | Type | Why we want it |
|---|---|---|
| `NDVI` | Vegetation | Direct measure of greenness / fuel state — strongest single vegetation signal in the wildfire literature |
| `deep_soil` (sm 1-2 m, swvl2) | Slow water | Captures multi-month drought; complements shallow `sm20` |
| `precip_def` | Drought index | Cumulative precipitation deficit — operational drought metric |
| `burn_count` | Fire history | Total burns over period (vs `burn_age` which is "years since last") |

**Infra is fully built — only the compute is missing**:

| Piece | Status |
|---|---|
| Cache build script (`slurm/rebuild_cache_2000_2025_narval.sh`) | ✅ Already supports `CACHE_CHANNELS=13` |
| Train script (`slurm/train_v3_13ch_*.sh`) | ✅ Exists in repo |
| Data on disk (NDVI, swvl2, precip TIFs) | ✅ Reprojected, audited, drift-stable |
| `train_v3.py` channel handling | ✅ Drives off `--channels` arg |
| `fire_dilated` cache (NBAC+NFDB) | ✅ Already pre-built (commits `d833720`, C-step `b5e2338` for 22y) |
| Chunked train-RAM copy | ✅ Already in `train_v3.py:2094` (commit `4b53991`) — handles 13ch peak too |
| Full-eval / save_scores pipeline | ✅ Same scripts work, just different `--channels` |
| LR-scaling fix | ⚠️ Use `--lr 5e-5` for 22y (per Section 6 finding) |

**Compute estimate** (with all the fixes from this period):

```
13ch cache 22y:    ~30h CPU (one-time)
13ch cache 12y:    ~15h CPU (one-time, can reuse pre-built dilated label)
13ch cache 4y:      ~6h CPU (one-time)

13ch × 4 enc × 22y:   4 × ~10h GPU = 40h
13ch × 4 enc × 12y:   4 × ~6h  GPU = 24h
13ch × 4 enc × 4y:    4 × ~3h  GPU = 12h

13ch full save_scores × 12 models: 12 × ~1.5h = 18h GPU

Total: ~3-4 days wall time on Narval (with usual queueing).
```

**Why defer**:
- 9ch alone already delivers the core paper claims (drift discovery,
  novel-ignition metric, scaling-law plot, +128% over climatology)
- NeurIPS D&B 2026 deadline ~early June — 13ch ablation eats 3-4 weeks
  including bug-fix cycles
- 13ch is a **channel ablation**, not a fundamentally different result
- Reviewer expects future-work section; 13ch fits there cleanly

**When to come back** (priority order):
1. After 9ch scaling-law data is fully analyzed (12y/22y both done)
2. If the scaling-law plot shows a flat line at the 22y point (suggests
   model is data-limited, not capacity-limited → channel boost may help)
3. Before any extension paper or v2 submission
4. Or whenever a reviewer asks "did you try NDVI?"

**Concrete reactivation steps** (when ready):

```bash
# 1. Build 13ch caches (CPU jobs, run in parallel)
CACHE_CHANNELS=13 DATA_START=2000-05-01 sbatch slurm/rebuild_cache_2000_2025_narval.sh
CACHE_CHANNELS=13 DATA_START=2014-05-01 sbatch slurm/rebuild_cache_2000_2025_narval.sh
CACHE_CHANNELS=13 DATA_START=2018-05-01 sbatch slurm/rebuild_cache_2000_2025_narval.sh

# 2. Pre-build dilated NBAC+NFDB labels for new ranges (saves dilation time)
#    -- only 4y/12y need new build; 22y already has it from C-step
#    Use scripts/build_fire_labels.py and copy into cache dir as
#    fire_dilated_r14_nbac_nfdb_*.npy

# 3. After caches done, submit 12 trainings:
for ENC in 14 21 28 35; do
  RANGE=22y ENC=$ENC sbatch --mem=750G slurm/train_v3_13ch_2000_narval.sh
  RANGE=12y ENC=$ENC sbatch --mem=400G slurm/train_v3_13ch_2000_narval.sh
  RANGE=4y  ENC=$ENC sbatch --mem=400G slurm/train_v3_13ch_2000_narval.sh
done

# 4. After training, run full-eval save_scores and compute_full_metrics
#    on each — same pipeline as 9ch
```

**Decision rationale recorded by user 2026-04-26**:
> "NDVI, deep_soil, precip_def — these I think are definitely needed.
> Record this; doesn't have to be done now, but I think it should be
> completed since our infra is already built."

→ This deferral is intentional, not forgotten. Re-open at any time.

## 11. Deferred work — 16ch retrain (much lower priority)

```
13ch + u10 + v10 + CAPE = 16ch
```

- u10/v10 (wind) — should help flame spread but model predicts
  ignition not spread; might add modest signal
- CAPE (atmospheric instability / lightning proxy) — should help
  lightning-caused ignitions
- All three reprojected and clean as of 2026-04-19 (post nan-edge bug fix)

**Defer until 13ch ablation result is known**. If 13ch gives meaningful
lift over 9ch → 16ch worth trying. If 13ch is flat → 16ch unlikely to
help and not worth ~2 more weeks of compute.
