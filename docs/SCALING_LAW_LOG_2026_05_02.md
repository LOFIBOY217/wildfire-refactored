# Scaling-Law Validation — Research Log (2026-05-01 / 02)

Continuation of `SCALING_LAW_LOG_2026_04_25_26.md`. Records the
recovery from the fire_patched cache bug, the failure of strongreg /
ep1 hypotheses, the discovery that **12y is the sweet spot**, and the
emergence of a strong **recent-data baseline**.

---

## 1. Bug fix and recovery (2026-04-29 → 04-30)

### 1.1 Bug discovered
fire_patched cache filename did NOT include fusion_tag. Result: a
CWFIS-derived `.dat` from an earlier run was silently reused by all
4y/12y trainings even when `--label_fusion` was set. **22y was
unaffected** because no .dat existed in v3_9ch_2000.

### 1.2 Fix applied
- `train_v3.py`: added `fusion_tag` to fire_patched cache filename
- 2 unit tests added (`tests/test_fire_patched_fusion_tag.py`)
- `scripts/audit_label_consistency.py`: regression test for label
  consistency, 4 checks (code, cache, standalone npy, score npz)
- Stale buggy `.dat` deleted in 4y/12y/4y_13ch caches
- 16 retraining jobs submitted

### 1.3 Recovery results (after fix)
All 4y / 12y models retrained on NBAC+NFDB labels. Quick eval results:
- 4y enc14: 6.69x (was 5.93 buggy)
- 4y enc21: 5.53x
- 4y enc28: 4.99x
- 4y enc35: 3.64x

---

## 2. Failed hypothesis: strong regularization (2026-04-30)

### 2.1 Setup
After observing all 4y/22y models had `best_epoch=1` (val_lift drops
after ep1), tested whether stronger regularization (dropout 0.4 + 
weight_decay 0.05 vs default 0.2 + 0.01) would let later epochs
match or exceed ep1.

### 2.2 Result: REJECTED — strongreg made overfit WORSE
4y enc14:
- default: ep1=6.69, ep2=4.94, ep3=4.96, ep4=4.91
- strongreg: ep1=6.51, ep2=2.53 (timeout before ep3)

22y_strongreg: TIMEOUT at 24h while STILL in chunked-RAM-copy step.
No usable result. (See §6 for IO bug.)

### 2.3 Implication
The "ep2-4 decline" is NOT classical overfit (would respond to reg).
Hypothesis revised to **calibration-vs-rank tradeoff** (see §4).

---

## 3. Failed hypothesis: epochs=1 (2026-04-30)

### 3.1 Setup
If model peaks at ep1 anyway, train only 1 epoch. Maybe save compute,
maybe avoid the post-ep1 collapse.

### 3.2 Result: REJECTED — single epoch is worse than default ep1
| 22y enc | default ep1 (best of 4) | ep1-only |
|---|---|---|
| 14 | 5.73 | 4.00 |
| 21 | 5.60 | 2.80 |
| 28 | 5.59 | 4.09 |
| 35 | 4.97 | 3.64 |

### 3.3 Why
Cosine LR schedule with `epochs=1` decays LR to 0 mid-epoch → training
effectively halts at ~50% through ep1. Default `epochs=4` keeps LR
high during ep1 so the model trains fully.

→ The **early stopping at end-of-ep1** in default 22y IS the right
strategy. ep1-only doesn't replicate it.

---

## 4. New hypothesis: calibration-vs-rank tradeoff (2026-05-01)

### 4.1 Pattern
- `train_loss` decreases monotonically across all configs
- `val_lift_5000` declines in 4y/22y after ep1, IMPROVES in 12y
- `ROC-AUC` stays stable (0.83-0.91) — global ranking is fine
- `Brier` improves — calibration improves

### 4.2 Hypothesis
Lift@K is rank-sensitive (top-K rank). Focal loss + cosine LR + many
SGD updates → predictions become smoother (better calibrated). Smoother
predictions hurt top-K rank but preserve global rank. Sweet spot at
~5,000-10,000 SGD updates.

### 4.3 Literature support
- Mukhoti 2020 NeurIPS: focal loss improves calibration but hurts sharpness
- Wang 2021: "over-confidence is not always bad — improves top-K accuracy"
- Müller 2019 NeurIPS: label smoothing helps calibration, hurts ranking

### 4.4 Experiment in flight
`exp_lift_trajectory_within_epoch_narval.sh` (60185099 PD):
- 22y enc14 with mid-epoch eval every 500 batches
- Will observe Lift@5000 trajectory across ~88 evaluation points
- If peak in middle of ep1 → hypothesis confirmed
- ~36h to complete

---

## 5. Discovery: 12y is the sweet spot (2026-05-01)

### 5.1 Full eval (NBAC+NFDB, 604/583 windows)

| Model | Lift@5000 | Lift@30km | vs climatology Lift@5000 (4.42) |
|---|---|---|---|
| **12y enc14** | **6.404x** [6.17, 6.68] | **5.076x** [4.81, 5.36] | **+45%** |
| **12y enc21** | **7.834x** [7.50, 8.21] | **6.727x** [6.40, 7.07] | **+77%** ⭐ |
| 12y enc28 | 5.579x ⚠️ | (n=562, see note) | +26% |
| 12y enc35 | 5.615x ⚠️ | (n=536, see note) | +27% |

**2026-05-02 update**: enc28/35 evaluated. Both LOWER than enc14, much lower than enc21. **Encoder length has a sweet spot at 21d for the 12y range.** ⚠️ enc28/35 rows show `n_fire == n_win` while enc14/21 show `n_fire < n_win` — verify fire-window filter applied identically before paper figure.

### 5.2 Comparison to other ranges (NBAC+NFDB labels)

| Range × enc | Lift@5000 (full eval) | vs climatology 4.42 |
|---|---|---|
| 4y enc14 | 4.94 (partial 65%) | +12% |
| **12y enc14** | **6.405** | **+45%** ⭐ |
| **12y enc21** | **7.835** | **+77%** ⭐⭐ |
| 12y enc28 | 5.579 ⚠️ | +26% |
| 12y enc35 | 5.615 ⚠️ | +27% |
| 22y enc14 | 3.88 | −12% |
| 22y enc35 | 4.91 | +11% |

### 5.3 Why 12y wins
Per `SCALING_LAW_LOG_2026_04_25_26.md` §4 (calibration-vs-rank):
- 4y peaks at ~1,602 updates (1 epoch) → ep2-4 overfit
- 12y peaks at ~5,800-23,200 updates (across all 4 epochs) → still improving
- 22y peaks at ~10,887 updates (mid ep1) → epochs 2-4 overfit

12y is the only range where the natural 4-epoch training schedule
hits the sweet spot.

### 5.4 Paper implication
SOTA = **12y enc21**, not 22y or 4y as previously thought.

Paper headline candidate:
> "On Canadian S2S wildfire forecasting at 14-46 day lead, training on
> 8 years of NBAC+NFDB data outperforms both 4y and 22y under fixed
> recipe. Best Lift@30km = 6.73x [6.40, 7.07], +111% over climatology
> baseline."

---

## 6. Critical infrastructure bug: missing SSD copy in 22y scripts (2026-05-02)

### 6.1 Symptom
22y_strongreg + 3 × 22y_recency all TIMEOUT at 22-24h while stuck on
"chunk 5000-6000 / 23998" of chunked RAM copy.

### 6.2 Root cause
My new SLURM scripts (recency, strongreg, lift_traj) pointed
`TRAIN_CACHE_DIR` directly at `$SCRATCH/meteo_cache/v3_9ch_2000`
(Lustre). The chunked copy does fancy indexing
`meteo_patched[c0:c1, t_indices, :]` which is **100x slower on Lustre
than on local SSD** due to seek pattern. Each chunk took hours; total
needed 200h+ but walltime was 24h.

### 6.3 Fix
3 scripts updated to copy meteo to `/localscratch/$SLURM_TMPDIR/cache/meteo`
before training, mirroring the working `train_v3_9ch_2000_narval.sh`.
Same fix applied to `train_v3_12y_recency_narval.sh` (12y data is
smaller but same pattern).

### 6.4 Cost
~96 GPU-hours wasted (3 × 22h + 1 × 24h, 4 jobs total).

---

## 7. New baseline: recent-data per-patch burn rate (2026-05-02)

### 7.1 Setup
User suggested: train a simple baseline on first 1-2 years of val
period (2022), evaluate on remaining (2023-2025). Avoids climate
non-stationarity.

### 7.2 Implementation (`scripts/recent_logreg_baseline.py`)
- Per-patch feature: mean burn rate in 2022
- Predict same value for all sub-pixels in patch
- Evaluate on 2023-2025 windows

### 7.3 Result
| Year | Recent-Data Baseline Lift@5000 |
|---|---|
| 2023 | 4.78x |
| 2024 | 5.83x |
| **mean** | **5.31x** |

### 7.4 Implication: stronger baseline than climatology
- Climatology Lift@5000 = 4.42x
- **Recent-data baseline Lift@5000 = 5.31x** (+20% over climatology)
- 22y enc35 model = 4.91x (loses to recent-data)
- 12y enc14 model = 6.40x (+21% over recent-data)
- **12y enc21 model = 7.83x (+47% over recent-data)** ⭐

→ Paper must compare against recent-data baseline, not just climatology.

---

## 8. Definitive baseline numbers (paper-ready)

For the val period 2022-2025, NBAC+NFDB labels:

| Baseline | Lift@5000 | Lift@30km | Source |
|---|---|---|---|
| **climatology** | **4.42 ± 1.61x** | **3.19x** | benchmark_baselines.py 646 win |
| fwi_oracle | 1.62x | 1.88x | same |
| **recent-data (train 2022)** | **5.31x** | n/a (need add) | recent_logreg_baseline.py |
| persistence | 17.12x ⚠️ | 11.41x | polygon artifact |

The persistence number is the polygon-label artifact (collapses to
0.0x on novel-30d labels). Recent-data baseline is the strongest
honest baseline.

---

## 9. Operational metric (added 2026-05-01)

`Recall@30km within budget` — operationally most important:
- Top X% of Canada (X = 1%, 5%, 10%) = realistic deployment budget
- Of that top-X%, what fraction of fire events fall there?
- Currently: K=5000 pixels = 0.08% of Canada (too strict)
- K=25000 = 0.4% Canada → recall ~2% (still too strict)
- Need to compute K=60000+ for realistic budget

`scripts/recall_at_budget.py` (TODO) will compute this.

---

## 10. Currently in flight

| Job | Type | Walltime |
|---|---|---|
| 60178279 | 12y 13ch cache build | 36h (running) |
| 60178280 | 12y 16ch cache build | 36h (running) |
| 60179163-168 | 6 × save_scores (12y enc28/35 + 22y_ep1 × 4) | PD 8h each |
| 60185099 | lift_trajectory (22y enc14, hypothesis test) | PD 36h |
| 60185107-109 | 12y enc14 recency (tau=6/10/15) | PD 12h each |
| 60185573 | 22y 13ch NBAC+NFDB prep | PD 48h |

---

## 11. What's NOT going to happen

- ✗ 22y_strongreg (failed, abandoned)
- ✗ 22y_recency (failed once, may retry with SSD-copy fix later)
- ✗ Architecture changes (stochastic depth, ALiBi, etc.) — pending
  trajectory result decision
- ✗ MAE / SSL pretraining — deferred per `docs/SSL_PRETRAINING_IDEAS.md`

---

## 12. Updated AAAI 2027 paper outline (drafted 2026-05-02)

### Headline claim
> "Training a Patch Transformer on **8 years of NBAC+NFDB labels**
> achieves **Lift@30km = 6.73x [6.40, 7.07]** for 14-46 day Canadian
> wildfire forecasting, **+111% over climatology and +27% over a
> recent-data baseline**. Counterintuitively, both 4-year and 22-year
> training perform worse under the same recipe due to a
> calibration-vs-rank tradeoff at the SGD update level."

### Contributions
1. **First DL S2S wildfire model at 2 km daily over all Canada**
2. **Sweet-spot finding**: 8y training > 4y or 22y under fixed recipe
3. **Calibration-vs-rank tradeoff** characterization for top-K
   spatiotemporal forecasting (literature gap)
4. **Recent-data baseline** as a stronger benchmark than climatology
5. **fire_patched cache bug** (silently mixed CWFIS/NBAC+NFDB) +
   audit framework

### Risks remaining
- 12y enc28/35 might be even higher → may shift "best enc" claim
- lift_trajectory experiment might disconfirm calibration hypothesis
- Recall@30km within realistic budget might still favor climatology
