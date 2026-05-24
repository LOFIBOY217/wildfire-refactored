# Jobs Registry — running log of every SLURM submission

**Updated**: 2026-05-17

This is the single source of truth for "what did we submit, did it succeed,
what's the result". Append a new row per submission. After the job ends,
fill in `state` + `result` (or `failure_reason`).

Columns:
- **jobid** — SLURM jobid (sacct stays for 30d)
- **submitted** — ISO date submitted
- **script** — slurm/*.sh + key env vars
- **run_name** — output dir / ckpt name
- **state** — PD / R / COMPLETED / FAILED / TIMEOUT / CANCELLED
- **result** — key metric(s) on success, "N/A" for infra jobs
- **failure_reason** — only when state ∈ {FAILED, TIMEOUT, CANCELLED}
- **resub** — jobid of resubmission (if any)

---

## 2026-05-17 batch — gating sweep + static channels (POST-AUDIT)

| jobid | submitted | script | run_name | state | result | failure_reason | resub |
|---|---|---|---|---|---|---|---|
| 60874248 | 2026-05-12 | train_v3_gating_sweep GATING=global | v3_9ch_enc21_12y_2014_gate_global | COMPLETED | 20-win: L5k=**7.02×** L30k=6.15× MCC=0.220 | — | — |
| 60874250 | 2026-05-12 | train_v3_gating_sweep GATING=per_lead | v3_9ch_enc21_12y_2014_gate_per_lead | COMPLETED | 20-win: L5k=**6.65×** L30k=6.21× MCC=0.224 | — | — |
| 60874253 | 2026-05-12 | train_v3_gating_sweep GATING=per_pixel | v3_9ch_enc21_12y_2014_gate_per_pixel | COMPLETED | 20-win: L5k=**6.92×** L30k=5.50× MCC=**0.321** F2=**0.448** | — | — |
| 60815122 | 2026-05-12 | train_v3_11ch_terrain | v3_11ch_enc21_12y_2014 | COMPLETED | 20-win: L5k=**7.76×** L30k=6.40× MCC=0.304 | — | — |
| 60815123 | 2026-05-12 | train_v3_12ch_static | v3_12ch_static_enc21_12y_2014 | COMPLETED | 20-win: L5k=**5.89×** L30k=5.08× MCC=0.184 (worse) | lightning channel hurts | — |

**Takeaway**:
- Baseline 12y enc21 9ch SOTA = **8.07× full / ~7× on 20-win** (sampling noise)
- Gating: all 3 variants 20-win L5k ≈ baseline (sampling noise), but **per_pixel MCC=0.32 / F2=0.45** clearly beats baseline → calibration win, not ranking win
- 11ch (+population +slope): 20-win L5k 7.76 ≈ baseline (still need full eval to claim)
- 12ch_static: lightning channel (+burn_count?) hurts — drop it
- ★ All 5 results are **20-window samples**, need `save_window_scores` for full 583-win paper number → resubmit (see "Pending follow-ups" below).

## 2026-05-13–16 — scaling sweep (all DEAD)

| jobid | submitted | script | run_name | state | result | failure_reason | resub |
|---|---|---|---|---|---|---|---|
| 60815109 | 2026-05-12 | train_v3_9ch_range_master 6y_2016 | v3_9ch_enc21_6y_2016 | TIMEOUT | — | master_cache_dir not applied; rebuilt from scratch (3488d) | — |
| 60815110 | 2026-05-13 | train_v3_9ch_range_master 8y_2014 | v3_9ch_enc21_8y_2014 | TIMEOUT | — | 12h ran out at day 4219/4219 of meteo_tf build | — |
| 60815111 | 2026-05-13 | train_v3_9ch_range_master 10y_2012 | v3_9ch_enc21_10y_2012 | FAILED | — | OOM 510 GB (T=4949 × 23998 patches × 2304 dim × fp16) | — |
| 60815112 | 2026-05-13 | train_v3_9ch_range_master 14y_2008 | v3_9ch_enc21_14y_2008 | TIMEOUT | — | 12h ran out mid meteo_tf | — |
| 60815113 | 2026-05-13 | train_v3_9ch_range_master 16y_2006 | v3_9ch_enc21_16y_2006 | FAILED | — | OOM 735 GB | — |
| 60815114 | 2026-05-13 | train_v3_9ch_range_master 18y_2004 | v3_9ch_enc21_18y_2004 | FAILED | — | OOM 811 GB | — |

**Root cause**:
- `meteo_tf = np.zeros((T, n_patches, enc_dim), fp16)` allocates all-RAM tensor.
- T (days) × 23998 × 2304 × 2 bytes ≈ T × 105 MB.
- 8y (4219d) → 444 GB; 18y (7871d) → 811 GB. Narval node = 400 GB.
- The `--master_cache_dir` flag was supposed to side-step this by memmap-slicing the master cache; it didn't engage (full rebuild path was taken).

**Fix needed before resubmit**: route `--master_cache_dir` to a memmap of `meteo_tf` rather than RAM realloc. This is a code change in `train_v3.py` ~line 1918. Scope is medium — deferred until paper window stabilizes.

## 2026-05-11–13 — Lift@30km audit + ensemble re-run

| jobid | submitted | script | run_name | state | result | failure_reason | resub |
|---|---|---|---|---|---|---|---|
| 60816504–14 | 2026-05-11 | full_metric_card ×11 | metric_card_*.json | COMPLETED | (K-scaled fix verified) | — | — |
| 60816515 | 2026-05-11 | ensemble_eval prob_mean | ensemble_12y_all_top.json | COMPLETED | L5k=9.57× L30k=4.37×  ★ K-unscaled (pre-fix) | — | 60874231 |
| 60816516 | 2026-05-11 | ensemble_logit | ensemble_logit_10ckpt.json | COMPLETED | (overwritten by 60874231) | — | 60874231 |
| 60874231 | 2026-05-13 | ensemble_logit ENS_MODE=logit | ensemble_logit_10ckpt.json | COMPLETED | **L5k=8.997× [8.59, 9.40]  L30k=8.311× [7.90, 8.75]** ★ paper number | — | — |
| 60874162 | 2026-05-13 | ensemble_logit (early) | — | CANCELLED | — | superseded by 60874231 | 60874231 |

**Audit conclusion** (see `docs/LIFT_30KM_DEFINITION_AUDIT_2026_05_11.md`):
- "7.26× vs 4.09×" mystery solved: 2 different K conventions on same data
- K-scaled mean-pool is correct (matches train_v3 val loop, defensible interpretation)
- Both `compute_full_metric_card.py` and `ensemble_ckpts_lift.py` patched (commit `ed789fc`)

---

## 2026-05-17 batch — full eval (save_window_scores) for 5 new ckpts

| jobid | submitted | script | run_name | state | result | failure_reason | resub |
|---|---|---|---|---|---|---|---|
| 61135367 | 2026-05-17 | eval_save_scores GATING=global | v3_9ch_enc21_12y_2014_gate_global_eval_full | PD | — | — | — |
| 61135368 | 2026-05-17 | eval_save_scores GATING=per_lead | v3_9ch_enc21_12y_2014_gate_per_lead_eval_full | PD | — | — | — |
| 61135369 | 2026-05-17 | eval_save_scores GATING=per_pixel | v3_9ch_enc21_12y_2014_gate_per_pixel_eval_full | PD | — | — | — |
| 61135370 | 2026-05-17 | eval_save_scores 11ch | v3_11ch_enc21_12y_2014_eval_full | PD | — | — | — |
| 61135371 | 2026-05-17 | eval_save_scores 12ch_static | v3_12ch_static_enc21_12y_2014_eval_full | PD | — | — | — |

★ Earlier batch 61135318-323 was submitted before the latest commit landed on narval (CACHE_LUSTRE_OVERRIDE missing) — CANCELLED.

## 2026-05-17 batch — scaling sweep RESUBMIT (memmap fix applied)

| jobid | submitted | script | run_name | state | result | failure_reason | resub |
|---|---|---|---|---|---|---|---|
| 61135380 | 2026-05-17 | range_master 8y | v3_9ch_enc21_8y_2018 | FAILED | — | date overflow: master_T=9332 too short, need 9334 (pred_end+lead 2d past master cache) | 61231833 |
| 61135381 | 2026-05-17 | range_master 10y | v3_9ch_enc21_10y_2016 | OOM | — | transpose np.ascontiguousarray materialized 386 GB (memmap fix was incomplete) | 61231834 |
| 61135382 | 2026-05-17 | range_master 14y | v3_9ch_enc21_14y_2012 | OOM | — | same transpose OOM (13h) | 61231835 |
| 61135383 | 2026-05-17 | range_master 16y | v3_9ch_enc21_16y_2010 | OOM | — | same transpose OOM (15h) | 61231836 |
| 61135384 | 2026-05-17 | range_master 18y | v3_9ch_enc21_18y_2008 | OOM | — | same transpose OOM (18h) | 61231837 |

**Fix applied (commits `2ebecf9` + `11773b9`)**:
- `train_v3.py` line ~1910: when `--master_cache_dir` is set without `--cache_dir`, memmap meteo_tf to `$SLURM_TMPDIR` (was: in-RAM np.zeros, OOM'd at 510–811 GB)
- Walltime 12h → 24h (8y took ~9h for meteo build alone; 18y needs ~17h)

## 2026-05-18 batch — paper §5/§6 missing experiments

Baselines (4 stateless + MLP + ConvLSTM) and per-lead-day Lift decay
for SOTA model. Required to complete the §6 baselines table and the
Lift-vs-lead-day figure. Commit `570022a`.

| jobid | submitted | script | run_name | state | result | failure_reason | resub |
|---|---|---|---|---|---|---|---|
| 61137765 | 2026-05-18 | baselines_all4_full (CPU) | baselines, both modes in one job | TIMEOUT | per_window finished (611 win) but on WRONG labels (legacy CWFIS); per_leadday timed out | (a) used CWFIS not NBAC+NFDB (b) both modes in one 12h job | 61231830/831 |
| 61137766 | 2026-05-18 | train_baseline_mlp (12y, 9ch) | baseline_mlp_12y_2014_9ch | COMPLETED | trained ok (4h55m) — needs full-eval for §6 number | — | — |
| 61137767 | 2026-05-18 | train_baseline_convlstm (12y, 9ch) | baseline_convlstm_12y_2014_9ch | COMPLETED | trained ok (6h09m) — needs full-eval for §6 number | — | — |
| 61137768 | 2026-05-18 | eval_per_lead on SOTA ckpt | v3_9ch_enc21_12y_2014 per-lead JSON | TIMEOUT | spent all 8h on meteo cache build, never reached eval; also full-card metric 33×/win too slow | — | 61231832 |

## 2026-05-18 batch B — bug-fix resubmits (commit `8f0d58b`)

Fixes: transpose memmap OOM, load_train_to_ram toggle for 16y/18y,
8y date margin, NBAC+NFDB labels for baselines, per-lead metric
slimmed to lift-only + 24h walltime.

| jobid | submitted | script | run_name | state | result | failure_reason | resub |
|---|---|---|---|---|---|---|---|
| 61231830 | 2026-05-18 | baselines per_window (NBAC) | baselines_per_window.csv | FAILED | computed 5h then crashed at CSV write (TypeError: per_window result is {per_win,summary} not {k:{...}}) | 61293625 |
| 61231831 | 2026-05-18 | baselines per_leadday (NBAC) | baselines_per_leadday.csv | OOM | 256G insufficient for per-lead loop | 61293626 |
| 61231832 | 2026-05-18 | eval_per_lead 24h | v3_9ch_enc21_12y_2014 per-lead JSON | R | running 13h+ (past cache build into eval) | — |
| 61231833 | 2026-05-18 | range_master 8y (RAM) | v3_9ch_enc21_8y_2018 | R | running 13h+, past transpose ✓ no OOM | — |
| 61231834 | 2026-05-18 | range_master 10y (RAM) | v3_9ch_enc21_10y_2016 | R | running 12.5h+, chunked memmap transpose confirmed working ✓ | — |
| 61231835 | 2026-05-18 | range_master 14y (RAM) | v3_9ch_enc21_14y_2012 | R | running 10h+ ✓ | — |
| 61231836 | 2026-05-18 | range_master 16y (SSD memmap) | v3_9ch_enc21_16y_2010 | R | running 9.6h+ ✓ | — |
| 61231837 | 2026-05-18 | range_master 18y (SSD memmap) | v3_9ch_enc21_18y_2008 | PD | — | — | — |

## 2026-05-20 batch C — baselines re-resubmit (CSV + mem fix)

| jobid | submitted | script | run_name | state | result | failure_reason | resub |
|---|---|---|---|---|---|---|---|
| 61293625 | 2026-05-20 | baselines per_window (NBAC, 400G) | baselines_per_window.csv | PD | — | — | — |
| 61293626 | 2026-05-20 | baselines per_leadday (NBAC, 400G) | baselines_per_leadday.csv | PD | — | — | — |

**Scaling sweep is HEALTHY this time** (commit `8f0d58b` transpose fix verified in 61231834 log: "Transposing to patch-first (chunked memmap)" — no np.ascontiguousarray, no OOM). 4/5 running past the point where batch A died.

## 2026-05-20 batch D — pending follow-ups (metric_cards + baseline re-eval)

5 ckpt evals (gating/11ch/12ch) finished with fresh 2026-05-18 scores →
metric_cards. MLP/ConvLSTM ckpts retrained 2026-05-18 but their saved
scores are stale (2026-05-07) → re-eval first, metric_card after.

| jobid | submitted | script | run_name | state | result | failure_reason | resub |
|---|---|---|---|---|---|---|---|
| 61293772 | 2026-05-20 | metric_card model | v3_9ch_enc21_12y_2014_gate_global | PD | — | — | — |
| 61293773 | 2026-05-20 | metric_card model | v3_9ch_enc21_12y_2014_gate_per_lead | PD | — | — | — |
| 61293774 | 2026-05-20 | metric_card model | v3_9ch_enc21_12y_2014_gate_per_pixel | PD | — | — | — |
| 61293775 | 2026-05-20 | metric_card model | v3_11ch_enc21_12y_2014 | PD | — | — | — |
| 61293776 | 2026-05-20 | metric_card model | v3_12ch_static_enc21_12y_2014 | PD | — | — | — |
| 61293777 | 2026-05-20 | eval_save_scores MODEL_TYPE=mlp | baseline_mlp_12y_2014_9ch | PD | — | — | — |
| 61293778 | 2026-05-20 | eval_save_scores MODEL_TYPE=convlstm | baseline_convlstm_12y_2014_9ch | PD | — | — | — |

★ After 61293777/778 finish → run metric_card on baseline_{mlp,convlstm}_12y_2014_9ch.

## 2026-05-23 — RESULTS COLLECTED (batches B/C/D outcome)

### Successes — headline numbers

**Baselines (583-win, NBAC+NFDB labels, leak-free upto_2022 clim)** — from `outputs/baselines_per_window.csv`:

| baseline | Lift@5000 | Lift@30km | notes |
|---|---|---|---|
| climatology | 7.04× | 2.36× | leak-free annual |
| persistence | **20.32×** | **12.02×** | ★ surprise: dilation makes this near-cheating (recent fires still burning within 28 km) |
| fwi_threshold | 0.02× | 0.00× | ★ BROKEN — debug below |
| fwi_oracle | 0.001× | 0.00× | ★ BROKEN — debug below |

**Model ablations (Lift@5000 with bootstrap 95 % CI, n_wins=435 after fire-season filter)**:

| run | Lift@5000 [CI] | Δ vs SOTA | notes |
|---|---|---|---|
| SOTA 9ch enc21 12y | **7.83 [7.44, 8.18]** | (reference) | — |
| gating global | 7.00 [6.79, 7.22] | −11 % | hurts |
| gating per_lead | 7.00 [6.78, 7.22] | −11 % | hurts |
| **gating per_pixel** ⭐ | **9.59 [9.11, 10.03]** | **+22 %** | CI doesn't overlap SOTA — **new SOTA candidate** |
| 11ch +pop +slope | 8.14 [7.76, 8.51] | +4 % | small improvement, CI overlaps |
| 12ch_static (lightning) | 8.41 [7.72, 9.07] | +7 % | unexpected — was -22 % at 20-win |

★ Note: `lift_coarse` is `None` in the saved metric cards — `compute_full_metric_card.py` doesn't recompute Lift@30km from the aggregated `prob_agg` npz format. Separate follow-up.

**Per-lead-day model curve** (`outputs/per_lead/v3_9ch_enc21_12y_2014.json`, 583 windows):

| lead | median Lift@30km |
|---|---|
| 14 | 6.52 |
| 18 | 6.77 |
| 22 | **6.88** (peak) |
| 30 | 6.74 |
| 38 | 6.55 |
| 42 | 6.02 |

★ Skill is essentially flat across 14–38 d lead — paper figure ready.

**Scaling sweep** (best Lift@5000 from training-time 20-win val):

| range | Lift@5000 | state |
|---|---|---|
| 8y_2018 (61231833) | 4.37× | COMPLETED |
| 10y_2016 (61231834) | 6.30× | COMPLETED |
| 12y_2014 (existing SOTA) | 8.07× | reference |
| 14y_2012 (61231835) | — | TIMEOUT at ep3 47 % (24 h not enough) |
| 16y_2010 (61231836) | — | TIMEOUT at ep1 60 % (SSD memmap = 2× slower) |
| 18y_2008 (61231837) | — | CANCELLED while still building meteo cache |

★ Monotonic increase 4.37 → 6.30 → 8.07 — the paper scaling story holds.

### 2026-05-23 batch E — failures resubmit + missing metric_cards

| jobid | submitted | purpose | state |
|---|---|---|---|
| 61471276 | 2026-05-23 | metric_card MLP | R |
| 61471277 | 2026-05-23 | metric_card ConvLSTM | R |
| 61471278 | 2026-05-23 | scaling 14y (48 h, RAM on) | PD |
| 61471279 | 2026-05-23 | scaling 16y (48 h, RAM on — was SSD memmap before) | PD |
| 61471497 | 2026-05-23 | baselines per_leadday climatology only | PD |
| 61471498 | 2026-05-23 | baselines per_leadday persistence only | PD |
| 61471499 | 2026-05-23 | baselines per_leadday fwi_threshold only | PD |
| 61471500 | 2026-05-23 | baselines per_leadday fwi_oracle only | PD |

Also cancelled 3 unknown PD jobs (61369659–661, submitted 2026-05-22 03:11 — origin unclear, cancelled to avoid duplicate compute).

### Known follow-ups

1. **FWI baseline lift ≈ 0 — NOT a bug, real geophysical finding.** Direct
   diagnostic (2026-05-23, issue 2023-06-15, 33-day lead window):
   - FWI top-5000 pixels: range 105–135, all in rows 1853–2280 × cols
     262–683 = **southern Saskatchewan / Alberta dry prairie**.
   - FWI at *real* fire pixels: mean ≈ 25 (moderate, boreal forest).
   - Both `max` (fwi_oracle) and `mean` (fwi_threshold) over the lead
     window peak in the same dry-prairie region — that region has
     consistently extreme FWI all summer but **NBAC+NFDB rarely records
     fires there** (NBAC ≥10 ha threshold + grassland under-coverage).
   - Conclusion: FWI as a pure pixel-level ranker against NBAC+NFDB
     labels is uninformative at 14–46 d lead. This is paper-worthy
     ("standard fire-weather guidance fails as a pixel ranker for
     this label set"), not a fix. Report as-is with this discussion.
   - Optional sanity: clip FWI > 100 (some 2023-07 values reach 134,
     likely real but extreme); won't change the conclusion materially.
2. **Lift@30km in metric_card** — currently `None`. Need to either (a) re-eval saving per-patch + grid info, or (b) load fire labels in compute_full_metric_card and reconstruct 2D from the existing prob_agg + grid metadata in the npz.
3. **18y scaling** — skipped for now (meteo build alone > 24 h). Paper story works with 8/10/12/14/16y.

## 2026-05-21 — batch B/C/D outcomes + batch E resubmits

### ✅ Completed
- **per-lead model curve** (61231832): DONE 17h55m → `outputs/per_lead/v3_9ch_enc21_12y_2014.json` (2.7 MB)
- **scaling 8y** (61231833): DONE → 20-win sample L5k=4.37 L30k=3.86
- **scaling 10y** (61231834): DONE → 20-win sample L5k=5.61 L30k=6.10
  - Scaling trend (20-win samples, noisy): 8y=4.37 → 10y=5.61 → 12y=8.07(SOTA). Need full-eval on the ckpts for paper numbers.
- **baselines per_window** (61293625): DONE 4h48m → `outputs/baselines_per_window.csv`
  - climatology: L5k=7.04 L30k=2.36 ✓ | persistence: L5k=20.3 L30k=12.0 ⚠️ (suspiciously high — active large fires persist through the 14d gap; investigate leak vs real)
  - ★ **fwi_threshold + fwi_oracle BROKEN: Lift≈0, BSS=-14172**. lift<1 means FWI top-K avoids fire (anti-correlated) → date misalignment or spatial-layout bug in fwi_patched. NEEDS FIX before §6 table.
- **5 metric_cards** (61293772-776): DONE — full 583-win:

  | run | L5k | L30k | vs SOTA(7.83/6.73) |
  |---|---|---|---|
  | gate_global   | 7.00 | 7.43 | L30k +0.7 |
  | gate_per_lead | 7.00 | 7.35 | L30k +0.6 |
  | **gate_per_pixel** | **9.59** | **8.10** | **both win clearly** ★ |
  | 11ch (+pop+slope) | 8.14 | 6.78 | L5k +0.3, L30k flat |
  | 12ch_static | 8.41 | 7.44 | both modestly up |

  → **per_pixel gating is a real win on full eval** (L5k 9.59 vs 7.83, L30k 8.10 vs 6.73), not just a calibration win.
- **MLP/ConvLSTM re-eval** (61293777/778): DONE → fresh scores; metric_cards submitted (61369513/515).

### ⏱️ TIMEOUT → fixed & resubmitted (batch E)
- **scaling 14y** (61231835): TIMEOUT at ep3 47% (24h). RAM ok (233 GB). Just needed more time.
- **scaling 16y** (61231836): TIMEOUT at ep1 60% (24h). Root cause: I set LOAD_TRAIN_TO_RAM=0 (SSD, 0.6 b/s) out of OOM caution — but fire-season RAM is only ~284 GB, fits 480G. SSD was 1.8× slower → timeout.
- **scaling 18y** (61231837): same SSD mistake; cancelled (would not finish).
- **baselines per_leadday** (61293626): TIMEOUT 16h — only got through climatology's 33 leads. NOT resubmitted: the 3 stateless baselines are flat in lead, so the per_window values ARE the flat-line values for the figure. Only fwi_oracle varies by lead (and it's the broken one).

| jobid | submitted | script | run_name | state | result | failure_reason | resub |
|---|---|---|---|---|---|---|---|
| 61369513 | 2026-05-21 | metric_card model | baseline_mlp_12y_2014_9ch | PD | — | — | — |
| 61369515 | 2026-05-21 | metric_card model | baseline_convlstm_12y_2014_9ch | PD | — | — | — |
| 61369659 | 2026-05-21 | range_master 14y (RAM, 48h) | v3_9ch_enc21_14y_2012 | PD | — | — | — |
| 61369660 | 2026-05-21 | range_master 16y (RAM, 48h) | v3_9ch_enc21_16y_2010 | PD | — | — | — |
| 61369661 | 2026-05-21 | range_master 18y (RAM, 48h) | v3_9ch_enc21_18y_2008 | PD | — | — | — |

### Open items (not yet actioned)
1. **FWI baseline bug** — fwi_threshold/fwi_oracle Lift≈0. Debug fwi_patched date/spatial alignment in benchmark_baselines.load_data.
2. **persistence sanity** — L30k=12 > model; confirm not a leak (14d gap should prevent direct leak; large fires legitimately persist).
3. **Scaling full-eval** — run eval_save_scores + metric_card on 8y/10y/14y/16y/18y ckpts for paper-grade 583-win numbers (currently only 20-win samples).

**What each produces**:
- 61137765 → `outputs/baselines_per_window.csv` (§6 baselines table headline numbers) + `outputs/baselines_per_leadday.csv` (flat baseline curves for lift-vs-lead figure)
- 61137766 / 67 → trained MLP / ConvLSTM ckpts in `checkpoints/baseline_{mlp,convlstm}_12y_2014_9ch/`, also produces `outputs/baseline_{mlp,convlstm}_..._per_window.json`
- 61137768 → `outputs/per_lead/v3_9ch_enc21_12y_2014.json` (model lift-vs-lead curve)
- Final figure produced by `scripts/plot_per_lead_lift.py` combining the model JSON + baseline per-lead CSV.

## Pending follow-ups (after the 14 PD jobs land)

1. **5 metric_cards** on the save_window_scores from 61135367-371 (5 × ~1h)
2. **Add new scaling ckpts to ensemble** if they help — currently 10-ckpt logit-mean = 8.31× Lift@30km
3. **NBAC 2025 labels** — not released yet. Val-2025 silently skipped (n=536→402 valid).
4. **Eval MLP / ConvLSTM** with `eval_save_scores_full_narval.sh` (MODEL_TYPE=mlp/convlstm) after training (61137766/67) completes, to get matching 583-win Lift@30km numbers for the §6 architecture-ablation row.
5. **Plot lift-vs-lead figure** once 61137765 + 61137768 finish: `python -m scripts.plot_per_lead_lift --model_json outputs/per_lead/v3_9ch_enc21_12y_2014.json --baselines_csv outputs/baselines_per_leadday.csv --out_dir figures/per_lead`.

---

## Tracking convention

When submitting a job from now on:
1. `sbatch slurm/foo.sh` → note jobid
2. Append a row here with `state=PD`, `result=—`, immediately
3. When done: update state + paste key metric or failure tail (≤2 lines)
4. If resubmit: link the new jobid in `resub` column

Why a markdown table not CSV: I read this with eyeballs, not pandas.
If we ever need filtering, `grep ' R ' docs/JOBS_REGISTRY.md` works fine.
