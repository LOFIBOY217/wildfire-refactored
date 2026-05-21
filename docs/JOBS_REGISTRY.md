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
