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

## Pending follow-ups (not yet submitted)

1. **save_window_scores for gating + static channels** — to get full 583-win numbers
   - 5 runs: gate_global, gate_per_lead, gate_per_pixel, 11ch, 12ch_static
   - Then 5 metric_cards on those
   - ETA: 5 × ~2h save_scores + 5 × ~1h metric_card

2. **Scaling sweep re-architect** — code fix for master_cache memmap path
   - Then resubmit 8y/10y/14y/16y/18y/20y with 24h walltime

3. **NBAC 2025 labels** — not released yet (per MEMORY). Val-2025 windows silently skipped (n=536→402 valid).

---

## Tracking convention

When submitting a job from now on:
1. `sbatch slurm/foo.sh` → note jobid
2. Append a row here with `state=PD`, `result=—`, immediately
3. When done: update state + paste key metric or failure tail (≤2 lines)
4. If resubmit: link the new jobid in `resub` column

Why a markdown table not CSV: I read this with eyeballs, not pandas.
If we ever need filtering, `grep ' R ' docs/JOBS_REGISTRY.md` works fine.
