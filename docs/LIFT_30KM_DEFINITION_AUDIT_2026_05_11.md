# Lift@30km Definition Audit — 2026-05-11

**For the paper window.** Read this before quoting any Lift@30km number.

---

## The problem

Two production pipelines computed something they both called "Lift@30km",
on the same checkpoint and same val set, but disagreed by a factor of 1.8×:

| Pipeline | What it called Lift@30km | Number on SOTA ckpt |
|---|---|---|
| `train_v3.py` val loop (uses `metrics.py:compute_coarsened_lift`) | "K-scaled, mean-pool" | **7.26×** |
| `compute_full_metric_card.py` + `ensemble_ckpts_lift.py` | "K-unscaled, max-pool" | **4.09×** |

---

## Audit: 4 combinations on identical data

Ran `scripts/audit_lift30km_definitions.py` on
`v3_9ch_enc21_12y_2014_climsim` saved scores (200 windows):

| # | Score pool | K | Mean Lift |
|---|---|---|---|
| 0 | (fine Lift@5000 control) | 5000 fine | 9.568 |
| **1** | **mean** | **K_fine // 15² = 22** | **9.122** ← train_v3 / paper |
| 2 | max | 22 | 8.073 |
| 3 | mean | 5000 | 4.412 |
| **4** | **max** | **5000** | **4.202** ← metric_card |

(K_fine=5000 throughout, pool factor=15 i.e. 30km tiles from 2km pixels.)

## Which axis dominates

| Axis | Impact |
|---|---|
| **K (22 vs 5000)** | 2.2× difference — DOMINANT |
| score pool (mean vs max) | 5–13% — minor |

## Interpretation

The pooled grid (after 15× coarsening) has ~27,360 cells covering Canada.

| Choice | Top-K selects | Area covered | Interpretation |
|---|---|---|---|
| K = 22 (scaled) | top 0.08% of cells | ~20,000 km² (Canada 0.2%) | "Most-confident hotspots" — discriminating |
| K = 5000 (unscaled) | top 18% of cells | 4.5M km² (Canada 45%) | "Broad geographic ranking" — too lenient, saturates |

**K=5000 unscaled is borderline meaningless** at 30km grid: it covers 45% of
Canada and inevitably includes all major fire zones regardless of model
skill. Every method (model / climatology / persistence) lands at ~4× under
this definition.

**K=22 scaled** correctly preserves the "patrol-budget" interpretation —
the same operational meaning as K=5000 at fine 2km resolution. This is
the convention used in `metrics.py` and what train_v3 reported all along.

## Conclusion: **K-scaled, mean-pool is correct** (method 1)

The previous published numbers (`metrics.py` route) are CONSISTENT and
DEFENSIBLE. The metric_card numbers were computed with a flawed K.

## What the paper should report

| Method | Lift@5000 (fine) | Lift@30km (K-scaled mean-pool) | Source |
|---|---|---|---|
| Random | 1.00× | 1.00× | def |
| Climatology (annual upto_Y) | 3.48× [3.41, 3.54] | **3.19×** | `benchmark_baselines.py` job 59600697 |
| ECMWF S2S forecast | 0.00× | 0.94× | metric_card (still valid — agreement at fine + degenerate at 30km) |
| **Single ckpt SOTA (climsim)** | **8.07×** [7.75, 8.42] | **7.26×** [6.93, 7.62] | `save_per_window_json` |
| **Ensemble 10-ckpt prob-mean** | **9.57×** [9.10, 10.02] | TBD ★ | needs re-run after metric fix |

★ = will be filled in by the re-submitted ensemble job (60816516 or similar).

### Headline claim (CORRECT and DEFENSIBLE)

> "Single-ckpt model 7.26× Lift@30km vs climatology 3.19× → **+127%**, CIs do not overlap.
> Ensemble 9.57× Lift@5000 vs single ckpt 8.07× → +18.6%, CIs do not overlap."

The MEMORY.md "+63%" was an earlier estimate using a different ckpt;
the post-2026-05-05 climsim SOTA gives a wider margin.

---

## Bug provenance

`compute_full_metric_card.py` line 119 (`lift_30km_pooled`) — used
`max(axis)` and `K=k_fine`. `ensemble_ckpts_lift.py` line 50
(`lift_30km`) — copied the same. Both were written 2026-05-03 onward
without cross-checking against `metrics.py:compute_coarsened_lift`.

Fixed in commit `ed789fc` (2026-05-11). 11 metric_cards + ensemble
re-submitted (jobs 60816505–60816516). New numbers expected ~1 hour.

---

## What to do if metric_card numbers still disagree after re-run

Sanity check ladder:

1. Verify the re-submitted metric_cards picked up the new code: check
   commit hash in log header (`PYTHONPATH=...wildfire-refactored` should
   contain commit `ed789fc` or later).
2. Spot-check a single window: load one window's npz, feed into both
   `lift_30km_pooled` (fixed) AND `metrics.py:compute_coarsened_lift`,
   compare.
3. If still disagreeing, investigate:
   - Canada-mask handling (does metric_card mask non-Canada coarse cells?)
   - Patch indexing offset between npz and label_2d
   - Window date alignment vs cluster_eval's `_compute_cluster_lift_k`
     (note: cluster_eval is a DIFFERENT metric — `cluster_lift_k` — not
     Lift@30km. Don't conflate them.)

---

## Files touched

- `scripts/audit_lift30km_definitions.py` — new audit tool, can be re-run
- `scripts/compute_full_metric_card.py:119-141` — fix
- `scripts/ensemble_ckpts_lift.py:50-65` — fix
- This file — paper-window reference
