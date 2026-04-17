# Post-Training Analysis Plan

Three analyses agreed on 2026-04-17 but deferred due to implementation
complexity. Each requires modification of `train_v3.py`'s val path or a
separate analysis script.

## Background

We have these top models (all 9ch, anti-overfit config, best at ep1):
- enc14 (Lift@5000 = 5.37x, ep3)
- enc21 (5.80x)
- enc28 (6.43x)
- enc35 (6.56x) ★ SOTA

Full evals for all four running 2026-04-17 (jobs 59455913/15/16/59511730).

---

## Analysis 1: Ensemble (average N models)

**Hypothesis**: Different encoder lengths capture different temporal patterns
(near-term vs multi-week drought). Averaging may reduce variance.

**Estimated ROI**: +0.2 to +0.5x Lift@5000 at zero training cost.

**Implementation**:
1. Add `--ensemble_ckpts` + `--ensemble_in_days` CLI args to `train_v3.py`.
2. Add `EnsembleModel(nn.Module)` class wrapping N submodels. Each submodel
   is fed `xb_enc[:, -d:, :]` (its own encoder_days). Forward averages logits.
3. In the `--eval_checkpoint` block, if `--ensemble_ckpts` is set, build
   N sub-models, wrap in EnsembleModel, skip single-ckpt loading.
4. Use outer `--in_days = max(ensemble_in_days)` for xb_enc extraction.

**Submission** (once implemented):
```bash
ssh narval
sbatch --time=12:00:00 slurm/eval_v3_checkpoint_narval.sh \
  --ensemble_ckpts checkpoints/v3_9ch_enc14/best_model.pt,\
checkpoints/v3_9ch_enc21/best_model.pt,\
checkpoints/v3_9ch_enc28/best_model.pt,\
checkpoints/v3_9ch_enc35/best_model.pt \
  --ensemble_in_days 14,21,28,35 \
  --in_days 35 \
  --full_val
```

**Key caveat**: Averaging logits (not sigmoid probs) — monotonic for ranking
metrics like Lift, but not calibrated for Brier. That's fine for our metric.

---

## Analysis 2: Coarsening (2km → 30km)

**Hypothesis**: Our Lift@5000 is at 2km grid, but adjacent pixels are
spatially correlated (fire clusters). A 30km grid matches S2S forecast
resolution and gives a more honest metric.

**Estimated finding**: coarsened Lift ~ 2-4x (lower than pixel 6.56x).
If coarsened Lift > climatology coarsened Lift, we have real signal.

**Implementation**:
1. In `_compute_val_lift_k_v3`, after computing per-window 2D prob map,
   also compute coarsened Lift:
   ```python
   factor = 15  # 30km from 2km
   H, W = prob_map.shape
   h2, w2 = H // factor, W // factor
   p_c = prob_map[:h2*factor, :w2*factor].reshape(h2, factor, w2, factor).mean(axis=(1,3))
   y_c = label_map[:h2*factor, :w2*factor].reshape(h2, factor, w2, factor).max(axis=(1,3))
   # Lift@K_coarse where K_coarse = K / factor² (same area)
   ```
2. Report both pixel Lift + coarsened Lift in output.

**Post-hoc alternative**: modify `_compute_val_lift_k_v3` to save per-window
2D prediction maps to disk. Do coarsening offline.

---

## Analysis 3: Per-Window Error Distribution

**Hypothesis**: Model fails worst on specific window types (e.g. early
season, BC, small fires). Identifying these guides next iteration.

**Estimated finding**: bottom 10% of val windows have Lift<2x. Common
properties: early season, <100 fires, remote regions.

**Implementation**:
1. In `_compute_val_lift_k_v3`, the per-window metrics are already computed
   but only mean±std is printed. Add `--save_per_window_json <path>` arg
   that dumps the full per-window list.
2. Post-hoc: sort by Lift, print bottom-10 + top-10 with (date, n_fire,
   baseline, Lift).
3. Grouping: by month, by year, by geographic quadrant.

---

## Priority Order

1. **Analysis 3** (easiest): 10-line change to save per-window JSON.
   Run with existing full-eval jobs. Do after they complete 2026-04-17.
2. **Analysis 1** (ensemble): 50-line change, likely best ROI.
   After cache rebuild + 2000-2025 training launched (to avoid conflicts).
3. **Analysis 2** (coarsening): 30-line change, needs care with K-scaling.
   Also post-2000-2025 training.

## Why deferred

- Current priority: 2000-2025 extended training (scaling law, higher ROI).
- GPU is saturated (4 encoder-sweep training jobs × 3 days).
- Modifying `train_v3.py` while 4 jobs import it is safe (Python caches
  imports at startup), but adds risk. Prefer to wait until sweep done.
- Smoke v5 already validated data pipeline — don't rush analysis work
  that could have been done when cache ready.

---

_Written 2026-04-17 during extended wait for ERA5 download completion._
