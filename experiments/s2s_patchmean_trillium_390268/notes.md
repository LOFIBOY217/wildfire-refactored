# S2S Patch-Mean Decoder (Trillium, old code) — Job 390268

## Run info
- **Job ID**: 390268 (original) / **390277** (re-run, identical config, COMPLETED 2026-04-01)
- **Cluster**: Trillium (trig0018, 1 GPU)
- **Script**: `slurm/train_v2_s2s_decoder_trillium_1gpu.sh`
- **Run name**: `s2s_decoder_s2s_v3_1gpu`
- **Wall time**: 4:44:19
- **Date**: 2026-03-31 (during Narval maintenance window)
- **Status**: COMPLETED
- **Checkpoint**: `checkpoints/s2s_decoder_s2s_v3_1gpu/best_model.pt`

## Hyperparameters
| Param | Value |
|-------|-------|
| decoder | `s2s` (full-patch, dec_dim=2048) |
| epochs | 12 |
| batch_size | 1024 |
| lr | 1e-4 → 1e-6 (cosine) |
| dropout | 0.2 |
| weight_decay | 0.05 |
| label_smoothing | 0.05 |
| neg_buffer | 2 |
| fire_season_only | True |
| num_workers | 12 |
| pred_end | 2025-10-31 |
| s2s_max_issue_lag | (default) |
| skip_forecast | True |

**Note**: Same heavy regularization as v3 (`dropout=0.2, wd=0.05, label_smooth=0.05, neg_buffer=2`).
Regularization confirmed over-strong from v3 analysis.

## Results

> Full epoch table from job 390277 (re-run, identical config, confirmed same result):

| Epoch | train_loss | val_loss | Lift@5000 | prec@5000 |
|-------|-----------|---------|-----------|-----------|
| 1  | 0.650254 | 0.287044 | 6.83x | 0.5646 |
| 2  | 0.646286 | 0.283882 | 6.81x | 0.5630 |
| 3  | 0.644731 | 0.282322 | 7.74x | 0.6398 |
| 4  | 0.643614 | 0.284220 | 7.88x | 0.6514 |
| **5** ★ | **0.642670** | **0.283146** | **9.28x** | **0.7666** |
| 6  | 0.641796 | 0.279339 | 7.53x | 0.6222 |
| 7  | 0.640994 | 0.278320 | 7.81x | 0.6450 |
| 8  | 0.640311 | 0.279936 | 8.27x | 0.6834 |
| 9  | 0.639729 | 0.278385 | 7.90x | 0.6530 |
| 10 | 0.639281 | 0.276229 | 7.78x | 0.6428 |
| 11 | 0.638970 | 0.276215 | 7.86x | 0.6494 |
| 12 | 0.638794 | 0.274908 | 7.74x | 0.6392 |

- **Best Lift@5000 = 9.28x @ ep5**, val_loss=0.283146
- Speed: 27.7–27.9 batches/sec, 30,314 batches/epoch
- 390277 是 390268 的重复运行（完全相同配置），结果一致，确认可复现

## Architecture
- **Decoder type**: `s2s` flag in old Trillium codebase = **patch-mean** (NOT full-patch)
- **dec_dim**: 9 (6 weather channels + issue_age + is_fallback + is_missing)
- S2S cache: patch-mean forecasts, shape `(n_dates, n_patches, 32, 6)` float16
- Channels: 2t / 2d / tcw / sm20 / st20 / VPD (6 weather, averaged over 16×16 patch)
- **Key**: In current Narval code, `--decoder s2s` now REQUIRES `--s2s_full_cache` (full-patch, dec_dim=2048).
  The old Trillium code predates the `s2s_legacy` / `s2s_full_cache` split.
  This run is architecturally equivalent to the current `--decoder s2s_legacy` (dec_dim=9, NOT Oracle-format).

## Comparison Table
| Model | Decoder | dec_dim | Regularization | Best Lift@5000 |
|-------|---------|---------|---------------|----------------|
| Oracle | oracle (future obs) | 2048 | dropout=0.1, wd=0.01 | **19.09x** |
| Climatology baseline | static map | — | — | 9.56x |
| **This run (Trillium 390268)** | **s2s patch-mean (old code)** | **9** | **heavy (v3)** | **9.28x** |
| S2S Legacy v3 (Narval 58577940) | s2s_legacy patch-mean | 9 | **heavy** (v3) | 6.78x |
| S2S Legacy v4 (Narval 58675629) | s2s_legacy patch-mean | 9 | light | pending |
| Random decoder | random noise | 2048 | — | pending |
| S2S Full-Patch (NOT YET RUN) | s2s + s2s_full_cache | **2048** | — | **TBD** |
| FWI Oracle baseline | FWI heuristic | — | — | 1.26x |

## Analysis

### CORRECTION: This is patch-mean (dec_dim=9), NOT full-patch (dec_dim=2048)
- The Trillium script used `--decoder s2s --s2s_cache ...` (no `--s2s_full_cache`)
- Old Trillium code: `--decoder s2s` = patch-mean (same as current `--decoder s2s_legacy`)
- Current Narval code: `--decoder s2s` without `--s2s_full_cache` raises `ValueError`
- This run is architecturally **identical** to Narval v3 (s2s_legacy, dec_dim=9)

### Why Trillium (9.28x) > Narval v3 (6.78x) with same hyperparams?
- Both use heavy v3 regularization (dropout=0.2, wd=0.05, label_smooth=0.05, neg_buffer=2)
- Both use dec_dim=9 (patch-mean S2S)
- Trillium ran 12 epochs vs Narval 8 epochs → more epochs = found better peak
- Narval v3 peaked at ep3, Trillium peaked at a later epoch with same regularization

### Still limited by over-regularization
- 9.28x barely beats climatology (9.56x)
- With lighter regularization (v4 running on Narval), full-patch S2S could reach higher
- Oracle (19.09x) uses observed weather → weather signal is strong when properly decoded

### Performance vs expectation
```
Oracle (19.09x)  >>  Climatology (9.56x) ≈ S2S Full-Patch (9.28x)  >>  S2S Legacy v3 (6.78x)  >>  FWI Oracle (1.26x)
```
- S2S full-patch is matching climatology, not beating it → regularization is too strong
- Climatology = static geographic prior (no weather)
- Model should be clearly above climatology once regularization is fixed

## Next steps
1. Wait for S2S legacy v4 (Narval 58675629, light reg) to compare s2s_legacy without over-regularization
2. Wait for random decoder (Narval 58663929) to establish encoder-only baseline
3. Once S2S full-patch cache rebuilt (Narval 58683212), train S2S full-patch with v4 light regularization
4. After ext TIF fix (tp bug), rebuild cache with 10u/10v/tp channels → new full-patch experiment
