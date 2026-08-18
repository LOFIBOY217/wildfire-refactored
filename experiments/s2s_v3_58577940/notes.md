# S2S Decoder v3 — Job 58577940

## Run info
- **Job ID**: 58577940
- **Run name**: `s2s_decoder_s2s_v3`
- **SLURM script**: `slurm/train_v2_s2s_decoder_narval.sh`
- **Node**: ng31301
- **Wall time**: 14:09:45
- **Date**: 2026-03-30 13:23 → 2026-03-31 03:33
- **Status**: COMPLETED

## Config
| Param | Value |
|-------|-------|
| decoder | s2s |
| data_start | 2018-05-01 |
| pred_start | 2022-05-01 |
| pred_end | 2025-10-31 |
| in_days | 7 |
| lead_start–end | 14–45 (32 days) |
| patch_size | 16 |
| dilate_radius | 14 px |
| neg_ratio | 20.0 |
| batch_size | 8192 |
| epochs | 8 |
| lr | 1e-4 → 1e-6 (cosine) |
| dropout | 0.2 |
| weight_decay | 0.05 |
| label_smoothing | 0.05 |
| neg_buffer | 2 |
| d_model / nhead | 256 / 8 |
| Parameters | 7,966,720 |

## Data
- Aligned dates: 2791 (2018-05-01 → 2025-12-20)
- Train windows: 1409 / Val windows: 1285
- Train samples: 31,041,338 (pos: 2,017,884 + neg: 29,023,454)
- Effective neg_ratio: ~14.4x (neg_buffer=2 excluded 2.77M buffer patches)
- S2S cache: 1676 dates (2017-01-02 → 2026-03-22), 100% coverage (exact+fallback, miss=0)

## Results

| Epoch | Train Loss | Val Loss | Lift@5000 | Prec@5000 |
|-------|-----------|----------|-----------|-----------|
| 1 | 0.6531 | 0.3417 | 5.37x | 0.4434 |
| 2 | 0.6481 | 0.3306 | 6.64x | 0.5490 |
| 3 | 0.6463 | 0.3264 | 6.78x | 0.5602 |
| 4 | 0.6452 | 0.3292 | 6.72x | 0.5548 |
| 5 | 0.6443 | 0.3325 | 6.78x | 0.5602 |
| 6 | 0.6437 | 0.3230 | 6.69x | 0.5528 |
| 7 | 0.6433 | 0.3244 | **7.17x** | **0.5924** |
| 8 | 0.6430 | 0.3246 | 6.68x | 0.5520 |

**Best**: Lift@5000=**7.17x** @ ep7, val_loss=0.3244
Checkpoint: `checkpoints/s2s_decoder_s2s_v3/best_model.pt`

## Analysis
- **Conclusion**: regularization is too strong; the model is suppressed and cannot learn well.
- train loss stays at 0.643-0.653 the whole time (not far from the random-guess ln2=0.693) and barely moves.
- dropout=0.2 + weight_decay=0.05 + label_smoothing=0.05 + neg_buffer=2 stacked together heavily dilute the learning signal.
- val loss bounces from ep4; after a brief dip at ep7 it rises again at ep8 — the model oscillates rather than converges.
- Reference target: previous best Lift@5000=19.09x (different data range, different hyperparameters).

## What to try next
- **v4** (`slurm/train_v2_s2s_decoder_narval_v4.sh`): dropout=0.1, weight_decay=0.01, no label_smoothing, no neg_buffer
- Submit after the Narval maintenance window ends (Apr 7 07:00)
