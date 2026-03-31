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
- **Conclusion**: 正则化过强，模型被压制，无法充分学习。
- train loss 全程停在 0.643–0.653（接近随机猜测 ln2=0.693 不远），几乎不动。
- dropout=0.2 + weight_decay=0.05 + label_smoothing=0.05 + neg_buffer=2 同时叠加，学习信号被严重稀释。
- val loss ep4 开始反弹，ep7 出现短暂反弹后 ep8 再次上升，模型振荡而非收敛。
- 对比目标：之前最优 Lift@5000=19.09x（不同数据范围，不同超参）。

## What to try next
- **v4** (`slurm/train_v2_s2s_decoder_narval_v4.sh`): dropout=0.1, weight_decay=0.01, 无 label_smoothing, 无 neg_buffer
- 等 Narval 维护结束 (Apr 7 07:00) 后提交
