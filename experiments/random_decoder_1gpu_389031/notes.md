# Random Decoder (1 GPU) — Job 389031

## Run info
- **Job ID**: 389031
- **Run name**: `random_decoder_reg_1gpu`
- **SLURM script**: `slurm/train_v2_random_trillium_1gpu.sh`
- **Cluster**: Trillium (trig0004), 1x H100
- **Wall time**: 18:08:35
- **Date**: 2026-03-30 02:17 → 2026-03-30 20:25
- **Status**: COMPLETED (exit 0)

## Config
| Param | Value |
|-------|-------|
| decoder | random (ablation — i.i.d. N(0,1) noise) |
| data_start | 2018-05-01 |
| pred_start | 2022-05-01 |
| pred_end | 2025-10-31 |
| in_days | 7 |
| lead_start–end | 14–46 (decoder_days=33) |
| patch_size | 16 |
| dilate_radius | 14 px |
| neg_ratio | 20.0 |
| neg_buffer | 2 |
| batch_size | 1024 |
| epochs | 12 |
| lr | 1e-4 (cosine decay) |
| dropout | 0.2 |
| weight_decay | 0.05 |
| label_smoothing | 0.05 |
| d_model / nhead | 256 / 8 |
| Parameters | 8,488,704 |

## Data
- Aligned dates: 2792 (2018-05-01 → 2025-12-21)
- Train days: 1461 / Val days: 1331
- Train windows: 1408 / Val windows: 1285 (fire season only: 621)
- Train pairs: 31,003,065 (pos: 2,055,193 + neg: 28,947,872)
- Effective neg_ratio: ~14.1x (neg_buffer=2 excluded 2.79M buffer patches)

## Results

| Epoch | Train Loss | Val Loss | Lift@5000 | Prec@5000 |
|-------|-----------|----------|-----------|-----------|
| 1 | 0.6530 | 0.2978 | 7.25x | 0.611 |
| 2 | 0.6514 | 0.2979 | 7.03x | 0.593 |
| 3 | 0.6502 | 0.2908 | 7.80x | 0.658 |
| **4** ★ | **0.6494** | **0.2891** | **7.93x** | **0.668** |
| 5 | 0.6486 | 0.2884 | 7.64x | 0.644 |
| 6 | 0.6479 | 0.2892 | 7.56x | 0.638 |
| 7 | 0.6472 | 0.2846 | 7.50x | 0.633 |
| 8 | 0.6465 | 0.2859 | 7.63x | 0.644 |
| 9 | 0.6459 | 0.2869 | 7.39x | 0.624 |
| 10 | 0.6454 | 0.2845 | 7.45x | 0.629 |
| 11 | 0.6451 | 0.2854 | 7.25x | 0.611 |
| 12 | 0.6449 | 0.2847 | 7.18x | 0.605 |

**Best**: Lift@5000=**7.93x** @ ep4, val_loss=0.2891
Checkpoint: `checkpoints/random_decoder_reg_1gpu/best_model.pt`

## Analysis
- **Lift 在 ep4 到顶，之后随 LR 下降而下滑** — 典型过正则化模式（与 v3 一致）
- val_loss 继续缓降至 ep10（0.2845），但 Lift 不跟随 — loss 与 Lift 解耦，模型学到了平均概率而非空间排序
- train loss 全程仅从 0.653 → 0.645，变化极小，说明模型容量被严重压制
- GPU VRAM 仅用 0.3/80 GB（实际极低利用率），瓶颈在 IO 和正则化而非模型复杂度
- ep10 grad max=1.0995 触发 CLIPPED，说明后期梯度不稳定

## 与其他实验对比

| 实验 | decoder | Lift@5000 | 正则化 |
|------|---------|-----------|--------|
| Oracle (Windows, ep1) | oracle (future ERA5) | **19.09x** | 轻 |
| S2S legacy v3 (Narval) | s2s (patch-mean, dec_dim=9) | 7.17x | 重 |
| **Random decoder (Trillium)** | random noise | **7.93x** | 重 |

**关键发现**: Random > S2S legacy — 随机噪声比 S2S 预报信息效果更好，说明旧 S2S 实现（patch-mean, dec_dim=9）提供的信息实际上在干扰模型，或与正则化叠加后负面效果更强。

## What to try next
- 减轻正则化（dropout=0.1, wd=0.01, 无 label_smoothing, 无 neg_buffer）对比公平
- 新 S2S full-patch decoder（dec_dim=2048，与 Oracle 同格式）训练完成后与此对比
- 目标：S2S full-patch 应显著超过 7.93x，接近 Oracle 19.09x
