# Random Decoder (Narval) — Job 58437222

## Run info
- **Job ID**: 58437222
- **Cluster**: Narval
- **Run name**: `s2s_decoder_random`
- **Date**: 2026-03-28
- **Status**: COMPLETED, ExitCode 0
- **Checkpoint**: `checkpoints/s2s_decoder_random/best_model.pt`

## Hyperparameters
| Param | Value |
|-------|-------|
| decoder | `random noise` (ablation) |
| pred_end | 2024-10-31 |
| lead_start–end | 14–46 (decoder_days=33) |
| epochs | 6 |
| batch_size | ~1024 (推测，16499 batches/epoch) |
| lr | 1e-4 |
| Parameters | 8,488,704 |

**Note**: decoder 为纯随机噪声，用于建立 encoder-only baseline。decoder 本身不携带任何天气信息。

## Results

| Epoch | Lift@5000 | prec@5000 | |
|-------|-----------|-----------|---|
| 1 | 5.89x | 0.4900 | |
| 2 | 7.55x | 0.6276 | |
| **3** | **8.15x** | **0.6778** | ★ Best |
| 4 | 8.07x | 0.6708 | |
| 5 | 7.36x | 0.6120 | |
| 6 | 7.97x | 0.6622 | |

- **Best Lift@5000 = 8.15x @ ep3**

## Analysis

### Random decoder > S2S legacy decoder？
Random (8.15x) > S2S Legacy v4 (7.80x) > S2S Legacy v3 (6.78x)

这说明：**S2S patch-mean 天气信号在 dec_dim=9 下几乎没有贡献，甚至略微干扰了训练**。
模型主要依赖 encoder（7天历史气象 + FWI + fire climatology）来预测，decoder 信息几乎不起作用。

### 注意事项
- pred_end=2024-10-31（比 v4 Narval 的 2025-10-31 短，val set 略不同）
- 评估仍为 20-sample 快速估算，有一定方差
- 当前在跑的 wf-dec-random (58759655) 使用 pred_end=2025-10-31，结果待更新

## Next
- 等 wf-dec-random (58759655) 完成后更新此结果（pred_end=2025-10-31 版本）
- 与 S2S full-patch (dec_dim=2048) 对比，验证高维 S2S decoder 是否有效
