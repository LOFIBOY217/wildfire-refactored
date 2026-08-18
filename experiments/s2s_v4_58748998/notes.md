# S2S Legacy v4 (Narval) — Job 58748998

## Run info
- **Job ID**: 58748998
- **Cluster**: Narval (ng20201, 1 GPU)
- **Script**: `slurm/resume_s2s_v4_narval.sh`
- **Run name**: `s2s_decoder_s2s_v4`
- **Wall time used**: 14:53:20 / 17:00:00
- **Date**: 2026-04-02
- **Status**: COMPLETED, ExitCode 0
- **Checkpoint**: `checkpoints/s2s_decoder_s2s_v4/best_model.pt`

## Hyperparameters
| Param | Value |
|-------|-------|
| decoder | `s2s_legacy` (patch-mean, dec_dim=9) |
| pred_end | 2025-10-31 |
| lead_start–end | 14–45 (decoder_days=32) |
| epochs | 8 |
| batch_size | 8192 |
| lr | 1e-4 → 1e-6 (cosine) |
| dropout | 0.1 |
| weight_decay | 0.01 |
| label_smoothing | 0.0 |
| neg_buffer | 0 |
| neg_ratio | 20.0 |
| Parameters | 7,966,720 |

**Note**: light regularization (v4), compared with v3 (dropout=0.2, wd=0.05, label_smooth=0.05, neg_buffer=2).

## Results

| Epoch | Lift@5000 | prec@5000 | val_loss | |
|-------|-----------|-----------|---------|---|
| 1 | 6.02x | — | 0.157314 | |
| 2 | 6.24x | 0.5154 | 0.165956 | |
| **3** | **7.80x** | **0.6446** | **0.188225** | ★ Best |
| 4 | 7.73x | 0.6384 | — | |
| 5 | 7.39x | 0.6104 | — | |
| 6 | 6.93x | 0.5724 | — | |
| 7 | 6.95x | 0.5744 | — | |
| 8 | 6.89x | 0.5692 | — | |

- **Best Lift@5000 = 7.80x @ ep3**
- Speed: ~2.7 batches/sec, 4128 batches/epoch (~25min/epoch)

## Analysis

### Compared with v3 (same architecture, different regularization)
| Experiment | Regularization | Best Lift@5000 | Best Epoch |
|------|--------|----------------|------------|
| v3 (58577940) | heavy (dropout=0.2, wd=0.05, ls=0.05, nb=2) | 6.78x | ep3 |
| **v4 (58748998)** | **light (dropout=0.1, wd=0.01)** | **7.80x** | **ep3** |

Light regularization yields a +1.02x gain, but still below Climatology (9.56x).

### Clear signs of overfitting
- After ep3, Lift keeps dropping (7.80 -> 6.89) while train_loss is still decreasing
- Suggests that s2s_legacy (dec_dim=9) model capacity is the bottleneck under the current data, not regularization

### Conclusion
- S2S patch-mean (dec_dim=9) cannot beat Climatology (9.56x) under either light or heavy regularization
- A full-patch decoder (dec_dim=2048) or oracle is needed to test whether the weather signal can be exploited effectively
- Next step: wait for wf-s2s-fp (58784155) and oracle (58777562) results
