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
| batch_size | ~1024 (inferred, 16499 batches/epoch) |
| lr | 1e-4 |
| Parameters | 8,488,704 |

**Note**: the decoder is pure random noise, used to establish an encoder-only baseline. The decoder itself carries no weather information.

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

### Random decoder > S2S legacy decoder?
Random (8.15x) > S2S Legacy v4 (7.80x) > S2S Legacy v3 (6.78x)

This shows: **the S2S patch-mean weather signal contributes almost nothing at dec_dim=9, and may even slightly interfere with training**.
The model relies mainly on the encoder (7-day historical meteorology + FWI + fire climatology) to predict; the decoder information barely matters.

### Caveats
- pred_end=2024-10-31 (shorter than v4 Narval's 2025-10-31, so a slightly different val set)
- Evaluation is still the 20-sample quick estimate, with some variance
- The currently running wf-dec-random (58759655) uses pred_end=2025-10-31; results pending

## Next
- Update this result once wf-dec-random (58759655) finishes (the pred_end=2025-10-31 version)
- Compare against S2S full-patch (dec_dim=2048) to check whether the high-dimensional S2S decoder is effective
