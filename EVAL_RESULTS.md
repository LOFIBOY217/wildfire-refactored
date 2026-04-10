# Evaluation Results — Wildfire Prediction

All results use **full validation** (811 fire-season windows, 2022-05–2024-10) unless noted.
K=5000 pixels (~2万km²). Goal: **V3 full-patch 16ch**.

---

## Best Results Summary

| Model | Best Epoch | Lift@5000 | PR-AUC | ROC-AUC | Architecture |
|-------|-----------|-----------|--------|---------|--------------|
| **Oracle (V2)** | ep2 | **10.01x** | **0.4107** | — | 8ch, BCE, oracle decoder (partial) |
| **S2S Legacy (Tri)** | ep4 | **7.38±4.74x** | 0.3481 | 0.8807 | 8ch, BCE, s2s_legacy dec |
| **S2S Legacy (V3)** | ep6 | **7.35±4.92x** | **0.3678** | 0.8897 | 8ch, BCE, s2s_legacy dec |
| **Climatology** | — | **4.20±1.34x** | — | — | Static fire frequency |
| **FWI Oracle** | — | **1.62±1.60x** | — | — | Same-day FWI ranking |
| **Null Input** | — | **~1.0x** | — | — | Random enc+dec (20-win) |

⚠️ All V2 results above are from pre-bug-fix runs. Valid because V2 did NOT use burn_age channel (V3 exclusive). V3 results pending.

---

## V2 Detailed Results

### V2 S2S Legacy — Trillium (`s2s_decoder_s2s_v3_1gpu`, 8ch BCE)

| Epoch | Lift@5000 | Prec@K | Recall@K | CSI@K | ETS@K | ROC-AUC | Brier | PR-AUC |
|-------|-----------|--------|----------|-------|-------|---------|-------|--------|
| 1 | 5.72±4.18x | 0.3894 | 0.0047 | 0.0046 | 0.0038 | 0.8906 | 0.0890 | 0.3460 |
| 2 | 6.73±3.97x | 0.4626 | 0.0055 | 0.0054 | 0.0046 | **0.8941** | 0.0869 | **0.3565** |
| 3 | 7.00±4.22x | 0.4669 | 0.0057 | 0.0056 | 0.0048 | 0.8880 | 0.0871 | 0.3559 |
| **4** | **7.38±4.74x** | **0.4858** | **0.0060** | **0.0060** | **0.0051** | 0.8807 | 0.0882 | 0.3481 |
| 5 | 6.87±4.21x | 0.4538 | 0.0056 | 0.0055 | 0.0047 | 0.8734 | 0.0890 | 0.3409 |
| 6 | 7.06±4.34x | 0.4683 | 0.0057 | 0.0057 | 0.0049 | 0.8581 | 0.0868 | 0.3542 |
| 7 | 7.07±4.25x | 0.4630 | 0.0058 | 0.0057 | 0.0049 | 0.8552 | 0.0883 | 0.3423 |
| 8-12 | (timeout at ep8 — 12h wall, overfitting after ep4) | | | | | | | |

Best: **ep4 Lift=7.38x** | ROC-AUC peaks ep2 then declines → overfitting from ep5+

### V2 S2S Legacy — Narval V3 code (`s2s_decoder_s2s_v3`, 8ch BCE)

| Epoch | Lift@5000 | Prec@K | Recall@K | CSI@K | ETS@K | ROC-AUC | Brier | PR-AUC |
|-------|-----------|--------|----------|-------|-------|---------|-------|--------|
| 1 | 6.09±3.79x | 0.4145 | 0.0050 | 0.0049 | 0.0041 | 0.8899 | 0.0898 | 0.3514 |
| 2 | 6.32±4.12x | 0.4257 | 0.0051 | 0.0051 | 0.0043 | **0.8912** | 0.0879 | 0.3626 |
| 3 | 7.00±4.88x | 0.4589 | 0.0057 | 0.0056 | 0.0048 | 0.8879 | 0.0882 | 0.3594 |
| 4 | 7.32±4.65x | 0.4809 | 0.0060 | 0.0059 | 0.0051 | 0.8890 | 0.0879 | 0.3575 |
| 5 | 7.10±4.64x | 0.4614 | 0.0058 | 0.0057 | 0.0049 | 0.8892 | 0.0894 | 0.3602 |
| **6** | **7.35±4.92x** | 0.4783 | 0.0060 | 0.0059 | 0.0051 | 0.8897 | **0.0863** | **0.3678** |
| 7 | 7.14±4.62x | 0.4681 | 0.0058 | 0.0058 | 0.0050 | 0.8904 | 0.0866 | 0.3666 |
| 8 | 7.29±4.75x | 0.4751 | 0.0059 | 0.0059 | 0.0051 | 0.8897 | 0.0871 | 0.3651 |

Best: **ep6 Lift=7.35x, PR-AUC=0.3678** | Stable plateau ep3-8, no severe overfitting

### V2 Oracle (`oracle_narval_v1`, 8ch BCE, partial eval)

| Epoch | Lift@5000 | Prec@K | PR-AUC | Note |
|-------|-----------|--------|--------|------|
| 2 | 10.01x | 0.4822 | 0.4107 | Job timed out during ep3 eval |

Upper bound with perfect future weather. 8 epoch checkpoints available, not fully evaluated.

---

## V3 Training Pipeline (Current Submission — Apr 10)

| Job | ID | State | Channels | Description |
|-----|----|----|----------|-------------|
| v3-full | 59128198 | PENDING | 16 | **Main target** — FWI/2t/2d/tcw/sm20/deep_soil/precip_def/u10/v10/CAPE/NDVI/population/slope/fire_clim/burn_age/burn_count |
| v3-13ch | 59128199 | PENDING | 13 | Subset of 16ch (no wind/CAPE) |
| v3-9s2s | 59128201 | PENDING | 9 | Simplified baseline |

**All three use:**
- Focal BCE loss (α=0.25, γ=2.0)
- Hard negative mining (hard_neg_fraction=0.5)
- neg_buffer=2 (exclude 5×5 neighborhood of positives)
- s2s_legacy decoder with decoder_ctx (dual-path)
- Fire-season-only training (May–October)
- Annual rolling fire_clim (no data leakage)
- All bug fixes applied (see Bug Log below)

---

## Bugs Discovered & Fixed (Apr 9-10)

### Silent bugs (most dangerous class — no crash, wrong model)

1. **burn_age nodata=9999 sentinel propagation** (HIGH severity)
   - `_load_static_channel` did not mask `src.nodata` → 99.8% of pixels had value 9999 → `log1p(9999)=9.21` dominating the channel → burn_age had near-zero discriminative power.
   - Fix: read `src.nodata` from TIF metadata, set those pixels to 0.

2. **burn_age temporal leakage** (HIGH severity)
   - `years_since_burn_2021.tif` contains ALL fires from 2021 (Jan-Dec). A sample from 2021-07 would see Sept-Dec 2021 fires via burn_age channel → data leakage.
   - Fix: use `cur_date.year - 1` (same logic as annual fire_clim "upto" convention).

3. **Hardcoded stats for burn_age/burn_count** (MEDIUM severity)
   - Code used hardcoded `mean=1.5, std=1.0` instead of computing from data.
   - Fix: compute stats from actual (post-nodata-mask) data.

### Loud bugs (crash the training loop)

4. **lead_end=46 > S2S cache leads=32** — Default `--lead_end 46` produces 33 decoder days, but S2S cache only has 32 leads (14-45). Only clamped for `--decoder s2s`, not `s2s_legacy`. → crash at first val batch.

5. **decoder_ctx missing in V3 validation** — V3 training augments decoder input with static context + lead time (17-dim total), but V3's val calls V2's `_compute_val_lift_k` which only produces 9-dim S2S decoder input. → shape mismatch crash at end of ep1.

6. **cluster_eval only supports ablation decoders** — V3's cluster-level eval code uses `_make_dec_ablation` which rejects `s2s_legacy`. → crash after pixel val.

7. **Python module caching stale code** — Fixed `train_v3.py` on disk, but long-running jobs had already imported old version at startup. Fixes didn't apply to running jobs. → Required cancel + resubmit for ALL V3 jobs to benefit from fixes.

8. **set -e prevents cache copy-back** — When training crashed, `set -euo pipefail` aborted before `copy cache back` logic ran → lost 15h of cache-building work twice.

### Self-inflicted bug from defensive fix

9. **Shape assertion bug** — Added `assert decoder_input.shape[-1] == self.dec_forecast_embed.in_features` to prevent bug #5, but `dec_forecast_embed` is `nn.Sequential` (not `nn.Linear`), has no `in_features` attribute. → smoke test crashed.

---

## Defensive Measures Added (Apr 9-10)

| # | Measure | File | Catches |
|---|---------|------|---------|
| 1 | Shape assertions in `model.forward()` | `src/models/s2s_hotspot.py` | Train/val dim mismatches at batch 1 |
| 2 | Git SHA + dirty status print at startup | `src/training/train_v3.py` | Stale-code bugs (running old imported version) |
| 3 | `_assert_channel_quality` (>99.9% same-value warning) | `src/training/train_v3.py` | Sentinel value propagation |
| 4 | Normalization stats sanity check (std<1e-3, \|mean\|>1e4) | `src/training/train_v3.py` | Stats computed from corrupt data |
| 5 | Val probe with fake tensors before loop | `src/training/train_v3.py` | Val path shape bugs in <1s |
| 6 | `smoke_test_v3_narval.sh` (1h SLURM job) | `slurm/` | Full train/val/cluster_eval path verification |
| 7 | `trap '_copy_back_cache' EXIT` in slurm | `slurm/train_v3_full_narval.sh` | Cache loss on crash/cancel/OOM |
| 8 | Memory-efficient PCA via covariance | `src/data_ops/processing/build_s2s_compressed_caches.py` | OOM from SVD on large sample matrix |
| 9 | Leakage regression tests (`tests/test_no_leakage.py`) | `tests/` | Static inspection of 6 critical code paths |

**Smoke test result (Apr 10)**: PASSED ✓ Lift@1000=3.70x on 2021 Jun-Nov tiny subset.

---

## Pending Evaluation (lower priority)

| Checkpoint | Epochs | Type | Note |
|------------|--------|------|------|
| oracle_narval_v1 | 8 | V2 oracle upper bound | Partial eval only |
| oracle_narval_v3 | 8 | V2 oracle (V3 code) | No eval yet |
| s2s_legacy_best_v1 | 16 | V2 long training | No eval yet |
| v3_focal_3ch | 8 | V3 focal 3ch (pre-bug-fix) | Invalid — burn_age bug |
| v3_null_input | 8 | V3 null baseline (pre-bug-fix) | Invalid — burn_age bug |

---

## Auxiliary Jobs

| Job | Status | Output |
|-----|--------|--------|
| s2s-compress (PCA) | RUNNING 59min | `s2s_pca128_cache.dat` (~300G) |
| s2s_multistat_cache | ✅ 58G | 24-dim per patch |
| s2s_subpatch4x4_cache | ✅ 309G | 128-dim (16 sub-blocks × 8ch) |
| s2s_full_patch_cache | ✅ 4.9T | 2048-dim (full 16×16×8) |
| s2s-fwi baseline | RUNNING 1d2h (53%) | Per-date S2S FWI TIFs |

---

## Key Conclusions (as of Apr 9)

1. **S2S Legacy V2 ceiling**: Lift ~7.4x, PR-AUC ~0.37 — confirmed across Trillium & Narval
2. **Oracle ceiling**: Lift ~10x, PR-AUC ~0.41 — even perfect weather only adds ~10% PR-AUC headroom
3. **Climatology dominance**: 4.2x Lift from static map alone → most signal is spatial prior
4. **V3 pending**: New architecture with focal loss + hard neg mining + decoder_ctx + 16 channels expected to push toward oracle ceiling
5. **V3 reliability improved**: 9 defensive measures now protect against 9 distinct bug categories discovered

---

## Disk Cleanup Log

| Date | Action | Space Freed |
|------|--------|-------------|
| 2026-04-07 | Deleted ERA5 raw GRIBs + idx (12676 files) | ~161G |
| 2026-04-07 | Deleted V2 memmap + fire caches (4 files) | ~302G |
| **Total freed** | | **~463G** |
| Pending | Delete s2s_processed after s2s-fwi completes | ~10.7T |

Scratch usage: 15TB / 20TB (75%) — will drop to ~5TB after s2s_processed deletion.

Last updated: 2026-04-10
