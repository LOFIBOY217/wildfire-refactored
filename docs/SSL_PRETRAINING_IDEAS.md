# Self-Supervised Pretraining Ideas (deferred — record only)

Recorded 2026-04-27. Not active work — to be revisited after 9ch 22y
full eval results land and we have a clearer picture of where the
performance ceiling is.

This file captures the analysis from the Q3/Q4 conversation so we
don't have to redo the literature scan and feasibility judgement next
time the topic comes up.

---

## Q3 — MAE (Masked Autoencoder, He et al. 2021) feasibility

**Standard recipe**:
```
input → patchify → randomly mask 75% of patches
encoder sees only visible 25% (no mask tokens)
small decoder + learnable mask token → reconstruct masked pixels
loss = MSE on masked pixels (per-patch normalized)
discard decoder; finetune encoder for downstream task
```

### What we already have (≈ half an MAE)
- ✅ Patch tokenizer: 16×16, n_patches = 23998
- ✅ Encoder/decoder transformer (`src/models/s2s_hotspot.py:151`,
  `nn.Transformer` with d_model=256, nhead=8, 4+4 layers)
- ✅ Patch embedding: `nn.Linear` enc_embed/dec_embed
- ✅ Sinusoidal positional encoding
- ✅ Massive unlabeled data: 23998 patches × ~9000 days × 9 channels

### What we need to add
1. `src/training/pretrain_mae.py` (~400 lines, mirror train_v3 layout)
2. Random masking logic (~10 lines)
3. Mask token: `nn.Parameter(torch.zeros(1, 1, d_model))`
4. Pixel reconstruction head: `nn.Linear(d_model, P²×C)`
5. Asymmetric decoder (depth 2 vs encoder 4) — MAE convention
6. Per-patch normalization (MAE paper's key trick — z-score before
   reconstruction loss)
7. Pretrain → finetune switching logic

### Effort estimate
- Implementation + debugging: **5-7 working days**
- First pretrain run (full data): 1-2 days GPU
- Finetune + experiments: 3-5 days

### Wildfire-specific traps
1. **Low information density** — most patches are ocean/tundra near-constants.
   MAE will spend capacity reconstructing the boring regions.
   *Fix*: mask only land patches (filter via `population > 0` or
   `fire_clim > 0`).
2. **Time dimension matters** — original MAE only masks spatial patches.
   We have `encoder_days=21`. Spatiotemporal MAE (mask in (space, time))
   is stronger but harder.
3. **Channel-set mismatch** — pretrain on 16ch, finetune on 9ch needs
   projection. Recommend pretrain on the largest set (16ch) and
   finetune by zeroing missing channels.
4. **Large supervised data may dilute SSL benefit** — MAE shines when
   labeled data is scarce. We have 22y of full supervision; SSL gains
   may be modest. Reference: He et al. ImageNet-1k (1M labeled) gets
   only +1.0~2.5% top-1 from MAE.

### Expected gain (rough)
- He et al. ImageNet-1k: +1.0~2.5% top-1
- SatMAE on Sentinel-2 (NeurIPS 2022): +2-5 mIoU
- Our case (estimate): Lift@5000 +5-15%, conditional on correct
  masking strategy. Without trap-1 fix, could easily be 0%.

---

## Q4 — Alternatives to MAE (ranked by feasibility)

### Tier 1: Transfer learning (lowest effort)

#### **Prithvi-100M** (NASA-IBM, 2023)
- 100M-param ViT, MAE-pretrained on HLS (Harmonized Landsat-Sentinel)
- HuggingFace: `ibm-nasa-geospatial/Prithvi-100M`
- Plug-and-play encoder weights for Earth observation
- **Difficulty**: low (a few days to project our 9ch onto 6 optical
  bands, swap into encoder, finetune)
- **Risk**: input mismatch — Prithvi expects optical reflectance, we
  feed FWI/temperature. Projection may waste most of the prior.

#### **ClimaX** (Microsoft, ICML 2023)
- Foundation model pretrained on CMIP6 + ERA5 reanalysis
- GitHub: `microsoft/ClimaX`
- Input modality matches us much better (atmospheric reanalysis vars)
- **Difficulty**: low-medium
- **Recommended over Prithvi if input alignment matters**

### Tier 2: Auxiliary supervision (simplest implementation)

#### **Multi-task auxiliary prediction**
- Add prediction heads for: 14d-future NDVI, soil moisture, FWI
- These targets are free (we already have the data)
- Forces encoder to learn general meteorological structure
- **Difficulty**: low (2-3 days, few extra heads + multi-loss)
- **Expected gain**: +1-3% (typical multi-task literature)
- **Why first**: cheapest validation of "does adding more signal help
  this encoder at all" — if not, MAE/SSL also unlikely to help

#### **Fire return interval pretraining**
- Predict expected return interval per patch (regression target
  derived from fire_clim history)
- Pseudo-supervised — label is auto-computed
- Probably less useful than multi-task above

### Tier 3: Other SSL paradigms

#### **SimMIM** (Microsoft, CVPR 2022)
- Like MAE but lighter decoder + L1 loss
- Often comparable to MAE, simpler to implement
- **Difficulty**: medium (similar to MAE -1 day complexity)

#### **TS2Vec / temporal contrastive**
- Treat same-patch-different-time as positive pair
- Different patch as negative
- Natural fit since we already have time series patches
- **Difficulty**: medium

#### **DINOv2** (Meta 2023)
- Self-distillation, no masking needed, no negatives
- Linear-separable representations stronger than MAE per paper
- **Difficulty**: medium-high

---

## Recommended sequence (if we ever activate this work)

```
Phase 0 (1 week):  multi-task auxiliary baseline
                   add NDVI/sm20 prediction heads to train_v3
                   measure: does +signal help current encoder?
                   
Phase 1 (2 weeks): if Phase 0 gains > +2%, build SimMIM
                   pretrain on 16ch unlabeled data
                   finetune for fire prediction
                   compare against current 9ch SOTA
                   
Phase 2 (open):    if SimMIM works, try Prithvi/ClimaX transfer for
                   benchmark; consider full MAE with spatiotemporal
                   masking
```

### Why not jump straight to MAE
- Wildfire data has low information density per patch (trap 1)
- Without mask-region fix, MAE may reconstruct ocean perfectly and
  learn nothing about fire-relevant signals
- Multi-task is a cheaper sanity check — if encoder doesn't even
  benefit from auxiliary supervised targets it has free access to,
  SSL is unlikely to magically extract more.

---

## Activation criteria

Start this work IF AND ONLY IF:
1. 9ch 22y full eval shows clear plateau (Lift@5000 < 6.5x ceiling)
2. 13ch/16ch sweep also shows plateau (channels alone aren't enough)
3. Year extension (1985-1999) data is available or ruled out
4. We have ~2 months of dedicated capacity

If ALL four hold, start with Phase 0 (multi-task) above.
