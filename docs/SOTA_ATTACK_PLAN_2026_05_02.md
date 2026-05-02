# SOTA Attack Plan — 2026-05-02

Goal: forget the paper for now. **Push Lift@5000 from 7.83x (12y enc21) to
the 10–15x range** by exhausting the joint design space (data × channels ×
loss × hparams × sampling × architecture × ensemble).

Baseline reference: climatology = 4.42x. "Remarkably better" = ≥ 3× baseline ≈ 13–15x.

---

## Current SOTA snapshot (2026-05-02 evening)

- **12y enc21 9ch (default recipe) = 7.83x Lift@5000** (full eval, 583 win)
- 12y enc14 + recency τ=6 = 7.01x
- 22y * = ≤ 4.91x  (calibration-vs-rank bottleneck, ep1 collapse)
- 4y enc14 9ch = ~5.5x (full eval ~4.94x partial)
- 4y enc14 13ch = 4.52x (channel-curse-of-dim under low data)

In-flight (2026-05-02): 12y enc21 + recency τ=6, 12y enc28/35 unified
metrics, lift_trajectory hypothesis test, 14y/6y/18y * 4 enc, 12y * 13ch
* 4 enc, 22y * 16ch * 4 enc + cache build.

---

## Decision-blocking experiments (resolve in 48h before committing more compute)

| # | Experiment | Status | What its result decides |
|---|------------|--------|-------------------------|
| 0a | lift_trajectory (60251421) | PD | If 22y has mid-epoch peak → mid-epoch ckpt selection unblocks 22y → potential jump from 4.9x to 7+x |
| 0b | 12y enc21 + recency τ=6 (60251780) | R | Confirms recency helps SOTA model. Predicted 8.3–8.7x |
| 0c | 12y enc28/35 unified metrics (60251779) | R | Confirms enc21 is true 12y best. If enc28 ≈ enc21 → 12y plateau |

---

## Attack tiers (ranked by expected ROI per GPU-hour)

### Tier 1 — Sampling strategy (HIGHEST ROI, low engineering cost)

#### 1.1 Hard-negative ratio sweep
- Current: hard_neg_fraction=0.5
- Try: 0.3 / 0.7 / 0.9
- Expected: +10–25%, 12y enc21 7.83 → 8.3–9.5
- Cost: 3 × 5h GPU
- Risk: > 0.9 may break calibration

#### 1.2 Cluster-aware (spatial) negative sampling  ★ STARTING TODAY
- Current: random patches across entire window
- Idea: sample negatives only from patches within R km (e.g. 200) of any
  positive in the same window. Forces model to learn a sharp spatial
  decision boundary instead of trivial boreal-forest-vs-tundra split.
- Implementation: ~50 lines in train_v3.py + `--neg_spatial_radius_km` flag
- Expected: +15–30%, 12y enc21 7.83 → 9–11
- Cost: ~1h impl + 1h regression + 5h GPU per config

#### 1.3 Boundary-weighted positive sampling
- Idea: positives near fire-cluster boundary carry more learning signal
  than cluster-centre pixels (which the model can over-fit easily).
- weight ∝ distance_to_cluster_centroid
- Expected: +5–10%

### Tier 2 — Loss function (directly attacks calibration-vs-rank)

#### 2.1 BCE + strong pos_weight
- Drop focal entirely. focal smooths logits → top-K rank degrades.
- Try: --loss_fn bce --pos_weight {25, 50, 100}
- Expected: 22y model (4.91 → 6.5–7.5), 12y enc21 (7.83 → 8.0–8.5)
- Cost: 4–6 configs × 6h GPU

#### 2.2 Top-K / listwise loss (lambda loss, NeuralSort, SoftRank)
- Loss directly optimises top-K rank instead of pixel-wise BCE/focal.
- Implementation: ~150 lines new loss + warmup
- Expected: 12y enc21 7.83 → 9–12
- Cost: 1 day impl + tune; 6h GPU per config
- Risk: listwise unstable; needs LR ↓, warmup ↑

#### 2.3 Focal γ sweep
- Current γ=2. Try {0.5, 1, 3, 5} × α ∈ {0.25, 0.5}
- γ↓ → calibration worse but rank better
- Expected: γ=0.5+α=0.5 may add +5–10%

### Tier 3 — Hyperparameters (depends on lift_trajectory result)

#### 3.1 LR schedule: cosine to lr/10 (not 0) + WSD
- Current: cosine to 0 → focal squashes logits in late epochs
- Change: cosine to lr_min = lr * 0.1, OR use Warmup-Stable-Decay
- Expected: 22y 4.9 → 6.5–7.5

#### 3.2 Mid-epoch ckpt selection
- Don't pick best at epoch end; pick at best mid-epoch val Lift@5000.
- Implementation: trivially using lift_trajectory CSV (once 60251421 lands)
- Expected: 22y 4.9 → 7+ (if mid-epoch peak exists)

#### 3.3 Batch / LR co-scaling
- Try {bs=2048, lr=5e-5} and {bs=8192, lr=2e-4}
- Expected: ±5%

### Tier 4 — Channels (already running)

#### 4.1 12y × 13ch (60252181-184) — predicted 7.83 → 8.5–9.5
#### 4.2 12y × 16ch (cache being built) — predicted 7.83 → 9–11 (best bet)
#### 4.3 7ch ablation (drop redundant humidity channels)
- 9ch has 2d + tcw + sm20 — three correlated humidity descriptors
- Test if 7ch (drop sm20 or tcw) reduces Hughes phenomenon
- Expected: ±5% (bias-variance trade)

### Tier 5 — Architecture (medium effort, uncertain return)

#### 5.1 Decoder depth: 4 → 6 / 8 / 10 (cheapest arch tweak)
#### 5.2 d_model: 256 → 384 / 512
#### 5.3 SwiGLU + RMSNorm (modern Transformer freebies)
#### 5.4 ALiBi or RoPE position encoding (better long-enc extrapolation)
#### 5.5 Multi-task auxiliary loss
- Aux head 1: next-day fire prob
- Aux head 2: burned-area regression
- Expected: +5–15%

### Tier 6 — Ensemble / TTA (final squeeze)

#### 6.1 Ensemble across (4y/6y/8y/14y/18y/22y) × (enc14/21/28/35)
- 24+ ckpts available after current sweep finishes
- Simple logit average
- Expected: single 7.83 → ensemble 9–11

#### 6.2 Self-distillation
- Train new model on best-ensemble soft labels
- Expected: +10–20%

#### 6.3 Test-time augmentation
- horizontal flip / channel dropout, average preds
- Expected: +2–5%

---

## Recommended sequencing

### This week (48h)
1. Wait on 0a/0b/0c — they redirect everything else.
2. **Implement cluster-aware spatial negative sampling (1.2)** — STARTED
3. Hard-neg ratio sweep (1.1) — once cluster-aware lands

### Next week (post-trajectory-result)
4. If 22y has mid-epoch peak → mid-epoch ckpt selection (3.2) → re-evaluate
   all 22y models; this could be the cheapest +50% in the entire program
5. 12y × 16ch enc21 once cache lands (4.2)
6. Lambda loss / top-K loss (2.2)

### Last week (squeeze)
7. Ensemble all ckpts (6.1) → +20–30% almost free
8. Self-distill best ensemble → train one model that captures it (6.2)

---

## Quantified bets (my expected SOTA)

| Path | Expected final SOTA | Probability |
|------|---------------------|-------------|
| Tier 0–1 only          | 9–10x   | 70% |
| + Tier 2 (lambda loss) | 10–12x  | 50% |
| + Tier 4 (12y 16ch)    | 11–13x  | 40% |
| + Tier 6 (ensemble)    | 13–16x  | 30% |
| Full stack + lucky     | 15–20x  | 15% |

Climatology = 4.42x. Target ≥ 3× baseline → 13–15x.

---

## Implementation log

- 2026-05-02 evening: cluster-aware negative sampling (1.2) — IN PROGRESS
  - File: src/training/train_v3.py
  - Flag: --neg_spatial_radius_km (float, default 0 = disabled, else
    restrict negatives per training window to patches within R km of any
    positive in same window)
  - Patch size = 16 px × 2 km/px = 32 km/patch → R_patches = ceil(R_km / 32)
  - Per-window dilation of pos_mask via binary_dilation, intersect with
    ~pos_mask (after neg_buffer dilation) for final candidate pool.
  - Then existing _sample_hard_negatives runs on the restricted neg_flat.
