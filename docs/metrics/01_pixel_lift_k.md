# Metric Card: Pixel Lift @ K

**Status**: primary metric (used for `best_model.pt` selection)
**Format**: per-window mean ± std, then bootstrap 95% CI across windows
**Implemented in**: [`src/training/train_s2s_hotspot_cwfis_v2.py:_compute_val_lift_k`](../../src/training/train_s2s_hotspot_cwfis_v2.py)
**Unit-tested**: covered indirectly via `tests/test_metrics.py`

---

## 1. Definition (math + plain language)

### Mathematical
For one validation window of `N` pixels with binary fire labels `y ∈ {0,1}^N` and predicted scores `s ∈ R^N`:

```
order   = argsort(s, descending)        # rank pixels by predicted score
top_K   = order[:K]                     # top K predictions
TP      = sum(y[top_K])                 # true positives in top K
P@K     = TP / K                        # precision at K
base    = sum(y) / N                    # baseline fire rate
Lift@K  = P@K / base
```

### Plain language
**"How many times more concentrated with fires is the model's top-K prediction set, compared to random selection?"**

If the model picks 5,000 pixels and 50% are real fires, while random selection would yield 5%, the lift is `0.5 / 0.05 = 10×`.

---

## 2. Intended Use

- **Primary decision metric** for ranking-based wildfire risk products
- Operational scenario: fire-management agency can dispatch limited monitoring/suppression resources to a fixed number of high-risk locations; this metric directly measures how well that dispatch performs
- Standard in imbalanced classification benchmarks (especially marketing, recommender systems, and rare-event prediction)

---

## 3. Inputs

| Field | Type | Notes |
|---|---|---|
| `score` | `float32 (N,)` | Per-pixel predicted probability or any monotone transform |
| `label` | `uint8 (N,)` | Binary, 1 = fire occurs in lead window |
| `K` | `int` | Top-K cutoff. We use K ∈ {1000, 2500, 5000, 10000, 25000} |

**N** is the count of valid pixels in one window. For our 2 km Canada grid: `N ≈ 2281 × 2709 ≈ 6.18M` (after fire-season filter and NaN masking, effective N varies per window).

---

## 4. Output Range and Interpretation

| Lift | Interpretation | Our 4y SOTA |
|---|---|---|
| 1.0 | Random predictor (no skill) | — |
| ≈ baseline ratio | Trivial baseline (climatology level) | climatology = 3.4 |
| 5–10× | Strong rare-event signal | **enc28 = 5.83** |
| > 20× | Suspect data leakage / overfitting | — |
| `1 / base` (≈ 20× for our data) | Theoretical maximum (perfect classifier) | — |

**Theoretical max**: `Lift = 1 / base`. With base ≈ 5%, max lift ≈ 20×. Anything close to this is suspicious.

---

## 5. Why We Use It

1. **Class imbalance**: positive-pixel rate is ~5% (fire-season) to 0.05% (full-year). Accuracy/F1 are misleading at these ratios.
2. **Decision matches operations**: fire managers act on **top-K locations**, not on full probability distributions. Lift@K measures the deployable quantity directly.
3. **Calibration-free**: doesn't require model output to be a calibrated probability. Any score with the same ranking gives the same lift.
4. **Single number**: easy to compare across runs.
5. **Established in our subfield**: used by [Joshi et al. 2025 (Nature Comm)](https://www.nature.com/articles/s41467-025-58097-7), [Pourmohamad et al. 2026 (Earth's Future)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025EF006935), and Canadian Forest Service's CWFIS internal evaluation.

---

## 6. How to Compute

```python
def lift_at_k(score_flat: np.ndarray, label_flat: np.ndarray, k: int) -> float:
    n_fire = int(label_flat.sum())
    if n_fire == 0:
        return float('nan')               # window with no fire → undefined
    base_rate = n_fire / label_flat.size
    order = np.argsort(score_flat)[::-1]  # descending
    tp = int(label_flat[order[:k]].sum())
    return (tp / k) / base_rate
```

We compute it per-validation-window, then aggregate as **mean ± standard deviation across windows**, plus bootstrap 95% CI (2000 resamples).

---

## 7. Properties / Strengths

- **Monotone in score**: any score that preserves ranking gives the same lift. Permits use of un-calibrated logits.
- **Decision-aligned**: the K corresponds directly to operational K (e.g., "monitor top 5000 patches today").
- **Rare-event sensitive**: doesn't get inflated by easy negatives.
- **Interpretable**: "lift = X" intuitively means "X times better than random".

---

## 8. Known Limitations and Failure Modes

1. **K selection is arbitrary**. Different K can give different rankings. We mitigate by reporting K ∈ {1000, 2500, 5000, 10000, 25000}.
2. **Cross-dataset incomparable**. Lift = 5× on one dataset can be easier or harder than 5× on another, because the baseline rates differ.
3. **Polygon-label artifact** (critical for our setup): NBAC fire labels are perimeter polygons; a single mega-fire of 10,000 ha contributes hundreds of positive pixels. Top-K pixel ranking is dominated by which model "captures the mega-fire". A trivial persistence baseline (past 7-day fire density) hits Lift@5000 = 17× on NBAC labels purely from this artifact. Documented in [`docs/SCALING_LAW_LOG_2026_04_25_26.md`](../SCALING_LAW_LOG_2026_04_25_26.md) Section 2.
4. **Mega-fire saturation**: the 4-year transformer already captures the dominant mega-fires; the 22-year transformer cannot improve pixel-level lift even though it has more spatial coverage. See Section 7 of the same log.
5. **Doesn't reward distinct event coverage**. A model that catches one mega-fire perfectly but misses 50 small fires can score the same as a model that catches all 51 fires partially. Use **Cluster Lift** (`02_cluster_lift.md`) to complement.
6. **Sensitive to dilation radius**. We dilate fire pixels by 14 px (~28 km). Larger dilation → more positive pixels → higher base rate → lower lift. K must be reported with `dilate_radius`.

---

## 9. How to Improve This Metric

### Training-side
- **Focal loss** (`α=0.25, γ=2.0`): downweights easy negatives, sharpens model around hard positives. Already in our setup.
- **Hard negative mining** (`hard_neg_fraction=0.5, neg_ratio=20`): forces the model to confront geographically/temporally similar non-fire pixels.
- **Cosine LR schedule with appropriately scaled lr**: prevents over-fitting from too-large learning steps. See `docs/SCALING_LAW_LOG_2026_04_25_26.md` Section 6 for the LR-rescaling diagnosis.
- **Larger model** (more attention heads / d_model): may help if data is not the bottleneck.

### Data-side
- **More diverse training fire events**: extends the model's spatial prior over rare fire regions.
- **Better label quality** (NBAC + NFDB fusion vs pure CWFIS): reduces label noise that the model must memorize.
- **Drift-stable labels**: avoids learning detection biases.

### Evaluation-side
- **Don't optimize this metric in isolation**. Combine with Cluster Lift, Novel-Ignition Lift, ROC-AUC, and Brier for a multi-faceted picture.
- **Larger validation pool**: full-eval (all ~700 windows) gives tighter confidence intervals than 20-window sample.

---

## 10. Comparison to Alternative Metrics

| Alternative | Pros vs Lift@K | Cons vs Lift@K |
|---|---|---|
| Accuracy | Simple | Useless under class imbalance |
| F1 | Threshold-aware | Requires threshold choice; symmetric over precision/recall |
| ROC-AUC | Threshold-free | Insensitive to class imbalance; doesn't reflect top-K decision |
| PR-AUC (AP) | Threshold-free, imbalanced-aware | Single number for entire curve, not the operational K |
| Recall@K | Same K decision | Doesn't penalize false positives proportionally |
| Brier score | Calibration | Penalizes magnitudes, not ranking |

**We use Lift@K as the headline number and report ROC-AUC + Brier alongside for completeness.**

---

## 11. Operational Meaning

Concrete scenario: a fire-management agency receives the model's daily output and selects the top 5,000 of ~6 million 2 km × 2 km cells across Canada to flag for monitoring. They send aerial recon, pre-position equipment, or issue fire-weather alerts.

- Lift@5000 = 5.83 means the **flagged cells contain 5.83× more fire-prone pixels than picking 5,000 cells at random**.
- Equivalently: random selection would yield ~250 fire pixels in 5,000; the model yields ~1,460.

This metric is what an operational fire-decision API would optimize.

---

## 12. Our Empirical Numbers (as of 2026-04-26)

Best 4y model (deployed `best_model.pt`, evaluated on save_scores 20-window sample):

| ENC | Lift@5000 (best epoch) | Bootstrap 95% CI |
|---|---|---|
| 14 | 5.66 | [4.79, 6.57] |
| 21 | 5.84 | [4.85, 6.95] |
| **28** ★ | **5.83** | [4.71, 7.06] |
| 35 | 5.73 | [4.80, 6.83] |

22y models (training-time eval, not yet apples-to-apples):

| ENC | Lift@5000 (best epoch) |
|---|---|
| 14 | 5.73 |
| 21 | 5.60 |
| 28 | 5.59 |
| 35 | 4.97 |

**Pixel Lift roughly plateaus from 4y → 22y.** See Cluster Lift for the metric where 22y wins.

---

## 13. References Using This Metric

- **Joshi et al. 2025**, *Global data-driven prediction of fire activity*, Nature Communications. [link](https://www.nature.com/articles/s41467-025-58097-7)
- **Pourmohamad et al. 2026**, *Daily ignition probability prediction across the western US*, Earth's Future. [link](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025EF006935)
- **Canadian Forest Service** internal CWFIS evaluation (private)
- More broadly: marketing / churn / fraud detection literature standardizes on Lift@K and Gain Charts as decision-aligned metrics.

---

## 14. Related Metric Cards

- [`02_cluster_lift.md`](02_cluster_lift.md) — event-level complement
- `03_novel_ignition_lift.md` — for "new fires only" evaluation (avoids polygon artifact)
- `04_lift_30km.md` — spatially coarsened version for event-level skill
- `05_roc_auc.md`, `06_pr_auc.md`, `07_brier_bss.md` — calibration / discrimination

---

## Changelog

- 2026-04-26 — initial card; reflects 9ch transformer + NBAC+NFDB labels.
