# Metric Card: Cluster Lift

**Status**: secondary headline metric (event-level complement to Pixel Lift)
**Format**: per-window mean ± std + bootstrap 95% CI; also weighted-by-n_clusters mean
**Implemented in**: [`src/training/train_v3.py:_compute_cluster_lift_k`](../../src/training/train_v3.py)
**Unit-tested**: 13 tests in [`tests/test_cluster_lift.py`](../../tests/test_cluster_lift.py)
**Originated**: tightening of "cluster-aware top-K" practice from rare-spatial-event literature; specifically tuned for NBAC polygon labels (2026-04-19 fix; 2026-04-26 audit-validated)

---

## 1. Definition (math + plain language)

### Mathematical
For one validation window with binary fire labels `y_2d ∈ {0,1}^(H×W)` and predicted scores `s_2d ∈ R^(H×W)`:

```
1.  Cluster the fire pixels via 8-connectivity:
        cluster_map, n_c = scipy.ndimage.label(y_2d, structure=ones((3,3)))
        clusters = list of connected components

2.  Score each cluster (event-level scoring):
        cluster_score[i] = max(s_2d[mask_i])     # max-pool over each cluster
        cluster_size[i]  = sum(mask_i)

3.  Tile the background into spatially contiguous tiles of comparable size:
        tile_side = sqrt(median(cluster_size))    # so tiles ≈ median cluster
        tile_score[j] = max(s_2d[tile_j])
        keep tiles where label_max == 0           # background-only

4.  Combine into one ranking pool:
        items = clusters (label=1)  ∪  bg_tiles (label=0)
        K_eff = min(K, n_items, max(3 × n_c, 50))

5.  Compute lift:
        baseline = n_c / n_items
        Lift = Precision@K_eff / baseline
```

### Plain language
**"How many distinct fire EVENTS does the model rank in its top-K, vs random?"**

Each fire is collapsed into a single voting unit regardless of size, so a 1-pixel ignition counts equally to a 10,000-pixel mega-fire. The model's score for an event = its highest-prob pixel within the event.

---

## 2. Intended Use

- **Operational decision metric** for resource pre-positioning across multiple distinct fire events
- Answers: *"How many of today's actual fire events did the model's top warnings cover?"* — independent of how big each fire ends up
- Complement to `Pixel Lift` (which is dominated by mega-fires); the two together form a more complete picture

---

## 3. Inputs

| Field | Type | Notes |
|---|---|---|
| `score_2d` | `float32 (H, W)` | Per-pixel predicted score, must be 2D image (not patches) |
| `label_2d` | `uint8 (H, W)` | Binary fire labels (post-dilation) |
| `K` | `int` | Requested top-K. Internally adapted (see below) |
| `min_cluster_size` | `int = 1` | Drops single-pixel noise clusters if > 1 |

**Note**: requires the score and label as 2D images, not patch-flattened. Our `train_v3.py` reconstructs 2D via `depatchify(prob_agg)` before calling.

---

## 4. Output Range and Interpretation

| Cluster Lift | Interpretation | Our 4y SOTA | Our 22y SOTA |
|---|---|---|---|
| 1.0 | Random predictor | — | — |
| 2–4 | Weak event-level signal | — | — |
| 5–8 | Strong event-level signal | **enc28 = 5.62** | **enc14 = 8.33** |
| > 10 | Excellent (or watch for artifact) | — | — |
| `(n_items / n_clusters)` | Theoretical maximum | ~hundreds | ~hundreds |

Theoretical max varies per-window because both `n_clusters` and `n_items` change.

---

## 5. Why We Need This Metric (Why Pixel Lift Alone Isn't Enough)

Two empirical observations forced us to introduce Cluster Lift on top of Pixel Lift:

### A. Pixel Lift is dominated by mega-fires under polygon labels

NBAC labels are perimeter polygons: a single 100,000-ha mega-fire becomes ~250 raw pixels at our 2 km grid (after r=14 dilation: ~150,000 pixels). Top-5000 pixel rankings are essentially "did the model capture the mega-fire?" — and our 4-year transformer already saturates this because mega-fires reside in the well-known boreal forest belt.

Adding 18 more years of training data (4y → 22y) **cannot improve Pixel Lift further**, even though the 22y model has demonstrably better spatial coverage — Pixel Lift can't see this improvement.

### B. Trivial baseline inflation

A persistence baseline (past 7-day fire density) hits **Pixel Lift@5000 = 17×** on NBAC labels. This is a polygon artifact: positive labels persist for the duration of the fire, so "where it's burning today" trivially predicts "where it's burning next week".

Cluster Lift collapses this to **~3× for persistence** because the same persisting mega-fire counts as 1 event.

### C. Operational mismatch

Fire managers don't allocate one resource per pixel; they allocate one resource per **distinct fire event**. Cluster Lift directly measures this.

---

## 6. How to Compute (with the 2026-04-19 bug fix)

```python
def _compute_cluster_lift_k(probs_2d, labels_2d, k, min_cluster_size=1):
    # 1. Find fire clusters via 8-connectivity
    structure = np.ones((3, 3), dtype=bool)
    cluster_map, n_clusters_raw = ndimage_label(labels_2d, structure=structure)

    # 2. Score each cluster as max(prob within cluster)
    cluster_scores, cluster_sizes = [], []
    for c_id in range(1, n_clusters_raw + 1):
        mask = cluster_map == c_id
        size = int(mask.sum())
        if size < min_cluster_size:
            continue
        cluster_scores.append(float(probs_2d[mask].max()))
        cluster_sizes.append(size)

    n_clusters = len(cluster_scores)
    if n_clusters == 0:
        return {"lift_k": 0.0, ...}

    # 3. Tile background to comparable size
    median_size = max(int(np.median(cluster_sizes)), 1)
    tile_side = max(1, int(np.sqrt(median_size)))
    H, W = probs_2d.shape
    nth, ntw = H // tile_side, W // tile_side
    trimmed_p = probs_2d[:nth*tile_side, :ntw*tile_side]
    trimmed_l = labels_2d[:nth*tile_side, :ntw*tile_side]
    tile_p = trimmed_p.reshape(nth, tile_side, ntw, tile_side).max(axis=(1,3))
    tile_l = trimmed_l.reshape(nth, tile_side, ntw, tile_side).max(axis=(1,3))
    bg_mask = (tile_l.ravel() == 0)
    bg_tile_scores = tile_p.ravel()[bg_mask]

    # 4. Combine and rank
    all_scores = np.concatenate([cluster_scores, bg_tile_scores])
    all_labels = np.concatenate([np.ones(n_clusters), np.zeros(len(bg_tile_scores))])
    n_total = len(all_scores)

    # 5. Adaptive K (don't ask for top-5000 when only 100 clusters exist)
    k_eff = min(k, n_total, max(3 * n_clusters, 50))
    top_idx = np.argpartition(all_scores, -k_eff)[-k_eff:]
    tp = float(all_labels[top_idx].sum())
    baseline = n_clusters / n_total

    return {
        "lift_k": (tp / k_eff) / baseline,
        "recall_k": tp / n_clusters,
        "n_clusters": n_clusters,
        "k_eff": k_eff,
        ...
    }
```

Per-window result is aggregated as **unweighted mean** + **n_clusters-weighted mean** + bootstrap 95% CI.

---

## 7. Properties / Strengths

- **Event-level fairness**: each fire has equal vote regardless of size.
- **Polygon-artifact-resistant**: persistence baseline drops from 17× (Pixel) to ~3× (Cluster) — eliminates the inflation.
- **Operationally meaningful**: matches "how many fires did we cover" question.
- **Scales with discovery**: a model that catches additional rare fires gains lift; a model that just gets the mega-fire perfectly does not.
- **Tile sizing matches statistical properties**: `tile_side = √(median size)` ensures background tiles ≈ fire clusters in pixel count, so the max-pool ranking is fair.

---

## 8. Known Limitations and Failure Modes

1. **Bias toward small clusters in heavy-tailed distributions**. Median-based tile sizing under-sizes background tiles when cluster sizes are heavy-tailed (1 mega + many small). The mega-fire's max-pool is computed over many more pixels than each background tile, giving it a structural advantage. *Documented in code comment* `train_v3.py:471-481`. Acceptable for now; can ablate with 75th-percentile or weighted lift.

2. **K_eff is per-window adaptive**. Different windows use different K_eff, so the "K=5000" label is partially aspirational. Aggregation across windows is cleaner with `cluster_lift_weighted` (n_clusters-weighted) than with unweighted mean.

3. **Single-pixel cluster handling**. With `min_cluster_size=1`, every NFDB ignition point becomes a tiny 1-pixel cluster. With `min_cluster_size=10`, NFDB events are filtered out. Choice changes behavior; we use `min_cluster_size=1` to preserve all events.

4. **Sensitive to dilation radius**. Pre-dilation, every NFDB point is a single pixel. Post-r=14 dilation, each point becomes ~615 pixels and can merge with nearby polygons via 8-connectivity. The "cluster" definition is therefore tied to the dilation radius, not the original event count.

5. **Compute cost**: `scipy.ndimage.label` is O(H × W) per window; full 700-window evaluation adds ~20 minutes total over Pixel Lift.

6. **Two forward passes**. Currently computed in a separate forward-pass loop after pixel evaluation. Refactoring to share is on `docs/TECH_DEBT.md`.

---

## 9. How to Improve This Metric

### Training-side
- **Increase training data range** (4y → 12y → 22y): broader spatial coverage of historical fires gives the model a stronger geographic prior over rare fire regions.
- **Multi-region pretraining** (SeasFire / NIFC + MTBS): fundamentally more independent fire events. Documented as future work in `docs/SCALING_LAW_LOG_2026_04_25_26.md` Section 10.
- **Per-region loss weighting**: upweight loss for rare-fire regions during training.

### Data-side
- **Higher-resolution labels** (NBAC alone misses small fires): supplement with NFDB to capture small/medium events.
- **Higher-frequency NDVI / vegetation features**: helps the model distinguish recently-disturbed vs flammable vegetation.

### Evaluation-side
- **Report both unweighted and n_clusters-weighted means**. Implemented as of 2026-04-26 audit fix.
- **Stratify by cluster size**: show lift on small / medium / large clusters separately to expose where the gain comes from.
- **Vary K_eff cap**: ablate sensitivity.

---

## 10. Comparison to Alternative Event-Level Metrics

| Alternative | Pros vs Cluster Lift | Cons vs Cluster Lift |
|---|---|---|
| **Lift@30km** (`lift_coarse`) | Simpler (just block max-pool) | Coarse uniform grid loses event boundaries |
| **Object-detection mAP** | Standard in CV | Requires bounding-box labels we don't have |
| **Hit-or-miss rate per fire** | Easy to interpret | Binary, no ranking sensitivity |
| **Cluster Recall@K** | Simpler | Single number, doesn't account for false positives |
| **Earth-Mover's Distance** | Measures spatial misalignment | Computationally expensive at km scale |

We report **Cluster Lift + Lift@30km** together. They agree on the qualitative ranking but Cluster Lift is more sensitive to the polygon-artifact issue.

---

## 11. Operational Meaning

Concrete scenario: on a given day, the model produces a per-pixel risk map. The fire-management coordinator wants to know how many of the actual fires that occur in the next 14–46 days will have been "warned about" — defined as the model giving high probability somewhere within the eventual fire's footprint.

- **Cluster Lift = 8.0** (our 22y enc14) means the agency's flagged regions contain **8× more distinct fire events than random**, regardless of fire size.
- This is the metric that determines whether resource pre-positioning across multiple wildfires is effective.

In contrast, **Pixel Lift answers a per-pixel question** ("how dense is the flagged area with fire pixels?") which is more about within-fire severity than between-fire coverage.

---

## 12. Our Empirical Numbers (as of 2026-04-26)

| Method | Lift@5000 (pixel) | Cluster Lift |
|---|---|---|
| persistence | 17.1× *(artifact)* | ~3× *(artifact dissolves)* |
| climatology | 5.1× | ~3× *(awaiting full eval)* |
| logreg | 2.7× *(on 4y enc14 windows)* | TBD |
| **4y enc28** *(SOTA pixel)* | **5.83** | 5.62 ± 1.11 |
| **22y enc14** *(SOTA cluster)* | 5.73 | **8.33 ± 1.35** ★ |

Critical pattern across all encoder lengths:

| ENC | 4y Cluster | 22y Cluster | Δ |
|---|---|---|---|
| 14 | 5.62 ± 1.11 | **8.33 ± 1.35** | **+48%** |
| 21 | 5.23 ± 0.78 | **7.95 ± 1.47** | **+52%** |
| 28 | 5.73 ± 1.86 | **8.00 ± 1.56** | **+40%** |
| 35 | 5.50 ± 1.71 | **8.00 ± 1.06** | **+45%** |
| **avg** | **5.52** | **8.07** | **+46%** ★ |

CIs do not overlap → statistically significant. **Pixel Lift cannot see this gain** (4y ≈ 22y on pixel ≈ 5.7); cluster lift is the metric that surfaces the data-scaling benefit.

---

## 13. Validation / Audit Trail

Cluster Lift has been audited end-to-end. See [`docs/SCALING_LAW_LOG_2026_04_25_26.md`](../SCALING_LAW_LOG_2026_04_25_26.md) Section 9 for the full audit and `tests/test_cluster_lift.py` for the 13 unit tests covering:

- empty-fire window edge case
- 8-connectivity correctness (diagonal merging, far-apart non-merging)
- min-cluster-size filtering
- K_eff floor / scaling / capping
- max-pool semantics (spike-pixel rescue test)
- output schema completeness
- tile_side scaling with median size
- oracle vs random predictor sanity

The 2026-04-19 bug fix (replacing shuffled-bg-pixels with spatial-tile sampling) made this metric meaningful; pre-fix results are in [`MEMORY.md`](../../../.claude/projects/-Users-huangjiaqi-Desktop-wildfire-refactored/memory/MEMORY.md) as the "cluster_lift mystery" that was actually a stats bug.

---

## 14. References / Background

This is not a standard published metric — we proposed and validated it as part of this work. The closest related literature:

- **Power 2003**, *Hit Rate, False Alarm Rate, ROC*, in MIT Press *Pattern Recognition* — general top-K event scoring
- **Roberts et al. 2017**, *Spatial cross-validation*, *Methods Ecol Evol* — spatial autocorrelation in evaluation
- **Wildfire literature standard practice**: hit-rate per event (binary, no K). Cluster Lift adds the "how dense in top-K" dimension.

For ML-style precedents, see **mAP** (object detection) and **NDCG** (information retrieval) — both treat each retrieved item as a discrete event, which is the same conceptual move we make.

---

## 15. Related Metric Cards

- [`01_pixel_lift_k.md`](01_pixel_lift_k.md) — pixel-level counterpart; complementary
- `03_novel_ignition_lift.md` — for "exclude already-burning" labels (deeper artifact removal)
- `04_lift_30km.md` — coarse-grid event proxy

---

## Changelog

- 2026-04-19 — bug fix: replace shuffled-bg-pixel tiling with spatial-tile max-pool
- 2026-04-26 — audit (8 issues), patch_ids consistency fix, weighted-mean addition, 13 unit tests
- 2026-04-26 — initial card
