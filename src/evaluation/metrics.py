"""
Evaluation Metrics
==================
Confusion matrix and derived metrics for wildfire forecast verification.

Metrics:
    - POD (Probability of Detection / Hit Rate)
    - FAR (False Alarm Ratio)
    - CSI (Critical Success Index / Threat Score)
    - Bias Score
    - Precision, F1 Score
    - Brier Score (probability-based)
    - AUC-ROC

Based on evaluate_with_confusion_matrix.py.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, brier_score_loss


def compute_confusion_metrics(y_true, y_pred_prob, threshold, nodata_mask=None, skip_auc=False):
    """
    Compute confusion matrix and derived metrics.

    Args:
        y_true: [N] binary labels (0/1)
        y_pred_prob: [N] predicted probabilities
        threshold: Probability threshold for binary classification
        nodata_mask: [N] boolean mask for valid pixels (True = valid)
        skip_auc: If True, skip roc_auc_score (caller will supply auc externally).
                  Use when AUC is computed once from a sub-sampled array to avoid
                  sorting the full ~6M-element array for every threshold.

    Returns:
        dict with confusion matrix and metrics, or None if no valid data
    """
    if nodata_mask is not None:
        y_true = y_true[nodata_mask]
        y_pred_prob = y_pred_prob[nodata_mask]

    valid = np.isfinite(y_true) & np.isfinite(y_pred_prob)
    y_true = y_true[valid]
    y_pred_prob = y_pred_prob[valid]

    if len(y_true) == 0:
        return None

    y_pred_binary = (y_pred_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()

    # POD (Probability of Detection) = Hit Rate = Recall
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # FAR (False Alarm Ratio)
    far = fp / (fp + tp) if (fp + tp) > 0 else 0.0

    # CSI (Critical Success Index) = Threat Score
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    # Bias Score
    bias = (tp + fp) / (tp + fn) if (tp + fn) > 0 else 0.0

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # F1 Score
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    # Brier Score
    brier = brier_score_loss(y_true, y_pred_prob)

    # AUC-ROC (skip when caller supplies a pre-computed sub-sampled value)
    if skip_auc:
        auc = float("nan")
    elif len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_pred_prob)
    else:
        auc = np.nan

    return {
        'threshold': threshold,
        'n_samples': len(y_true),
        'n_fires': int(y_true.sum()),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'pod': float(pod),
        'far': float(far),
        'csi': float(csi),
        'bias': float(bias),
        'precision': float(precision),
        'f1': float(f1),
        'brier': float(brier),
        'auc': float(auc)
    }


# ============================================================================
# Rare-event / imbalanced-classification metrics (2026-04-17)
#
# Pure functions — depend only on (scores, labels), no model/training state.
# Designed to be called from:
#   - Training val loop (train_s2s_hotspot_cwfis_v2._compute_val_lift_k)
#   - Standalone eval (train_v3 --eval_checkpoint)
#   - Baseline eval (evaluation.benchmark_baselines)
#   - Any post-hoc analysis notebook
#
# See docs/ANALYSIS_PLAN.md for rationale and literature references.
# ============================================================================


def compute_ranking_metrics(scores, labels, k):
    """Top-K ranking metrics (single K).

    Args:
        scores: (N,) float — predicted probabilities
        labels: (N,) binary — 0/1 ground truth
        k:      int — top-K cutoff (will be clamped to N if larger)

    Returns dict with:
        lift_k, precision_k, recall_k, csi_k, ets_k,
        n_total, n_fire, baseline, tp, k_eff
    """
    scores = np.asarray(scores, dtype=np.float32).ravel()
    labels = np.asarray(labels, dtype=np.float32).ravel()
    n_total = len(scores)
    n_fire = int(labels.sum())

    if n_total == 0 or n_fire == 0:
        return dict(lift_k=0.0, precision_k=0.0, recall_k=0.0,
                    csi_k=0.0, ets_k=0.0,
                    n_total=n_total, n_fire=n_fire, baseline=0.0,
                    tp=0, k_eff=0)

    baseline = n_fire / n_total
    k_eff = min(k, n_total)
    top_idx = np.argpartition(scores, -k_eff)[-k_eff:]
    tp = float(labels[top_idx].sum())
    fp = k_eff - tp
    fn = n_fire - tp
    precision_k = tp / k_eff
    recall_k = tp / n_fire
    lift_k = precision_k / baseline if baseline > 0 else 0.0
    csi_k = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    tp_random = k_eff * baseline
    denom = tp + fp + fn - tp_random
    ets_k = (tp - tp_random) / denom if denom > 0 else 0.0

    return dict(lift_k=float(lift_k), precision_k=float(precision_k),
                recall_k=float(recall_k), csi_k=float(csi_k),
                ets_k=float(ets_k),
                n_total=int(n_total), n_fire=int(n_fire),
                baseline=float(baseline),
                tp=int(tp), k_eff=int(k_eff))


def compute_imbalanced_metrics(scores, labels):
    """F1 / F2 / MCC at F1-optimal threshold + PR-AUC + ROC-AUC.

    Rare-event standards per MDPI 2025 / Sofaer 2019.
    """
    from sklearn.metrics import (average_precision_score, roc_auc_score,
                                 fbeta_score, matthews_corrcoef,
                                 precision_recall_curve)
    scores = np.asarray(scores, dtype=np.float32).ravel()
    labels = np.asarray(labels, dtype=np.float32).ravel()

    if labels.sum() == 0 or labels.sum() == len(labels):
        return dict(pr_auc=0.0, roc_auc=0.0,
                    f1=0.0, f2=0.0, mcc=0.0,
                    optimal_threshold=0.5)

    try:
        pr_auc = float(average_precision_score(labels, scores))
    except Exception:
        pr_auc = 0.0
    try:
        roc_auc = float(roc_auc_score(labels, scores))
    except Exception:
        roc_auc = 0.0

    try:
        precs, recs, thrs = precision_recall_curve(labels, scores)
        f1s = 2 * precs * recs / (precs + recs + 1e-12)
        opt_idx = int(np.argmax(f1s[:-1]))
        opt_thr = float(thrs[opt_idx]) if opt_idx < len(thrs) else 0.5
        y_pred = (scores >= opt_thr).astype(np.int32)
        y_int = labels.astype(np.int32)
        f1 = float(fbeta_score(y_int, y_pred, beta=1, zero_division=0))
        f2 = float(fbeta_score(y_int, y_pred, beta=2, zero_division=0))
        mcc = float(matthews_corrcoef(y_int, y_pred))
    except Exception:
        f1 = f2 = mcc = 0.0
        opt_thr = 0.5

    return dict(pr_auc=pr_auc, roc_auc=roc_auc,
                f1=f1, f2=f2, mcc=mcc,
                optimal_threshold=opt_thr)


def compute_brier_decomposition(scores, labels, n_bins=10):
    """Brier + Reliability/Resolution decomposition + BSS.

    Brier Skill Score (BSS) vs climatology reference:
        BSS = 1 - Brier_model / Brier_climatology
    where Brier_climatology = p(1-p), the Brier of always-predict-baseline.
    BSS > 0 means the model beats climatology.

    Reliability: lower = better calibrated (forecast probs match observed freq).
    Resolution:  higher = sharper (useful discrimination).
    """
    scores = np.asarray(scores, dtype=np.float32).ravel()
    labels = np.asarray(labels, dtype=np.float32).ravel()
    n_total = len(scores)

    if n_total == 0:
        return dict(brier=0.0, reliability=0.0, resolution=0.0,
                    bss=0.0, brier_climatology=0.0)

    brier = float(np.mean((scores - labels) ** 2))
    y_mean = float(labels.mean())
    brier_clim = y_mean * (1 - y_mean)
    bss = 1.0 - brier / brier_clim if brier_clim > 0 else 0.0

    # Reliability + Resolution decomposition (Murphy 1973)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(scores, bin_edges) - 1, 0, n_bins - 1)
    reliability, resolution = 0.0, 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        p_b = float(scores[mask].mean())
        y_b = float(labels[mask].mean())
        reliability += (n_b / n_total) * (p_b - y_b) ** 2
        resolution += (n_b / n_total) * (y_b - y_mean) ** 2

    return dict(brier=brier, reliability=reliability, resolution=resolution,
                bss=float(bss), brier_climatology=float(brier_clim))


def compute_coarsened_lift(scores_2d, labels_2d, factor, k_fine=5000):
    """30km-equivalent Lift after spatial coarsening — removes spatial autocor.

    Args:
        scores_2d:  (H, W) probability array
        labels_2d:  (H, W) binary label (fire anywhere in this cell)
        factor:     aggregation factor (e.g., 15 → 15×15 cells, 2km → 30km)
        k_fine:     K used at fine (2km) resolution; auto-scaled to coarse

    Aggregation: mean(prob), max(label) within each coarse cell.

    Returns: dict(lift_coarse, baseline_coarse, n_fire_coarse, k_coarse)
    """
    scores_2d = np.asarray(scores_2d)
    labels_2d = np.asarray(labels_2d)

    if scores_2d.ndim != 2 or labels_2d.ndim != 2 or factor <= 1:
        return dict(lift_coarse=0.0, baseline_coarse=0.0,
                    n_fire_coarse=0, k_coarse=0)

    H, W = scores_2d.shape
    h2, w2 = H // factor, W // factor
    if h2 == 0 or w2 == 0:
        return dict(lift_coarse=0.0, baseline_coarse=0.0,
                    n_fire_coarse=0, k_coarse=0)

    p = scores_2d[:h2 * factor, :w2 * factor]
    y = labels_2d[:h2 * factor, :w2 * factor]
    p_c = p.reshape(h2, factor, w2, factor).mean(axis=(1, 3))
    y_c = y.reshape(h2, factor, w2, factor).max(axis=(1, 3))

    p_flat = p_c.ravel()
    y_flat = y_c.ravel().astype(np.float32)
    n_total_c = y_flat.size
    n_fire_c = int(y_flat.sum())

    if n_fire_c == 0:
        return dict(lift_coarse=0.0, baseline_coarse=0.0,
                    n_fire_coarse=0, k_coarse=0)

    baseline_c = n_fire_c / n_total_c
    k_c = max(1, k_fine // (factor * factor))
    k_c = min(k_c, n_total_c)
    top_idx = np.argpartition(p_flat, -k_c)[-k_c:]
    prec_c = y_flat[top_idx].sum() / k_c
    lift_c = prec_c / baseline_c if baseline_c > 0 else 0.0

    return dict(lift_coarse=float(lift_c),
                baseline_coarse=float(baseline_c),
                n_fire_coarse=int(n_fire_c),
                k_coarse=int(k_c))


def compute_all_metrics(scores, labels, k_values=(5000,),
                        spatial_shape=None, coarsen_factor=15):
    """One-stop metric computation for a single (window or aggregated) sample.

    Args:
        scores:         1D (N,) or 2D (H, W) array of probabilities
        labels:         1D (N,) or 2D (H, W) binary array (same shape as scores)
        k_values:       iterable of K values for top-K metrics
        spatial_shape:  (H, W) for coarsening. If None, attempts to infer from
                        scores.shape if it's 2D. If still unavailable, skips
                        coarsening.
        coarsen_factor: downsample factor for coarsened Lift (default 15 = 30km)

    Returns:
        flat dict with all metrics. Top-K metrics are namespaced by K:
          lift_k, precision_k, ... refer to k_values[0] (primary K)
          lift_k_5000, lift_k_1000, ... for all K
        Other metrics: pr_auc, roc_auc, f1, f2, mcc, bss, brier,
          reliability, resolution, lift_coarse
    """
    scores_arr = np.asarray(scores)
    labels_arr = np.asarray(labels)

    # Determine spatial shape for coarsening
    if spatial_shape is None and scores_arr.ndim == 2:
        spatial_shape = scores_arr.shape

    # Flatten for scalar metrics
    scores_flat = scores_arr.ravel()
    labels_flat = labels_arr.ravel().astype(np.float32)

    result = {}

    # Top-K ranking family (one set per K)
    k_list = list(k_values) if not isinstance(k_values, int) else [k_values]
    primary_k = k_list[0]
    for k in k_list:
        rk = compute_ranking_metrics(scores_flat, labels_flat, k)
        if k == primary_k:
            # Unnamespaced primary
            for key, val in rk.items():
                result[key] = val
        # Also namespaced
        for key in ["lift_k", "precision_k", "recall_k", "csi_k", "ets_k"]:
            result[f"{key}_{k}"] = rk[key]

    # Threshold-free + imbalanced metrics
    imb = compute_imbalanced_metrics(scores_flat, labels_flat)
    result.update(imb)

    # Brier + decomposition + BSS
    brier = compute_brier_decomposition(scores_flat, labels_flat)
    result.update(brier)

    # Coarsened Lift (if possible)
    if spatial_shape is not None:
        H, W = spatial_shape
        if scores_flat.size == H * W:
            coarse = compute_coarsened_lift(
                scores_flat.reshape(H, W),
                labels_flat.reshape(H, W),
                coarsen_factor, primary_k)
            result.update(coarse)
        else:
            result.update(dict(lift_coarse=0.0, baseline_coarse=0.0,
                               n_fire_coarse=0, k_coarse=0))
    else:
        result.update(dict(lift_coarse=0.0, baseline_coarse=0.0,
                           n_fire_coarse=0, k_coarse=0))

    return result


def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=42):
    """Bootstrap 95% CI for a list of per-sample metric values.

    Args:
        values: list of scalar metric values (one per val window)
        n_boot: number of bootstrap resamples
        alpha:  1 - confidence level (0.05 → 95% CI)

    Returns:
        dict(mean, ci_low, ci_high)
    """
    vals = np.asarray(values, dtype=np.float64)
    if len(vals) == 0:
        return dict(mean=0.0, ci_low=0.0, ci_high=0.0)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    N = len(vals)
    for i in range(n_boot):
        idx = rng.integers(0, N, N)
        boots[i] = vals[idx].mean()
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return dict(mean=float(vals.mean()), ci_low=lo, ci_high=hi)
