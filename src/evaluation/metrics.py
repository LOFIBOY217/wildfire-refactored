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


def compute_confusion_metrics(y_true, y_pred_prob, threshold, nodata_mask=None):
    """
    Compute confusion matrix and derived metrics.

    Args:
        y_true: [N] binary labels (0/1)
        y_pred_prob: [N] predicted probabilities
        threshold: Probability threshold for binary classification
        nodata_mask: [N] boolean mask for valid pixels (True = valid)

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

    # AUC-ROC
    if len(np.unique(y_true)) > 1:
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
