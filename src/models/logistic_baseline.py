"""
Logistic Regression Baseline for Wildfire Prediction
=====================================================
Uses 3 engineered features from a 7-day history window:
    1. fwi_max_norm: Max FWI (clipped to 150, normalized by 30)
    2. dryness_norm: Max dewpoint depression (T - Td), normalized
    3. recent_fire: Binary indicator of fire in history window

Based on simple_logistic_7day.py.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


# Default feature engineering parameters
FWI_MAX_CLIP = 150.0
DRYNESS_OFFSET = 5.0
DRYNESS_SCALE = 15.0


def compute_features(fwi_window, t2m_window, d2m_window, fire_window,
                     fwi_max_clip=FWI_MAX_CLIP,
                     dryness_offset=DRYNESS_OFFSET,
                     dryness_scale=DRYNESS_SCALE):
    """
    Compute features from 7-day history window.

    Args:
        fwi_window: [T, H, W] FWI values
        t2m_window: [T, H, W] 2m temperature (K)
        d2m_window: [T, H, W] 2m dewpoint temperature (K)
        fire_window: [T, H, W] binary fire indicators
        fwi_max_clip: Max FWI clip value
        dryness_offset: Dewpoint depression offset (K)
        dryness_scale: Dewpoint depression scale (K)

    Returns:
        features: [H, W, 3] array
    """
    # Feature 1: Max FWI (clipped and normalized)
    fwi_max = np.max(fwi_window, axis=0)
    fwi_max = np.clip(fwi_max, 0, fwi_max_clip)
    fwi_max_norm = fwi_max / 30.0

    # Feature 2: Max dewpoint depression (T - Td)
    dew_depression = t2m_window - d2m_window  # [T, H, W]
    dryness_max = np.max(dew_depression, axis=0)
    dryness_norm = (dryness_max - dryness_offset) / dryness_scale
    dryness_norm = np.clip(dryness_norm, 0, 5)

    # Feature 3: Recent fire indicator
    recent_fire = (np.sum(fire_window, axis=0) > 0).astype(np.float32)

    return np.stack([fwi_max_norm, dryness_norm, recent_fire], axis=-1)


def sample_training_data(features, labels, n_samples, nodata_value=-9999):
    """
    Sample pixels for training with balanced positive/negative examples.

    Args:
        features: [H, W, 3]
        labels: [H, W] binary
        n_samples: Target number of samples
        nodata_value: Value to exclude

    Returns:
        X: [n, 3] feature samples (or None)
        y: [n] label samples (or None)
    """
    H, W = labels.shape

    valid_mask = (features[:, :, 0] != nodata_value)
    valid_mask &= (labels != nodata_value)

    pos_indices = np.where(valid_mask & (labels == 1))
    neg_indices = np.where(valid_mask & (labels == 0))

    n_pos = len(pos_indices[0])
    n_neg = len(neg_indices[0])

    if n_pos == 0 and n_neg == 0:
        return None, None

    if n_pos > 0:
        pos_samples = min(n_pos, n_samples // 2)
        pos_idx = np.random.choice(n_pos, size=pos_samples, replace=False)

        neg_samples = min(n_neg, n_samples - pos_samples)
        neg_idx = np.random.choice(n_neg, size=neg_samples, replace=False)

        rows = np.concatenate([pos_indices[0][pos_idx], neg_indices[0][neg_idx]])
        cols = np.concatenate([pos_indices[1][pos_idx], neg_indices[1][neg_idx]])
    else:
        n_samples = min(n_neg, n_samples)
        neg_idx = np.random.choice(n_neg, size=n_samples, replace=False)
        rows = neg_indices[0][neg_idx]
        cols = neg_indices[1][neg_idx]

    X = features[rows, cols]  # [n, 3]
    y = labels[rows, cols]    # [n]
    return X, y


def build_logistic_model(max_iter=500, class_weight='balanced', random_state=42):
    """
    Create a configured LogisticRegression model.

    Returns:
        sklearn LogisticRegression instance
    """
    return LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state
    )
