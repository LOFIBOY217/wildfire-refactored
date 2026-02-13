"""
Normalization Utilities
=======================
Data standardization with safety checks for NaN/Inf/extreme values.
"""

import numpy as np


def standardize(train_frames, all_frames, clip_range=(-10, 10)):
    """
    Standardize data using training set statistics.

    Assumes NoData has already been cleaned (no NaN in input).

    Args:
        train_frames: numpy array used to compute mean and std
        all_frames: numpy array to standardize
        clip_range: (min, max) tuple to clip standardized values

    Returns:
        standardized: Standardized array
        mu: Mean used for standardization
        sd: Standard deviation used for standardization
    """
    mu = float(np.mean(train_frames))
    sd = float(np.std(train_frames) + 1e-6)

    # Safety checks
    if np.isnan(mu) or np.isinf(mu):
        print(f"[Warning] Standardization mean is abnormal ({mu}), using 0")
        mu = 0.0
    if np.isnan(sd) or np.isinf(sd) or sd < 1e-6:
        print(f"[Warning] Standardization std is abnormal ({sd}), using 1")
        sd = 1.0

    print(f"[Standardization] mean: {mu:.4f}, std: {sd:.4f}")

    standardized = (all_frames - mu) / sd

    # Clip extreme values
    if clip_range is not None:
        standardized = np.clip(standardized, clip_range[0], clip_range[1])

    return standardized, mu, sd
