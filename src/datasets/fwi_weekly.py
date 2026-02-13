"""
FWI Weekly Dataset
==================
Sliding-window dataset for 7-day FWI prediction using patch-based Transformer.

Loads FWI GeoTIFF rasters, builds input/output windows, and patchifies
each frame for per-patch temporal modeling.

Based on pytorch_transformer_fwi20260129.py FWIWeeklyDataset.
"""

import math
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.patch_utils import patchify, build_windows


class FWIWeeklyDataset(Dataset):
    """
    Dataset for weekly FWI forecasting.

    Each sample is a pair of patch sequences:
        X: (in_days, patch_dim) - input patch time series
        Y: (out_days, patch_dim) - target patch time series

    Args:
        frames: (T, H, W) standardized FWI frames
        windows: List of (start, mid, end) index tuples from build_windows()
        in_days: Number of input days
        out_days: Number of output days
        patch_size: Patch edge length
        augment: Whether to apply horizontal flip augmentation
    """

    def __init__(self, frames, windows, in_days, out_days, patch_size, augment=False):
        self.frames = frames
        self.windows = windows
        self.in_days = in_days
        self.out_days = out_days
        self.patch_size = patch_size
        self.augment = augment

        # Patchify all windows upfront
        X_list, Y_list = [], []
        self.grid = None
        self.hw = None

        for (s, m, e) in windows:
            X = frames[s:m]  # (in_days, H, W)
            Y = frames[m:e]  # (out_days, H, W)
            Xp, hw, grid = patchify(X, patch_size)
            Yp, _, _ = patchify(Y, patch_size)
            if self.grid is None:
                self.hw, self.grid = hw, grid
            X_list.append(Xp)
            Y_list.append(Yp)

        # Concatenate: each row = one patch trajectory
        self.X = np.concatenate(X_list, axis=0)  # (N, in_days, P*P)
        self.Y = np.concatenate(Y_list, axis=0)  # (N, out_days, P*P)

        # Optional horizontal flip augmentation
        if augment and len(self.X) > 0:
            p = int(math.sqrt(self.X.shape[-1]))
            Xf = self.X.copy().reshape(-1, self.in_days, p, p)
            Yf = self.Y.copy().reshape(-1, self.out_days, p, p)
            Xf = Xf[..., ::-1, :].reshape(self.X.shape)
            Yf = Yf[..., ::-1, :].reshape(self.Y.shape)
            self.X = np.concatenate([self.X, Xf], axis=0)
            self.Y = np.concatenate([self.Y, Yf], axis=0)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])  # (in_days, P*P)
        y = torch.from_numpy(self.Y[idx])  # (out_days, P*P)
        return x, y
