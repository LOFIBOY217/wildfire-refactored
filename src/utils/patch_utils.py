"""
Patchify / Depatchify Utilities
================================
Convert spatial grids to/from patch sequences for Transformer input.

Uses the most general version (from train_s2s_transformer.py) that supports
both 3D (D, H, W) and 4D (D, H, W, C) inputs.
"""

import numpy as np
import torch


def patchify(frames, patch_size):
    """
    Convert frame sequence to patch sequences.

    Args:
        frames: (D, H, W) or (D, H, W, C) numpy array
        patch_size: Patch edge length

    Returns:
        patches: (num_patches, D, patch_size*patch_size*C) numpy array
        hw: (Hc, Wc) cropped spatial dimensions
        grid: (nph, npw) patch grid dimensions
    """
    if frames.ndim == 3:
        frames = frames[..., np.newaxis]

    D, H, W, C = frames.shape

    # Crop to patch-aligned dimensions
    Hc = H - (H % patch_size)
    Wc = W - (W % patch_size)
    frames = frames[:, :Hc, :Wc, :]

    nph, npw = Hc // patch_size, Wc // patch_size

    # Reshape: (D, nph, patch, npw, patch, C) -> (nph, npw, D, patch, patch, C)
    x = frames.reshape(D, nph, patch_size, npw, patch_size, C)
    x = x.transpose(1, 3, 0, 2, 4, 5)  # (nph, npw, D, patch, patch, C)

    # Flatten: (nph*npw, D, patch*patch*C)
    x = x.reshape(nph * npw, D, patch_size * patch_size * C)

    return x, (Hc, Wc), (nph, npw)


def depatchify(patches, grid, patch_size, hw, num_channels=1):
    """
    Convert patch sequences back to frame sequence.

    Args:
        patches: (num_patches, D, patch_size*patch_size*C) numpy or torch tensor
        grid: (nph, npw) patch grid dimensions
        patch_size: Patch edge length
        hw: (Hc, Wc) target spatial dimensions
        num_channels: Number of channels C

    Returns:
        frames: (D, Hc, Wc, C) numpy array. If C==1, squeezed to (D, Hc, Wc).
    """
    if isinstance(patches, torch.Tensor):
        patches = patches.cpu().numpy()

    nph, npw = grid
    num_patches, D, _ = patches.shape

    # Reshape: (nph, npw, D, patch, patch, C)
    x = patches.reshape(nph, npw, D, patch_size, patch_size, num_channels)

    # Transpose: (D, nph, patch, npw, patch, C)
    x = x.transpose(2, 0, 3, 1, 4, 5)

    # Reshape: (D, nph*patch, npw*patch, C)
    x = x.reshape(D, nph * patch_size, npw * patch_size, num_channels)

    Hc, Wc = hw
    result = x[:, :Hc, :Wc, :]

    # Squeeze channel dim if single-channel
    if num_channels == 1:
        result = result[..., 0]

    return result


def build_windows(frames, in_days, out_days):
    """
    Build sliding windows for time series prediction.

    Args:
        frames: Array with first dimension as time
        in_days: Number of input days
        out_days: Number of output days

    Returns:
        List of (start, mid, end) index tuples
    """
    T = frames.shape[0]
    windows = []
    for t in range(T - in_days - out_days + 1):
        windows.append((t, t + in_days, t + in_days + out_days))
    return windows
