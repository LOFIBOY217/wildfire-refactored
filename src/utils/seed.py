"""
Random Seed Utility
===================
Set random seeds for reproducibility across Python, NumPy, and PyTorch.
"""

import random
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
