"""
Tests for FocalBCELoss reduction modes (added 2026-04-30 for recency weighting).
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.losses import FocalBCELoss


def test_focal_default_is_mean():
    """Default reduction='mean' returns scalar (backward compat)."""
    loss_fn = FocalBCELoss()
    logits = torch.randn(4, 33, 256)
    targets = torch.randint(0, 2, logits.shape).float()
    out = loss_fn(logits, targets)
    assert out.dim() == 0, f"Default 'mean' should be scalar, got shape {out.shape}"


def test_focal_none_returns_per_element():
    """reduction='none' returns per-element loss."""
    loss_fn = FocalBCELoss(reduction="none")
    logits = torch.randn(4, 33, 256)
    targets = torch.randint(0, 2, logits.shape).float()
    out = loss_fn(logits, targets)
    assert out.shape == logits.shape, \
        f"'none' should match logits shape, got {out.shape}"


def test_focal_sample_mean_returns_per_sample():
    """reduction='sample_mean' returns shape (B,) — one scalar per batch sample."""
    loss_fn = FocalBCELoss(reduction="sample_mean")
    B = 4
    logits = torch.randn(B, 33, 256)
    targets = torch.randint(0, 2, logits.shape).float()
    out = loss_fn(logits, targets)
    assert out.shape == (B,), f"'sample_mean' should be (B,), got {out.shape}"


def test_recency_weighted_loss_changes_with_weights():
    """Weighted loss differs when weights differ."""
    loss_fn = FocalBCELoss(reduction="sample_mean")
    torch.manual_seed(0)
    B = 8
    logits = torch.randn(B, 33, 256)
    targets = torch.randint(0, 2, logits.shape).float()

    # Equal weights -> matches default mean (up to numerics)
    sw_uniform = torch.ones(B)
    loss_uniform = (loss_fn(logits, targets) * sw_uniform).mean()

    # Recency-like decay weights (more weight on later samples)
    sw_recency = torch.exp(-torch.arange(B).float() / 2.0).flip(0)  # last sample has weight 1.0
    sw_recency = sw_recency / sw_recency.mean()                      # normalize
    loss_recency = (loss_fn(logits, targets) * sw_recency).mean()

    # Should differ since weights are different
    assert not torch.allclose(loss_uniform, loss_recency, atol=1e-4), \
        "Uniform vs recency-weighted should differ"


def test_focal_mean_equals_sample_mean_then_mean():
    """Mathematical consistency: mean over all == sample_mean → mean."""
    torch.manual_seed(42)
    B = 4
    logits = torch.randn(B, 33, 256)
    targets = torch.randint(0, 2, logits.shape).float()

    loss_mean = FocalBCELoss(reduction="mean")(logits, targets)
    loss_sm = FocalBCELoss(reduction="sample_mean")(logits, targets).mean()

    assert torch.allclose(loss_mean, loss_sm, atol=1e-6), \
        f"reduction='mean' should equal sample_mean.mean(), got {loss_mean} vs {loss_sm}"


if __name__ == "__main__":
    test_focal_default_is_mean()
    test_focal_none_returns_per_element()
    test_focal_sample_mean_returns_per_sample()
    test_recency_weighted_loss_changes_with_weights()
    test_focal_mean_equals_sample_mean_then_mean()
    print("All recency-loss tests passed.")
