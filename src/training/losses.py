"""
Loss functions for wildfire prediction V3 training.

Provides:
  - FocalBCELoss: Focal modulation of BCE for hard-example mining
  - ApproxNDCGLoss: Differentiable NDCG surrogate for ranking optimisation
  - HybridLoss: Weighted combination of focal + ranking losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalBCELoss(nn.Module):
    """Focal Binary Cross-Entropy Loss.

    Down-weights well-classified (easy) samples so the model focuses on
    hard negatives near the decision boundary.  Particularly effective
    for extreme class imbalance (fire pixels << background pixels).

    Parameters
    ----------
    alpha : float
        Weighting factor for the positive class.  ``1 - alpha`` is applied
        to negatives.  Default 0.25.
    gamma : float
        Focusing parameter.  ``gamma = 0`` recovers standard BCE.
        Higher values increase down-weighting of easy samples.  Default 2.0.
    pos_weight : torch.Tensor | None
        Optional additional positive-class weight (same semantics as
        ``nn.BCEWithLogitsLoss(pos_weight=...)``).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        # Store pos_weight as a buffer so it moves with .to(device)
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Element-wise BCE (no reduction)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # p_t = probability assigned to the *correct* class
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)

        # Focal modulation: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t).pow(self.gamma)

        # Alpha balance: alpha for positives, (1-alpha) for negatives
        alpha_weight = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        loss = alpha_weight * focal_weight * bce

        # Optional pos_weight for additional class-imbalance correction
        if self.pos_weight is not None:
            pw = targets * (self.pos_weight - 1.0) + 1.0
            loss = loss * pw

        return loss.mean()


class ApproxNDCGLoss(nn.Module):
    """Differentiable surrogate for NDCG (Normalised Discounted Cumulative Gain).

    Approximates the non-differentiable rank function using a sigmoid on
    pairwise score differences, then computes a smooth DCG.  Optimising
    this loss directly encourages the model to rank fire pixels above
    background pixels — matching the Top-K evaluation metric.

    To keep memory manageable (the pairwise matrix is O(N^2)), a random
    subsample of elements is used each forward pass.

    Parameters
    ----------
    temperature : float
        Controls sigmoid sharpness for rank approximation.  Lower values
        produce sharper (closer to true rank) but noisier gradients.
    subsample : int
        Maximum number of elements to use for pairwise comparison.
    """

    def __init__(self, temperature: float = 1.0, subsample: int = 50_000):
        super().__init__()
        self.temperature = temperature
        self.subsample = subsample

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Flatten to 1-D
        s = logits.reshape(-1)
        y = targets.reshape(-1)

        n = s.size(0)
        if n == 0:
            return s.new_tensor(0.0)

        # Subsample to bound memory at O(K^2)
        if n > self.subsample:
            idx = torch.randperm(n, device=s.device)[: self.subsample]
            s = s[idx]
            y = y[idx]
            n = self.subsample

        # Skip if no positive labels in subsample (NDCG undefined)
        if y.sum() < 1.0:
            return s.new_tensor(0.0)

        # Approximate ranks via sigmoid on pairwise score differences
        # rank_i ≈ sum_j sigmoid((s_j - s_i) / temperature) + 1
        diff = s.unsqueeze(0) - s.unsqueeze(1)  # (K, K): diff[j, i] = s_j - s_i
        approx_rank = torch.sigmoid(diff / self.temperature).sum(dim=0) + 1.0  # (K,)

        # DCG with approximate ranks
        discount = 1.0 / torch.log2(approx_rank + 1.0)
        dcg = (y * discount).sum()

        # Ideal DCG (sorted by true labels — deterministic)
        k = n
        ideal_rank = torch.arange(1, k + 1, device=s.device, dtype=s.dtype)
        ideal_discount = 1.0 / torch.log2(ideal_rank + 1.0)
        sorted_y, _ = y.sort(descending=True)
        idcg = (sorted_y * ideal_discount).sum()

        ndcg = dcg / (idcg + 1e-8)
        return 1.0 - ndcg


class HybridLoss(nn.Module):
    """Weighted combination of Focal BCE and ApproxNDCG losses.

    ``loss = (1 - rank_weight) * focal + rank_weight * ranking``

    Parameters
    ----------
    rank_weight : float
        Weight given to the ranking component.  Default 0.3.
    focal_alpha, focal_gamma : float
        Passed to ``FocalBCELoss``.
    pos_weight : torch.Tensor | None
        Passed to ``FocalBCELoss``.
    rank_temperature : float
        Passed to ``ApproxNDCGLoss``.
    rank_subsample : int
        Passed to ``ApproxNDCGLoss``.
    """

    def __init__(self, rank_weight: float = 0.3,
                 focal_alpha: float = 0.25, focal_gamma: float = 2.0,
                 pos_weight: torch.Tensor | None = None,
                 rank_temperature: float = 1.0, rank_subsample: int = 50_000):
        super().__init__()
        self.rank_weight = rank_weight
        self.focal = FocalBCELoss(alpha=focal_alpha, gamma=focal_gamma,
                                  pos_weight=pos_weight)
        self.ranking = ApproxNDCGLoss(temperature=rank_temperature,
                                      subsample=rank_subsample)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal_loss = self.focal(logits, targets)
        rank_loss = self.ranking(logits, targets)
        return (1.0 - self.rank_weight) * focal_loss + self.rank_weight * rank_loss
