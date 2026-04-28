"""
Unit tests for _compute_cluster_lift_k in train_v3.

Covers:
  - basic correctness (trivial perfect predictor)
  - empty-fire window
  - single-cluster window
  - mega + small mix (the documented bias case)
  - 8-connectivity correctness
  - K capping logic (k_eff = min(k, n_total, max(3*n_clusters, 50)))
  - Score = max-over-cluster (not mean)
  - Output schema completeness
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.train_v3 import _compute_cluster_lift_k


# ── helpers ───────────────────────────────────────────────────────────────

def _make_map(H=100, W=100, fire_pixels=None):
    """Build a (H, W) label map with fire at given coordinates."""
    label = np.zeros((H, W), dtype=np.uint8)
    if fire_pixels is not None:
        for (r, c) in fire_pixels:
            if 0 <= r < H and 0 <= c < W:
                label[r, c] = 1
    return label


def _perfect_pred(label_2d):
    """Score = label exactly (perfect oracle)."""
    return label_2d.astype(np.float32)


def _random_pred(label_2d, seed=0):
    """Score = uniform random (no signal)."""
    rng = np.random.default_rng(seed)
    return rng.random(label_2d.shape).astype(np.float32)


def _inverted_pred(label_2d):
    """Score = 1 - label (anti-oracle)."""
    return (1 - label_2d).astype(np.float32)


# ── tests ─────────────────────────────────────────────────────────────────

class TestEmptyFire:
    def test_no_fire_returns_zero(self):
        label = _make_map(100, 100, fire_pixels=None)
        prob = np.full_like(label, 0.5, dtype=np.float32)
        result = _compute_cluster_lift_k(prob, label, k=50)
        assert result["n_clusters"] == 0
        assert result["lift_k"] == 0.0
        assert result["precision_k"] == 0.0


class TestSingleCluster:
    def test_perfect_single_blob(self):
        # 10x10 fire blob in 100x100 map
        label = np.zeros((100, 100), dtype=np.uint8)
        label[10:20, 10:20] = 1
        prob = _perfect_pred(label)
        result = _compute_cluster_lift_k(prob, label, k=50)
        assert result["n_clusters"] == 1
        assert result["n_clusters_raw"] == 1
        assert result["recall_k"] == 1.0  # caught the 1 cluster
        assert result["lift_k"] > 0.0

    def test_inverted_predictor_zero_recall(self):
        label = np.zeros((100, 100), dtype=np.uint8)
        label[10:20, 10:20] = 1
        prob = _inverted_pred(label)  # high prob OUTSIDE fire
        result = _compute_cluster_lift_k(prob, label, k=10)
        # Fire cluster has score 0; bg tiles all have score ≈ 1
        # Top 10 should be ALL bg tiles → 0 fire clusters caught
        assert result["recall_k"] == 0.0
        assert result["lift_k"] == 0.0


class TestEightConnectivity:
    def test_diagonal_pixels_merge_into_one_cluster(self):
        label = np.zeros((100, 100), dtype=np.uint8)
        # 4 diagonally-adjacent pixels — should merge with 8-connectivity
        label[50, 50] = 1
        label[51, 51] = 1
        label[52, 52] = 1
        label[53, 53] = 1
        prob = _perfect_pred(label)
        result = _compute_cluster_lift_k(prob, label, k=50, min_cluster_size=1)
        assert result["n_clusters_raw"] == 1, \
            "4-pixel diagonal stripe should be 1 cluster under 8-connectivity"

    def test_far_apart_pixels_stay_separate(self):
        label = np.zeros((100, 100), dtype=np.uint8)
        label[10, 10] = 1
        label[80, 80] = 1
        label[10, 80] = 1
        prob = _perfect_pred(label)
        result = _compute_cluster_lift_k(prob, label, k=50, min_cluster_size=1)
        assert result["n_clusters_raw"] == 3, \
            "3 isolated pixels should remain 3 clusters"


class TestMinClusterSize:
    def test_filters_small_clusters(self):
        label = np.zeros((100, 100), dtype=np.uint8)
        # 1 big cluster (4 px) + 2 single-pixel "noise"
        label[10, 10] = 1; label[10, 11] = 1; label[11, 10] = 1; label[11, 11] = 1
        label[50, 50] = 1
        label[80, 80] = 1
        prob = _perfect_pred(label)
        # min_cluster_size=2 → drop the 2 single-pixel clusters
        result = _compute_cluster_lift_k(prob, label, k=50, min_cluster_size=2)
        assert result["n_clusters_raw"] == 3
        assert result["n_clusters"] == 1
        # min_cluster_size=1 → keep all
        result = _compute_cluster_lift_k(prob, label, k=50, min_cluster_size=1)
        assert result["n_clusters"] == 3


class TestKEff:
    def test_k_eff_floor_50(self):
        # Single cluster → k_eff = max(3*1, 50) = 50
        label = np.zeros((100, 100), dtype=np.uint8)
        label[10:13, 10:13] = 1  # 9-px cluster
        prob = _perfect_pred(label)
        result = _compute_cluster_lift_k(prob, label, k=5000)
        assert result["k_eff"] == 50

    def test_k_eff_3x_n_clusters(self):
        # 30 small clusters → k_eff = max(3*30, 50) = 90
        label = np.zeros((100, 100), dtype=np.uint8)
        for i in range(30):
            r, c = (i // 6) * 15 + 5, (i % 6) * 15 + 5
            label[r, c] = 1
        prob = _perfect_pred(label)
        result = _compute_cluster_lift_k(prob, label, k=5000)
        assert result["n_clusters_raw"] == 30
        assert result["k_eff"] == 90

    def test_k_eff_capped_by_requested_k(self):
        # 200 clusters → 3*200 = 600, but cap at requested k=300
        label = np.zeros((200, 200), dtype=np.uint8)
        for i in range(200):
            r, c = (i // 20) * 10 + 5, (i % 20) * 10 + 5
            label[r, c] = 1
        prob = _perfect_pred(label)
        result = _compute_cluster_lift_k(prob, label, k=300)
        assert result["n_clusters_raw"] == 200
        assert result["k_eff"] == 300


class TestScoreIsMaxOverCluster:
    def test_max_pooled_not_mean(self):
        # Cluster has mixed prob values; cluster score should be MAX
        label = np.zeros((100, 100), dtype=np.uint8)
        label[10:20, 10:20] = 1
        prob = np.full_like(label, 0.0, dtype=np.float32)
        # Most cluster pixels have prob 0.1, ONE pixel has 0.99
        prob[10:20, 10:20] = 0.1
        prob[15, 15] = 0.99   # the spike
        result = _compute_cluster_lift_k(prob, label, k=10)
        # If max-pool: cluster score = 0.99 → ranked above bg tiles → caught
        # If mean-pool: cluster score ≈ 0.1 → maybe below bg tiles
        assert result["recall_k"] == 1.0, \
            "Max-pool should let the 0.99 spike rescue the cluster"


class TestOutputSchema:
    def test_returns_all_expected_keys(self):
        label = np.zeros((100, 100), dtype=np.uint8)
        label[10:20, 10:20] = 1
        prob = _perfect_pred(label)
        result = _compute_cluster_lift_k(prob, label, k=50)
        expected = {"lift_k", "precision_k", "recall_k", "n_clusters",
                    "n_clusters_raw", "baseline", "n_items", "k_eff",
                    "median_cluster_size", "cl_pr_auc"}
        assert expected.issubset(set(result.keys()))


class TestBackgroundTileSizing:
    def test_tile_side_grows_with_median_cluster_size(self):
        # Small clusters → small tile_side
        label1 = np.zeros((100, 100), dtype=np.uint8)
        for i in range(20):
            r, c = (i // 5) * 18 + 5, (i % 5) * 18 + 5
            label1[r:r+2, c:c+2] = 1  # 4-px clusters
        prob = _perfect_pred(label1)
        r1 = _compute_cluster_lift_k(prob, label1, k=50)
        # Large clusters → larger tile_side
        label2 = np.zeros((100, 100), dtype=np.uint8)
        for i in range(4):
            r, c = (i // 2) * 40 + 5, (i % 2) * 40 + 5
            label2[r:r+10, c:c+10] = 1  # 100-px clusters
        prob = _perfect_pred(label2)
        r2 = _compute_cluster_lift_k(prob, label2, k=50)
        assert r2["median_cluster_size"] > r1["median_cluster_size"]


class TestPerfectPredictorLiftAboveOne:
    def test_oracle_beats_random(self):
        # Make a non-trivial map: 5 clusters scattered
        label = np.zeros((200, 200), dtype=np.uint8)
        for r0, c0 in [(20, 20), (50, 80), (100, 30), (150, 150), (180, 100)]:
            label[r0:r0+5, c0:c0+5] = 1
        oracle = _perfect_pred(label)
        random_pred = _random_pred(label, seed=42)
        oracle_r = _compute_cluster_lift_k(oracle, label, k=100)
        random_r = _compute_cluster_lift_k(random_pred, label, k=100)
        assert oracle_r["lift_k"] > random_r["lift_k"], \
            "Oracle predictor should beat random predictor"
        assert oracle_r["recall_k"] >= random_r["recall_k"]


class TestTileSizePercentile:
    def test_default_is_median_backward_compat(self):
        # Default should match prior behavior (50th percentile = median)
        label = np.zeros((200, 200), dtype=np.uint8)
        label[10:20, 10:20] = 1   # 100-px cluster
        label[50:55, 50:55] = 1   # 25-px cluster
        prob = _perfect_pred(label)
        r_default = _compute_cluster_lift_k(prob, label, k=50)
        r_p50 = _compute_cluster_lift_k(prob, label, k=50,
                                         tile_size_percentile=50)
        assert r_default["lift_k"] == r_p50["lift_k"]
        assert r_default["tile_side"] == r_p50["tile_side"]

    def test_higher_percentile_grows_tile_side(self):
        # Skewed distribution: many small + a few medium + 1 mega.
        # Mix designed so p90 lands on medium clusters (not small).
        label = np.zeros((300, 300), dtype=np.uint8)
        label[10:80, 10:80] = 1   # 4900-px mega cluster
        # 8 medium clusters (size 100 each)
        for i in range(8):
            r = (i // 4) * 30 + 150
            c = (i % 4) * 30 + 30
            label[r:r+10, c:c+10] = 1   # 100-px clusters
        # 6 small clusters (9 px each)
        for i in range(6):
            r = (i // 3) * 20 + 220
            c = (i % 3) * 20 + 200
            label[r:r+3, c:c+3] = 1   # 9-px clusters
        prob = _perfect_pred(label)
        r_p50 = _compute_cluster_lift_k(prob, label, k=200,
                                         tile_size_percentile=50)
        r_p90 = _compute_cluster_lift_k(prob, label, k=200,
                                         tile_size_percentile=90)
        assert r_p50["tile_side"] <= r_p90["tile_side"], \
            "Higher percentile should produce larger or equal tile_side"
        # On heavy-tail mixed distribution: p90 strictly larger
        assert r_p90["tile_side"] > r_p50["tile_side"], \
            f"On heavy-tail data p90 should beat p50, got {r_p90['tile_side']} vs {r_p50['tile_side']}"


class TestStratifiedLift:
    def test_returns_three_size_bins(self):
        label = np.zeros((300, 300), dtype=np.uint8)
        # 1 small (size 9), 1 medium (size 400), 1 large (size 4900)
        label[10:13, 10:13] = 1            # 9 px
        label[50:70, 50:70] = 1            # 400 px
        label[150:220, 150:220] = 1        # 4900 px
        prob = _perfect_pred(label)
        r = _compute_cluster_lift_k(prob, label, k=200,
                                     return_stratified=True)
        assert "lift_k_small" in r
        assert "lift_k_medium" in r
        assert "lift_k_large" in r
        assert r["n_clusters_small"] == 1
        assert r["n_clusters_medium"] == 1
        assert r["n_clusters_large"] == 1

    def test_no_stratified_when_flag_off(self):
        label = np.zeros((100, 100), dtype=np.uint8)
        label[10:20, 10:20] = 1
        prob = _perfect_pred(label)
        r = _compute_cluster_lift_k(prob, label, k=50)
        assert "lift_k_small" not in r
        assert "lift_k_medium" not in r
        assert "lift_k_large" not in r

    def test_perfect_predictor_lifts_all_bins(self):
        label = np.zeros((300, 300), dtype=np.uint8)
        label[10:13, 10:13] = 1
        label[50:70, 50:70] = 1
        label[150:220, 150:220] = 1
        prob = _perfect_pred(label)
        r = _compute_cluster_lift_k(prob, label, k=200,
                                     return_stratified=True)
        # Perfect predictor: all three bins should have lift > 1
        for bin_name in ["small", "medium", "large"]:
            v = r[f"lift_k_{bin_name}"]
            if v == v:  # not nan
                assert v > 1.0, f"{bin_name} bin lift {v} should be > 1.0"

    def test_empty_bin_returns_nan(self):
        # Only small clusters; medium/large bins should be NaN
        label = np.zeros((100, 100), dtype=np.uint8)
        for i in range(10):
            r = (i // 3) * 20 + 10
            c = (i % 3) * 20 + 10
            label[r, c] = 1   # 1-px
        prob = _perfect_pred(label)
        r = _compute_cluster_lift_k(prob, label, k=50,
                                     return_stratified=True,
                                     min_cluster_size=1)
        assert r["n_clusters_small"] >= 1
        assert r["n_clusters_medium"] == 0
        assert r["n_clusters_large"] == 0
        assert r["lift_k_medium"] != r["lift_k_medium"]  # nan check
        assert r["lift_k_large"] != r["lift_k_large"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
