"""
Unit tests for src/evaluation/metrics.py — rare-event eval metrics.

Tests use small hand-verifiable inputs. Each metric is verified against
a value computed by hand or from the metric's definition.

Run: python -m pytest tests/test_metrics.py -v
"""
import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_ranking_metrics,
    compute_imbalanced_metrics,
    compute_brier_decomposition,
    compute_coarsened_lift,
    compute_all_metrics,
    bootstrap_ci,
)


# ---------- compute_ranking_metrics ---------- #

def test_ranking_perfect_prediction():
    """5 fires, top-5 scores are all fires → Lift = 100/baseline."""
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
                      dtype=np.float32)
    labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    r = compute_ranking_metrics(scores, labels, k=5)

    assert r['tp'] == 5
    assert r['k_eff'] == 5
    assert r['precision_k'] == 1.0
    assert r['recall_k'] == 1.0
    assert r['baseline'] == 0.5  # 5/10
    assert r['lift_k'] == 2.0    # perfect: 1.0 / 0.5
    assert r['csi_k'] == 1.0     # no FP, no FN


def test_ranking_zero_fires():
    """No positives → all metrics 0."""
    scores = np.array([0.9, 0.5, 0.1], dtype=np.float32)
    labels = np.array([0, 0, 0], dtype=np.float32)
    r = compute_ranking_metrics(scores, labels, k=2)
    assert r['n_fire'] == 0
    assert r['lift_k'] == 0.0


def test_ranking_k_larger_than_N():
    """K > N should clamp to N."""
    scores = np.array([0.5, 0.3, 0.8], dtype=np.float32)
    labels = np.array([1, 0, 1], dtype=np.float32)
    r = compute_ranking_metrics(scores, labels, k=100)
    assert r['k_eff'] == 3
    assert r['tp'] == 2
    assert r['precision_k'] == pytest.approx(2/3)


def test_ranking_known_example():
    """
    100 samples, 4 fires at indices [0,1,2,3] with scores [1.0,0.9,0.8,0.7].
    Other 96 samples random but all < 0.7.
    K=10 should catch all 4 fires → precision=0.4, lift=0.4/0.04=10x.
    """
    scores = np.concatenate([
        np.array([1.0, 0.9, 0.8, 0.7], dtype=np.float32),
        np.random.default_rng(0).uniform(0, 0.5, 96).astype(np.float32),
    ])
    labels = np.zeros(100, dtype=np.float32)
    labels[:4] = 1.0
    r = compute_ranking_metrics(scores, labels, k=10)
    assert r['tp'] == 4
    assert r['precision_k'] == pytest.approx(0.4)
    assert r['baseline'] == pytest.approx(0.04)
    assert r['lift_k'] == pytest.approx(10.0)
    # CSI = TP / (TP + FP + FN) = 4 / (4 + 6 + 0) = 0.4
    assert r['csi_k'] == pytest.approx(0.4)
    # ETS: tp_random = 10 * 0.04 = 0.4; ETS = (4-0.4)/(10-0.4) = 3.6/9.6
    assert r['ets_k'] == pytest.approx(3.6 / 9.6, rel=1e-5)


# ---------- compute_imbalanced_metrics ---------- #

def test_imbalanced_perfect_separation():
    """Perfect ranking → PR-AUC = 1.0, ROC-AUC = 1.0, MCC = 1.0."""
    scores = np.array([0.9, 0.8, 0.1, 0.2], dtype=np.float32)
    labels = np.array([1, 1, 0, 0], dtype=np.float32)
    m = compute_imbalanced_metrics(scores, labels)
    assert m['pr_auc'] == pytest.approx(1.0)
    assert m['roc_auc'] == pytest.approx(1.0)
    assert m['mcc'] == pytest.approx(1.0)
    assert m['f1'] == pytest.approx(1.0)
    assert m['f2'] == pytest.approx(1.0)


def test_imbalanced_random_predictions():
    """Random scores on balanced data → ROC-AUC ≈ 0.5."""
    rng = np.random.default_rng(0)
    labels = np.array([0, 1] * 50, dtype=np.float32)
    scores = rng.uniform(0, 1, 100).astype(np.float32)
    m = compute_imbalanced_metrics(scores, labels)
    assert 0.3 < m['roc_auc'] < 0.7  # not perfect, not anti-correlated


def test_imbalanced_no_positives():
    """No positives → all zero."""
    scores = np.array([0.5, 0.5], dtype=np.float32)
    labels = np.array([0, 0], dtype=np.float32)
    m = compute_imbalanced_metrics(scores, labels)
    assert m['f1'] == 0.0
    assert m['mcc'] == 0.0


# ---------- compute_brier_decomposition ---------- #

def test_brier_perfect_calibration():
    """Predict exactly true labels → Brier = 0, BSS = 1."""
    scores = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    labels = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    b = compute_brier_decomposition(scores, labels)
    assert b['brier'] == pytest.approx(0.0)
    assert b['bss'] == pytest.approx(1.0)


def test_brier_climatology_reference():
    """Predict baseline everywhere → Brier = p(1-p), BSS = 0."""
    labels = np.array([0, 0, 0, 1], dtype=np.float32)  # baseline = 0.25
    scores = np.full(4, 0.25, dtype=np.float32)
    b = compute_brier_decomposition(scores, labels)
    # Brier = mean((0.25-y)²) = (0.0625 + 0.0625 + 0.0625 + 0.5625) / 4 = 0.1875
    # brier_clim = 0.25 * 0.75 = 0.1875
    # BSS = 1 - 0.1875/0.1875 = 0
    assert b['brier'] == pytest.approx(0.1875)
    assert b['brier_climatology'] == pytest.approx(0.1875)
    assert b['bss'] == pytest.approx(0.0, abs=1e-10)


def test_brier_worse_than_climatology():
    """Predict opposite of truth → negative BSS (worse than climatology)."""
    labels = np.array([1, 0, 1, 0], dtype=np.float32)
    scores = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    b = compute_brier_decomposition(scores, labels)
    assert b['bss'] < 0  # definitely worse than just predicting baseline


# ---------- compute_coarsened_lift ---------- #

def test_coarsen_downsample_factor2():
    """6x6 → 3x3 with factor 2. Check aggregation logic."""
    scores_2d = np.array([
        [0.9, 0.9, 0.1, 0.1, 0.1, 0.1],
        [0.9, 0.9, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.1, 0.9, 0.9],
        [0.1, 0.1, 0.1, 0.1, 0.9, 0.9],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    ], dtype=np.float32)
    labels_2d = np.zeros((6, 6), dtype=np.float32)
    labels_2d[0, 0] = 1.0  # one fire in top-left 2x2 cell
    labels_2d[2, 4] = 1.0  # one fire in middle-right 2x2 cell

    # Coarse grid: 3x3
    # Scores: mean of each 2x2 block:
    #   Top-left   = 0.9
    #   Top-right areas have 0.1
    #   Bottom-right (row 2-3, col 4-5) = 0.9
    # Labels: max = 1 for cells (0,0) and (1,2), 0 elsewhere
    # n_fire = 2, n_total = 9, baseline = 2/9
    # k_fine = 1 → k_coarse = 1 / 4 = 0 → clamped to 1
    r = compute_coarsened_lift(scores_2d, labels_2d, factor=2, k_fine=4)
    assert r['n_fire_coarse'] == 2
    # Top-K (K=1 after scaling) should catch one of the high-scoring cells
    # Since both fire cells have score 0.9 (highest), Lift should be high
    assert r['lift_coarse'] > 1.0


def test_coarsen_1d_input_returns_zero():
    """1D input should return 0s, not crash."""
    r = compute_coarsened_lift(np.array([0.5, 0.5]), np.array([1, 0]),
                               factor=2, k_fine=1)
    assert r['lift_coarse'] == 0.0


# ---------- compute_all_metrics (integration) ---------- #

def test_all_metrics_integration():
    """Complete integration test on a toy 4x4 problem."""
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, (4, 4)).astype(np.float32)
    scores[0, 0] = 0.95
    scores[3, 3] = 0.92
    labels = np.zeros((4, 4), dtype=np.float32)
    labels[0, 0] = 1.0
    labels[3, 3] = 1.0

    result = compute_all_metrics(scores, labels, k_values=[4],
                                 spatial_shape=(4, 4), coarsen_factor=2)

    # Expected top-4 captures both fires
    assert result['tp'] == 2
    assert result['n_fire'] == 2
    assert result['precision_k'] == pytest.approx(0.5)
    assert result['lift_k'] == pytest.approx(0.5 / (2 / 16))  # = 4
    # Has all metrics
    for key in ['pr_auc', 'roc_auc', 'f1', 'f2', 'mcc',
                'bss', 'brier', 'reliability', 'resolution',
                'lift_coarse']:
        assert key in result


def test_all_metrics_namespaced_keys():
    """Multiple K values should produce namespaced keys."""
    scores = np.array([0.9, 0.5, 0.1, 0.8, 0.3], dtype=np.float32)
    labels = np.array([1, 0, 0, 1, 0], dtype=np.float32)
    result = compute_all_metrics(scores, labels, k_values=[2, 3])
    assert 'lift_k_2' in result
    assert 'lift_k_3' in result
    # Primary K = first in list
    assert result['lift_k'] == result['lift_k_2']


# ---------- bootstrap_ci ---------- #

def test_bootstrap_ci_basic():
    """Bootstrap CI of constant array → low == high == mean."""
    vals = [5.0, 5.0, 5.0, 5.0, 5.0]
    ci = bootstrap_ci(vals, n_boot=100)
    assert ci['mean'] == 5.0
    assert ci['ci_low'] == 5.0
    assert ci['ci_high'] == 5.0


def test_bootstrap_ci_includes_mean():
    """95% CI should contain the mean of the underlying data."""
    rng = np.random.default_rng(0)
    vals = rng.normal(loc=10.0, scale=1.0, size=100).tolist()
    ci = bootstrap_ci(vals, n_boot=500, alpha=0.05)
    assert ci['ci_low'] < ci['mean'] < ci['ci_high']
    # CI should be tight for N=100
    assert (ci['ci_high'] - ci['ci_low']) < 1.0


def test_bootstrap_ci_empty():
    """Empty input → all zeros, no crash."""
    ci = bootstrap_ci([])
    assert ci['mean'] == 0.0
    assert ci['ci_low'] == 0.0
    assert ci['ci_high'] == 0.0


# ---------- Integration: verify refactored callers work ---------- #

def test_benchmark_baselines_caller():
    """benchmark_baselines._compute_metrics now delegates to metrics.py —
    smoke-test that it returns expected structure for both per-K and
    shared metrics."""
    from src.evaluation.benchmark_baselines import _compute_metrics

    scores = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.5], dtype=np.float32)
    labels = np.array([1, 1, 0, 0, 1, 0], dtype=np.float32)
    results = _compute_metrics(scores, labels, k_values=[3, 5])

    # Both K's should be present
    assert 3 in results and 5 in results
    # Each K dict has all standard metrics
    for k_res in results.values():
        for key in ['lift_k', 'precision_k', 'recall_k', 'csi_k', 'ets_k',
                    'pr_auc', 'roc_auc', 'brier',
                    'f1', 'f2', 'mcc', 'bss',
                    'reliability', 'resolution', 'lift_coarse',
                    'n_fire', 'n_total', 'baseline']:
            assert key in k_res, f"Missing metric: {key}"

    # Values should be valid (non-negative for all-positive metrics)
    assert results[3]['lift_k'] > 0
    assert 0 <= results[3]['precision_k'] <= 1
    # BSS for non-degenerate case should be a finite number
    assert np.isfinite(results[3]['bss'])


def test_benchmark_baselines_zero_fires():
    """Verify refactored caller doesn't crash on all-negative labels."""
    from src.evaluation.benchmark_baselines import _compute_metrics
    scores = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    labels = np.array([0, 0, 0], dtype=np.float32)
    results = _compute_metrics(scores, labels, k_values=[2])
    assert results[2]['lift_k'] == 0.0
    assert results[2]['n_fire'] == 0
