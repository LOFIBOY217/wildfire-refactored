"""Test spatial-radius negative mining isolates the candidate pool to
patches within R patches of any positive in the same training window.

This mirrors the per-window dilation logic added to train_v3.py (no
import — replicates the math standalone)."""
import numpy as np
from scipy.ndimage import binary_dilation


def _spatial_neg_mask(win_pos_patches, n_patches, grid, R_patches):
    nrow, ncol = grid
    struct = np.ones((2 * R_patches + 1, 2 * R_patches + 1), dtype=bool)
    out = np.zeros(n_patches, dtype=bool)
    grid_pos = np.zeros(n_patches, dtype=bool)
    grid_pos[win_pos_patches] = True
    dilated = binary_dilation(grid_pos.reshape(nrow, ncol), structure=struct)
    out[:] = dilated.flatten()
    return out


def test_single_pos_dilates_to_disk():
    nrow, ncol = 10, 10
    n_patches = nrow * ncol
    pos = np.array([55])  # row 5 col 5
    R = 2
    mask = _spatial_neg_mask(pos, n_patches, (nrow, ncol), R)
    # dilation of single point with (2R+1)x(2R+1) square = 5x5 = 25 patches
    assert mask.sum() == 25
    # The center is included; positives are still in the mask (caller subtracts)
    assert mask[55] is np.True_ or mask[55] == True


def test_two_pos_dilations_overlap():
    nrow, ncol = 10, 10
    n_patches = nrow * ncol
    pos = np.array([55, 56])  # adjacent
    R = 1
    mask = _spatial_neg_mask(pos, n_patches, (nrow, ncol), R)
    # Two adjacent dilations of 3x3 each, overlapping in 6 cells -> total 3*3 + 3*3 - 6 = 12
    assert mask.sum() == 12


def test_no_pos_yields_empty():
    nrow, ncol = 10, 10
    n_patches = nrow * ncol
    mask = _spatial_neg_mask(np.array([], dtype=int), n_patches, (nrow, ncol), 3)
    assert mask.sum() == 0


def test_subtraction_removes_positives():
    """End-to-end: spatial mask & ~pos_mask = real negative candidates."""
    nrow, ncol = 10, 10
    n_patches = nrow * ncol
    pos = np.array([55, 56])
    pos_mask = np.zeros(n_patches, dtype=bool)
    pos_mask[pos] = True
    R = 1
    spatial = _spatial_neg_mask(pos, n_patches, (nrow, ncol), R)
    neg_candidates = spatial & ~pos_mask
    # 12 in disk - 2 positives = 10
    assert neg_candidates.sum() == 10
    # No positive should remain
    assert not neg_candidates[55] and not neg_candidates[56]


def test_radius_ceil():
    """Patch resolution 32 km/patch. R=200 km -> 7 patches per direction."""
    patch_km = 32.0
    R_km_in = [50, 100, 200, 250]
    expected = [2, 4, 7, 8]   # ceil(R/32)
    got = [max(1, int(np.ceil(r / patch_km))) for r in R_km_in]
    assert got == expected, f"got={got} expected={expected}"


if __name__ == "__main__":
    test_single_pos_dilates_to_disk()
    test_two_pos_dilations_overlap()
    test_no_pos_yields_empty()
    test_subtraction_removes_positives()
    test_radius_ceil()
    print("OK — spatial neg mining math verified")
