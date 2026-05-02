"""Regression test for master-cache slice reuse.

Builds two synthetic memmap caches:
  - master cache: T=20 days, range 2000-01-01..2000-01-20
  - per-range cache: T=10 days, range 2000-01-06..2000-01-15

Verifies that slicing the master from t_offset=5 yields exactly the
per-range cache contents.
"""
import os
import tempfile
import numpy as np


def test_master_slice_matches_per_range():
    n_patches, enc_dim = 4, 3
    T_master = 20
    T_sub = 10
    t_offset = 5

    rng = np.random.RandomState(0)
    master = rng.randn(n_patches, T_master, enc_dim).astype(np.float16)
    per_range = master[:, t_offset:t_offset + T_sub, :].copy()

    with tempfile.TemporaryDirectory() as td:
        master_path = os.path.join(td, "master_pf.dat")
        per_path = os.path.join(td, "per_pf.dat")

        m_mm = np.memmap(master_path, dtype='float16', mode='w+',
                         shape=master.shape)
        m_mm[:] = master
        m_mm.flush()
        del m_mm

        p_mm = np.memmap(per_path, dtype='float16', mode='w+',
                         shape=per_range.shape)
        p_mm[:] = per_range
        p_mm.flush()
        del p_mm

        # Replicate train_v3 logic
        master_loaded = np.memmap(master_path, dtype='float16', mode='r',
                                  shape=(n_patches, T_master, enc_dim))
        sliced = np.array(master_loaded[:, t_offset:t_offset + T_sub, :])

        per_loaded = np.memmap(per_path, dtype='float16', mode='r',
                               shape=(n_patches, T_sub, enc_dim))
        per_arr = np.array(per_loaded)

        assert sliced.shape == per_arr.shape
        assert np.array_equal(sliced, per_arr), \
            f"sliced master != per-range cache; max diff={np.abs(sliced.astype(np.float32) - per_arr.astype(np.float32)).max()}"


def test_fire_patched_slice():
    """Time-first layout: fire_patched is (T, n_patches, out_dim)."""
    n_patches, out_dim = 4, 256
    T_master = 20
    T_sub = 10
    t_offset = 5

    rng = np.random.RandomState(1)
    master = rng.randint(0, 2, size=(T_master, n_patches, out_dim)).astype(np.uint8)
    per_range = master[t_offset:t_offset + T_sub].copy()

    with tempfile.TemporaryDirectory() as td:
        master_path = os.path.join(td, "fire_master.dat")
        m_mm = np.memmap(master_path, dtype='uint8', mode='w+', shape=master.shape)
        m_mm[:] = master
        m_mm.flush()
        del m_mm

        master_loaded = np.memmap(master_path, dtype='uint8', mode='r',
                                  shape=(T_master, n_patches, out_dim))
        sliced = master_loaded[t_offset:t_offset + T_sub]
        assert np.array_equal(np.array(sliced), per_range)


if __name__ == "__main__":
    test_master_slice_matches_per_range()
    test_fire_patched_slice()
    print("OK — master-cache slice matches per-range cache")
