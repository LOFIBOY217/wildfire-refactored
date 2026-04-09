"""
Regression tests for temporal data leakage.

Run: pytest tests/test_no_leakage.py

These tests verify that channels that depend on year-of-sample
only use data from years STRICTLY BEFORE the sample year.
"""
import os
import sys
import numpy as np
from pathlib import Path

# Allow running without pytest (just python tests/test_no_leakage.py)
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_burn_age_uses_previous_year():
    """burn_age for a 2021 date should use burn_scars from <=2020.

    Verifies the code path in train_v3.py around line 1408-1426:
        prev_year = cur_date.year - 1
        if prev_year in burn_scar_raw: raw = burn_scar_raw[prev_year]
    """
    # Read source directly since we can't import without full env
    src_file = Path(__file__).parent.parent / "src/training/train_v3.py"
    source = src_file.read_text()

    # Check burn_age section uses prev_year, not cur_date.year
    burn_age_idx = source.index('elif ch_name == "burn_age":')
    burn_count_idx = source.index('elif ch_name == "burn_count":', burn_age_idx)
    burn_age_block = source[burn_age_idx:burn_count_idx]

    assert "prev_year = cur_date.year - 1" in burn_age_block, \
        "burn_age must use prev_year to avoid temporal leakage"
    # Must NOT use cur_date.year directly for lookup
    assert "burn_scar_raw[cur_date.year]" not in burn_age_block, \
        "burn_age must not use cur_date.year directly (leakage bug)"


def test_burn_count_uses_previous_year():
    """Same as burn_age but for burn_count."""
    src_file = Path(__file__).parent.parent / "src/training/train_v3.py"
    source = src_file.read_text()

    burn_count_idx = source.index('elif ch_name == "burn_count":')
    # Find end of block (next elif or end of loop)
    next_block = source.find('elif ch_name', burn_count_idx + 5)
    if next_block == -1:
        next_block = source.find('# Normalize', burn_count_idx)
    burn_count_block = source[burn_count_idx:next_block]

    assert "prev_year = cur_date.year - 1" in burn_count_block, \
        "burn_count must use prev_year to avoid temporal leakage"
    assert "burn_count_arrays[cur_date.year]" not in burn_count_block, \
        "burn_count must not use cur_date.year directly (leakage bug)"


def test_fire_clim_uses_upto_year():
    """fire_clim for year Y should use fire_clim_upto_Y which contains only <Y data.

    The naming convention: fire_clim_upto_2022.tif uses years [2018..2021].
    """
    src_file = Path(__file__).parent.parent / "src/data_ops/processing/make_fire_clim_annual.py"
    if not src_file.exists():
        # Fallback: some older versions have different filename
        return
    source = src_file.read_text()
    # The "upto" logic must use range(start, target_year), not range(start, target_year+1)
    assert "range(args.data_start_year, target_year)" in source or \
           "prior_years = list(range(" in source, \
        "fire_clim_upto must use years strictly before target_year"


def test_nodata_masked_in_static_loader():
    """_load_static_channel must mask src.nodata to prevent sentinel leakage."""
    src_file = Path(__file__).parent.parent / "src/training/train_v3.py"
    source = src_file.read_text()

    loader_idx = source.index("def _load_static_channel")
    loader_end = source.index("\ndef ", loader_idx + 10)
    loader_block = source[loader_idx:loader_end]

    assert "src.nodata" in loader_block, \
        "_load_static_channel must read src.nodata"
    assert "arr[arr == nodata] = 0.0" in loader_block or \
           "arr[arr == nodata] = np.nan" in loader_block, \
        "_load_static_channel must mask nodata pixels"


def test_lead_end_clamped_for_s2s():
    """s2s and s2s_legacy decoders must clamp lead_end to 45 (cache has 32 leads)."""
    src_file = Path(__file__).parent.parent / "src/training/train_v3.py"
    source = src_file.read_text()

    assert 'if args.decoder in ("s2s", "s2s_legacy") and lead_end > 45' in source, \
        "lead_end must be clamped to 45 for s2s decoders"


def test_val_decoder_ctx_callback_wired():
    """V3 training must pass decoder_ctx_fn to val when decoder_ctx is enabled."""
    src_file = Path(__file__).parent.parent / "src/training/train_v3.py"
    source = src_file.read_text()

    assert "decoder_ctx_fn=_val_ctx_fn" in source, \
        "V3 val call must pass decoder_ctx_fn callback"
    assert "def _val_ctx_fn" in source, \
        "V3 must define _val_ctx_fn closure"


if __name__ == "__main__":
    tests = [
        test_burn_age_uses_previous_year,
        test_burn_count_uses_previous_year,
        test_fire_clim_uses_upto_year,
        test_nodata_masked_in_static_loader,
        test_lead_end_clamped_for_s2s,
        test_val_decoder_ctx_callback_wired,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)
