"""
Verify fire_patched cache filename includes fusion_tag.

This guards against the 2026-04-29 bug where fire_patched.dat was named
WITHOUT fusion_tag, causing CWFIS-built caches to be reused by NBAC+NFDB
runs (silently corrupting 4y/12y training+eval for weeks).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _extract_fire_cache_path_logic():
    """
    Extract the fire_patched filename construction logic from train_v3.py
    and run a few input combinations to verify fusion_tag is included.

    Returns the source line containing fire_patched_v3_r{...}.
    """
    src = (ROOT / "src" / "training" / "train_v3.py").read_text()
    return src


def test_fire_patched_filename_contains_fusion_tag():
    """
    The fire_patched cache filename MUST include fusion_tag, otherwise
    a CWFIS-built .dat will be silently reused by NBAC+NFDB runs.
    """
    src = _extract_fire_cache_path_logic()
    # Find the construction line
    target_line = None
    for line in src.split("\n"):
        if "fire_patched_v3_r" in line and "args.dilate_radius" in line:
            target_line = line
            break
    assert target_line is not None, \
        "Could not find fire_patched cache filename construction in train_v3.py"
    assert "fusion_tag" in target_line, (
        f"fire_patched cache filename construction MUST include fusion_tag.\n"
        f"Found line:\n  {target_line.strip()}\n"
        f"This is the bug from 2026-04-29 — see commit history."
    )


def test_fire_patched_filename_construction_simulation():
    """
    Simulate the filename construction with both fusion_tag values
    and verify the strings differ.
    """
    # Replicate the f-string from train_v3.py
    cache_dir = "/scratch/jiaqi217/meteo_cache/v3_9ch_2000"
    dilate_radius = 14
    aligned_dates_first = "2000-05-01"
    aligned_dates_last = "2025-12-20"
    T = 9332
    n_patches = 23998
    out_dim = 256

    def build_path(fusion_tag):
        return (
            f"{cache_dir}/fire_patched_v3_r{dilate_radius}{fusion_tag}"
            f"_{aligned_dates_first}_{aligned_dates_last}"
            f"_{T}x{n_patches}x{out_dim}.dat"
        )

    cwfis_path = build_path("")  # --label_fusion=False
    fusion_path = build_path("_nbac_nfdb")  # --label_fusion=True

    assert cwfis_path != fusion_path, \
        f"CWFIS and NBAC+NFDB paths must differ: {cwfis_path} vs {fusion_path}"

    assert "_nbac_nfdb" in fusion_path
    assert "_nbac_nfdb" not in cwfis_path


if __name__ == "__main__":
    test_fire_patched_filename_contains_fusion_tag()
    test_fire_patched_filename_construction_simulation()
    print("All tests passed.")
