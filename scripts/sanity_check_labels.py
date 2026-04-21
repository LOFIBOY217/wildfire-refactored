#!/usr/bin/env python3
"""
Sanity check a built fire-label .npy + sidecar .json from
build_fire_labels.py before using for training.

Checks:
  1. Shape matches expected (T, H, W) uint8
  2. positive_rate is in expected range (0.01% - 10%)
  3. Per-year positive count has no suspicious 0-years or extreme jumps
  4. For nbac_nfdb scheme: NBAC prescribed polygons excluded (check
     NFDB cause filter worked, size filter worked — via provenance JSON)
  5. For nbac_nfdb scheme: positive rate should be HIGHER than cwfis
     (drift correction should add positive pixels)

Usage:
    python scripts/sanity_check_labels.py \\
        --label data/fire_labels/fire_labels_nbac_nfdb_..._r14.npy

    # compare to another scheme:
    python scripts/sanity_check_labels.py \\
        --label data/fire_labels/fire_labels_nbac_nfdb_..._r14.npy \\
        --baseline data/fire_labels/fire_labels_cwfis_..._r14.npy
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_meta(npy_path: Path):
    json_path = npy_path.with_suffix(".json")
    if not json_path.exists():
        print(f"  [FAIL] sidecar JSON not found: {json_path}")
        return None
    with open(json_path) as f:
        return json.load(f)


def check_one(npy_path: Path):
    print(f"\n=== SANITY CHECK: {npy_path.name} ===")
    if not npy_path.exists():
        print(f"  [FAIL] file not found")
        return False

    meta = load_meta(npy_path)
    if meta is None:
        return False

    # Load with mmap (don't pull 60 GB into RAM)
    arr = np.load(npy_path, mmap_mode="r")
    T, H, W = arr.shape
    print(f"  shape:           ({T}, {H}, {W})")
    print(f"  dtype:           {arr.dtype}")
    print(f"  scheme:          {meta.get('scheme')}")
    print(f"  date_range:      {meta.get('date_range')}")
    print(f"  dilate_radius:   {meta.get('dilate_radius')}")
    print(f"  raw_positive:    {meta.get('raw_positive'):,}")
    print(f"  dilated_positive:{meta.get('dilated_positive'):,}")
    print(f"  positive_rate:   {meta.get('positive_rate'):.4%}")

    ok = True

    # Check 1: shape sanity
    if (H, W) != (2281, 2709):
        print(f"  [FAIL] H×W {H}×{W} != expected 2281×2709")
        ok = False
    if arr.dtype != np.uint8:
        print(f"  [FAIL] dtype {arr.dtype} != uint8")
        ok = False

    # Check 2: positive rate sane
    pr = meta.get("positive_rate", 0)
    if pr < 1e-5:
        print(f"  [FAIL] positive rate {pr:.6%} < 0.001% — "
              "probably empty / bug")
        ok = False
    elif pr > 0.10:
        print(f"  [WARN] positive rate {pr:.4%} > 10% — "
              "very dense labels, may hurt baseline ratio")
    else:
        print(f"  [PASS] positive rate {pr:.4%} in normal range")

    # Check 3: per-year distribution
    yearly = meta.get("yearly_positive", {})
    if yearly:
        yrs = sorted((int(y), int(v)) for y, v in yearly.items())
        print("\n  per-year positive count:")
        max_val = max(v for _, v in yrs) or 1
        for y, v in yrs:
            bar = "█" * max(1, int(40 * v / max_val))
            print(f"    {y}: {v:>12,}  {bar}")
        # Flag 0-years and extreme jumps
        prev = None
        for y, v in yrs:
            if v == 0:
                print(f"  [FAIL] year {y} has ZERO positive pixels!")
                ok = False
            if prev is not None and prev > 0 and v / prev > 100:
                print(f"  [WARN] year {y}: {v / prev:.1f}× jump from {y-1}")
            prev = v

    # Check 4: filters worked (for nbac_nfdb)
    scheme = meta.get("scheme", "")
    filters = meta.get("filters", {})
    if scheme == "nbac_nfdb":
        if filters.get("exclude_prescribed"):
            print(f"  [PASS] prescribed burns excluded (per provenance)")
        else:
            print(f"  [WARN] prescribed burns INCLUDED — label will contain "
                  "managed-burn targets (not intended for wildfire prediction)")
        min_ha = filters.get("nfdb_min_size_ha", 0)
        print(f"  [INFO] NFDB min_size_ha = {min_ha}")
        if min_ha == 0:
            print(f"    → includes ~319k <1ha micro-fires (noisy)")
        elif min_ha == 1.0:
            print(f"    → excludes ~319k <1ha noise (recommended)")

    # Check 5: sample spot-check — load 3 random days, print pos counts
    rng = np.random.default_rng(42)
    sample_days = rng.choice(T, size=min(5, T), replace=False)
    print(f"\n  sample days (random):")
    for t in sorted(sample_days):
        n_pos = int(np.asarray(arr[t]).sum())
        print(f"    t={t:>5}: positive pixels = {n_pos:,}")

    return ok


def compare_two(new_path: Path, old_path: Path):
    print(f"\n=== CROSS-CHECK: new vs old ===")
    new_meta = load_meta(new_path)
    old_meta = load_meta(old_path)
    if new_meta is None or old_meta is None:
        return False

    new_pr = new_meta["positive_rate"]
    old_pr = old_meta["positive_rate"]
    print(f"  old ({old_meta['scheme']}) positive rate: {old_pr:.4%}")
    print(f"  new ({new_meta['scheme']}) positive rate: {new_pr:.4%}")
    print(f"  ratio new/old: {new_pr / max(old_pr, 1e-9):.1f}×")

    if new_pr > old_pr * 1.5:
        print(f"  [PASS] new label has >1.5× more positives — "
              "drift correction active")
    elif new_pr > old_pr:
        print(f"  [PASS] new label has more positives (modest)")
    else:
        print(f"  [WARN] new label has FEWER positives than old — "
              "unexpected; investigate")

    # Per-year comparison
    new_yr = new_meta.get("yearly_positive", {})
    old_yr = old_meta.get("yearly_positive", {})
    if new_yr and old_yr:
        all_years = sorted(set(new_yr) | set(old_yr), key=int)
        print("\n  per-year comparison (new_count / old_count ratio):")
        for y in all_years:
            n = new_yr.get(y, 0)
            o = old_yr.get(y, 0)
            ratio = n / max(o, 1)
            flag = ""
            if int(y) <= 2011 and ratio > 10:
                flag = "  ★ drift fix (early era, big new/old)"
            print(f"    {y}: new={n:>12,} / old={o:>12,}  ratio={ratio:.2f}×{flag}")

    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, type=Path,
                    help="fire_labels_*.npy file to check")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="Optional: another label to compare (old vs new)")
    args = ap.parse_args()

    ok = check_one(args.label)

    if args.baseline:
        ok_base = check_one(args.baseline)
        if ok and ok_base:
            compare_two(args.label, args.baseline)

    print(f"\n=== {'PASS — safe for training' if ok else 'FAIL — fix before training'} ===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
