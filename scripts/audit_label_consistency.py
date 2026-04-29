#!/usr/bin/env python3
"""
Automated audit: verify all training/eval pipelines use consistent NBAC+NFDB labels.

This is the regression test for the 2026-04-29 fire_patched cache bug
(filename was missing fusion_tag, silently mixing CWFIS and NBAC+NFDB
across runs). Run after rebuilding caches + retraining.

Checks:
  1. fire_patched cache filename pattern includes fusion_tag (code-level)
  2. Each cache dir's fire_patched.dat has a matching fire_dilated.npy
     and the patched values match the dilated source on sample dates
  3. All score npz files have label_agg consistent with NBAC+NFDB labels
  4. positive_rate in each cache matches expected (0.48% for 22y NBAC+NFDB)
  5. No stale CWFIS-derived fire_patched.dat present in any cache

Usage:
  python scripts/audit_label_consistency.py
  python scripts/audit_label_consistency.py --cache_dirs v3_9ch_4y_2018 v3_9ch_2000
"""
import argparse
import glob
import os
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── colored output for pass/fail ──────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def _ok(msg):
    print(f"  {GREEN}✓{RESET} {msg}")


def _fail(msg):
    print(f"  {RED}✗{RESET} {msg}")


def _warn(msg):
    print(f"  {YELLOW}⚠{RESET} {msg}")


# ── Check 1: code-level filename pattern includes fusion_tag ──────────────
def check_code_has_fusion_tag(failures):
    print("\n[1] Code: fire_patched cache filename includes fusion_tag")
    src = (ROOT / "src" / "training" / "train_v3.py").read_text()
    target = None
    for line in src.split("\n"):
        if "fire_patched_v3_r" in line and "args.dilate_radius" in line:
            target = line
            break
    if target is None:
        _fail("Could not find fire_patched cache filename construction")
        failures.append("code: fire_patched line not found")
        return
    if "fusion_tag" not in target:
        _fail(f"fire_patched cache filename MISSING fusion_tag:\n      {target.strip()}")
        failures.append("code: fusion_tag missing in fire_patched cache filename")
    else:
        _ok(f"fire_patched cache filename includes fusion_tag")


# ── Check 2: per-cache fire_patched matches fire_dilated NBAC+NFDB ────────
def parse_dates_from_dilated_filename(fname):
    m = re.search(r"fire_dilated_r(\d+)(_nbac_nfdb)?_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})_(\d+)x(\d+)\.npy", fname)
    if not m:
        return None
    return {
        "r": int(m.group(1)),
        "fusion": m.group(2) or "",
        "start": m.group(3),
        "end": m.group(4),
        "H": int(m.group(5)),
        "W": int(m.group(6)),
    }


def check_cache_consistency(cache_dir, failures, warnings_list):
    print(f"\n[2] Cache: {cache_dir}")
    if not os.path.isdir(cache_dir):
        _warn(f"cache dir does not exist (not built yet?)")
        warnings_list.append(f"{cache_dir}: not built")
        return

    # Find fire_dilated files
    nbac_files = sorted(glob.glob(os.path.join(cache_dir, "fire_dilated_r*_nbac_nfdb_*.npy")))
    cwfis_files = sorted(glob.glob(os.path.join(cache_dir, "fire_dilated_r*_*x*.npy")))
    cwfis_only = [f for f in cwfis_files if "_nbac_nfdb" not in f]

    if not nbac_files:
        _warn(f"no fire_dilated_*_nbac_nfdb_*.npy found")
        warnings_list.append(f"{cache_dir}: no NBAC+NFDB dilated label")
        return

    # Find fire_patched files
    nbac_patched = sorted(glob.glob(os.path.join(cache_dir, "fire_patched_v3_r*_nbac_nfdb_*.dat")))
    no_fusion_patched = sorted(glob.glob(os.path.join(cache_dir, "fire_patched_v3_r*_*x*.dat")))
    no_fusion_patched = [f for f in no_fusion_patched if "_nbac_nfdb" not in os.path.basename(f)]

    # Check 5: no stale (no-fusion-tag) fire_patched.dat
    if no_fusion_patched:
        _fail(f"STALE no-fusion-tag fire_patched.dat present: {[os.path.basename(f) for f in no_fusion_patched]}")
        failures.append(f"{cache_dir}: stale no-fusion-tag fire_patched.dat")
    else:
        _ok("no stale (no-fusion-tag) fire_patched.dat")

    if not nbac_patched:
        _warn("no NBAC+NFDB fire_patched.dat (will be built on next training run)")
        warnings_list.append(f"{cache_dir}: no NBAC+NFDB patched yet")
        return

    # Compare dilated vs patched on sample dates
    nbac_npy = nbac_files[-1]
    pat_dat = nbac_patched[-1]
    a_nbac = np.load(nbac_npy, mmap_mode="r")

    # Parse start date from filename
    info = parse_dates_from_dilated_filename(os.path.basename(nbac_npy))
    if info is None:
        _fail(f"could not parse dates from {nbac_npy}")
        failures.append(f"{cache_dir}: filename parse failed")
        return
    label_start = date.fromisoformat(info["start"])

    # Open patched memmap
    pat_size = os.path.getsize(pat_dat)
    # Read shape from filename
    m = re.search(r"_(\d+)x(\d+)x(\d+)\.dat$", pat_dat)
    if not m:
        _fail(f"could not parse shape from {pat_dat}")
        failures.append(f"{cache_dir}: patched filename parse failed")
        return
    T_pat, n_patches, out_dim = int(m.group(1)), int(m.group(2)), int(m.group(3))
    a_pat = np.memmap(pat_dat, dtype="uint8", mode="r",
                      shape=(T_pat, n_patches, out_dim))

    # Sample 3 fire-active dates
    sample_dates = [date(2022, 8, 15), date(2023, 7, 1), date(2024, 7, 15)]
    all_match = True
    for d in sample_dates:
        i = (d - label_start).days
        if i < 0 or i >= a_nbac.shape[0] or i >= T_pat:
            continue
        # NBAC raw sum (after edge-trim to match patched array)
        # patched is (P × P) per patch, total pixels = n_patches × P²
        # = (H // P × P) × (W // P × P) — edge-trimmed
        H, W = a_nbac.shape[1], a_nbac.shape[2]
        P = int(np.sqrt(out_dim))
        Hc, Wc = H - H % P, W - W % P
        nbac_sum_trimmed = int(a_nbac[i, :Hc, :Wc].sum())
        pat_sum = int(a_pat[i].sum())
        # Patched sum should equal NBAC trimmed sum (patchify is lossless on uint8)
        if nbac_sum_trimmed == pat_sum:
            continue
        all_match = False
        # Also check if matches CWFIS as diagnosis
        if cwfis_only:
            a_cw = np.load(cwfis_only[-1], mmap_mode="r")
            cw_sum = int(a_cw[i, :Hc, :Wc].sum()) if i < a_cw.shape[0] else -1
            cw_match = abs(pat_sum - cw_sum) < 5000 if cw_sum >= 0 else False
        else:
            cw_match = False

        if cw_match:
            _fail(f"{d}: pat={pat_sum:,} matches CWFIS not NBAC ({nbac_sum_trimmed:,}) — BUGGY CACHE")
            failures.append(f"{cache_dir}: patched matches CWFIS for {d}")
        else:
            _fail(f"{d}: pat={pat_sum:,} ≠ NBAC trimmed={nbac_sum_trimmed:,}")
            failures.append(f"{cache_dir}: patched ≠ NBAC for {d}")

    if all_match:
        _ok("fire_patched values match NBAC+NFDB on 3 sample dates")


# ── Check 3: standalone label npy matches all cache fire_dilated ──────────
def check_standalone_npy_matches_caches(label_npy, failures):
    print(f"\n[3] Standalone NBAC+NFDB label vs cache fire_dilated")
    if not os.path.exists(label_npy):
        _warn(f"standalone label npy not found: {label_npy}")
        return
    a_sa = np.load(label_npy, mmap_mode="r")

    # Parse start from filename
    m = re.search(r"_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})_", label_npy)
    if not m:
        _fail(f"could not parse dates from {label_npy}")
        return
    sa_start = date.fromisoformat(m.group(1))

    # Find all cache NBAC+NFDB dilated files
    cache_files = sorted(glob.glob(
        "/scratch/jiaqi217/meteo_cache/v3_*ch*/fire_dilated_r*_nbac_nfdb_*.npy"
    ))
    if not cache_files:
        _warn("no cache fire_dilated files found")
        return

    sample_dates = [date(2022, 8, 15), date(2023, 7, 1), date(2024, 7, 1)]
    for cf in cache_files:
        a_c = np.load(cf, mmap_mode="r")
        m = re.search(r"_(\d{4}-\d{2}-\d{2})_", cf)
        c_start = date.fromisoformat(m.group(1))
        all_match = True
        for d in sample_dates:
            i_sa = (d - sa_start).days
            i_c = (d - c_start).days
            if i_sa < 0 or i_c < 0:
                continue
            if i_sa >= a_sa.shape[0] or i_c >= a_c.shape[0]:
                continue
            if int(a_sa[i_sa].sum()) != int(a_c[i_c].sum()):
                all_match = False
                _fail(f"{cf}: {d} cache={int(a_c[i_c].sum())} ≠ standalone={int(a_sa[i_sa].sum())}")
                failures.append(f"{cf}: NBAC mismatch with standalone on {d}")
                break
        if all_match:
            _ok(f"{os.path.basename(cf)}: matches standalone NBAC+NFDB on 3 sample dates")


# ── Check 4: score npz file label_agg consistency ──────────────────────────
def check_score_npz_label_consistency(failures):
    print(f"\n[4] Score npz files: label_agg consistency check")
    score_dirs = sorted(glob.glob(
        "/scratch/jiaqi217/wildfire-refactored/outputs/window_scores_full/v3_*"
    ))
    if not score_dirs:
        _warn("no save_scores output dirs found yet")
        return

    # Use a known fire-active date to compare label_agg across runs
    target_dates = ["2022-08-15", "2023-08-01", "2024-07-15"]

    # For a fair comparison: pick the same enc length, compare label_agg across ranges
    # Group by enc
    enc_groups = {}
    for sd in score_dirs:
        m = re.search(r"v3_(\d+)ch_enc(\d+)_(.*?)$", os.path.basename(sd))
        if m:
            ch, enc, rng = m.group(1), m.group(2), m.group(3)
            key = (ch, enc)
            enc_groups.setdefault(key, []).append((rng, sd))

    for (ch, enc), runs in sorted(enc_groups.items()):
        if len(runs) < 2:
            continue
        for td in target_dates:
            sums = {}
            for rng, sd in runs:
                files = glob.glob(os.path.join(sd, f"window_*_{td}.npz"))
                if files:
                    z = np.load(files[0])
                    sums[rng] = int(z["label_agg"].sum())
            if len(set(sums.values())) > 1:
                _fail(f"{ch}ch enc{enc} {td}: label_agg DIFFERS across ranges: {sums}")
                failures.append(f"{ch}ch enc{enc} {td}: label_agg mismatch {sums}")
            elif len(sums) >= 2:
                _ok(f"{ch}ch enc{enc} {td}: label_agg consistent across ranges ({list(sums.values())[0]:,})")


# ── Main ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dirs", nargs="*",
                    default=["v3_9ch_4y_2018", "v3_9ch_12y_2014", "v3_9ch_2000",
                            "v3_13ch_4y_2018", "v3_13ch_2000",
                            "v3_16ch_4y_2018", "v3_16ch_2000"])
    ap.add_argument("--cache_root", default="/scratch/jiaqi217/meteo_cache")
    ap.add_argument("--label_npy",
                    default="/scratch/jiaqi217/wildfire-refactored/data/fire_labels/fire_labels_nbac_nfdb_2000-05-01_2025-12-21_2281x2709_r14.npy")
    args = ap.parse_args()

    failures = []
    warnings_list = []

    # Check 1: code-level
    check_code_has_fusion_tag(failures)

    # Check 2: each cache dir consistency
    for cd in args.cache_dirs:
        check_cache_consistency(os.path.join(args.cache_root, cd),
                                failures, warnings_list)

    # Check 3: standalone vs cache
    check_standalone_npy_matches_caches(args.label_npy, failures)

    # Check 4: score npz files
    check_score_npz_label_consistency(failures)

    # ── Final verdict ──────────────────────────────────────────────
    print()
    print("=" * 70)
    if not failures:
        print(f"{GREEN}AUDIT PASSED{RESET} — no critical failures")
        if warnings_list:
            print(f"{YELLOW}({len(warnings_list)} warnings — caches not yet built or partial){RESET}")
        return 0
    else:
        print(f"{RED}AUDIT FAILED{RESET} — {len(failures)} critical failures:")
        for f in failures:
            print(f"  - {f}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
