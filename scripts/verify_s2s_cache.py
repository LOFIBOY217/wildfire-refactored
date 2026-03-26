#!/usr/bin/env python3
"""
Quick sanity checks on the S2S decoder cache before training.

Usage:
    python scripts/verify_s2s_cache.py \
        --cache /scratch/jiaqi217/meteo_cache/s2s_decoder_cache.dat \
        --data-start 2018-05-01 \
        --pred-start 2022-05-01 \
        --pred-end   2024-10-31
"""
import argparse
import os
import sys
from datetime import date, timedelta

import numpy as np

S2S_N_LEADS = 32
S2S_N_CHANNELS = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="Path to s2s_decoder_cache.dat")
    ap.add_argument("--data-start", default="2018-05-01")
    ap.add_argument("--pred-start", default="2022-05-01")
    ap.add_argument("--pred-end", default="2024-10-31")
    ap.add_argument("--patch-size", type=int, default=16)
    ap.add_argument("--n-patches", type=int, default=23998,
                    help="Expected n_patches (2272//16 * 2704//16 = 23998)")
    args = ap.parse_args()

    cache_path = args.cache
    dates_path = cache_path + ".dates.npy"

    # --- Check files exist ---
    print("=" * 60)
    print("S2S DECODER CACHE VERIFICATION")
    print("=" * 60)
    for f in [cache_path, dates_path]:
        if not os.path.exists(f):
            print(f"FAIL: File not found: {f}")
            sys.exit(1)
        sz = os.path.getsize(f) / 1e9
        print(f"  OK: {f}  ({sz:.3f} GB)")

    # --- Load dates ---
    s2s_dates = np.load(dates_path, allow_pickle=True)
    n_dates = len(s2s_dates)
    print(f"\n[1] DATE COVERAGE")
    print(f"  Total dates in cache: {n_dates}")
    print(f"  Range: {s2s_dates[0]} .. {s2s_dates[-1]}")

    # Check sorted
    for i in range(1, n_dates):
        if s2s_dates[i] <= s2s_dates[i - 1]:
            print(f"  WARN: Dates not sorted at index {i}: {s2s_dates[i-1]} >= {s2s_dates[i]}")
            break
    else:
        print("  OK: Dates are sorted")

    # Check duplicates
    unique = set(s2s_dates)
    if len(unique) < n_dates:
        print(f"  WARN: {n_dates - len(unique)} duplicate dates found!")
    else:
        print("  OK: No duplicate dates")

    # --- Check shape ---
    n_patches = args.n_patches
    expected_bytes = n_dates * n_patches * S2S_N_LEADS * S2S_N_CHANNELS * 2
    actual_bytes = os.path.getsize(cache_path)
    print(f"\n[2] SHAPE CHECK")
    print(f"  Expected shape: ({n_dates}, {n_patches}, {S2S_N_LEADS}, {S2S_N_CHANNELS})")
    print(f"  Expected size: {expected_bytes / 1e9:.3f} GB")
    print(f"  Actual size:   {actual_bytes / 1e9:.3f} GB")
    if actual_bytes != expected_bytes:
        print(f"  FAIL: Size mismatch! Diff = {actual_bytes - expected_bytes} bytes")
        sys.exit(1)
    else:
        print("  OK: Size matches")

    # --- Open memmap and spot-check ---
    cache = np.memmap(cache_path, dtype='float16', mode='r',
                      shape=(n_dates, n_patches, S2S_N_LEADS, S2S_N_CHANNELS))

    print(f"\n[3] VALUE CHECKS (random sample of 20 dates)")
    rng = np.random.default_rng(42)
    sample_idxs = rng.choice(n_dates, size=min(20, n_dates), replace=False)
    sample_idxs.sort()

    all_zero_dates = []
    nan_dates = []
    inf_dates = []
    for idx in sample_idxs:
        row = cache[idx]  # (n_patches, 32, 6)
        row_f32 = row.astype(np.float32)
        n_nan = np.isnan(row_f32).sum()
        n_inf = np.isinf(row_f32).sum()
        n_zero = (row_f32 == 0).sum()
        total = row_f32.size
        zero_frac = n_zero / total

        if n_nan > 0:
            nan_dates.append((s2s_dates[idx], n_nan))
        if n_inf > 0:
            inf_dates.append((s2s_dates[idx], n_inf))
        if zero_frac > 0.99:
            all_zero_dates.append((s2s_dates[idx], zero_frac))

    if nan_dates:
        print(f"  WARN: {len(nan_dates)} sampled dates have NaN values:")
        for d, n in nan_dates[:5]:
            print(f"    {d}: {n} NaN")
    else:
        print("  OK: No NaN in sampled dates")

    if inf_dates:
        print(f"  WARN: {len(inf_dates)} sampled dates have Inf values:")
        for d, n in inf_dates[:5]:
            print(f"    {d}: {n} Inf")
    else:
        print("  OK: No Inf in sampled dates")

    if all_zero_dates:
        print(f"  WARN: {len(all_zero_dates)} sampled dates are >99% zeros:")
        for d, f in all_zero_dates:
            print(f"    {d}: {f*100:.1f}% zeros")
    else:
        print("  OK: No nearly-all-zero dates in sample")

    # --- Check per-channel stats for a few dates ---
    print(f"\n[4] CHANNEL STATISTICS (5 random dates, patch-averaged)")
    ch_names = ["2t", "2d", "tcw", "sm20", "st20", "VPD"]
    for idx in sample_idxs[:5]:
        row = cache[idx].astype(np.float32)  # (n_patches, 32, 6)
        # Mean across patches and leads for each channel
        ch_means = np.nanmean(row, axis=(0, 1))
        ch_stds = np.nanstd(row, axis=(0, 1))
        stats_str = "  ".join(f"{ch_names[c]}={ch_means[c]:+.2f}±{ch_stds[c]:.2f}"
                              for c in range(S2S_N_CHANNELS))
        print(f"  {s2s_dates[idx]}: {stats_str}")

    # --- Check date coverage vs training period ---
    print(f"\n[5] COVERAGE vs TRAINING PERIOD")
    def _parse(s):
        parts = s.split("-")
        return date(int(parts[0]), int(parts[1]), int(parts[2]))

    pred_start = _parse(args.pred_start)
    pred_end = _parse(args.pred_end)
    data_start = _parse(args.data_start)

    s2s_date_set = set(str(d) for d in s2s_dates)

    # Check train period: data_start .. pred_start-1
    train_dates = []
    cur = data_start
    while cur < pred_start:
        train_dates.append(str(cur))
        cur += timedelta(days=1)
    train_hits = sum(1 for d in train_dates if d in s2s_date_set)
    print(f"  Train period ({data_start} → {pred_start - timedelta(days=1)}):")
    print(f"    {train_hits}/{len(train_dates)} days have S2S data ({train_hits/len(train_dates)*100:.1f}%)")

    # Check val period: pred_start .. pred_end
    val_dates = []
    cur = pred_start
    while cur <= pred_end:
        val_dates.append(str(cur))
        cur += timedelta(days=1)
    val_hits = sum(1 for d in val_dates if d in s2s_date_set)
    print(f"  Val period ({pred_start} → {pred_end}):")
    print(f"    {val_hits}/{len(val_dates)} days have S2S data ({val_hits/len(val_dates)*100:.1f}%)")

    # Note about S2S issue frequency
    print(f"\n  NOTE: ECMWF S2S issues forecasts on Mon/Thu (2017-2022), daily (2023+).")
    print(f"  Not every calendar day will have S2S data — this is expected.")
    print(f"  Windows without S2S coverage will get all-zero decoder input.")

    # Count how many windows would get zero decoder input
    if val_hits == 0:
        print(f"\n  CRITICAL: Val period has ZERO S2S dates! Check data range.")
    elif val_hits < len(val_dates) * 0.2:
        print(f"\n  WARN: Val period has <20% S2S coverage. Many val windows will be zero-filled.")

    print(f"\n{'=' * 60}")
    print("VERIFICATION COMPLETE")
    print(f"{'=' * 60}")
    del cache


if __name__ == "__main__":
    main()
