"""
Verify S2S Forecast Skill vs ERA5 Observations
===============================================
For each lead time (14–45 days), computes Pearson r, RMSE, and bias between
ECMWF S2S patch-mean forecasts and ERA5 observations over Canada.

This provides empirical evidence for the low forecast skill of S2S at
extended range, explaining why the s2s_legacy decoder underperforms the
random decoder in fire prediction experiments.

Channels verified: 2t, 2d, tcw, sm20, st20
  (VPD is S2S-only derived; skipped for direct obs comparison)

Usage:
    python -m src.data_ops.validation.verify_s2s_skill \\
        --s2s-cache /scratch/jiaqi217/meteo_cache/s2s_decoder_cache.dat \\
        --era5-dir data/ecmwf_observation \\
        --reference data/fwi_data/fwi_20250615.tif \\
        --n-dates 100 \\
        --out-csv results/s2s_skill.csv

Output:
    Per-lead table:  lead | r_2t | r_2d | r_tcw | r_sm20 | r_st20 | r_mean
    Also saves CSV for plotting.
"""

import argparse
import os
import sys
from datetime import date, timedelta

import numpy as np
import rasterio

# S2S cache layout constants (must match build_s2s_decoder_cache.py)
S2S_LEADS     = list(range(14, 46))   # 14..45 inclusive, 32 values
S2S_CHANNELS  = ["2t", "2d", "tcw", "sm20", "st20", "VPD"]
ERA5_CHANNELS = ["2t", "2d", "tcw", "sm20", "st20"]   # VPD not a direct obs file
PATCH_SIZE    = 16


def _get_grid(reference_tif):
    with rasterio.open(reference_tif) as src:
        H, W = src.height, src.width
    Hc = H - H % PATCH_SIZE
    Wc = W - W % PATCH_SIZE
    nph = Hc // PATCH_SIZE
    npw = Wc // PATCH_SIZE
    return H, W, Hc, Wc, nph, npw


def _load_era5_patch_means(era5_dir, date_str, channels, nph, npw):
    """
    Load ERA5 obs tifs for `date_str`, compute patch means.
    Returns (n_patches, len(channels)) float32, or None if any channel missing.
    """
    date_nodash = date_str.replace("-", "")
    n_patches = nph * npw
    out = np.zeros((n_patches, len(channels)), dtype=np.float32)

    for ci, ch in enumerate(channels):
        tif_path = os.path.join(era5_dir, ch, f"{ch}_{date_nodash}.tif")
        if not os.path.exists(tif_path):
            return None
        with rasterio.open(tif_path) as src:
            arr = src.read(1).astype(np.float32)   # (H, W)

        H_arr, W_arr = arr.shape
        Hc = H_arr - H_arr % PATCH_SIZE
        Wc = W_arr - W_arr % PATCH_SIZE
        arr = arr[:Hc, :Wc]

        # Patch means: reshape to (nph, P, npw, P) → mean over P×P
        patches = arr.reshape(nph, PATCH_SIZE, npw, PATCH_SIZE)
        patch_means = patches.mean(axis=(1, 3))          # (nph, npw)
        out[:, ci] = patch_means.ravel()

    return out


def _pearson_r(a, b):
    """Pearson r between 1-D arrays a and b, ignoring NaN pairs."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return np.nan
    a, b = a[mask], b[mask]
    da, db = a - a.mean(), b - b.mean()
    denom = np.sqrt((da ** 2).sum() * (db ** 2).sum())
    if denom == 0:
        return np.nan
    return float(np.dot(da, db) / denom)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s2s-cache",  required=True,
                    help="Path to s2s_decoder_cache.dat")
    ap.add_argument("--era5-dir",   required=True,
                    help="Directory with ecmwf_observation/{channel}/ sub-dirs")
    ap.add_argument("--reference",  required=True,
                    help="Reference GeoTIFF for grid dimensions (e.g. fwi_20250615.tif)")
    ap.add_argument("--n-dates",    type=int, default=100,
                    help="Number of S2S issue dates to sample (default 100)")
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--out-csv",    default=None,
                    help="Path to save per-lead results CSV")
    args = ap.parse_args()

    # ── Grid ──────────────────────────────────────────────────────────────────
    H, W, Hc, Wc, nph, npw = _get_grid(args.reference)
    n_patches = nph * npw
    print(f"Grid: {H}x{W} → cropped {Hc}x{Wc}  patches={n_patches} ({nph}x{npw})")

    # ── Load S2S cache ────────────────────────────────────────────────────────
    dates_file = args.s2s_cache + ".dates.npy"
    if not os.path.exists(dates_file):
        sys.exit(f"ERROR: S2S dates file not found: {dates_file}")
    s2s_dates = np.load(dates_file, allow_pickle=True)
    n_s2s_dates = len(s2s_dates)
    print(f"S2S cache: {n_s2s_dates} issue dates  ({s2s_dates[0]} .. {s2s_dates[-1]})")

    cache = np.memmap(args.s2s_cache, dtype="float16", mode="r",
                      shape=(n_s2s_dates, n_patches, 32, 6))
    print(f"S2S cache shape: {cache.shape}  ({os.path.getsize(args.s2s_cache)/1e9:.1f} GB)")

    # ── Sample issue dates ────────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    n_sample = min(args.n_dates, n_s2s_dates)
    sample_idxs = sorted(rng.choice(n_s2s_dates, size=n_sample, replace=False))
    print(f"Sampling {n_sample} issue dates (seed={args.seed})\n")

    # ── Per-lead accumulators ─────────────────────────────────────────────────
    n_leads    = len(S2S_LEADS)
    n_era5_ch  = len(ERA5_CHANNELS)
    all_r      = [[] for _ in range(n_leads)]   # all_r[lead_i] = list of (n_era5_ch,) arrays
    all_rmse   = [[] for _ in range(n_leads)]
    all_bias   = [[] for _ in range(n_leads)]
    skipped    = 0

    for sample_num, date_idx in enumerate(sample_idxs):
        issue_str = str(s2s_dates[date_idx])
        try:
            issue_dt = date.fromisoformat(issue_str)
        except ValueError:
            skipped += 1
            continue

        if (sample_num + 1) % 20 == 0 or sample_num == 0:
            print(f"  Processing date {sample_num+1}/{n_sample}: {issue_str}")

        for li, lead in enumerate(S2S_LEADS):
            verif_dt  = issue_dt + timedelta(days=lead)
            verif_str = verif_dt.isoformat()

            era5 = _load_era5_patch_means(args.era5_dir, verif_str, ERA5_CHANNELS, nph, npw)
            if era5 is None:
                skipped += 1
                continue

            # S2S forecast for this date × lead: (n_patches, 6)
            # Only use the 5 ERA5-comparable channels (skip VPD at index 5)
            s2s_lead = cache[date_idx, :, li, :n_era5_ch].astype(np.float32)  # (n_patches, 5)

            # Filter: remove patches where S2S is all-zero (missing forecast)
            valid_patch = ~np.all(s2s_lead == 0, axis=1)
            if valid_patch.sum() < 100:
                skipped += 1
                continue

            s2s_v  = s2s_lead[valid_patch]   # (n_valid, 5)
            era5_v = era5[valid_patch]        # (n_valid, 5)

            r_vec    = np.array([_pearson_r(s2s_v[:, c], era5_v[:, c]) for c in range(n_era5_ch)])
            rmse_vec = np.sqrt(np.nanmean((s2s_v - era5_v) ** 2, axis=0))
            bias_vec = np.nanmean(s2s_v - era5_v, axis=0)

            all_r[li].append(r_vec)
            all_rmse[li].append(rmse_vec)
            all_bias[li].append(bias_vec)

    # ── Aggregate and print ───────────────────────────────────────────────────
    print(f"\nSkipped {skipped} (date/lead/missing)\n")

    header = f"{'lead':>5}  " + "  ".join(f"r_{ch:>5}" for ch in ERA5_CHANNELS) + "  r_mean  n_dates"
    print(header)
    print("-" * len(header))

    rows = []
    for li, lead in enumerate(S2S_LEADS):
        if len(all_r[li]) == 0:
            continue
        r_mat   = np.array(all_r[li])    # (n, 5)
        r_mean  = np.nanmean(r_mat, axis=0)
        r_avg   = float(np.nanmean(r_mean))
        n_dates = len(all_r[li])
        row = [lead] + list(r_mean) + [r_avg, n_dates]
        rows.append(row)
        cols = "  ".join(f"{v:>7.4f}" for v in r_mean)
        print(f"  {lead:>3}  {cols}  {r_avg:>6.4f}  {n_dates:>5}")

    # ── Summary ───────────────────────────────────────────────────────────────
    if rows:
        r_means = [r[-2] for r in rows]
        print(f"\nOverall mean r across all leads and channels: {np.mean(r_means):.4f}")
        print(f"Lead 14 mean r: {rows[0][-2]:.4f}")
        print(f"Lead 45 mean r: {rows[-1][-2]:.4f}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        import csv
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["lead"] + [f"r_{ch}" for ch in ERA5_CHANNELS] + ["r_mean", "n_dates"])
            for row in rows:
                writer.writerow([f"{v:.6f}" if isinstance(v, float) else v for v in row])
        print(f"\nSaved to {args.out_csv}")


if __name__ == "__main__":
    main()
