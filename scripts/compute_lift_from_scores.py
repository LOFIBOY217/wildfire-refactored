#!/usr/bin/env python3
"""
Compute Lift@K (total + novel-ignition) from per-window per-pixel score
arrays dumped by train_v3 --save_window_scores_dir.

Each window is a .npz file containing:
  prob_agg   : (n_patches, P²) float16 model output
  label_agg  : (n_patches, P²) uint8  ground-truth fire (max over lead window)
  hs, he, ts, te : window indices
  win_date   : str
  win_idx    : int

For novel-ignition we also need the FULL fire stack to look up "was this
patch burning in the lookback window?". We re-derive this from the same
fire_label_npy used during training.

Usage:
    python -m scripts.compute_lift_from_scores \\
        --scores_dir outputs/window_scores/v3_9ch_enc28_4y_2018/ \\
        --fire_label_npy data/fire_labels/fire_labels_nbac_nfdb_*.npy \\
        --pred_start 2022-05-01 --pred_end 2024-10-31 \\
        --output_csv outputs/model_novel_lift.csv
"""
import argparse
import csv
import glob
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def lift_at_k(score_flat, label_flat, k):
    n_fire = int(label_flat.sum())
    n = label_flat.size
    if n_fire == 0:
        return float('nan'), float('nan'), 0
    base_rate = n_fire / n
    order = np.argsort(score_flat)[::-1]
    tp = int(label_flat[order[:k]].sum())
    prec = tp / k
    return prec / base_rate, prec, n_fire


def patchify(frame_2d, P):
    """(H, W) → (n_patches, P²)."""
    H, W = frame_2d.shape
    Hc, Wc = H - H % P, W - W % P
    nph, npw = Hc // P, Wc // P
    out = frame_2d[:Hc, :Wc].reshape(nph, P, npw, P)
    return out.transpose(0, 2, 1, 3).reshape(nph * npw, P * P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dir", required=True,
                    help="Directory with window_NNNN_DATE.npz files")
    ap.add_argument("--fire_label_npy", required=True,
                    help="Same .npy used at training time (NBAC+NFDB stack)")
    ap.add_argument("--label_start_date", default="2000-05-01",
                    help="Date corresponding to fire_label_npy[0] "
                         "(check sidecar JSON for actual start)")
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--lookback_days_list", nargs="+", type=int,
                    default=[7, 30, 90])
    ap.add_argument("--k_values", nargs="+", type=int,
                    default=[1000, 5000, 10000])
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--run_name", default=None,
                    help="Label for this run in output CSV (default = "
                         "basename of scores_dir)")
    args = ap.parse_args()

    run_name = args.run_name or os.path.basename(args.scores_dir.rstrip("/"))
    print(f"=== compute lift from scores: {run_name} ===")

    # Load full fire stack (mmap so we don't read 6GB)
    fire_full = np.load(args.fire_label_npy, mmap_mode="r")  # (T, H, W) uint8
    print(f"  fire_full shape: {fire_full.shape}  dtype: {fire_full.dtype}")
    label_start = date.fromisoformat(args.label_start_date)

    # Try to read sidecar JSON for accurate label_start
    sidecar = str(args.fire_label_npy).rsplit(".", 1)[0] + ".json"
    if os.path.exists(sidecar):
        import json
        with open(sidecar) as f:
            prov = json.load(f)
        label_start = date.fromisoformat(prov["date_range"][0])
        print(f"  label_start (from sidecar): {label_start}")

    P = args.patch_size

    # Iterate window files
    files = sorted(glob.glob(os.path.join(args.scores_dir, "window_*.npz")))
    print(f"  found {len(files)} window files")
    if not files:
        sys.exit("ERROR: no window_*.npz files in scores_dir")

    rows = []
    for fpath in files:
        z = np.load(fpath)
        prob = z["prob_agg"].astype(np.float32)        # (n_patches, P²)
        label_total = z["label_agg"].astype(np.uint8)  # (n_patches, P²)
        win_date_str = str(z["win_date"])
        hs, he, ts, te = int(z["hs"]), int(z["he"]), int(z["ts"]), int(z["te"])
        n_patches = prob.shape[0]

        # Compute novel labels by patchifying the recent-fire frame
        # NOTE: hs/he/ts/te are indices into the TRAINING date axis, not
        # absolute calendar days. We map back via win_date (which is the
        # forecast issue date = the date of patch index `he-1` in training).
        win_date = date.fromisoformat(win_date_str) if win_date_str else None

        if win_date is None:
            print(f"  skip {fpath}: no win_date")
            continue

        # Map win_date to absolute index in fire_full
        # win_date = date corresponding to the encoder-end (he-1 in training T axis)
        # forecast period = lead_start (14d) to lead_end (45d) AFTER win_date
        # encoder period = past 7 days BEFORE win_date
        rec = {
            "run": run_name,
            "win_date": win_date_str,
            "n_fire_total": int(label_total.sum()),
        }

        for lookback in args.lookback_days_list:
            # past_window: [win_date - lookback - 7, win_date]  (encoder + lookback)
            past_start = win_date - timedelta(days=lookback + 7)
            past_end = win_date  # exclusive at win_date+1
            past_start_idx = (past_start - label_start).days
            past_end_idx = (past_end - label_start).days + 1
            past_start_idx = max(0, past_start_idx)
            past_end_idx = min(fire_full.shape[0], past_end_idx)
            if past_start_idx >= past_end_idx:
                rec[f"lift_novel_{lookback}d_5000"] = float('nan')
                continue
            # Build "burning recently" map (H, W)
            burn_recent = (fire_full[past_start_idx:past_end_idx]
                           .max(axis=0).astype(np.uint8))
            # Patchify
            burn_recent_p = patchify(burn_recent, P)        # (n_patches, P²)
            if burn_recent_p.shape != label_total.shape:
                print(f"  shape mismatch {burn_recent_p.shape} vs "
                      f"{label_total.shape}; skip")
                continue
            # Novel = total fire AND not burning recently
            novel = (label_total > 0) & (burn_recent_p == 0)
            novel = novel.astype(np.uint8)

            score_flat = prob.reshape(-1)
            label_total_flat = label_total.reshape(-1)
            novel_flat = novel.reshape(-1)
            n_novel = int(novel_flat.sum())
            n_total = int(label_total_flat.sum())

            for k in args.k_values:
                lift_t, prec_t, _ = lift_at_k(score_flat, label_total_flat, k)
                lift_n, prec_n, _ = lift_at_k(score_flat, novel_flat, k)
                if k == 5000 and lookback == args.lookback_days_list[0]:
                    rec[f"lift_total_{k}"] = lift_t
                rec[f"lift_novel_{lookback}d_{k}"] = lift_n
            rec[f"n_novel_{lookback}d"] = n_novel
            rec[f"n_total"] = n_total
            rec[f"novel_frac_{lookback}d"] = (n_novel / max(n_total, 1))
        rows.append(rec)
        print(f"  {win_date_str}: n_total={rec.get('n_total', 0):>8,}  "
              + " ".join(f"novel_{lb}={rec.get(f'lift_novel_{lb}d_5000', 0):.2f}x"
                         for lb in args.lookback_days_list))

    # Aggregate + write CSV
    print(f"\n  writing {args.output_csv}")
    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)

    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY  {run_name}")
    print("=" * 70)
    for lookback in args.lookback_days_list:
        for k in args.k_values:
            key = f"lift_novel_{lookback}d_{k}"
            vals = [r[key] for r in rows
                    if key in r and not (isinstance(r[key], float)
                                          and np.isnan(r[key]))]
            if vals:
                print(f"  novel_{lookback}d  L@{k}  "
                      f"= {np.mean(vals):.2f} ± {np.std(vals):.2f}x  "
                      f"(n={len(vals)})")
    if "lift_total_5000" in rows[0]:
        vals = [r["lift_total_5000"] for r in rows
                if "lift_total_5000" in r and not np.isnan(r["lift_total_5000"])]
        if vals:
            print(f"\n  total      L@5000 = {np.mean(vals):.2f} ± {np.std(vals):.2f}x  "
                  f"(n={len(vals)}) [for reference]")


if __name__ == "__main__":
    main()
