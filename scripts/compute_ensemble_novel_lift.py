#!/usr/bin/env python3
"""
Ensemble novel-ignition Lift from per-checkpoint score dumps.

Reuses the per-window score dumps in outputs/window_scores_full/<run>/
(the same dirs used to build the 10-checkpoint ensemble Lift). For each
window we PROBABILITY-MEAN the prob_agg arrays across the member dirs
(matching the "ensemble_prob" variant that won standard Lift@5000), then
compute standard + novel-30d/7d/90d Lift exactly like
scripts/compute_lift_from_scores.py.

Note: only prob-mean is possible post-hoc — the npz store aggregated
probabilities (max over lead), not pre-sigmoid logits, so logit-mean
cannot be reconstructed here.

Usage:
    python -m scripts.compute_ensemble_novel_lift \\
        --scores_dirs outputs/window_scores_full/v3_9ch_enc21_12y_2014 \\
                      outputs/window_scores_full/v3_9ch_enc21_12y_2014_climsim \\
                      ... (10 dirs) \\
        --fire_label_npy data/fire_labels/fire_labels_nbac_nfdb_*.npy \\
        --output_csv outputs/model_novel_lift_ensemble_full.csv
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

from scripts.compute_lift_from_scores import lift_at_k, patchify  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dirs", nargs="+", required=True,
                    help="Member checkpoint score dirs to prob-mean.")
    ap.add_argument("--fire_label_npy", required=True)
    ap.add_argument("--label_start_date", default="2000-05-01")
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--lookback_days_list", nargs="+", type=int, default=[7, 30, 90])
    ap.add_argument("--k_values", nargs="+", type=int, default=[1000, 5000, 10000])
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--run_name", default="ensemble_prob_10ckpt")
    args = ap.parse_args()

    P = args.patch_size
    dirs = [d for d in args.scores_dirs if os.path.isdir(d)]
    print(f"=== ensemble novel lift: {args.run_name} ===")
    print(f"  members ({len(dirs)}):")
    for d in dirs:
        print(f"    {d}  ({len(glob.glob(os.path.join(d, 'window_*.npz')))} npz)")
    if len(dirs) < 2:
        sys.exit("ERROR: need >= 2 member dirs")

    fire_full = np.load(args.fire_label_npy, mmap_mode="r")
    print(f"  fire_full shape: {fire_full.shape}")
    label_start = date.fromisoformat(args.label_start_date)
    sidecar = str(args.fire_label_npy).rsplit(".", 1)[0] + ".json"
    if os.path.exists(sidecar):
        import json
        label_start = date.fromisoformat(json.load(open(sidecar))["date_range"][0])
        print(f"  label_start (sidecar): {label_start}")

    # Window filenames present in the FIRST dir; require in all others.
    base_files = sorted(os.path.basename(f)
                        for f in glob.glob(os.path.join(dirs[0], "window_*.npz")))
    print(f"  {len(base_files)} candidate windows")

    rows = []
    for bf in base_files:
        paths = [os.path.join(d, bf) for d in dirs]
        if not all(os.path.exists(p) for p in paths):
            continue  # window missing in some member → skip
        # prob-mean across members; label + indices from the first member
        probs = []
        z0 = np.load(paths[0])
        for p in paths:
            probs.append(np.load(p)["prob_agg"].astype(np.float32))
        prob = np.mean(probs, axis=0)                       # (n_patches, P²)
        label_total = z0["label_agg"].astype(np.uint8)
        win_date_str = str(z0["win_date"])
        if not win_date_str:
            continue
        win_date = date.fromisoformat(win_date_str)

        rec = {"run": args.run_name, "win_date": win_date_str,
               "n_fire_total": int(label_total.sum())}

        for lookback in args.lookback_days_list:
            past_start = win_date - timedelta(days=lookback + 7)
            ps_idx = max(0, (past_start - label_start).days)
            pe_idx = min(fire_full.shape[0], (win_date - label_start).days + 1)
            if ps_idx >= pe_idx:
                rec[f"lift_novel_{lookback}d_5000"] = float("nan")
                continue
            burn_recent = fire_full[ps_idx:pe_idx].max(axis=0).astype(np.uint8)
            burn_recent_p = patchify(burn_recent, P)
            if burn_recent_p.shape != label_total.shape:
                continue
            novel = ((label_total > 0) & (burn_recent_p == 0)).astype(np.uint8)

            score_flat = prob.reshape(-1)
            label_flat = label_total.reshape(-1)
            novel_flat = novel.reshape(-1)
            for k in args.k_values:
                lift_t, _, _ = lift_at_k(score_flat, label_flat, k)
                lift_n, _, _ = lift_at_k(score_flat, novel_flat, k)
                if k == 5000 and lookback == args.lookback_days_list[0]:
                    rec[f"lift_total_{k}"] = lift_t
                rec[f"lift_novel_{lookback}d_{k}"] = lift_n
            rec[f"n_novel_{lookback}d"] = int(novel_flat.sum())
            rec["n_total"] = int(label_flat.sum())
            rec[f"novel_frac_{lookback}d"] = int(novel_flat.sum()) / max(int(label_flat.sum()), 1)
        rows.append(rec)
        print(f"  {win_date_str}: "
              + " ".join(f"novel_{lb}={rec.get(f'lift_novel_{lb}d_5000', 0):.2f}x"
                         for lb in args.lookback_days_list))

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n  wrote {args.output_csv}  ({len(rows)} windows)")

    print("\n" + "=" * 60 + f"\nSUMMARY  {args.run_name}\n" + "=" * 60)
    for lookback in args.lookback_days_list:
        key = f"lift_novel_{lookback}d_5000"
        vals = [r[key] for r in rows if key in r and not np.isnan(r[key])]
        if vals:
            print(f"  novel_{lookback}d L@5000 = {np.mean(vals):.2f} ± {np.std(vals):.2f}x  (n={len(vals)})")
    tt = [r["lift_total_5000"] for r in rows if "lift_total_5000" in r]
    if tt:
        print(f"  standard L@5000        = {np.mean(tt):.2f} ± {np.std(tt):.2f}x")


if __name__ == "__main__":
    main()
