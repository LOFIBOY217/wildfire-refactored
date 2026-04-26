#!/usr/bin/env python3
"""
Score separability analysis — how cleanly does the model split fire
pixels (label=1) from non-fire pixels (label=0)?

Computes per window + aggregated:
  mean_pos, mean_neg, ratio       — basic separation
  median_pos, median_neg          — robust to outliers
  cohen_d                          — standardized effect size
  ks_distance                      — max gap between empirical CDFs
  pos_q90 / neg_q90               — top tail comparison

Also for NOVEL labels (excluding patches burning in past 30 days),
to check whether separation holds for genuinely new ignitions vs
fire-continuation pixels.

Output:
  outputs/separability_<run>.csv         per-window
  outputs/separability_<run>_summary.csv aggregated mean ± CI

Usage:
  python -m scripts.analyze_score_separability \\
      --scores_dir outputs/window_scores/v3_9ch_enc28_4y_2018/ \\
      --fire_label_npy data/fire_labels/fire_labels_nbac_nfdb_*.npy \\
      --output_csv outputs/separability_enc28_4y.csv
"""
import argparse
import csv
import glob
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compute_lift_from_scores import patchify  # noqa: E402


def cohen_d(pos, neg):
    if len(pos) < 2 or len(neg) < 2:
        return float('nan')
    mp, mn = pos.mean(), neg.mean()
    sp, sn = pos.std(ddof=1), neg.std(ddof=1)
    pooled = np.sqrt(((len(pos) - 1) * sp ** 2 + (len(neg) - 1) * sn ** 2)
                     / (len(pos) + len(neg) - 2))
    if pooled < 1e-12:
        return float('nan')
    return float((mp - mn) / pooled)


def ks_distance(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    a = np.sort(pos); b = np.sort(neg)
    all_v = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, all_v, side="right") / len(a)
    cdf_b = np.searchsorted(b, all_v, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def separability(score_flat, label_flat, prefix):
    pos = score_flat[label_flat == 1].astype(np.float32)
    neg = score_flat[label_flat == 0].astype(np.float32)
    if len(pos) == 0:
        return {f"{prefix}_n_pos": 0}
    rec = {
        f"{prefix}_n_pos": len(pos),
        f"{prefix}_n_neg": len(neg),
        f"{prefix}_mean_pos": float(pos.mean()),
        f"{prefix}_mean_neg": float(neg.mean()),
        f"{prefix}_ratio": float(pos.mean() / max(neg.mean(), 1e-9)),
        f"{prefix}_median_pos": float(np.median(pos)),
        f"{prefix}_median_neg": float(np.median(neg)),
        f"{prefix}_q90_pos": float(np.quantile(pos, 0.9)),
        f"{prefix}_q90_neg": float(np.quantile(neg, 0.9)),
        f"{prefix}_q10_pos": float(np.quantile(pos, 0.1)),
        f"{prefix}_q10_neg": float(np.quantile(neg, 0.1)),
        f"{prefix}_cohen_d": cohen_d(pos, neg),
        f"{prefix}_ks": ks_distance(pos, neg),
    }
    return rec


def build_novel(win_date, label_total_p, fire_full, label_start, lookback_days, P):
    past_start = win_date - timedelta(days=lookback_days + 7)
    past_start_idx = max(0, (past_start - label_start).days)
    past_end_idx = min(fire_full.shape[0], (win_date - label_start).days + 1)
    burn_recent = fire_full[past_start_idx:past_end_idx].max(axis=0).astype(np.uint8)
    burn_recent_p = patchify(burn_recent, P)
    if burn_recent_p.shape != label_total_p.shape:
        return None
    return ((label_total_p > 0) & (burn_recent_p == 0)).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dir", required=True)
    ap.add_argument("--fire_label_npy", required=True)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--lookback_days", type=int, default=30)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--summary_csv", default=None)
    args = ap.parse_args()

    if args.summary_csv is None:
        args.summary_csv = args.output_csv.replace(".csv", "_summary.csv")

    P = args.patch_size
    print(f"=== separability analysis: {args.scores_dir} ===")

    fire_full = np.load(args.fire_label_npy, mmap_mode="r")
    sidecar = str(args.fire_label_npy).rsplit(".", 1)[0] + ".json"
    label_start = date.fromisoformat("2000-05-01")
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            label_start = date.fromisoformat(json.load(f)["date_range"][0])

    files = sorted(glob.glob(os.path.join(args.scores_dir, "window_*.npz")))
    print(f"  {len(files)} window files")

    rows = []
    for fpath in files:
        z = np.load(fpath)
        prob = z["prob_agg"].astype(np.float32)
        label_total = z["label_agg"].astype(np.uint8)
        win_date_str = str(z["win_date"])
        if not win_date_str:
            continue
        win_date = date.fromisoformat(win_date_str)

        rec = {"win_date": win_date_str, "n_total_fire": int(label_total.sum())}

        # Total label separability
        rec.update(separability(prob.reshape(-1), label_total.reshape(-1), "total"))

        # Novel label separability
        novel = build_novel(win_date, label_total, fire_full, label_start,
                             args.lookback_days, P)
        if novel is not None:
            rec.update(separability(prob.reshape(-1), novel.reshape(-1),
                                     f"novel{args.lookback_days}d"))

        rows.append(rec)
        if len(rows) % 50 == 0 or len(rows) == 1:
            print(f"  done {len(rows)}/{len(files)}")

    # Per-window CSV
    fieldnames = sorted({k for r in rows for k in r.keys()})
    fieldnames = (["win_date", "n_total_fire"]
                  + [k for k in fieldnames if k not in ("win_date", "n_total_fire")])
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"\n  per-window → {args.output_csv}  ({len(rows)} rows)")

    # Summary with bootstrap CI
    rng = np.random.default_rng(0)
    summary_rows = []
    for col in fieldnames:
        if col in ("win_date", "n_total_fire"):
            continue
        vals = np.array([r[col] for r in rows
                          if col in r and not (isinstance(r[col], float)
                                                and np.isnan(r[col]))])
        if len(vals) == 0:
            continue
        boot = np.empty(2000)
        for i in range(2000):
            boot[i] = vals[rng.integers(0, len(vals), len(vals))].mean()
        lo, hi = np.percentile(boot, [2.5, 97.5])
        summary_rows.append({
            "metric": col, "mean": float(vals.mean()),
            "ci_lo": float(lo), "ci_hi": float(hi), "n": len(vals),
        })

    with open(args.summary_csv, "w") as f:
        wr = csv.DictWriter(f, fieldnames=["metric", "mean", "ci_lo", "ci_hi", "n"])
        wr.writeheader()
        for r in summary_rows:
            wr.writerow(r)
    print(f"  summary  → {args.summary_csv}")

    # Print headline
    print("\n" + "=" * 70)
    print("HEADLINE separability")
    print("=" * 70)
    by_m = {r["metric"]: r for r in summary_rows}
    for prefix in ["total", f"novel{args.lookback_days}d"]:
        print(f"\n  {prefix}:")
        for stat in ["mean_pos", "mean_neg", "ratio", "cohen_d", "ks",
                     "q90_pos", "q90_neg"]:
            key = f"{prefix}_{stat}"
            if key in by_m:
                r = by_m[key]
                print(f"    {stat:14s} = {r['mean']:7.4f}  "
                      f"[{r['ci_lo']:7.4f}, {r['ci_hi']:7.4f}]  n={r['n']}")


if __name__ == "__main__":
    main()
