"""
Ensemble all available save_window_scores npz files. For each window,
average prob_agg across selected ckpts, then compute Lift@5000 and
Lift@30km on the ensembled scores. Reports per-window + bootstrap CI.

Usage:
  python -m scripts.ensemble_ckpts_lift \\
      --score_dirs outputs/window_scores_full/v3_9ch_enc21_12y_2014 \\
                   outputs/window_scores_full/v3_9ch_enc21_12y_2014_climsim \\
                   outputs/window_scores_full/v3_13ch_enc28_12y_2014 \\
      --output outputs/ensemble_lift.json
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


def list_score_files(score_dir, pred_start_iso, pred_end_iso):
    pat = re.compile(r"window_\d+_(\d{4}-\d{2}-\d{2})\.npz$")
    out = {}
    for f in sorted(Path(score_dir).glob("window_*.npz")):
        m = pat.search(f.name)
        if not m:
            continue
        d = m.group(1)
        if d < pred_start_iso or d > pred_end_iso:
            continue
        out[d] = f
    return out


def lift_at_k(s, y, k):
    valid = np.isfinite(s) & np.isfinite(y)
    s = s[valid]; y = y[valid]
    if len(s) == 0 or y.sum() == 0:
        return float("nan")
    base = float(y.mean())
    if base <= 0:
        return float("nan")
    k = min(k, len(s))
    top = np.argpartition(-s, k - 1)[:k]
    return float(y[top].mean() / base)


def lift_30km(score_2d, label_2d, k, pool=15):
    H, W = score_2d.shape
    Hp, Wp = H // pool, W // pool
    s = score_2d[:Hp * pool, :Wp * pool].reshape(Hp, pool, Wp, pool).max(axis=(1, 3))
    y = label_2d[:Hp * pool, :Wp * pool].reshape(Hp, pool, Wp, pool).max(axis=(1, 3))
    return lift_at_k(s.flatten(), y.flatten(), k)


def patches_to_2d(patch_arr, n_rows, n_cols, P):
    return (patch_arr.reshape(n_rows, n_cols, P, P)
            .transpose(0, 2, 1, 3).reshape(n_rows * P, n_cols * P))


def bootstrap_ci(values, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.asarray([v for v in values if np.isfinite(v)])
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), size=len(arr))
        means.append(arr[idx].mean())
    return float(arr.mean()), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score_dirs", nargs="+", required=True)
    ap.add_argument("--pred_start", default="2022-05-01")
    ap.add_argument("--pred_end", default="2025-10-31")
    ap.add_argument("--n_rows", type=int, default=142)
    ap.add_argument("--n_cols", type=int, default=169)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--k", type=int, default=5000)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    # Build dict[score_dir] -> dict[date] -> npz path
    files_by_dir = {}
    for sd in args.score_dirs:
        files_by_dir[sd] = list_score_files(sd, args.pred_start, args.pred_end)
        print(f"  {Path(sd).name}: {len(files_by_dir[sd])} windows")

    # Common dates across ALL dirs
    common_dates = sorted(set.intersection(*[set(d.keys()) for d in files_by_dir.values()]))
    print(f"\nCommon windows across {len(args.score_dirs)} ckpts: {len(common_dates)}")

    if not common_dates:
        print("[ERROR] no common windows")
        sys.exit(1)

    NR, NC, P = args.n_rows, args.n_cols, args.patch_size

    per_window = []
    for di, date in enumerate(common_dates):
        # Load + average prob_agg across all ckpts (logit-mean, not prob-mean,
        # using log-of-prob since outputs are sigmoid'd; we approximate by
        # averaging the prob — close enough for top-K rank purposes).
        prob_sum = None
        label = None
        for sd, files in files_by_dir.items():
            npz = np.load(files[date])
            p = npz["prob_agg"].astype(np.float32)
            if prob_sum is None:
                prob_sum = p.copy()
                label = npz["label_agg"]
            else:
                prob_sum += p
        prob_avg = prob_sum / len(files_by_dir)

        # 2D reshape
        score_2d = patches_to_2d(prob_avg, NR, NC, P)
        label_2d = (patches_to_2d(label, NR, NC, P) > 0).astype(np.uint8)

        if label_2d.sum() == 0:
            continue
        l5k = lift_at_k(score_2d.flatten(), label_2d.flatten(), args.k)
        l30 = lift_30km(score_2d, label_2d, args.k)
        per_window.append({"win_date": date, "lift_5000": l5k, "lift_30km": l30})

        if (di + 1) % 100 == 0:
            print(f"  [{di+1}/{len(common_dates)}]")

    print(f"\nValid windows: {len(per_window)}")
    if not per_window:
        sys.exit(1)

    l5 = [w["lift_5000"] for w in per_window]
    l30 = [w["lift_30km"] for w in per_window]
    m5, lo5, hi5 = bootstrap_ci(l5)
    m30, lo30, hi30 = bootstrap_ci(l30)
    print(f"\nEnsemble of {len(args.score_dirs)} ckpts on {len(per_window)} windows:")
    print(f"  Lift@{args.k}    : {m5:.3f}× [{lo5:.3f}, {hi5:.3f}]")
    print(f"  Lift@30 km    : {m30:.3f}× [{lo30:.3f}, {hi30:.3f}]")

    out = {"n_ckpts": len(args.score_dirs), "ckpt_dirs": args.score_dirs,
           "n_windows": len(per_window),
           "lift_5000": {"mean": m5, "ci_lo": lo5, "ci_hi": hi5},
           "lift_30km": {"mean": m30, "ci_lo": lo30, "ci_hi": hi30},
           "per_window": per_window}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
