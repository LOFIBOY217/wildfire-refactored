#!/usr/bin/env python3
"""
Ensemble (prob-mean) FULL metric suite from member score dumps.

Mirrors compute_ensemble_novel_lift.py's prob-mean over the 10 member
dirs, but instead of novel Lift it computes the full per-window metric
suite (lift_k, f2, mcc, bss, brier, pr_auc, roc_auc) via
src.evaluation.metrics, writing a per_window JSON in the SAME shape as
the model FULL JSONs (sota_FULL_per_window.json) so the Fig 2-B plot
reads it uniformly.

lift_coarse (30km) is NOT recomputed here (needs spatial grid); pull it
from the existing ensemble_prob.json (4.37x) when needed.

Usage:
    python -m scripts.compute_ensemble_full_metrics \\
        --scores_dirs <10 member dirs> \\
        --output_json outputs/ensemble_prob_FULL_per_window.json
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import (  # noqa: E402
    compute_ranking_metrics, compute_imbalanced_metrics,
    compute_brier_decomposition,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dirs", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=5000)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--run_name", default="ensemble_prob_10ckpt")
    args = ap.parse_args()

    dirs = [d for d in args.scores_dirs if os.path.isdir(d)]
    print(f"=== ensemble full metrics: {args.run_name} ({len(dirs)} members) ===")

    def date_index(d):
        out = {}
        for f in glob.glob(os.path.join(d, "window_*.npz")):
            out[os.path.basename(f)[:-4].split("_")[-1]] = f
        return out

    midx = [date_index(d) for d in dirs]
    common = set(midx[0])
    for mi in midx[1:]:
        common &= set(mi)
    common = sorted(common)
    print(f"  {len(common)} dates in all members")

    per_window = []
    for ds in common:
        paths = [mi[ds] for mi in midx]
        z0 = np.load(paths[0])
        probs = [np.load(p)["prob_agg"].astype(np.float32) for p in paths]
        if len({pr.shape for pr in probs}) != 1:
            continue
        prob = np.mean(probs, axis=0).reshape(-1)
        label = z0["label_agg"].astype(np.uint8).reshape(-1)
        if label.sum() == 0:
            continue
        rk = compute_ranking_metrics(prob, label, args.k)
        im = compute_imbalanced_metrics(prob, label)
        br = compute_brier_decomposition(prob, label)
        per_window.append({
            "date": str(z0["win_date"]) or ds,
            "lift_k": rk["lift_k"], "precision_k": rk["precision_k"],
            "recall_k": rk["recall_k"], "n_fire": rk["n_fire"],
            "pr_auc": im["pr_auc"], "roc_auc": im["roc_auc"],
            "f1": im["f1"], "f2": im["f2"], "mcc": im["mcc"],
            "brier": br["brier"], "bss": br["bss"],
            "reliability": br["reliability"], "resolution": br["resolution"],
        })
        if len(per_window) % 50 == 0:
            print(f"  ...{len(per_window)} windows")

    if not per_window:
        sys.exit("ERROR: no windows with fire")

    out = {"run_name": args.run_name, "k": args.k,
           "n_windows_with_fire": len(per_window), "per_window": per_window}
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f)
    print(f"\n  wrote {args.output_json}  ({len(per_window)} windows)")

    print("=" * 56 + f"\nSUMMARY {args.run_name}\n" + "=" * 56)
    for m in ["lift_k", "f2", "mcc", "bss", "brier", "pr_auc", "roc_auc"]:
        v = np.array([w[m] for w in per_window], dtype=float)
        v = v[np.isfinite(v)]
        print(f"  {m:10s} = {v.mean():.4f}")


if __name__ == "__main__":
    main()
