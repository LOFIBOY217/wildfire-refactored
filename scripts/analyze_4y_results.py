#!/usr/bin/env python3
"""
Post-hoc analysis of 4y NBAC training results.

Reads per_window JSON files for the four 4y enc {14, 21, 28, 35} runs
and produces:
  1. Per-month Lift@5000 + Lift@30km (which months does the model do best?)
  2. Bootstrap CI for headline Lift@5000 and Lift@30km
  3. Lift@30km summary (event-level metric, more meaningful for ops)
  4. Per-window full trajectory (which windows are easy vs hard)
  5. Markdown summary table for paper Table 1

Usage:
    python scripts/analyze_4y_results.py --outputs_dir outputs/ \\
        --runs v3_9ch_enc14_4y_2018 v3_9ch_enc21_4y_2018 \\
               v3_9ch_enc28_4y_2018 v3_9ch_enc35_4y_2018
"""
import argparse
import json
import os
from collections import defaultdict
from datetime import date as Date
from pathlib import Path

import numpy as np


def bootstrap_ci(values, n_boot=2000, ci=95, seed=0):
    """Bootstrap mean + percentile CI."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    boot_means = np.empty(n_boot)
    n = len(arr)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = arr[idx].mean()
    lo, hi = np.percentile(boot_means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return arr.mean(), lo, hi


def load_run(json_path):
    """Return list of per-window dicts + run summary."""
    with open(json_path) as f:
        d = json.load(f)
    return d["per_window"], d.get("summary", {})


def per_month_breakdown(per_window):
    """Group windows by month, compute mean lift."""
    by_month = defaultdict(list)
    for w in per_window:
        m = Date.fromisoformat(w["date"]).month
        by_month[m].append(w)
    out = {}
    for m in sorted(by_month):
        wins = by_month[m]
        out[m] = {
            "n_windows": len(wins),
            "lift_5000_mean": float(np.mean([w["lift_k"] for w in wins])),
            "lift_5000_std": float(np.std([w["lift_k"] for w in wins])),
            "lift_30km_mean": float(np.mean([w["lift_coarse"] for w in wins])),
            "n_fire_mean": float(np.mean([w["n_fire"] for w in wins])),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs_dir", default="outputs")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="Run names (without _per_window.json suffix)")
    ap.add_argument("--out_md", default="outputs/analysis_4y.md")
    args = ap.parse_args()

    print("=" * 70)
    print("4y NBAC TRAINING RESULTS — POST-HOC ANALYSIS")
    print("=" * 70)

    md_lines = ["# 4y NBAC training analysis (post-hoc)\n"]

    runs_data = {}
    for r in args.runs:
        path = os.path.join(args.outputs_dir, f"{r}_per_window.json")
        if not os.path.exists(path):
            print(f"  SKIP {r}: {path} not found")
            continue
        per_win, summary = load_run(path)
        runs_data[r] = {"per_win": per_win, "summary": summary}
        print(f"  loaded {r}: {len(per_win)} windows")

    if not runs_data:
        print("No runs loaded — exiting.")
        return

    # ── Section 1: Headline Lift@5000 + Lift@30km with bootstrap CI ────
    print("\n" + "=" * 70)
    print("1. Headline metrics with bootstrap 95% CI")
    print("=" * 70)
    md_lines.append("## 1. Headline metrics (bootstrap 95% CI, n=2000)\n")
    md_lines.append("| Run | Lift@5000 | Lift@30km | n_win |")
    md_lines.append("|---|---|---|---|")
    for r, d in runs_data.items():
        lifts5k = [w["lift_k"] for w in d["per_win"]]
        lifts30 = [w["lift_coarse"] for w in d["per_win"]]
        m5, lo5, hi5 = bootstrap_ci(lifts5k)
        m30, lo30, hi30 = bootstrap_ci(lifts30)
        s5 = f"{m5:.2f}x [{lo5:.2f}, {hi5:.2f}]"
        s30 = f"{m30:.2f}x [{lo30:.2f}, {hi30:.2f}]"
        print(f"  {r:35s} L@5000={s5}  L@30km={s30}  n={len(lifts5k)}")
        md_lines.append(f"| {r} | {s5} | {s30} | {len(lifts5k)} |")
    md_lines.append("")

    # ── Section 2: Per-month breakdown ─────────────────────────────────
    print("\n" + "=" * 70)
    print("2. Per-month breakdown (which months does model perform best?)")
    print("=" * 70)
    md_lines.append("## 2. Per-month Lift@5000 (mean of windows in that month)\n")
    md_lines.append("| Run | May | Jun | Jul | Aug | Sep | Oct |")
    md_lines.append("|---|---|---|---|---|---|---|")
    for r, d in runs_data.items():
        by_m = per_month_breakdown(d["per_win"])
        row = [r]
        for m in [5, 6, 7, 8, 9, 10]:
            if m in by_m:
                row.append(f"{by_m[m]['lift_5000_mean']:.2f}x (n={by_m[m]['n_windows']})")
            else:
                row.append("—")
        print(f"  {r}: " + "  ".join(row[1:]))
        md_lines.append("| " + " | ".join(row) + " |")
    md_lines.append("")

    # ── Section 3: Per-month Lift@30km ─────────────────────────────────
    md_lines.append("## 3. Per-month Lift@30km (event-level)\n")
    md_lines.append("| Run | May | Jun | Jul | Aug | Sep | Oct |")
    md_lines.append("|---|---|---|---|---|---|---|")
    for r, d in runs_data.items():
        by_m = per_month_breakdown(d["per_win"])
        row = [r]
        for m in [5, 6, 7, 8, 9, 10]:
            if m in by_m:
                row.append(f"{by_m[m]['lift_30km_mean']:.2f}x")
            else:
                row.append("—")
        md_lines.append("| " + " | ".join(row) + " |")
    md_lines.append("")

    # ── Section 4: Best vs worst windows per run ──────────────────────
    print("\n" + "=" * 70)
    print("3. Best & worst windows per run (Lift@5000 extremes)")
    print("=" * 70)
    md_lines.append("## 4. Best & worst windows (extreme cases)\n")
    md_lines.append("| Run | Best date | Best L@5000 | Worst date | Worst L@5000 |")
    md_lines.append("|---|---|---|---|---|")
    for r, d in runs_data.items():
        sorted_wins = sorted(d["per_win"], key=lambda w: w["lift_k"])
        worst = sorted_wins[0]
        best = sorted_wins[-1]
        print(f"  {r}: best={best['date']}({best['lift_k']:.2f}x)  "
              f"worst={worst['date']}({worst['lift_k']:.2f}x)")
        md_lines.append(f"| {r} | {best['date']} | {best['lift_k']:.2f}x | "
                        f"{worst['date']} | {worst['lift_k']:.2f}x |")
    md_lines.append("")

    # ── Section 5: Comparison vs baselines ─────────────────────────────
    md_lines.append("## 5. Vs baselines (NBAC labels, 20-win sample)\n")
    md_lines.append("| Method | Lift@5000 | vs climatology | vs logreg |")
    md_lines.append("|---|---|---|---|")
    md_lines.append("| fwi_oracle | 1.10x | -78% | -79% |")
    md_lines.append("| **climatology** | **5.09x** | 0 | -3% |")
    md_lines.append("| **logreg (6 feats)** | **5.24x** | +3% | 0 |")
    for r, d in runs_data.items():
        lifts5k = [w["lift_k"] for w in d["per_win"]]
        m5 = float(np.mean(lifts5k))
        gain_clim = (m5 - 5.09) / 5.09 * 100
        gain_lr = (m5 - 5.24) / 5.24 * 100
        md_lines.append(f"| {r} | {m5:.2f}x | {gain_clim:+.1f}% | {gain_lr:+.1f}% |")
    md_lines.append("")

    # ── Save markdown ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write("\n".join(md_lines))
    print(f"\nMarkdown summary -> {args.out_md}")


if __name__ == "__main__":
    main()
