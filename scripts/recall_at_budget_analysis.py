"""
Analyse Recall@budget per_window JSON files to produce three metrics
that isolate the dynamic / month-to-month value of a forecast model
beyond static climatology:

  1. Climatology-relative ΔRecall@budget (paired per-window)
       ΔR_method = Recall_method - Recall_climatology
       Tells us: "given the same patrol budget, how many MORE fire events
       does the method capture than the always-fixed climatology prior?"

  2. Per-year stratified Recall@budget
       Reports recall per validation year × method.
       Identifies which year (e.g. 2023 QC megafire anomaly) the model
       beats the static climatology.

  3. Per-month stratified Recall@budget
       Same as (2) but by calendar month.
       Reveals seasons where dynamic forecasting matters most.

Reads from outputs/recall_at_budget_<method>_per_window.json
(produced by recall_at_budget.py and recall_at_budget_baselines.py).

Usage:
  python -m scripts.recall_at_budget_analysis \\
      --per_window_jsons \\
        outputs/recall_at_budget_enc21_12y_per_window.json:model \\
        outputs/recall_at_budget_climatology_per_window.json:climatology \\
        outputs/recall_at_budget_persistence_per_window.json:persistence \\
        outputs/recall_at_budget_ecmwf_s2s_per_window.json:ecmwf_s2s \\
      --output_prefix outputs/recall_at_budget_analysis
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_pw_input(s):
    """Parse 'path:tag' → (path, tag)."""
    if ":" not in s:
        raise ValueError(f"Expected 'path:tag', got {s}")
    path, tag = s.rsplit(":", 1)
    return path, tag


def load_per_window(path):
    """Load per_window JSON → dict keyed by win_date."""
    with open(path, "r") as f:
        rows = json.load(f)
    out = {}
    for r in rows:
        out[r["win_date"]] = r
    return out


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


def get_recalls(window_dict, win_date, budgets):
    """Return list of recalls (one per budget) for a single window, or None if missing."""
    if win_date not in window_dict:
        return None
    by = window_dict[win_date]["by_budget"]
    out = []
    for B in budgets:
        match = next((b for b in by if abs(b["budget"] - B) < 1e-9), None)
        if match is None:
            return None
        out.append(match["recall"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_window_jsons", nargs="+", required=True,
                    help="path:tag pairs, e.g. outputs/foo.json:model")
    ap.add_argument("--budgets", type=float, nargs="+",
                    default=[0.001, 0.005, 0.01, 0.05, 0.10])
    ap.add_argument("--clim_tag", default="climatology",
                    help="Tag of the climatology baseline (used as reference for Δ-Recall)")
    ap.add_argument("--output_prefix", required=True)
    args = ap.parse_args()

    # Load all per-window JSONs
    methods = []
    win_data = {}
    for s in args.per_window_jsons:
        path, tag = parse_pw_input(s)
        win_data[tag] = load_per_window(path)
        methods.append(tag)
        print(f"  loaded {tag}: {len(win_data[tag])} windows from {path}")

    if args.clim_tag not in win_data:
        print(f"[ERROR] climatology tag '{args.clim_tag}' not in inputs")
        sys.exit(1)

    # Common windows (intersection across all methods)
    common = set(win_data[methods[0]].keys())
    for m in methods[1:]:
        common &= set(win_data[m].keys())
    common = sorted(common)
    print(f"\nCommon windows: {len(common)}\n")

    # Build a 3D array: (n_method, n_window, n_budget) of recalls
    n_m, n_w, n_b = len(methods), len(common), len(args.budgets)
    R = np.full((n_m, n_w, n_b), np.nan, dtype=np.float64)
    for mi, m in enumerate(methods):
        for wi, w in enumerate(common):
            r = get_recalls(win_data[m], w, args.budgets)
            if r is not None:
                R[mi, wi, :] = r

    clim_idx = methods.index(args.clim_tag)

    # ── Metric 1: Climatology-relative ΔRecall@budget ──────────────────
    print("=" * 70)
    print("METRIC 1 — Climatology-relative ΔRecall@budget (paired per window)")
    print("=" * 70)
    print(f"{'Budget':>8} | " + " | ".join(f"{m:>15}" for m in methods))
    delta_summary = {"budgets": list(args.budgets), "by_method": {}}
    for mi, m in enumerate(methods):
        delta_summary["by_method"][m] = []
    for bi, B in enumerate(args.budgets):
        line = [f"{B*100:6.2f}%"]
        for mi, m in enumerate(methods):
            if m == args.clim_tag:
                line.append(f"{'(reference)':>15}")
                delta_summary["by_method"][m].append({
                    "budget": float(B), "delta_mean": 0.0,
                    "delta_ci_lo": 0.0, "delta_ci_hi": 0.0,
                })
            else:
                deltas = R[mi, :, bi] - R[clim_idx, :, bi]
                mean, lo, hi = bootstrap_ci(deltas)
                line.append(f"{mean*100:+6.2f}% [{lo*100:+5.2f},{hi*100:+5.2f}]")
                delta_summary["by_method"][m].append({
                    "budget": float(B), "delta_mean": mean,
                    "delta_ci_lo": lo, "delta_ci_hi": hi,
                })
        print(" | ".join(line))

    # ── Metric 2: Per-year recall ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("METRIC 2 — Per-year Recall@budget (mean over windows in year)")
    print("=" * 70)
    yr_buckets = defaultdict(list)
    for wi, w in enumerate(common):
        y = int(w[:4])
        yr_buckets[y].append(wi)
    years = sorted(yr_buckets.keys())
    year_summary = {"years": years, "by_budget": {}}
    for B in args.budgets:
        year_summary["by_budget"][f"{B}"] = {y: {} for y in years}
    print(f"\nFocus on Budget = 5% (paper-relevant operational scale)")
    bi5 = args.budgets.index(0.05) if 0.05 in args.budgets else None
    if bi5 is not None:
        print(f"{'year':>6} | n_win | " + " | ".join(f"{m:>11}" for m in methods))
        for y in years:
            idxs = np.array(yr_buckets[y])
            line = [f"{y}", f"{len(idxs):>5}"]
            for mi, m in enumerate(methods):
                vals = R[mi, idxs, bi5]
                line.append(f"{np.nanmean(vals)*100:6.2f}%   ")
            print(" | ".join(line))
    # Save full per-budget per-year results
    for bi, B in enumerate(args.budgets):
        for y in years:
            idxs = np.array(yr_buckets[y])
            year_summary["by_budget"][f"{B}"][y] = {
                "n": int(len(idxs)),
                "by_method": {m: float(np.nanmean(R[mi, idxs, bi]))
                              for mi, m in enumerate(methods)},
            }

    # ── Metric 3: Per-month recall ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("METRIC 3 — Per-month Recall@budget (mean over windows in calendar month)")
    print("=" * 70)
    mo_buckets = defaultdict(list)
    for wi, w in enumerate(common):
        mo = int(w[5:7])
        mo_buckets[mo].append(wi)
    months = sorted(mo_buckets.keys())
    month_summary = {"months": months, "by_budget": {}}
    for B in args.budgets:
        month_summary["by_budget"][f"{B}"] = {m: {} for m in months}
    if bi5 is not None:
        print(f"\nFocus on Budget = 5%")
        print(f"{'month':>6} | n_win | " + " | ".join(f"{m:>11}" for m in methods))
        for mo in months:
            idxs = np.array(mo_buckets[mo])
            line = [f"{mo:02d}", f"{len(idxs):>5}"]
            for mi, m in enumerate(methods):
                vals = R[mi, idxs, bi5]
                line.append(f"{np.nanmean(vals)*100:6.2f}%   ")
            print(" | ".join(line))
    for bi, B in enumerate(args.budgets):
        for mo in months:
            idxs = np.array(mo_buckets[mo])
            month_summary["by_budget"][f"{B}"][mo] = {
                "n": int(len(idxs)),
                "by_method": {m: float(np.nanmean(R[mi, idxs, bi]))
                              for mi, m in enumerate(methods)},
            }

    # ── Metric 4: Anomaly-window stratified recall ──────────────────────
    # Stratify windows by climatology recall@5% (bottom = "climatology
    # fails to predict this window's fires" = anomaly windows). Compare
    # each method's recall on these strata.
    print("\n" + "=" * 70)
    print("METRIC 4 — Anomaly-window stratified Recall@5% (B requested)")
    print("=" * 70)
    anomaly_summary = {"by_stratum": {}}
    if bi5 is not None:
        clim_recall_w = R[clim_idx, :, bi5]
        # Thresholds: bottom-25%, mid-50%, top-25% by climatology recall
        q25, q75 = np.nanpercentile(clim_recall_w, [25, 75])
        strata = {
            "anomaly (clim fails, bottom 25%)":
                np.where(clim_recall_w <= q25)[0],
            "normal (clim mid, 25-75%)":
                np.where((clim_recall_w > q25) & (clim_recall_w < q75))[0],
            "easy (clim top 25%)":
                np.where(clim_recall_w >= q75)[0],
        }
        print(f"\n{'Stratum':<40} | n_win | " + " | ".join(f"{m:>11}" for m in methods))
        for stratum_name, idxs in strata.items():
            line = [f"{stratum_name:<40}", f"{len(idxs):>5}"]
            for mi, m in enumerate(methods):
                vals = R[mi, idxs, bi5]
                line.append(f"{np.nanmean(vals)*100:6.2f}%   ")
            print(" | ".join(line))
            anomaly_summary["by_stratum"][stratum_name] = {
                "n": int(len(idxs)),
                "by_method": {m: float(np.nanmean(R[mi, idxs, bi5]))
                              for mi, m in enumerate(methods)},
            }
        # Also report per-budget
        print(f"\nFull per-budget breakdown for the ANOMALY stratum:")
        anomaly_idxs = strata["anomaly (clim fails, bottom 25%)"]
        print(f"{'Budget':>8} | " + " | ".join(f"{m:>11}" for m in methods))
        anomaly_summary["per_budget_anomaly"] = []
        for bi, B in enumerate(args.budgets):
            line = [f"{B*100:6.2f}%"]
            row = {"budget": float(B), "n": int(len(anomaly_idxs)),
                   "by_method": {}}
            for mi, m in enumerate(methods):
                vals = R[mi, anomaly_idxs, bi]
                m_val = float(np.nanmean(vals))
                line.append(f"{m_val*100:6.2f}%   ")
                row["by_method"][m] = m_val
            print(" | ".join(line))
            anomaly_summary["per_budget_anomaly"].append(row)

    # ── Save outputs ───────────────────────────────────────────────────
    out_dir = Path(args.output_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(f"{args.output_prefix}_delta.json", "w") as f:
        json.dump(delta_summary, f, indent=2)
    with open(f"{args.output_prefix}_year.json", "w") as f:
        json.dump(year_summary, f, indent=2)
    with open(f"{args.output_prefix}_month.json", "w") as f:
        json.dump(month_summary, f, indent=2)
    with open(f"{args.output_prefix}_anomaly.json", "w") as f:
        json.dump(anomaly_summary, f, indent=2)
    print(f"\nWrote {args.output_prefix}_{{delta,year,month,anomaly}}.json")


if __name__ == "__main__":
    main()
