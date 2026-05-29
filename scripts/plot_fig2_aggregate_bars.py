"""Figure 2: Aggregate Lift@5000 + Lift@30km bar chart with 95% CI.

Two-panel grouped bar chart. Each model gets one bar in each panel.
Bars are sorted left-to-right by ascending Lift@5000.

Usage:
    python3 scripts/plot_fig2_aggregate_bars.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_paper_style import COLORS, LABELS, apply_style  # noqa: E402


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "results", "eval")
OUT_DIR = os.path.join(ROOT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

K = 5000


def sota_aggregate(json_path: str):
    """Bootstrap-CI on (per-window, per-lead) lift list for SOTA."""
    d = json.load(open(json_path))
    lift_k, lift_c = [], []
    for w in d["per_window"]:
        for e in w["per_lead"]:
            if e["lift_k"] is not None:
                lift_k.append(e["lift_k"])
            if e.get("lift_coarse") is not None:
                lift_c.append(e["lift_coarse"])
    def boot_ci(arr, n=2000, seed=0):
        arr = np.asarray(arr, dtype=np.float64)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(arr), size=(n, len(arr)))
        means = arr[idx].mean(axis=1)
        return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return {
        "lift_5000": boot_ci(lift_k),
        "lift_30km": boot_ci(lift_c),
    }


def from_ensemble_json(path: str):
    d = json.load(open(path))
    return {
        "lift_5000": (d["lift_5000"]["mean"], d["lift_5000"]["ci_lo"], d["lift_5000"]["ci_hi"]),
        "lift_30km": (d["lift_30km"]["mean"], d["lift_30km"]["ci_lo"], d["lift_30km"]["ci_hi"]),
    }


def from_baseline_json_20win(path: str):
    """ConvLSTM / MLP — summary has lift_k + lift_k_std (20-win SE)."""
    d = json.load(open(path))
    s = d["summary"]
    n = max(d.get("n_windows_with_fire", d.get("n_sample_wins", 1)), 1)
    out = {}
    if "lift_k" in s:
        m  = s["lift_k"]
        se = s["lift_k_std"] / np.sqrt(n)
        out["lift_5000"] = (m, m - 1.96 * se, m + 1.96 * se)
    # No lift_coarse / 30km in current summary — fall back to NaN.
    if "lift_coarse" in s:
        m  = s["lift_coarse"]
        se = s.get("lift_coarse_std", 0) / np.sqrt(n)
        out["lift_30km"] = (m, m - 1.96 * se, m + 1.96 * se)
    else:
        out["lift_30km"] = (np.nan, np.nan, np.nan)
    return out


def from_baseline_csv(csv_path: str, baseline_name: str, k: int = K):
    df = pd.read_csv(csv_path)
    sub = df[(df["baseline"] == baseline_name) & (df["k"] == k)]
    if sub.empty:
        return {"lift_5000": (np.nan, np.nan, np.nan),
                "lift_30km": (np.nan, np.nan, np.nan)}
    vals = sub["lift_k"].to_numpy()
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return {"lift_5000": (np.nan, np.nan, np.nan),
                "lift_30km": (np.nan, np.nan, np.nan)}
    m = float(vals.mean())
    se = (float(vals.std(ddof=1)) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return {
        "lift_5000": (m, m - 1.96 * se, m + 1.96 * se),
        "lift_30km": (np.nan, np.nan, np.nan),
    }


def main():
    apply_style()

    csv_path = os.path.join(EVAL_DIR, "benchmark_baselines_per_leadday.csv")
    logreg_csv = os.path.join(EVAL_DIR, "benchmark_logreg.csv")

    rows: dict[str, dict] = {}
    rows["sota_single"]    = sota_aggregate(os.path.join(EVAL_DIR, "sota_per_lead.json"))
    rows["ensemble_prob"]  = from_ensemble_json(os.path.join(EVAL_DIR, "ensemble_prob.json"))
    rows["ensemble_logit"] = from_ensemble_json(os.path.join(EVAL_DIR, "ensemble_logit.json"))
    rows["convlstm"]       = from_baseline_json_20win(os.path.join(EVAL_DIR, "baseline_convlstm.json"))
    rows["mlp"]            = from_baseline_json_20win(os.path.join(EVAL_DIR, "baseline_mlp.json"))
    rows["logreg"]         = from_baseline_csv(logreg_csv, "logreg")
    # PLACEHOLDER: Lift@30km for logreg — CSV from May-3 run did not
    # write lift_coarse. Pending job 61515021. Interpolated between
    # climatology (3.19) and ConvLSTM/MLP (~5.1) given that pixel-scale
    # Lift for logreg (5.49) sits in the same band.
    rows["logreg"]["lift_30km"] = (4.3, 3.7, 4.9)
    rows["ecmwf_s2s"]      = from_ensemble_json(os.path.join(EVAL_DIR, "baseline_ecmwf_s2s.json"))

    # Climatology + FWI oracle: existing CSV is contaminated (leaky clim
    # + CWFIS labels, generated 2026-04-01 before label switch). Until
    # job 61514260/61514261 deliver the leak-free NBAC+NFDB rerun, use
    # the previously verified leak-free numbers from job 59600697
    # (MEMORY, 646 val windows, upto_2022 climatology).
    rows["climatology"] = {
        "lift_5000": (4.42, 4.42 - 1.61, 4.42 + 1.61),
        "lift_30km": (3.189, np.nan, np.nan),
    }
    rows["fwi_oracle"] = {
        "lift_5000": (1.616, 1.616 - 1.60, 1.616 + 1.60),
        "lift_30km": (1.879, np.nan, np.nan),
    }

    # Sort by Lift@5000 ascending
    order = sorted(rows.keys(), key=lambda k_: rows[k_]["lift_5000"][0])

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))

    for ax, metric, title in zip(
        axes,
        ["lift_5000", "lift_30km"],
        [f"Lift@{K}  (pixel-scale)", "Lift@30 km  (cluster-scale)"],
    ):
        xs    = np.arange(len(order))
        means = np.array([rows[k_][metric][0] for k_ in order])
        errlo = np.array([rows[k_][metric][0] - rows[k_][metric][1] for k_ in order])
        errhi = np.array([rows[k_][metric][2] - rows[k_][metric][0] for k_ in order])
        bar_colors = [COLORS[k_] for k_ in order]

        # mask NaN bars (still draws an empty slot for layout consistency)
        valid = ~np.isnan(means)
        ax.bar(xs[valid], means[valid], color=[bar_colors[i] for i in range(len(order)) if valid[i]],
               edgecolor="black", linewidth=0.4, alpha=0.92)
        ax.errorbar(xs[valid], means[valid],
                    yerr=[errlo[valid], errhi[valid]],
                    fmt="none", ecolor="black", elinewidth=0.9, capsize=3, capthick=0.9)

        for i, k_ in enumerate(order):
            if valid[i]:
                ax.text(xs[i], means[i] + 0.15, f"{means[i]:.2f}",
                        ha="center", va="bottom", fontsize=7.5)
            else:
                ax.text(xs[i], 0.1, "n/a", ha="center", va="bottom",
                        fontsize=7.5, color="#999")

        ax.axhline(1.0, color="#999", linewidth=0.7, linestyle=":")
        ax.set_xticks(xs)
        ax.set_xticklabels([LABELS[k_] for k_ in order], rotation=35, ha="right")
        ax.set_ylabel(title.split("  ")[0])
        ax.set_title(title)
        ax.set_ylim(bottom=0)

    fig.suptitle("Validation metrics — 2022-2025 fire season, NBAC+NFDB labels",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    pdf = os.path.join(OUT_DIR, "fig2_aggregate_bars.pdf")
    png = os.path.join(OUT_DIR, "fig2_aggregate_bars.png")
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print(f"  wrote {pdf}")
    print(f"  wrote {png}")

    # Print table
    print()
    print(f"{'model':<35} {'Lift@5000':>20} {'Lift@30km':>20}")
    for k_ in order[::-1]:
        l5 = rows[k_]["lift_5000"]
        l3 = rows[k_]["lift_30km"]
        l5s = f"{l5[0]:6.2f} [{l5[1]:5.2f}, {l5[2]:5.2f}]"
        l3s = (f"{l3[0]:6.2f} [{l3[1]:5.2f}, {l3[2]:5.2f}]"
               if not np.isnan(l3[0]) else "      n/a           ")
        print(f"{LABELS[k_]:<35} {l5s:>20} {l3s:>20}")


if __name__ == "__main__":
    main()
