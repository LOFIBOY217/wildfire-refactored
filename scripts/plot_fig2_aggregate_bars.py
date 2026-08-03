"""Figure 2: Aggregate Lift@5000 + Lift@30km, TRUE full-window (435 win) data.

Every bar uses leak-free NBAC+NFDB full-window evaluation:
  - learned models (single / MLP / ConvLSTM): per-window mean from *_FULL files
    (same 435 windows, same dates, same loss -> isolates architecture)
  - ensemble: 11-member + gating, prob-mean and logit-mean
  - climatology / FWI oracle / persistence: baselines_per_window_leakfree.csv
  - logreg: benchmark_logreg.csv (30 km not available -> n/a)
  - ECMWF S2S: baseline_ecmwf_s2s.json

Persistence is a DEGENERATE baseline (copies the recent fire mask forward;
its novel-ignition Lift is 0, see Fig 2-H). It is drawn on a clipped axis with
a hatch so it does not dominate the scale.

Usage:
    python3 scripts/plot_fig2_aggregate_bars.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_paper_style import COLORS, LABELS, apply_style  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "results", "eval")
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

YMAX = {"lift_5000": 11.0, "lift_30km": 9.5}  # axis cap; persistence is clipped
N_BASELINE_WIN = 435  # for baseline CSV standard-error approximation


def boot_ci(arr, n=2000, seed=0):
    arr = np.asarray([x for x in arr if x is not None], dtype=np.float64)
    if len(arr) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def from_pw(path, ka="lift_k", kb="lift_coarse"):
    d = json.load(open(path))
    pw = d["per_window"]
    return {"lift_5000": boot_ci([w.get(ka) for w in pw]),
            "lift_30km": boot_ci([w.get(kb) for w in pw])}


def from_baseline_csv(name):
    df = pd.read_csv(os.path.join(EVAL, "baselines_per_window_leakfree.csv"))
    r = df[(df.baseline == name) & (df.k == 5000)].iloc[0]

    def ci(m, sd):
        se = (sd / np.sqrt(N_BASELINE_WIN)) if sd and not np.isnan(sd) else 0.0
        return (float(m), float(m - 1.96 * se), float(m + 1.96 * se))

    return {"lift_5000": ci(r.lift_k, r.lift_k_std),
            "lift_30km": ci(r.lift_coarse, r.lift_coarse_std)}


def main():
    apply_style()

    rows = {}
    rows["sota_single"] = from_pw(os.path.join(EVAL, "sota_FULL_per_window.json"))
    rows["mlp"] = from_pw(os.path.join(EVAL, "mlp_FULL.json"))
    rows["convlstm"] = from_pw(os.path.join(EVAL, "convlstm_FULL.json"))
    # Single-model figure: ensembles intentionally excluded (would overstate
    # deployable single-model performance; see model_color_mapping / SOTA notes).
    rows["climatology"] = from_baseline_csv("climatology")
    rows["fwi_oracle"] = from_baseline_csv("fwi_oracle")
    rows["persistence"] = from_baseline_csv("persistence")

    lr = pd.read_csv(os.path.join(EVAL, "benchmark_logreg.csv"))
    lrr = lr[(lr.k == 5000) & (lr.lift_k.notna())].iloc[-1]
    rows["logreg"] = {"lift_5000": (float(lrr.lift_k), np.nan, np.nan),
                      "lift_30km": (np.nan, np.nan, np.nan)}

    e = json.load(open(os.path.join(EVAL, "baseline_ecmwf_s2s.json")))
    rows["ecmwf_s2s"] = {
        "lift_5000": (e["lift_5000"]["mean"], e["lift_5000"]["ci_lo"], e["lift_5000"]["ci_hi"]),
        "lift_30km": (e["lift_30km"]["mean"], e["lift_30km"]["ci_lo"], e["lift_30km"]["ci_hi"]),
    }

    lab = dict(LABELS)

    order = sorted(rows, key=lambda k: (rows[k]["lift_5000"][0]
                   if not np.isnan(rows[k]["lift_5000"][0]) else -1))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for ax, metric, title in zip(
        axes, ["lift_5000", "lift_30km"],
        ["Lift@5000  (pixel-scale)", "Lift@30 km  (cluster-scale)"],
    ):
        ym = YMAX[metric]
        for i, k in enumerate(order):
            m, lo, hi = rows[k][metric]
            if np.isnan(m):
                ax.text(i, 0.1, "n/a", ha="center", va="bottom", fontsize=7.5, color="#999")
                continue
            degen = (k == "persistence")
            ax.bar(i, m, color=COLORS[k], edgecolor="black", linewidth=0.4,
                   alpha=0.5 if degen else 0.92, hatch="//" if degen else None)
            clipped = m > ym
            if (not np.isnan(lo)) and (not clipped):
                ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="black",
                            elinewidth=0.9, capsize=3, capthick=0.9)
            if clipped:
                ax.annotate(f"{m:.1f}  ↑\n(degenerate,\nnovel = 0)", xy=(i, ym * 0.93),
                            ha="center", va="top", fontsize=6.8, color="#333")
            else:
                ax.text(i, m + 0.12, f"{m:.2f}", ha="center", va="bottom", fontsize=7.5)

        ax.axhline(1.0, color="#999", linewidth=0.7, linestyle=":")
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels([lab[k] for k in order], rotation=35, ha="right")
        ax.set_ylabel(title.split("  ")[0])
        ax.set_title(title)
        ax.set_ylim(0, ym)

    fig.suptitle("Validation metrics, 2022-2025 fire season, NBAC+NFDB labels (full-window, leak-free)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_aggregate_bars.pdf"))
    fig.savefig(os.path.join(OUT, "fig2_aggregate_bars.png"))
    plt.close(fig)

    print(f"{'model':<34}{'L@5000':>12}{'L@30km':>12}")
    for k in order[::-1]:
        a, b = rows[k]["lift_5000"][0], rows[k]["lift_30km"][0]
        af = f"{a:.2f}" if not np.isnan(a) else "n/a"
        bf = f"{b:.2f}" if not np.isnan(b) else "n/a"
        print(f"{lab[k]:<34}{af:>12}{bf:>12}")
    print("\n  wrote figures/fig2_aggregate_bars.{png,pdf}")


if __name__ == "__main__":
    main()
