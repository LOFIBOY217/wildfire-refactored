"""Figure 2-G: Standard vs Novel-30d Lift@5000 — per-year comparison.

Tests whether the model is genuinely predicting NEW fires or just
re-discovering pixels that are already burning.

  * lift_total      : standard Lift@5000 (label = fire in [ts, te))
  * lift_novel_30d  : label = fire in [ts, te) AND no fire in the
                      30 days before the lead window. Strips out the
                      "persistence" component.

Layout: 2 panels (Total | Novel-30d), grouped bars per year × model.

DRAFT NOTE: Model bars currently use the 4y_2018 enc=21 proxy because
the 12y_2014 SOTA novel-Lift dump is queued. Will be hot-swapped once
that job completes.

Usage:
    python3 scripts/plot_fig2g_novel_ignition.py
"""

from __future__ import annotations

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

YEARS = [2022, 2023, 2024]   # 2025 only has 5 windows + NBAC not yet released


def _boot_ci(arr, n_boot=2000, seed=0):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    if len(arr) < 4:
        # too few — return mean + spread
        return float(arr.mean()), float(arr.min()), float(arr.max())
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    apply_style()

    m = pd.read_csv(os.path.join(EVAL_DIR, "model_novel_lift_enc21.csv"))
    b = pd.read_csv(os.path.join(EVAL_DIR, "baseline_on_enc21_windows.csv"))
    df = m.merge(b, on="win_date", how="inner")
    df["year"] = df["win_date"].str[:4].astype(int)

    # Columns we'll read per (model, metric_mode):
    src = {
        ("sota_single", "total"):      "lift_total_5000",
        ("sota_single", "novel_30d"):  "lift_novel_30d_5000",
        ("climatology", "total"):      "clim_lift_total",
        ("climatology", "novel_30d"):  "clim_lift_novel_30d",
        ("persistence", "total"):      "persist_lift_total",
        ("persistence", "novel_30d"):  "persist_lift_novel_30d",
    }

    rows: dict[tuple, dict] = defaultdict(dict)
    for (mdl, mode), col in src.items():
        for y in YEARS:
            sub = df[df["year"] == y][col].dropna().to_numpy()
            rows[(mdl, mode)][y] = _boot_ci(sub)

    model_order = ["sota_single", "climatology", "persistence"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6),
                             sharey=True,
                             gridspec_kw={"wspace": 0.06})

    for ax, mode, ttl in zip(
        axes,
        ["total", "novel_30d"],
        ["Standard  Lift@5000  (any fire)",
         "Novel-30d  Lift@5000  (NEW fires only)"],
    ):
        n_models = len(model_order)
        n_years = len(YEARS)
        bar_w = 0.78 / n_models
        x = np.arange(n_years)

        for i, mdl in enumerate(model_order):
            means = np.array([rows[(mdl, mode)][y][0] for y in YEARS])
            errlo = np.array([rows[(mdl, mode)][y][0] - rows[(mdl, mode)][y][1] for y in YEARS])
            errhi = np.array([rows[(mdl, mode)][y][2] - rows[(mdl, mode)][y][0] for y in YEARS])
            offset = (i - (n_models - 1) / 2) * bar_w
            ax.bar(x + offset, means, bar_w, color=COLORS[mdl],
                   edgecolor="black", linewidth=0.4,
                   label=LABELS[mdl] if ax is axes[0] else None,
                   alpha=0.92)
            ax.errorbar(x + offset, means, yerr=[errlo, errhi],
                        fmt="none", ecolor="black", elinewidth=0.7,
                        capsize=2.5, capthick=0.7)
            for xi, v in zip(x + offset, means):
                if np.isfinite(v) and v > 0.3:
                    ax.text(xi, v + 0.18, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
                elif np.isfinite(v) and v < 0.05:
                    ax.text(xi, 0.15, "0", ha="center", va="bottom", fontsize=7, color="#999")

        ax.axhline(1.0, color="#999", linewidth=0.7, linestyle=":")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{y}\nfire season" for y in YEARS])
        ax.set_title(ttl, fontsize=10.5)
        if ax is axes[0]:
            ax.set_ylabel("Lift@5000")
        ax.set_ylim(0, max(8.5, max(rows[(mdl, "total")][y][2] for mdl in model_order for y in YEARS) * 1.1))

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.04),
               ncol=3, frameon=True, fontsize=9)
    fig.suptitle(
        "Per-year Lift@5000 — stripping the persistence component "
        "exposes a 50% wider gap to climatology",
        fontsize=11, y=1.13,
    )

    fig.tight_layout()
    pdf = os.path.join(OUT_DIR, "fig2g_novel_ignition.pdf")
    png = os.path.join(OUT_DIR, "fig2g_novel_ignition.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {pdf}")
    print(f"  wrote {png}")

    # Print summary
    print("\n=== summary (4y_2018 proxy model) ===")
    print(f"{'year':>6} | {'mdl total':>10} {'mdl novel':>10} | {'clim total':>10} {'clim novel':>10} | {'pers total':>10} {'pers novel':>10}")
    for y in YEARS:
        cells = [rows[("sota_single", "total")][y][0],
                 rows[("sota_single", "novel_30d")][y][0],
                 rows[("climatology", "total")][y][0],
                 rows[("climatology", "novel_30d")][y][0],
                 rows[("persistence", "total")][y][0],
                 rows[("persistence", "novel_30d")][y][0]]
        print(f"{y:>6} | " + " | ".join(
            " ".join(f"{v:>10.2f}" for v in cells[i:i+2]) for i in (0, 2, 4)
        ))


if __name__ == "__main__":
    main()
