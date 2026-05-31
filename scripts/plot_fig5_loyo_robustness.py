#!/usr/bin/env python3
"""Fig 5 — Forward-chaining LOYO robustness.

Two-panel horizontal forest plot. Each fold (val year 2020..2024) trains on
2014 -> (year-1) and evaluates that year's fire season at full window. The
macro mean (per-fold equal weight) and its spread are shown as a summary row,
and the single-split SOTA full-eval value is drawn as a vertical reference
line. The macro mean landing on the single-split line is the key result: the
single time-split evaluation is NOT optimistically biased.

Source data: results/eval/loyo/LOYO_SUMMARY.txt (committed, cross-validated by
two independent aggregations). Numbers are reproduced here as constants.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from plot_paper_style import apply_style, COLORS

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)

# --- LOYO results (from results/eval/loyo/LOYO_SUMMARY.txt) ---
FOLDS = ["2020", "2021", "2022", "2023", "2024"]
LIFT5K = {"2020": 8.54, "2021": 10.38, "2022": 9.07, "2023": 4.80, "2024": 7.47}
LIFT30 = {"2020": 7.81, "2021": 8.67, "2022": 8.99, "2023": 4.74, "2024": 5.86}

# macro = per-fold equal weight (mean +/- std); micro = pooled-window bootstrap CI
MACRO5K = (8.05, 2.10)          # mean, std
MACRO30 = (7.21, 1.84)
MICRO5K = (8.05, 7.04, 9.05)    # mean, lo, hi
MICRO30 = (7.21, 6.32, 8.19)

# single time-split SOTA, full-window eval (the value the paper headlines)
SINGLE5K = 7.83
SINGLE30 = 6.73


def _panel(ax, vals, macro, micro, single, title, xlabel, color):
    """One forest panel: fold rows (bottom->top) + macro summary row on top."""
    years = FOLDS
    y_fold = list(range(len(years)))          # 0..4
    y_macro = len(years) + 0.8                 # summary row, with a gap

    # single-split reference line (label at the bottom to avoid the macro row)
    ax.axvline(single, color="0.35", ls="--", lw=1.3, zorder=1)
    ax.text(single, -0.62, f"single-split\nfull eval\n{single:.2f}",
            ha="center", va="top", fontsize=8.0, color="0.30")

    # per-fold dots
    for yr, y in zip(years, y_fold):
        ax.plot(vals[yr], y, "o", ms=8, color=color, zorder=3)
        ax.annotate(f"{vals[yr]:.2f}", (vals[yr], y),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8.5, color=color)

    # macro summary: mean dot + std whisker (and micro CI as thin error bar)
    m_mean, m_std = macro
    mi_mean, mi_lo, mi_hi = micro
    ax.errorbar(mi_mean, y_macro, xerr=[[mi_mean - mi_lo], [mi_hi - mi_mean]],
                fmt="none", ecolor=color, elinewidth=1.2, capsize=4, zorder=2)
    ax.errorbar(m_mean, y_macro, xerr=m_std, fmt="D", ms=9, color=color,
                ecolor=color, elinewidth=2.4, capsize=5, zorder=4)
    ax.annotate(f"macro {m_mean:.2f}$\\pm${m_std:.2f}", (m_mean, y_macro),
                textcoords="offset points", xytext=(0, 11),
                ha="center", fontsize=9, color=color, fontweight="bold")

    ax.set_yticks(y_fold + [y_macro])
    ax.set_yticklabels([f"val {y}" for y in years] + ["MACRO"])
    ax.set_ylim(-1.7, y_macro + 1.2)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight="bold")
    ax.axhline(len(years) - 0.4, color="0.85", lw=0.8, zorder=0)
    ax.grid(axis="x", color="0.92", lw=0.7, zorder=0)
    ax.set_axisbelow(True)


def main():
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    _panel(axes[0], LIFT5K, MACRO5K, MICRO5K, SINGLE5K,
           "Pixel scale", "Lift@5000", COLORS["sota_single"])
    _panel(axes[1], LIFT30, MACRO30, MICRO30, SINGLE30,
           "Cluster scale (30 km)", "Lift@30km", COLORS["ensemble_prob"])
    fig.suptitle("Forward-chaining leave-one-year-out robustness",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(OUT, f"fig5_loyo_robustness.{ext}")
        fig.savefig(p)
        print("wrote", os.path.abspath(p))


if __name__ == "__main__":
    main()
