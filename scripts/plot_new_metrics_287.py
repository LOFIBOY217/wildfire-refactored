"""Two candidate new figures (287-window 2023/24), NOT touching the existing set:
  fig_reliability_287 — reliability diagram + ECE (calibration)
  fig_rev_287         — Relative Economic Value curve (cost-loss decision value)

Reads results/eval/new_metrics_287.json. Canonical colors/labels/order from
plot_paper_style. fwi_oracle has no calibrated probability -> omitted from both
(it appears only in the rank/Lift figures). No CI (project rule).

Usage: python3 scripts/plot_new_metrics_287.py
"""
from __future__ import annotations
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_paper_style import COLORS, SHORT, BAR_ORDER, apply_style  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = os.path.join(ROOT, "results", "eval", "new_metrics_287.json")
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

# line style by group so 8 curves stay legible
STYLE = {
    "fcnhead": "-", "convstem_novel": "-", "convstem": "-", "flatten": "-",
    "convlstm": "--", "mlp": "--", "climatology": ":", "persistence": ":",
}


def _models(d, need_key):
    return [k for k in BAR_ORDER if k in d and d[k].get(need_key) is not None]


def reliability(d):
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.plot([0, 1], [0, 1], color="#888", lw=1.0, ls="--", zorder=0,
            label="perfect calibration")
    for k in _models(d, "reliability"):
        rel = np.array(d[k]["reliability"])          # [conf, obs, frac]
        conf, obs = rel[:, 0], rel[:, 1]
        ece = d[k]["ece"]
        ax.plot(conf, obs, STYLE.get(k, "-"), color=COLORS[k], lw=1.7,
                marker="o", ms=3.5, alpha=0.9,
                label=f"{SHORT.get(k, k).replace(chr(10), ' ')} (ECE={ece:.3f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fire frequency")
    ax.set_title("Reliability (calibration), 287-window 2023/24\n"
                 "closer to the diagonal = better calibrated")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=7.2, loc="upper left", frameon=True, ncol=1)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_reliability_287.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig_reliability_287.{png,pdf}")


def rev(d):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for k in _models(d, "rev_value"):
        a = np.array(d[k]["rev_alpha"]); v = np.array(d[k]["rev_value"])
        peak = d[k]["rev_peak"]
        ax.plot(a, v, STYLE.get(k, "-"), color=COLORS[k], lw=1.8, alpha=0.92,
                label=f"{SHORT.get(k, k).replace(chr(10), ' ')} (peak={peak:.2f})")
    ax.axhline(0.0, color="#888", lw=0.9, ls="--", zorder=0)
    ax.set_xlabel("Cost / loss ratio  (user decision threshold)")
    ax.set_ylabel("Relative economic value  (0 = climatology, 1 = perfect)")
    ax.set_title("Relative economic value vs cost/loss ratio, 287-window 2023/24\n"
                 "value over a leak-free climatology baseline")
    ax.set_xlim(0, 1); ax.set_ylim(-0.15, 0.8)
    ax.legend(fontsize=7.4, loc="upper right", frameon=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_rev_287.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig_rev_287.{png,pdf}")


def main():
    apply_style()
    d = json.load(open(J))
    reliability(d)
    rev(d)


if __name__ == "__main__":
    main()
