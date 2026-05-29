"""Figure 2-H: Standard vs Novel-ignition Lift@5000, cross-model.

The key operational argument: a trivial persistence baseline wins
"standard" Lift (it predicts where fire is ALREADY burning, and NBAC
mega-fires persist for weeks), but its skill VANISHES on novel
ignitions — fires starting in pixels that had no fire in the prior
30 days. Our model holds its skill on novel ignitions; persistence
collapses to zero.

Two panels, bars per model:
  LEFT  : standard Lift@5000 (any fire in the window)
  RIGHT : novel-30d Lift@5000 (NEW fires only)

DATA: baselines from benchmark_novel.csv (20-win sample). SOTA from
model_novel_lift_SOTA_full.csv (full window) if present, else the
4y_2018 proxy (model_novel_lift_enc21.csv). Swap is automatic.

Usage:
    python3 scripts/plot_fig2h_novel_crossmodel.py
"""

from __future__ import annotations

import os
import sys

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


def baseline_vals(csv_path):
    """Return {baseline: {'total': lift, 'novel_30d': lift}}."""
    b = pd.read_csv(csv_path)
    out = {}
    for bl in ["persistence", "climatology", "fwi_oracle"]:
        d = {}
        for mode in ["total", "novel_30d"]:
            r = b[(b.baseline == bl) & (b.label_mode == mode) & (b.k == K)]
            d[mode] = float(r.lift_mean.values[0]) if len(r) else np.nan
        out[bl] = d
    return out


def model_vals(csv_path):
    """Read a model_novel_lift_*_full.csv → mean total + novel_30d Lift@5000."""
    m = pd.read_csv(csv_path)
    return {
        "total":     float(m["lift_total_5000"].mean()),
        "novel_30d": float(m["lift_novel_30d_5000"].mean()),
    }


def sota_vals():
    """Prefer real full-window SOTA novel CSV; fall back to 4y proxy."""
    full = os.path.join(EVAL_DIR, "model_novel_lift_v3_9ch_enc21_12y_2014_full.csv")
    full2 = os.path.join(EVAL_DIR, "model_novel_lift_SOTA_full.csv")
    proxy = os.path.join(EVAL_DIR, "model_novel_lift_enc21.csv")
    if os.path.exists(full):
        return model_vals(full), "full-window"
    if os.path.exists(full2):
        return model_vals(full2), "full-window"
    return model_vals(proxy), "4y proxy (preliminary)"


# Optional DL-baseline novel CSVs (present once their re-eval + compute lands)
DL_MODELS = {
    "convlstm": "model_novel_lift_baseline_convlstm_12y_2014_9ch_full.csv",
    "mlp":      "model_novel_lift_baseline_mlp_12y_2014_9ch_full.csv",
}


def main():
    apply_style()
    base = baseline_vals(os.path.join(EVAL_DIR, "benchmark_novel.csv"))
    sota, sota_tag = sota_vals()

    rows = {
        "sota_single": sota,
        "climatology": base["climatology"],
        "persistence": base["persistence"],
        "fwi_oracle":  base["fwi_oracle"],
    }
    # add ConvLSTM / MLP novel if their CSVs have landed
    for key, fname in DL_MODELS.items():
        p = os.path.join(EVAL_DIR, fname)
        if os.path.exists(p):
            rows[key] = model_vals(p)
    # order by standard Lift descending so persistence (highest) is first
    order = sorted(rows, key=lambda m: rows[m]["total"], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.7), sharey=True,
                             gridspec_kw={"wspace": 0.06})
    for ax, mode, ttl in zip(
        axes, ["total", "novel_30d"],
        ["Standard Lift@5000\n(any fire — rewards persistence)",
         "Novel-30d Lift@5000\n(NEW fires only — the operational task)"],
    ):
        xs = np.arange(len(order))
        vals = [rows[m][mode] for m in order]
        cols = [COLORS[m] for m in order]
        ax.bar(xs, vals, color=cols, edgecolor="black", linewidth=0.4, alpha=0.92)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.25, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)
        ax.axhline(1.0, color="#999", linewidth=0.7, linestyle=":")
        ax.set_xticks(xs)
        ax.set_xticklabels([LABELS[m] for m in order], rotation=25, ha="right")
        ax.set_title(ttl, fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Lift@5000")
        ax.set_ylim(0, max(max(rows[m]["total"] for m in order) * 1.12, 8))

    # collapse arrow annotation on persistence
    pi = order.index("persistence")
    axes[1].annotate("persistence\ncollapses to 0", xy=(pi, 0.2),
                     xytext=(pi - 0.1, 4.5), fontsize=8, ha="center", color="#C0392B",
                     arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.0))

    fig.suptitle(f"Novel-ignition Lift exposes trivial persistence  "
                 f"(SOTA: {sota_tag})", fontsize=11, y=1.02)
    fig.tight_layout()
    pdf = os.path.join(OUT_DIR, "fig2h_novel_crossmodel.pdf")
    png = os.path.join(OUT_DIR, "fig2h_novel_crossmodel.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {pdf}\n  wrote {png}")

    print(f"\n=== values (SOTA = {sota_tag}) ===")
    print(f"{'model':<28} {'standard':>10} {'novel_30d':>10}")
    for m in order:
        print(f"{LABELS[m]:<28} {rows[m]['total']:>10.2f} {rows[m]['novel_30d']:>10.2f}")


if __name__ == "__main__":
    main()
