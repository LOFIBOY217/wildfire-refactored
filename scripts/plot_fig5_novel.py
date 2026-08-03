"""Figure 5 — Novel-ignition Lift exposes trivial persistence.
Two panels: Standard Lift@5000 (any fire, rewards persistence) vs Novel-30d
Lift@5000 (NEW fires only, the operational task). 287-window 2023/24, all models
on the SAME windows (from score dumps). Persistence collapses 20+->0 on novel.
Canonical model->color. Reads results/eval/dbl_287.csv.

Usage: python3 scripts/plot_fig5_novel.py
"""
from __future__ import annotations
import os, sys
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_paper_style import COLORS, LABELS, apply_style  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "eval", "dbl_287.csv")
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

# fixed left->right order (persistence first to mirror the original), then models
ORDER = ["persistence", "fcnhead", "convstem_novel", "convstem", "flatten",
         "convlstm", "climatology", "mlp", "fwi_oracle"]


def main():
    apply_style()
    df = pd.read_csv(CSV).set_index("key")
    order = [k for k in ORDER if k in df.index]
    df = df.reindex(order).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    panels = [("std_l5", "std_l5lo", "std_l5hi",
               "Standard Lift@5000\n(any fire — rewards persistence)"),
              ("nov_l5", "nov_l5lo", "nov_l5hi",
               "Novel-30d Lift@5000\n(NEW fires only — the operational task)")]
    ymax = max(df["std_l5"].max(), df["nov_l5"].max()) * 1.12
    for ax, (col, lo, hi, title) in zip(axes, panels):
        for i, r in df.iterrows():
            k = r["key"]; v = r[col]
            ax.bar(i, v, color=COLORS.get(k, "#888"), edgecolor="black", linewidth=0.5, alpha=0.92)
            if lo in df.columns and np.isfinite(r.get(lo, np.nan)) and k != "persistence":
                ax.errorbar(i, v, yerr=[[max(v - r[lo], 0)], [max(r[hi] - v, 0)]], fmt="none",
                            ecolor="black", elinewidth=0.8, capsize=2.5)
            ax.text(i, v + 0.2, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)
        # annotate persistence collapse on the novel panel
        if col == "nov_l5" and "persistence" in order:
            pi = order.index("persistence")
            ax.annotate("persistence\ncollapses to 0", xy=(pi, 0.15), xytext=(pi + 1.1, ymax * 0.4),
                        fontsize=8.5, color="#C0392B",
                        arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.1))
        ax.axhline(1.0, ls=":", color="grey", lw=0.8)
        ax.set_xticks(np.arange(len(df)))
        ax.set_xticklabels([LABELS.get(k, k) for k in df["key"]], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Lift@5000")
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, ymax)
    fig.suptitle("Novel-Ignition Lift Exposes Trivial Persistence "
                 "(2023/24 test set, 287 windows, all models on the same windows)", fontsize=12.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig5_novel.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    cols = [c for c in ["key", "std_l5", "nov_l5", "std_l30", "nov_l30"] if c in df.columns]
    print(df[cols].to_string(index=False))
    print("\nwrote figures/fig5_novel.{png,pdf}")


if __name__ == "__main__":
    main()
