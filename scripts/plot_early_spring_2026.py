"""2026 early-spring out-of-sample comparison (issues on/before 2026-04-06).
Standard vs novel-30d Lift@5000, FULL model set: our Transformers + ConvLSTM/MLP
learned baselines + climatology / persistence / FWI-oracle physical baselines.

Two stories in one figure:
  - conv-stem family holds ~11-12x on BOTH panels; fcnhead (in-dist SOTA) and
    flatten collapse OOS in early spring.
  - persistence looks like it "wins" the standard panel (20.9x, degenerate —
    copies recently-burning fire) but collapses to 0 on novel ignitions; FWI
    oracle is ~0 even restricted to Canada (high-FWI southern prairie != where
    boreal fires start); MLP fails OOS (~0.4).

Canonical model->color (plot_paper_style). Reads results/eval/early_spring_2026.csv.
Usage: python3 scripts/plot_early_spring_2026.py
"""
from __future__ import annotations
import os, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_paper_style import COLORS, LABELS, apply_style  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "eval", "early_spring_2026.csv")
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
ORDER = ["convstem_novel", "convstem", "convlstm", "fcnhead", "flatten",
         "mlp", "climatology", "persistence", "fwi_oracle"]
DEGENERATE = {"persistence"}   # copies already-burning fire
YCAP = 14.0                    # clip the degenerate persistence bar


def main():
    apply_style()
    df = pd.read_csv(CSV).set_index("key")
    order = [k for k in ORDER if k in df.index]
    df = df.reindex(order).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    panels = [("total_l5", "total_lo", "total_hi", "Standard Lift@5000  (any fire)"),
              ("novel_l5", "novel_lo", "novel_hi", "Novel-30d Lift@5000  (new ignitions)")]
    for ax, (col, lo, hi, title) in zip(axes, panels):
        for i, r in df.iterrows():
            k = r["key"]; v = r[col]
            degen = k in DEGENERATE
            vplot = min(v, YCAP)
            ax.bar(i, vplot, color=COLORS.get(k, "#888"), edgecolor="black",
                   linewidth=0.5, alpha=0.55 if degen else 0.92,
                   hatch="//" if degen else None)
            clipped = v > YCAP
            if not clipped and lo in df.columns and np.isfinite(r.get(lo, np.nan)):
                ax.errorbar(i, v, yerr=[[max(v - r[lo], 0)], [max(r[hi] - v, 0)]], fmt="none",
                            ecolor="black", elinewidth=0.8, capsize=3)
            if clipped:
                ax.text(i, YCAP - 0.4, f"↑{v:.1f}\n(degen.)", ha="center", va="top",
                        fontsize=8, color="#333")
            elif v < 0.05:
                ax.text(i, 0.25, "0.0", ha="center", va="bottom", fontsize=8.5,
                        color=COLORS.get(k, "#333"), fontweight="bold")
            else:
                ax.text(i, v + YCAP * 0.015, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)
        ax.axhline(1.0, ls=":", color="grey", lw=0.8)
        ax.set_xticks(np.arange(len(df)))
        ax.set_xticklabels([LABELS.get(k, k) for k in df["key"]], rotation=32, ha="right", fontsize=8)
        ax.set_ylabel("Lift@5000")
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, YCAP)
    fig.suptitle("2026 early-spring out-of-sample (issues ≤ Apr 6): conv-stem holds on both, "
                 "persistence collapses on novel, FWI ≈ 0",
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_early_spring_2026.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(df.to_string(index=False))
    print("\nwrote figures/fig_early_spring_2026.{png,pdf}")


if __name__ == "__main__":
    main()
