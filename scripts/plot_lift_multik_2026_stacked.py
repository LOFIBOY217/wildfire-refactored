"""2026 out-of-sample multi-budget lift, total (top) and novel ignition (bottom)
stacked in one figure sharing the nine model x axis. Total uses the window mean,
novel uses the fire weighted mean, matching Figures 8 and 9 done separately.

Usage: python3 scripts/plot_lift_multik_2026_stacked.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_paper_style import COLORS, SHORT, BAR_ORDER, apply_style  # noqa: E402
from plot_lift_multik import load_values, shade, KS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")
W = 0.26


def draw(ax, vals, ym, title):
    for gi, m in enumerate(BAR_ORDER):
        for j, K in enumerate(KS):
            v = vals[m].get(K, np.nan)
            x = gi + (j - 1) * W
            if not np.isfinite(v):
                ax.text(x, 0.05, "n/a", ha="center", va="bottom", fontsize=6, color="#999")
                continue
            degen = v > ym
            ax.bar(x, min(v, ym), width=W, color=shade(COLORS[m], K), edgecolor="black",
                   linewidth=0.4, alpha=0.55 if degen else 0.95, hatch="//" if degen else None)
            if degen:
                if j == 1:
                    ax.annotate(f"{v:.0f} ↑", xy=(gi, ym * 0.95), ha="center", va="top",
                                fontsize=6.5, color="#333")
            else:
                ax.text(x, v + ym * 0.01, f"{v:.1f}", ha="center", va="bottom", fontsize=6.2)
    ax.axhline(1.0, color="#999", linewidth=0.7, linestyle=":")
    ax.set_ylabel("Lift")
    ax.set_ylim(0, ym)
    ax.set_title(title)


def main():
    apply_style()
    tot = load_values("2026", "total", "macro")
    nov = load_values("2026", "novel", "micro")
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(12.8, 9.6), sharex=True)
    draw(a0, tot, 11.0, "Total fire")
    draw(a1, nov, 9.5, "Novel ignition")
    a1.set_xticks(range(len(BAR_ORDER)))
    a1.set_xticklabels([SHORT.get(m, m) for m in BAR_ORDER], rotation=35, ha="right", fontsize=8)
    leg = [Patch(facecolor=shade("#5b5b5b", K), edgecolor="black", label=f"Lift@{K//1000}k") for K in KS]
    a0.legend(handles=leg, frameon=False, fontsize=8, loc="upper right", title="alert budget")
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_lift_multik_2026_stacked.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig_lift_multik_2026_stacked.png")


if __name__ == "__main__":
    main()
