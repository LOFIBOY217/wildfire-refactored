"""Lift@K curves (K = 1000..10000) for the learned models, 287-window
in-distribution eval. Two panels: total fire (left) and novel ignition (right).
One line per model, canonical colors. Shows that headline Lift@5000 is a single
slice of a curve, and that MLP's novel curve sits at/above ours in-distribution
(a memorizing baseline) — the point sharpened out-of-distribution elsewhere.

Reads results/eval/mlp_analysis.json (pulled from narval dump analysis).
Usage: python3 scripts/plot_lift_curve_287.py
"""
from __future__ import annotations
import json, os, sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_paper_style import COLORS, SHORT, apply_style  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results", "eval", "mlp_analysis.json")
OUT = os.path.join(ROOT, "figures")

# Draw order: ours first (on top), then learned baselines. MLP emphasized.
ORDER = ["fcnhead", "convstem_novel", "convstem", "flatten", "convlstm", "mlp"]


def main():
    apply_style()
    d = json.load(open(SRC))
    Ks = d["K"]
    panels = [("lift_tot", "Total fire"), ("lift_nov", "Novel ignition")]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharex=True)
    for ax, (field, title) in zip(axes, panels):
        for m in ORDER:
            ys = [d[field][m][str(k)] if str(k) in d[field][m] else d[field][m][k]
                  for k in Ks]
            is_mlp = (m == "mlp")
            ax.plot(Ks, ys, marker="o", markersize=4,
                    color=COLORS.get(m, "#888"),
                    linewidth=2.6 if is_mlp else 1.8,
                    linestyle="--" if is_mlp else "-",
                    zorder=5 if is_mlp else 3,
                    label=SHORT.get(m, m).replace("\n", " "))
        ax.axvline(5000, color="#bbb", linewidth=0.8, linestyle=":")
        ax.text(5000, ax.get_ylim()[1], " @5000", fontsize=7, color="#999",
                va="top", ha="left")
        ax.axhline(1.0, color="#999", linewidth=0.7, linestyle=":")
        ax.set_xlabel("K  (top-K pixels ranked by predicted probability)")
        ax.set_ylabel("Lift@K")
        ax.set_title(f"{title}  (287-window, in-distribution)")
        ax.set_xticks(Ks)
        ax.set_xticklabels([f"{k//1000}k" for k in Ks])
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_lift_curve_287.{ext}"),
                    dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig_lift_curve_287.{png,pdf}")


if __name__ == "__main__":
    main()
