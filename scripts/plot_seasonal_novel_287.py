"""Seasonal decomposition of novel-ignition skill (Lift@5000) on the 287-window
in-distribution test. Shows the crossover: our patch-transformer models win
early-to-mid season (May-Jul, when new ignitions spread across the boreal), while
the MLP baseline wins late season (Aug-Sep, when late novel fires cluster into a
few predictable blocks). The two are seasonally complementary -> direct evidence
for why the ensemble beats every single model, and why MLP's high SEASON-AVERAGE
novel Lift is not uniform skill but late-season concentration.

Reads results/eval/mlp_analysis.json.
Usage: python3 scripts/plot_seasonal_novel_287.py
"""
from __future__ import annotations
import json, os, sys, collections
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_paper_style import COLORS, SHORT, apply_style  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results", "eval", "mlp_analysis.json")
OUT = os.path.join(ROOT, "figures")
MONTHS = {4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep"}
MODELS = ["fcnhead", "convstem_novel", "convstem", "mlp"]


def main():
    apply_style()
    d = json.load(open(SRC))
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    nwin = collections.defaultdict(int)
    for r in d["perwin"]:
        if r["n_novel"] == 0:
            continue
        mo = r["month"]
        nwin[mo] += 1
        for m in MODELS:
            if r.get(m) is not None:
                by[mo][m].append(r[m])
    months = sorted(MONTHS)
    x = np.arange(len(months))

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for m in MODELS:
        ys = [np.mean(by[mo][m]) if by[mo][m] else np.nan for mo in months]
        emph = (m == "mlp")
        ax.plot(x, ys, marker="o", markersize=6,
                color=COLORS.get(m, "#888"),
                linewidth=2.8 if emph else 1.9,
                linestyle="--" if emph else "-",
                zorder=5 if emph else 3,
                label=SHORT.get(m, m).replace("\n", " "))
    ax.axhline(1.0, color="#999", linewidth=0.7, linestyle=":")
    # annotate the two regimes
    ax.axvspan(-0.4, 3.4, color="#C0392B", alpha=0.05, zorder=0)
    ax.axvspan(3.6, 5.4, color="#7B6F9E", alpha=0.08, zorder=0)
    ax.text(1.5, ax.get_ylim()[1] * 0.98, "ours win\n(spreading new ignitions)",
            ha="center", va="top", fontsize=8, color="#8B2E28", style="italic")
    ax.text(4.5, ax.get_ylim()[1] * 0.98, "MLP wins\n(clustered late fires)",
            ha="center", va="top", fontsize=8, color="#4A3F6B", style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{MONTHS[mo]}\n(n={nwin[mo]})" for mo in months])
    ax.set_ylabel("Novel Lift@5000  (mean over windows in month)")
    ax.set_title("Novel-ignition skill by month  (287-window in-distribution test)")
    ax.legend(frameon=False, fontsize=8.5, loc="center left")
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_seasonal_novel_287.{ext}"),
                    dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig_seasonal_novel_287.{png,pdf}")


if __name__ == "__main__":
    main()
