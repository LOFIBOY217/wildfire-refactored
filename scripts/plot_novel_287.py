"""Novel-Ignition Lift Reveals True Forecast Skill Beyond Fire Persistence.
Single-panel novel-30d Lift@5000. DL models = 287-window 2023/24 (real);
climatology/FWI/persistence = benchmark_novel; MLP = model_novel_lift CSV.
Canonical model->color. Persistence collapses to ~0 on novel (predicts only
already-burning fire) — the whole point of the figure.

Usage: python3 scripts/plot_novel_287.py
"""
from __future__ import annotations
import os, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_paper_style import COLORS, LABELS, apply_style  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "eval", "novel_287.csv")
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)


def main():
    apply_style()
    df = pd.read_csv(CSV).sort_values("lift_5000").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for i, r in df.iterrows():
        k = r["key"]; v = r["lift_5000"]
        degen = int(r.get("degenerate", 0)) == 1
        ax.bar(i, v, color=COLORS.get(k, "#888"), edgecolor="black", linewidth=0.4,
               alpha=0.55 if degen else 0.92, hatch="//" if degen else None)
        ax.text(i, v + 0.08, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(1.0, color="#999", linewidth=0.7, linestyle=":")
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels([LABELS.get(k, k) for k in df["key"]], rotation=35, ha="right")
    ax.set_ylabel("Novel-ignition Lift@5000")
    ax.set_ylim(0, max(df["lift_5000"]) * 1.16)
    ax.legend(handles=[plt.matplotlib.patches.Patch(facecolor="grey", hatch="//", alpha=0.55,
              edgecolor="black", label="Persistence = predicts already-burning fire (novel ≈ 0)")],
              loc="upper right", fontsize=8, frameon=False)
    fig.suptitle("Novel-Ignition Lift Reveals True Forecast Skill Beyond Fire Persistence\n"
                 "(2023/24 test set, novel-30d target; baselines reused as in Fig 4)",
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_novel_287.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(df.to_string(index=False))
    print("\nwrote figures/fig_novel_287.{png,pdf}")


if __name__ == "__main__":
    main()
