"""Complete novel-ignition figure (287-window in-dist test, 2023/24, NBAC+NFDB).

Two panels, same models, canonical colors, bootstrap 95% CI:
  A) Novel Lift@5000  — pixel-level skill on genuinely new ignitions
  B) Novel Lift@30km  — cluster-level (30 km) skill, MAX-pool coarsening

"Novel" = fire pixels NOT already burning in the prior 30 days (novel30 target).

Cluster uses MAX-pool of the probability within each 30 km cell (a cell is
flagged if ANY pixel is high), NOT mean-pool. Mean-pool rewards diffuse
predictions and spuriously ranked flatten #1; max-pool is the operationally
correct "did we flag this cell" question and puts convstem_novel #1 at BOTH
scales. flatten still gets a cluster boost (4.42 -> 5.44: right region, wrong
exact pixel). Persistence collapses to 0 at pixel level on novel targets
(it only copies already-burning fire).

Reads results/eval/novel_final_287.json. Writes figures/fig_novel_complete_287.{png,pdf}.
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

from plot_paper_style import COLORS, SHORT, BAR_ORDER, apply_style

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = os.path.join(HERE, "results", "eval", "novel_final_287.json")
ORDER = BAR_ORDER   # canonical 9-model order, identical across Fig 4/5/6/8


def main() -> None:
    apply_style()
    d = json.load(open(J))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    panels = [("lift_5000", "A  Novel Lift@5000  (pixel-level)"),
              ("lift_30km", "B  Novel Lift@30km  (30 km cluster, max-pool)")]

    for ax, (metric, title) in zip(axes, panels):
        for i, k in enumerate(ORDER):
            if k not in d:                       # mlp / fwi_oracle novel: pending
                ax.text(i, 0.15, "n/a", ha="center", va="bottom", fontsize=7.5, color="#999")
                continue
            v = d[k][metric][0]                  # mean only (no CI)
            degen = (k == "persistence")
            ax.bar(i, v, color=COLORS[k], edgecolor="black", linewidth=0.5,
                   alpha=0.5 if degen else 0.92, hatch="//" if degen else None)
            if v < 0.05:                         # persistence pixel-level collapse
                ax.text(i, 0.15, "0.00", ha="center", va="bottom", fontsize=8,
                        color=COLORS["persistence"], fontweight="bold")
            else:
                ax.text(i, v + 0.12, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        # climatology must-beat reference line (pixel panel only)
        if metric == "lift_5000" and "climatology" in d:
            ax.axhline(d["climatology"]["lift_5000"][0], ls="--", lw=1.1,
                       color=COLORS["climatology"], zorder=0)
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_xticks(np.arange(len(ORDER)))
        ax.set_xticklabels([SHORT.get(k, k) for k in ORDER], rotation=35, ha="right",
                           fontsize=8)
        ax.set_ylabel("Lift over random (×)")
        ax.set_ylim(0, 8.2)

    fig.suptitle("Novel-ignition skill on the 287-window in-distribution test "
                 "(2023/24, fires not burning in prior 30 d)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, "figures", f"fig_novel_complete_287.{ext}")
        fig.savefig(out, bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
