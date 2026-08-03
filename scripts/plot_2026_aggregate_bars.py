#!/usr/bin/env python3
"""2026 out-of-sample aggregate bar chart (Lift@5000 + Lift@30 km), reference-style.

Reads outputs/agg_2026_bars.csv (produced on Narval by plot_2026_aggregate.py, which
recomputes neural models + ensembles per-window from the 2026 forecast tifs and reads
baselines/logreg from their eval JSONs). Renders the two-panel figure locally.

Usage: python3 scripts/plot_2026_aggregate_bars.py [path/to/agg_2026_bars.csv]
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_paper_style import COLORS  # canonical model->color (keep consistent)

# CSV keys in agg_2026_*.csv -> canonical color keys in plot_paper_style.COLORS
KEY2COLOR = {"fwi": "fwi_oracle", "clim": "climatology", "mlp": "mlp",
             "pers": "persistence", "fcnhead": "fcnhead", "flatten": "flatten",
             "convlstm": "convlstm", "convstem": "convstem",
             "convstem_novel": "convstem_novel", "logreg": "logreg",
             "ecmwf_s2s": "ecmwf_s2s"}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "outputs", "agg_2026_bars.csv")
SUPTITLE = sys.argv[2] if len(sys.argv) > 2 else \
    "2026 out-of-sample forecast, CIFFC size-circle labels (strict novel window, 16 issues)"
OUTNAME = sys.argv[3] if len(sys.argv) > 3 else "fig_2026_aggregate_bars"
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(CSV)
# SINGLE-MODEL figure: drop ensembles (reporting ensemble as "better" misleads).
df = df[~df["key"].astype(str).str.startswith("ens")].copy()
df = df.sort_values("l5").reset_index(drop=True)

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(17, 8))


def draw(ax, col, lo, hi, title, ylab):
    for i, r in df.iterrows():
        is_pers = str(r["key"]) == "pers"
        color = COLORS.get(KEY2COLOR.get(str(r["key"]), ""), "#888888")
        ax.bar(i, r[col], color=color, edgecolor="black", lw=0.7,
               hatch=("//" if is_pers else None), alpha=(0.55 if is_pers else 0.9))
        yl, yh = r[lo], r[hi]
        if np.isfinite(yl):
            ax.errorbar(i, r[col], yerr=[[r[col] - yl], [yh - r[col]]], fmt="none",
                        ecolor="black", capsize=3, lw=1)
        if np.isfinite(r[col]):
            top = yh if np.isfinite(yh) else r[col]
            ax.text(i, top + 0.12, f"{r[col]:.2f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(1.0, ls=":", color="grey", lw=1)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["label"], rotation=35, ha="right", fontsize=9)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(ylab)
    ax.margins(y=0.12)


draw(axes[0], "l5", "l5lo", "l5hi", "Lift@5000  (pixel-scale)", "Lift@5000")
draw(axes[1], "l30", "l30lo", "l30hi", "Lift@30 km  (cluster-scale)", "Lift@30 km")
fig.suptitle(SUPTITLE, fontsize=14, y=0.99)
axes[0].legend(handles=[Patch(facecolor="grey", hatch="//", alpha=0.55, edgecolor="black",
               label="Persistence = predicts already-burning fire (not new ignitions)")],
               loc="upper left", fontsize=8, frameon=False)
fig.tight_layout(rect=[0, 0, 1, 0.97])
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"{OUTNAME}.{ext}"), dpi=160, bbox_inches="tight")
print(f"saved figures/{OUTNAME}.png / .pdf")
