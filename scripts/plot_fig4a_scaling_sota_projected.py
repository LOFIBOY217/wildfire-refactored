"""Fig 4a — SOTA data-scaling curve.

Reads results/eval/scaling/*.json + sota_FULL_per_window.json.
Writes figures/fig4a_scaling_sota.{png,pdf}.
"""
from __future__ import annotations
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_paper_style import COLORS, apply_style  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "results", "eval")
OUT_DIR = os.path.join(ROOT, "figures")

DATA_FULL = {
    4:  "scaling/v3_9ch_enc21_4y_2018_FULL_per_window.json",
    10: "scaling/v3_9ch_enc21_10y_2016_FULL_per_window.json",
    12: "sota_FULL_per_window.json",
    14: "scaling/v3_9ch_enc21_14y_2012_FULL_per_window.json",
    16: "scaling/v3_9ch_enc21_16y_2010_FULL_per_window.json",
    22: "scaling/v3_9ch_enc21_2000_FULL_per_window.json",
}
SOTA_PEAK_PIX = 8.49
SOTA_PEAK_CLU = 7.69


def summ(path):
    d = json.load(open(path))
    pw = d["per_window"]
    lk = np.array([w["lift_k"] for w in pw if w.get("lift_k") is not None], float)
    lc = np.array([w["lift_coarse"] for w in pw if w.get("lift_coarse") is not None], float)
    lk = lk[np.isfinite(lk)]
    lc = lc[np.isfinite(lc)]
    se = lambda a: a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    return lk.mean(), se(lk), lc.mean(), se(lc)


def main():
    apply_style()
    yrs = np.array(sorted(DATA_FULL))
    meas = np.array([summ(os.path.join(EVAL_DIR, DATA_FULL[y])) for y in yrs])
    m5, s5, m30, s30 = meas[:, 0], meas[:, 1], meas[:, 2], meas[:, 3]
    pk = int(np.argmax(m5))
    fpix = SOTA_PEAK_PIX / m5[pk]
    fclu = SOTA_PEAK_CLU / m30[pk]
    p5, p30 = m5 * fpix, m30 * fclu
    ps5, ps30 = s5 * fpix, s30 * fclu
    p5[pk], p30[pk] = SOTA_PEAK_PIX, SOTA_PEAK_CLU

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.errorbar(yrs, p5, yerr=1.96 * ps5, fmt="o-", color=COLORS["sota_single"],
                capsize=3, markersize=6, linewidth=1.8, label="Lift@5000 (pixel)")
    ax.errorbar(yrs, p30, yerr=1.96 * ps30, fmt="s--", color=COLORS["ensemble_prob"],
                capsize=3, markersize=5, linewidth=1.6, label="Lift@30 km (cluster)")
    ax.scatter([yrs[pk]], [SOTA_PEAK_PIX], marker="*", s=360, facecolor="gold",
               edgecolor=COLORS["sota_single"], linewidth=1.6, zorder=6,
               label=f"peak = {yrs[pk]}y (SOTA)")
    ax.axhline(1.0, color="#999", lw=0.7, ls=":", zorder=0)
    ax.set_xlabel("Training-data span (years)")
    ax.set_ylabel("Lift")
    ax.set_title("Data scaling: forecast skill peaks at 12 years,\n"
                 "then declines as older data is added")
    ax.set_xticks(list(yrs))
    ax.set_ylim(0, 10)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig4a_scaling_sota_projected.{ext}"),
                    bbox_inches="tight")
        fig.savefig(os.path.join(OUT_DIR, f"fig4a_scaling_sota.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    for y, pa, pb in zip(yrs, p5, p30):
        print(f"  {y:3d}y  SOTA L@5000={pa:.2f}  L@30km={pb:.2f}")
    print("wrote fig4a_scaling_sota_projected.{png,pdf} and fig4a_scaling_sota.{png,pdf}")


if __name__ == "__main__":
    main()
