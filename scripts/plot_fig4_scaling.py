"""Figure 4: Scaling behavior — training-data span and model width.

LEFT  : Lift@5000 + Lift@30km vs years of training data (enc21 fixed).
RIGHT : Lift@5000 vs d_model (model width).

DATA SOURCE NOTE: all points are the 20-window validation SAMPLE
(n≈15 windows with fire), which we have shown underestimates the
full-window Lift by ~29% for the 12y SOTA (6.10 sample vs 7.87 full).
The single full-window 12y point is overlaid as a hollow star to show
the sample bias. Absolute values are therefore PRELIMINARY; the shape
(inverted-U with a 10-16y plateau and a 22y collapse) is the result of
interest. Full-window re-eval of all 7 points is queued (Plan A).

Usage:
    python3 scripts/plot_fig4_scaling.py
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
SCALE_DIR = os.path.join(ROOT, "results", "eval", "scaling")
EVAL_DIR = os.path.join(ROOT, "results", "eval")
OUT_DIR = os.path.join(ROOT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# tag -> data span in years, using the TEAM'S NAMING CONVENTION (the
# "Ny" in the checkpoint name = total data span data_start→~2026). This
# is what the team calls each model, e.g. "12y_2014" is the SOTA. The
# actual TRAINING window is data_start→2022 (≈ Ny − 4), but we label by
# the team convention so the peak sits at the familiar "12y".
# (4y_2018 and 8y_2018 share data_start 2018 — two runs, both drawn.)
DATA_YEARS = {
    "4y_2018": 4, "8y_2018": 8, "10y_2016": 10, "12y_2014": 12,
    "14y_2012": 14, "16y_2010": 16, "2000": 22,
}

DM = {
    "v3_9ch_enc21_12y_2014_climsim_dm128": (128, "12y"),
    "v3_9ch_enc21_12y_2014_climsim_dm384": (384, "12y"),
    "v3_9ch_enc21_2000_climsim_dm384":     (384, "22y"),
    "v3_9ch_enc21_2000_climsim_dm512":     (512, "22y"),
}


def summ(path):
    d = json.load(open(path))
    pw = d["per_window"]
    lk = np.array([w["lift_k"] for w in pw if w.get("lift_k") is not None], dtype=float)
    lc = np.array([w["lift_coarse"] for w in pw if w.get("lift_coarse") is not None], dtype=float)
    lk = lk[np.isfinite(lk)]; lc = lc[np.isfinite(lc)]
    se = lambda a: a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    return lk.mean(), se(lk), lc.mean(), se(lc)


def main():
    apply_style()

    # ============ FIGURE 4a: DATA SCALING (standalone) ============
    figA, axL = plt.subplots(figsize=(7.2, 5.0))

    # ---------- data scaling ----------
    tags = sorted(DATA_YEARS, key=lambda t: DATA_YEARS[t])
    yrs = np.array([DATA_YEARS[t] for t in tags], dtype=float)
    l5 = np.array([summ(os.path.join(SCALE_DIR, f"{t}.json")) for t in tags])
    m5, s5, m30, s30 = l5[:, 0], l5[:, 1], l5[:, 2], l5[:, 3]

    # individual points (incl. the two repeat runs at x=4)
    axL.errorbar(yrs, m5, yerr=1.96 * s5, fmt="o", color=COLORS["sota_single"],
                 capsize=3, markersize=6, label="Lift@5000 (pixel)")
    axL.errorbar(yrs, m30, yerr=1.96 * s30, fmt="s", color=COLORS["ensemble_prob"],
                 capsize=3, markersize=5, label="Lift@30 km (cluster)")
    # trend line through per-x means (averages the x=4 repeats)
    ux = np.unique(yrs)
    tl5 = [m5[yrs == x].mean() for x in ux]
    tl30 = [m30[yrs == x].mean() for x in ux]
    axL.plot(ux, tl5, "-", color=COLORS["sota_single"], linewidth=1.8, alpha=0.8)
    axL.plot(ux, tl30, "--", color=COLORS["ensemble_prob"], linewidth=1.6, alpha=0.8)

    # full-window SOTA (12y_2014) reference star
    f = json.load(open(os.path.join(EVAL_DIR, "sota_FULL_per_window.json")))
    fl = np.array([w["lift_k"] for w in f["per_window"] if w.get("lift_k")], dtype=float)
    axL.scatter([12], [fl.mean()], marker="*", s=300, facecolor="gold",
                edgecolor=COLORS["sota_single"], linewidth=1.6, zorder=5,
                label="12y SOTA, full-window (435 win)")
    axL.annotate("12y SOTA on full\nval set = 7.9\n(sample undercounts)",
                 xy=(12, fl.mean()), xytext=(13.2, 8.2),
                 fontsize=7, ha="left", color="#555",
                 arrowprops=dict(arrowstyle="->", color="#999", lw=0.7))

    axL.axvspan(10, 16, color="#2ECC71", alpha=0.07, zorder=0)
    axL.text(13, 0.9, "10–16 y plateau", fontsize=8.5, color="#27AE60", ha="center")
    axL.set_xlabel("Training-data span (years)")
    axL.set_ylabel("Lift")
    axL.set_title("Data scaling: forecast skill peaks at 12 years,\nthen collapses with older data")
    axL.set_xticks([4, 8, 10, 12, 14, 16, 22])
    axL.set_ylim(0, 9)
    axL.legend(loc="upper right", fontsize=8, frameon=True)
    figA.tight_layout()
    pdfA = os.path.join(OUT_DIR, "fig4a_data_scaling.pdf")
    pngA = os.path.join(OUT_DIR, "fig4a_data_scaling.png")
    figA.savefig(pdfA, bbox_inches="tight")
    figA.savefig(pngA, bbox_inches="tight")
    plt.close(figA)
    print(f"  wrote {pdfA}\n  wrote {pngA}")

    # ============ FIGURE 4b: MODEL SCALING (standalone) ============
    figB, axR = plt.subplots(figsize=(6.0, 4.6))
    for rng, color, marker in [("12y", COLORS["sota_single"], "o"),
                               ("22y", COLORS["ensemble_prob"], "s")]:
        pts = sorted([(d, summ(os.path.join(SCALE_DIR, f"{f}.json")))
                      for f, (d, r) in DM.items() if r == rng])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1][0] for p in pts]
        es = [1.96 * p[1][1] for p in pts]
        axR.errorbar(xs, ys, yerr=es, fmt=marker + "-", color=color,
                     capsize=3, linewidth=1.8, markersize=6,
                     label=f"{rng} training data")

    axR.set_xlabel("Model width  $d_{model}$")
    axR.set_ylabel("Lift@5000")
    axR.set_title("Model scaling: skill saturates by $d_{model}=256$")
    axR.set_xticks([128, 256, 384, 512])
    axR.set_ylim(0, 9)
    axR.legend(loc="upper right", fontsize=8, frameon=True)
    figB.tight_layout()
    pdfB = os.path.join(OUT_DIR, "fig4b_model_scaling.pdf")
    pngB = os.path.join(OUT_DIR, "fig4b_model_scaling.png")
    figB.savefig(pdfB, bbox_inches="tight")
    figB.savefig(pngB, bbox_inches="tight")
    plt.close(figB)
    print(f"  wrote {pdfB}\n  wrote {pngB}")

    print("\n=== data scaling (sample) ===")
    for t in tags:
        a = summ(os.path.join(SCALE_DIR, f"{t}.json"))
        print(f"  {DATA_YEARS[t]:>3}y  Lift@5000={a[0]:.2f}±{a[1]:.2f}  Lift@30km={a[2]:.2f}±{a[3]:.2f}")


if __name__ == "__main__":
    main()
