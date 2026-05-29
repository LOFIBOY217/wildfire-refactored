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

# tag -> TRUE training-data span (data_start → pred_start 2022-05-01).
# NOTE: the tag's "Ny" counts data_start→2026 (total span incl. val),
# but the model only TRAINS on data_start→2022. We plot real training
# years. 4y_2018 and 8y_2018 are two runs of the SAME 4-year span
# (2018→2022) → repro/seed check at x=4.
TRAIN_YEARS = {
    "4y_2018": 4, "8y_2018": 4, "10y_2016": 6, "12y_2014": 8,
    "14y_2012": 10, "16y_2010": 12, "2000": 22,
}
DATA_YEARS = TRAIN_YEARS  # keep old name working below

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
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                                   gridspec_kw={"wspace": 0.22})

    # ---------- LEFT: data scaling ----------
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

    # repro note for the two 4y runs
    axL.annotate("two runs,\nsame 4y data\n(Δ=0.14)", xy=(4, 4.3), xytext=(4.4, 1.4),
                 fontsize=6.8, ha="left", color="#777",
                 arrowprops=dict(arrowstyle="->", color="#bbb", lw=0.6))

    # full-window SOTA (= 8y training) reference star
    f = json.load(open(os.path.join(EVAL_DIR, "sota_FULL_per_window.json")))
    fl = np.array([w["lift_k"] for w in f["per_window"] if w.get("lift_k")], dtype=float)
    axL.scatter([8], [fl.mean()], marker="*", s=280, facecolor="none",
                edgecolor=COLORS["sota_single"], linewidth=1.8, zorder=5,
                label="8y full-window (435 win)")
    axL.annotate("sample under-\nestimates ~29%", xy=(8, fl.mean()), xytext=(9.5, 8.4),
                 fontsize=7, ha="left", color="#555",
                 arrowprops=dict(arrowstyle="->", color="#999", lw=0.7))

    axL.axvspan(6, 12, color="#2ECC71", alpha=0.07, zorder=0)
    axL.text(9, 0.9, "6–12 y plateau", fontsize=8, color="#27AE60", ha="center")
    axL.set_xlabel("Years of TRAINING data (data_start → 2022; enc21 fixed)")
    axL.set_ylabel("Lift")
    axL.set_title("Data scaling — peaks ~8 y, collapses at 22 y")
    axL.set_xticks([4, 6, 8, 10, 12, 22])
    axL.set_ylim(0, 9)
    axL.legend(loc="upper right", fontsize=7.3, frameon=True)

    # ---------- RIGHT: model scaling ----------
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
    axR.set_title("Model scaling — saturates early")
    axR.set_xticks([128, 256, 384, 512])
    axR.set_ylim(0, 9)
    axR.legend(loc="upper right", fontsize=8, frameon=True)

    fig.suptitle("Scaling behavior (PRELIMINARY — 20-window sample; full-window re-eval queued)",
                 fontsize=10.5, y=1.02)

    pdf = os.path.join(OUT_DIR, "fig4_scaling.pdf")
    png = os.path.join(OUT_DIR, "fig4_scaling.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {pdf}\n  wrote {png}")

    print("\n=== data scaling (sample) ===")
    for t in tags:
        a = summ(os.path.join(SCALE_DIR, f"{t}.json"))
        print(f"  {DATA_YEARS[t]:>3}y  Lift@5000={a[0]:.2f}±{a[1]:.2f}  Lift@30km={a[2]:.2f}±{a[3]:.2f}")


if __name__ == "__main__":
    main()
