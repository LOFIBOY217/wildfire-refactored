"""Figure 4 (scaling): training-data span, model width, and encoder length.

All three panels are full-window (>=339 win) mean Lift, same aggregation as the
bar figure. Definitive.

4a (data scaling):   Lift vs years of training data (enc21, dm256 fixed).
                     Peak at 12y, declining as older data is added.
4b (model scaling):  Lift vs model width d_model (enc21 fixed). The 12y line
                     (128/256/384) plateaus over 256-384; the 22y line
                     (384/512) sits well below, re-confirming 12y > 22y.
4c (encoder length): Lift vs encoder history length in_days (dm256, 12y fixed).
                     Peak at a 21-day window. NOTE: longer encoders consume more
                     lead-in frames, so the usable val-window count shrinks with
                     in_days (n=463 at 7d down to n=339 at 56d); the sweep is
                     therefore not on an identical window set, though the 21-day
                     peak is robust.

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
EVAL_DIR = os.path.join(ROOT, "results", "eval")
OUT_DIR = os.path.join(ROOT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# 4a data scaling: years -> full-window per-window JSON (relative to EVAL_DIR).
DATA_FULL = {
    4:  "scaling/v3_9ch_enc21_4y_2018_FULL_per_window.json",
    10: "scaling/v3_9ch_enc21_10y_2016_FULL_per_window.json",
    12: "sota_FULL_per_window.json",
    14: "scaling/v3_9ch_enc21_14y_2012_FULL_per_window.json",
    16: "scaling/v3_9ch_enc21_16y_2010_FULL_per_window.json",
    22: "scaling/v3_9ch_enc21_2000_FULL_per_window.json",
}

# 4b model scaling (full-window): file stem -> (d_model, data-span tag).
# dm256 == the enc21/12y SOTA checkpoint (sota_FULL).
DM_FULL = {
    "scaling/v3_9ch_enc21_12y_2014_climsim_dm128_FULL_per_window": (128, "12y"),
    "sota_FULL_per_window":                                        (256, "12y"),
    "scaling/v3_9ch_enc21_12y_2014_climsim_dm384_FULL_per_window": (384, "12y"),
    "scaling/v3_9ch_enc21_2000_climsim_dm384_FULL_per_window":     (384, "22y"),
    "scaling/v3_9ch_enc21_2000_climsim_dm512_FULL_per_window":     (512, "22y"),
}

# 4c encoder length (full-window): in_days -> file stem.
ENC_FULL = {
    7:  "scaling/v3_9ch_enc7_12y_2014_FULL_per_window",
    10: "scaling/v3_9ch_enc10_12y_2014_FULL_per_window",
    14: "scaling/v3_9ch_enc14_12y_2014_FULL_per_window",
    21: "scaling/v3_9ch_enc21_12y_2014_FULL_per_window",
    28: "scaling/v3_9ch_enc28_12y_2014_FULL_per_window",
    35: "scaling/v3_9ch_enc35_12y_2014_FULL_per_window",
    42: "scaling/v3_9ch_enc42_12y_2014_FULL_per_window",
    56: "scaling/v3_9ch_enc56_12y_2014_FULL_per_window",
}


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

    # ===== 4a: DATA SCALING =====
    yrs = np.array(sorted(DATA_FULL))
    vals = np.array([summ(os.path.join(EVAL_DIR, DATA_FULL[y])) for y in yrs])
    m5, s5, m30, s30 = vals[:, 0], vals[:, 1], vals[:, 2], vals[:, 3]

    figA, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.errorbar(yrs, m5, yerr=1.96 * s5, fmt="o-", color=COLORS["sota_single"],
                capsize=3, markersize=6, linewidth=1.8, label="Lift@5000 (pixel)")
    ax.errorbar(yrs, m30, yerr=1.96 * s30, fmt="s--", color=COLORS["ensemble_prob"],
                capsize=3, markersize=5, linewidth=1.6, label="Lift@30 km (cluster)")
    pk = int(np.argmax(m5))
    ax.scatter([yrs[pk]], [m5[pk]], marker="*", s=320, facecolor="gold",
               edgecolor=COLORS["sota_single"], linewidth=1.5, zorder=5,
               label=f"peak = {yrs[pk]}y (SOTA)")
    ax.axhline(1.0, color="#999999", linewidth=0.7, linestyle=":", zorder=0)
    ax.set_xlabel("Training-data span (years)")
    ax.set_ylabel("Lift")
    ax.set_title("Data scaling: forecast skill peaks at 12 years,\n"
                 "then declines as older data is added")
    ax.set_xticks(list(yrs))
    ax.set_ylim(0, 9)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    figA.tight_layout()
    figA.savefig(os.path.join(OUT_DIR, "fig4a_data_scaling.pdf"), bbox_inches="tight")
    figA.savefig(os.path.join(OUT_DIR, "fig4a_data_scaling.png"), bbox_inches="tight")
    plt.close(figA)

    # ===== 4b: MODEL SCALING (full-window) =====
    figB, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.axvspan(256, 384, color="#FFF3CC", alpha=0.7, zorder=0)
    for rng, color, marker in [("12y", COLORS["sota_single"], "o"),
                               ("22y", COLORS["ensemble_prob"], "s")]:
        pts = sorted([(dm, summ(os.path.join(EVAL_DIR, f"{f}.json")))
                      for f, (dm, r) in DM_FULL.items() if r == rng])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1][0] for p in pts]
        es = [1.96 * p[1][1] for p in pts]
        ax.errorbar(xs, ys, yerr=es, fmt=marker + "-", color=color, capsize=3,
                    linewidth=1.8, markersize=6, label=f"{rng} training data")
    ax.text(320, 0.35, "plateau", ha="center", va="bottom", fontsize=8,
            color="#9A7D0A", style="italic")
    ax.set_xlabel("Model width  $d_{model}$")
    ax.set_ylabel("Lift@5000")
    ax.set_title("Model scaling: skill plateaus over $d_{model}=256$ to $384$\n"
                 "(full-window, 12-year training data)")
    ax.set_xticks([128, 256, 384, 512])
    ax.set_ylim(0, 9)
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    figB.tight_layout()
    figB.savefig(os.path.join(OUT_DIR, "fig4b_model_scaling.pdf"), bbox_inches="tight")
    figB.savefig(os.path.join(OUT_DIR, "fig4b_model_scaling.png"), bbox_inches="tight")
    plt.close(figB)

    # ===== 4c: ENCODER LENGTH SWEEP (full-window) =====
    encs = np.array(sorted(ENC_FULL))
    evals = np.array([summ(os.path.join(EVAL_DIR, ENC_FULL[e] + ".json")) for e in encs])
    e5, es5, e30, es30 = evals[:, 0], evals[:, 1], evals[:, 2], evals[:, 3]

    figC, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.errorbar(encs, e5, yerr=1.96 * es5, fmt="o-", color=COLORS["sota_single"],
                capsize=3, markersize=6, linewidth=1.8, label="Lift@5000 (pixel)")
    ax.errorbar(encs, e30, yerr=1.96 * es30, fmt="s--", color=COLORS["ensemble_prob"],
                capsize=3, markersize=5, linewidth=1.6, label="Lift@30 km (cluster)")
    pk = int(np.argmax(e5))
    ax.scatter([encs[pk]], [e5[pk]], marker="*", s=320, facecolor="gold",
               edgecolor=COLORS["sota_single"], linewidth=1.5, zorder=5,
               label=f"peak = {encs[pk]} d (SOTA)")
    ax.set_xlabel("Encoder history length  (days)")
    ax.set_ylabel("Lift")
    ax.set_title("Encoder length: forecast skill peaks at a 21-day\n"
                 "history window (full-window, 12-year training data)")
    ax.set_xticks(list(encs))
    ax.set_ylim(0, 9)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.text(0.015, 0.03,
            "Usable val windows shrink with encoder length (n=463 at 7d to n=339 at 56d).",
            transform=ax.transAxes, fontsize=6.3, color="#666", va="bottom")
    figC.tight_layout()
    figC.savefig(os.path.join(OUT_DIR, "fig4c_encoder_length.pdf"), bbox_inches="tight")
    figC.savefig(os.path.join(OUT_DIR, "fig4c_encoder_length.png"), bbox_inches="tight")
    plt.close(figC)

    # ===== console summary =====
    print("=== 4a data scaling (full-window mean) ===")
    for y, a, b in zip(yrs, m5, m30):
        print(f"  {y:3d}y  L@5000={a:.2f}  L@30km={b:.2f}")
    print("=== 4b model scaling (full-window mean) ===")
    for f, (dm, r) in sorted(DM_FULL.items(), key=lambda kv: (kv[1][1], kv[1][0])):
        a, _, b, _ = summ(os.path.join(EVAL_DIR, f"{f}.json"))
        print(f"  {r:>3s}  dm{dm:<4d} L@5000={a:.2f}  L@30km={b:.2f}")
    print("=== 4c encoder length (full-window mean) ===")
    for e, a, b in zip(encs, e5, e30):
        print(f"  enc{e:<3d} L@5000={a:.2f}  L@30km={b:.2f}")
    print("  wrote fig4a_data_scaling, fig4b_model_scaling, fig4c_encoder_length")


if __name__ == "__main__":
    main()
