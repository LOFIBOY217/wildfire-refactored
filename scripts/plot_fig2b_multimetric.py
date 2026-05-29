"""Figure 2-B: multi-metric comparison (F2 / MCC / BSS / PR-AUC).

Four panels, bars per model. Story:
  - Trivial persistence wins rank/continuation metrics (BSS, MCC, PR-AUC)
    by exploiting multi-week NBAC mega-fire persistence — a degenerate
    baseline.
  - On recall-weighted detection (F2), the Patch Transformer leads ALL
    models, including persistence. (Brier + ROC-AUC, reported in text,
    are also won by SOTA.)

Models with full multi-metric: SOTA, ConvLSTM, MLP (FULL JSONs),
climatology/persistence/fwi (baselines CSV). Ensemble auto-included
when outputs ensemble_prob_FULL_per_window.json lands. LogReg omitted
(only Lift+PR-AUC available; stays in the Lift figure).

Usage:
    python3 scripts/plot_fig2b_multimetric.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_paper_style import COLORS, LABELS, apply_style  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "results", "eval")
OUT_DIR = os.path.join(ROOT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

METRICS = [("f2", "F2  (recall-weighted detection)", "hi"),
           ("mcc", "MCC", "hi"),
           ("bss", "BSS  (skill vs climatology)", "hi"),
           ("pr_auc", "PR-AUC", "hi")]


def boot_ci(arr, n=2000, seed=0):
    arr = np.asarray(arr, float); arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bm = arr[rng.integers(0, len(arr), size=(n, len(arr)))].mean(1)
    return float(arr.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))


def from_json(path):
    pw = json.load(open(path))["per_window"]
    out = {}
    for m, _, _ in METRICS:
        out[m] = boot_ci([w[m] for w in pw if w.get(m) is not None])
    return out


def main():
    apply_style()
    rows = {}
    rows["sota_single"] = from_json(os.path.join(EVAL_DIR, "sota_FULL_per_window.json"))
    rows["convlstm"] = from_json(os.path.join(EVAL_DIR, "convlstm_FULL.json"))
    rows["mlp"] = from_json(os.path.join(EVAL_DIR, "mlp_FULL.json"))
    ens = os.path.join(EVAL_DIR, "ensemble_prob_FULL_per_window.json")
    if os.path.exists(ens):
        rows["ensemble_prob"] = from_json(ens)

    # baselines (point values from CSV, no per-window CI)
    b = pd.read_csv(os.path.join(EVAL_DIR, "baselines_per_window_leakfree.csv"))
    for bl in ["climatology", "persistence", "fwi_oracle"]:
        r = b[(b.baseline == bl) & (b.k == 5000)]
        if len(r):
            rows[bl] = {m: (float(r[m].values[0]), np.nan, np.nan)
                        for m, _, _ in METRICS if m in r.columns}

    order = ["ensemble_prob", "sota_single", "convlstm", "mlp",
             "persistence", "climatology", "fwi_oracle"]
    order = [m for m in order if m in rows]

    fig, axes2d = plt.subplots(2, 2, figsize=(9.5, 8.4))
    axes = axes2d.flatten()
    for ax, (m, title, _) in zip(axes, METRICS):
        xs = np.arange(len(order))
        means = np.array([rows[k].get(m, (np.nan,)*3)[0] for k in order])
        lo = np.array([rows[k].get(m, (np.nan,)*3)[0] - rows[k].get(m, (np.nan,)*3)[1] for k in order])
        hi = np.array([rows[k].get(m, (np.nan,)*3)[2] - rows[k].get(m, (np.nan,)*3)[0] for k in order])
        cols = [COLORS[k] for k in order]
        valid = ~np.isnan(means)
        ax.bar(xs[valid], means[valid], color=[cols[i] for i in range(len(order)) if valid[i]],
               edgecolor="black", linewidth=0.4, alpha=0.92)
        ev = ~np.isnan(lo) & ~np.isnan(hi)
        if ev.any():
            ax.errorbar(xs[ev], means[ev], yerr=[lo[ev], hi[ev]], fmt="none",
                        ecolor="black", elinewidth=0.8, capsize=2.5)
        # mark best (winner)
        if valid.any():
            bi = int(np.nanargmax(means))
            ax.scatter([xs[bi]], [means[bi]], marker="v", s=40,
                       color="black", zorder=6)
        for i, k in enumerate(order):
            # only label in-range values (degenerate BSS≈-14172 handled below)
            if valid[i] and -1.0 <= means[i] <= 1.5:
                ax.text(xs[i], means[i], f" {means[i]:.2f}", rotation=90,
                        ha="center", va="bottom", fontsize=7)
        if m == "bss":
            ax.axhline(0, color="#999", lw=0.7, ls=":")
            ax.set_ylim(-0.65, 0.32)   # clip: fwi_oracle BSS≈-14172 is off-scale
            for i, k in enumerate(order):
                v = rows[k].get(m, (np.nan,)*3)[0]
                if np.isfinite(v) and v < -0.65:
                    ax.text(xs[i], -0.6, f"{v:.0f}\n(off-scale)", rotation=0,
                            ha="center", va="bottom", fontsize=6, color="#C0392B")
        ax.set_xticks(xs)
        ax.set_xticklabels([LABELS[k] for k in order], rotation=40, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9.5)
    fig.suptitle("Multi-metric comparison (full 435-window validation, NBAC+NFDB)\n"
                 "▼ = best.  Persistence wins rank metrics (degenerate); "
                 "our models win recall-weighted F2.", fontsize=10, y=1.0)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"fig2b_multimetric.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  wrote figures/fig2b_multimetric.{pdf,png}")
    print(f"\n  models: {order}")
    for m, _, _ in METRICS:
        best = max((k for k in order if not np.isnan(rows[k].get(m,(np.nan,))[0])),
                   key=lambda k: rows[k][m][0], default=None)
        print(f"  {m:8s} best = {best} ({rows[best][m][0]:.3f})")


if __name__ == "__main__":
    main()
