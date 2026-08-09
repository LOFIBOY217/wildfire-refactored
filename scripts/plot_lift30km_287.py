"""Cluster scale companion figure. Two panels side by side, total Lift@30km on the
left and novel ignition Lift@30km on the right, all nine models. Total uses the
window mean and novel uses the fire weighted mean, matching Figures 4 and 5. All
values recomputed with the official compute_coarsened_lift (factor 15) so the
learned models and the physical baselines share one method.

Data: results/eval/l30_287_learned.json + results/eval/l30_287_phys.json
Usage: python3 scripts/plot_lift30km_287.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_paper_style import COLORS, SHORT, BAR_ORDER, apply_style  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "results", "eval")
OUT = os.path.join(ROOT, "figures")
LEARNED = BAR_ORDER[:6]
PHYS = BAR_ORDER[6:]


def agg(rec, how):
    a = np.array(rec["l30"], float); w = np.array(rec["n"], float); ok = np.isfinite(a)
    if how == "macro":
        return float(a[ok].mean()) if ok.any() else np.nan
    okm = ok & (w > 0)
    return float(np.average(a[okm], weights=w[okm])) if okm.any() else np.nan


def load(target, how):
    dl = json.load(open(os.path.join(EVAL, "l30_287_learned.json")))["res"][target]
    dp = json.load(open(os.path.join(EVAL, "l30_287_phys.json")))["res"][target]
    pmap = {"climatology": "climatology", "persistence": "persistence", "fwi_oracle": "fwi_oracle"}
    vals = {m: agg(dl[m], how) for m in LEARNED}
    for m in PHYS:
        vals[m] = agg(dp[pmap[m]], how)
    return vals


def main():
    apply_style()
    tot = load("total", "macro")
    nov = load("novel30", "micro")
    ym = 8.0  # shared y scale across both panels; degenerate persistence is clipped
    panels = [(tot, "Total fire"), (nov, "Novel ignition")]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), sharey=True)
    for ax, (vals, title) in zip(axes, panels):
        for i, m in enumerate(BAR_ORDER):
            v = vals.get(m, np.nan)
            if not np.isfinite(v):
                ax.text(i, 0.05, "n/a", ha="center", va="bottom", fontsize=7, color="#999")
                continue
            degen = v > ym
            ax.bar(i, min(v, ym), color=COLORS[m], edgecolor="black", linewidth=0.4,
                   alpha=0.55 if degen else 0.92, hatch="//" if degen else None)
            if degen:
                ax.annotate(f"{v:.0f} ↑", xy=(i, ym * 0.95), ha="center", va="top",
                            fontsize=7, color="#333")
            else:
                ax.text(i, v + ym * 0.01, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
        ax.axhline(1.0, color="#999", linewidth=0.7, linestyle=":")
        ax.set_xticks(range(len(BAR_ORDER)))
        ax.set_xticklabels([SHORT.get(m, m) for m in BAR_ORDER], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Lift@30 km")
        ax.set_ylim(0, ym)
        ax.set_title(title)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_lift30km_287.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("total 30km (macro):", {m: round(tot[m], 2) for m in BAR_ORDER})
    print("novel 30km (micro):", {m: round(nov[m], 2) for m in BAR_ORDER})
    print("wrote figures/fig_lift30km_287.png")


if __name__ == "__main__":
    main()
