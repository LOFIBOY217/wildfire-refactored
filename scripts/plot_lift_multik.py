"""Multi-budget Lift bar figure (grouped bars, Lift@1k / 5k / 10k per model as
dark->light shades of each model's canonical hue). One consistent 9-model set,
canonical colors / order / short labels. Shows that the headline Lift is stable
across the operational budget, not a single convenient K.

Data cube (assembled here):
  learned 6 : results/eval/multik_287_learned.json  (per-window -> macro or micro)
  physical 3: results/eval/novel_baselines_multik_287.csv  (canonical, macro)
  2026 all 9: results/eval/multik_2026.json          (per-window -> macro or micro)

Usage:
  python3 scripts/plot_lift_multik.py 287 total macro
  python3 scripts/plot_lift_multik.py 287 novel micro
  python3 scripts/plot_lift_multik.py 2026 total micro
  python3 scripts/plot_lift_multik.py 2026 novel micro
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_paper_style import COLORS, SHORT, BAR_ORDER, apply_style  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "results", "eval")
OUT = os.path.join(ROOT, "figures")
KS = [1000, 5000, 10000]
DOMAIN = 23998 * 256                        # scored grid cells
PCT = {k: 100.0 * k / DOMAIN for k in KS}   # budget as % of domain
LEARNED = BAR_ORDER[:6]
PHYS = BAR_ORDER[6:]                         # climatology, persistence, fwi_oracle
YMAX = {"total": 11.0, "novel": 9.5}


def shade(c, k):
    rgb = np.array(mcolors.to_rgb(c))
    if k == KS[0]:      rgb = rgb * 0.72                 # smallest budget -> darkest
    elif k == KS[-1]:   rgb = rgb + (1 - rgb) * 0.48     # largest budget -> lightest
    return tuple(np.clip(rgb, 0, 1))


def agg_learned(rec, K, how):
    a = np.array(rec["lift_k"][str(K)], float)
    w = np.array(rec["n"], float)
    ok = np.isfinite(a)
    if how == "macro":
        return float(a[ok].mean()) if ok.any() else np.nan
    okm = ok & (w > 0)
    return float(np.average(a[okm], weights=w[okm])) if okm.any() else np.nan


def load_values(period, target, how):
    """Return {model: {K: lift}} for the 9 canonical models."""
    tkey = "total" if target == "total" else "novel30"
    vals = {}
    if period == "287":
        dl = json.load(open(os.path.join(EVAL, "multik_287_learned.json")))["res"]
        for m in LEARNED:
            vals[m] = {K: agg_learned(dl[tkey][m], K, how) for K in KS}
        csv = pd.read_csv(os.path.join(EVAL, "novel_baselines_multik_287.csv"))
        lm = "total" if target == "total" else "novel_30d"
        pmap = {"climatology": "climatology", "persistence": "persistence", "fwi_oracle": "fwi_oracle"}
        for m in PHYS:
            sub = csv[(csv.baseline == pmap[m]) & (csv.label_mode == lm)]
            vals[m] = {K: float(sub[sub.k == K]["lift_mean"].iloc[0]) if (sub.k == K).any() else np.nan for K in KS}
    else:  # 2026
        r2 = json.load(open(os.path.join(EVAL, "multik_2026.json")))["res"][tkey]
        namemap = {"climatology": "clim", "persistence": "pers", "fwi_oracle": "fwi"}
        for m in BAR_ORDER:
            key = namemap.get(m, m)
            if key not in r2:
                vals[m] = {K: np.nan for K in KS}; continue
            vals[m] = {K: agg_learned(r2[key], K, how) for K in KS}
    return vals


def main():
    period, target, how = (sys.argv[1:] + ["287", "total", "macro"])[:3]
    apply_style()
    vals = load_values(period, target, how)
    ym = YMAX["total" if target == "total" else "novel"]
    w = 0.26
    fig, ax = plt.subplots(figsize=(12.8, 5.6))
    for gi, m in enumerate(BAR_ORDER):
        for j, K in enumerate(KS):
            v = vals[m].get(K, np.nan)
            x = gi + (j - 1) * w
            if not np.isfinite(v):
                ax.text(x, 0.1, "n/a", ha="center", va="bottom", fontsize=6, color="#999")
                continue
            degen = v > ym
            ax.bar(x, min(v, ym), width=w, color=shade(COLORS[m], K), edgecolor="black",
                   linewidth=0.4, alpha=0.55 if degen else 0.95, hatch="//" if degen else None)
            if degen:
                if j == 1:
                    ax.annotate(f"{v:.0f} ↑", xy=(gi, ym * 0.95), ha="center",
                                va="top", fontsize=6.5, color="#333")
            else:
                ax.text(x, v + ym * 0.008, f"{v:.1f}", ha="center", va="bottom", fontsize=6.2)
    ax.axhline(1.0, color="#999", linewidth=0.7, linestyle=":")
    ax.set_xticks(range(len(BAR_ORDER)))
    ax.set_xticklabels([SHORT.get(m, m) for m in BAR_ORDER], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Lift")
    ax.set_ylim(0, ym)
    ax.set_title("Novel-ignition lift across the operational budget"
                 if target != "total" else "Lift across the operational budget")
    # legend: shades = budgets, with operational % framing
    from matplotlib.patches import Patch
    leg = [Patch(facecolor=shade("#5b5b5b", K), edgecolor="black",
                 label=f"Lift@{K//1000}k") for K in KS]
    ax.legend(handles=leg, frameon=False, fontsize=8, loc="upper right", title="alert budget")
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    name = f"fig_lift_multik_{period}_{target}_{how}"
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{name}.png")
    for m in BAR_ORDER:
        print(f"  {m:16s} " + "  ".join(f"{K//1000}k={vals[m].get(K, float('nan')):.2f}" for K in KS))


if __name__ == "__main__":
    main()
