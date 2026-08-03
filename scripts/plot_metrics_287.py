"""Wildfire prediction validation across F2, MCC, Brier Skill Score, PR-AUC.
2x2 layout, fixed model order across panels. ALL models (DL + MLP + persistence +
climatology + FWI oracle) on the SAME 287-window 2023/24 in-distribution test —
baselines recomputed with the identical compute_all_metrics pipeline as the DL
per-window evals (convstem cross-check reproduces eval_convstem_2324 exactly).
Canonical model->color (plot_paper_style). BSS is n/a for FWI oracle (not a
probability -> Brier undefined). Persistence is degenerate (tops f2/mcc/pr_auc by
copying already-burning fire); hatched. On these matched windows persistence BSS
is negative — only fcnhead and MLP achieve BSS > 0.

Usage: python3 scripts/plot_metrics_287.py
"""
from __future__ import annotations
import os, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_paper_style import COLORS, LABELS, apply_style  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "eval", "metrics_287_ci.csv")  # mean + bootstrap 95% CI
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

# Fixed order across all four panels (ours first by quality, then baselines).
ORDER = ["fcnhead", "convstem_novel", "convstem", "flatten", "convlstm",
         "mlp", "persistence", "climatology", "fwi_oracle"]
PANELS = [("f2", "F2  (recall-weighted detection)"),
          ("mcc", "MCC"),
          ("bss", "Brier Skill Score  (skill vs climatology)"),
          ("pr_auc", "PR-AUC")]


def main():
    apply_style()
    df = pd.read_csv(CSV).set_index("key").reindex(ORDER).reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (col, title) in zip(axes.ravel(), PANELS):
        vmax = df[col].max(skipna=True)
        for i, r in df.iterrows():
            k = r["key"]; v = r[col]
            degen = int(r.get("degenerate", 0)) == 1
            if pd.isna(v):
                ax.text(i, 0.0, "n/a", ha="center", va="bottom", fontsize=8, color="#999")
                continue
            ax.bar(i, v, color=COLORS.get(k, "#888"), edgecolor="black", linewidth=0.4,
                   alpha=0.55 if degen else 0.92, hatch="//" if degen else None)
            lo, hi = r.get(f"{col}_lo", np.nan), r.get(f"{col}_hi", np.nan)
            if np.isfinite(lo) and np.isfinite(hi):
                ax.errorbar(i, v, yerr=[[max(v - lo, 0)], [max(hi - v, 0)]], fmt="none",
                            ecolor="#222", elinewidth=0.9, capsize=2.5)
            off = 0.012 * (vmax if vmax else 1)
            ytxt = (hi if np.isfinite(hi) else v) + off if v >= 0 else (lo if np.isfinite(lo) else v) - off
            ax.text(i, ytxt, f"{v:.2f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)
        if col == "bss":
            ax.axhline(0.0, color="#444", linewidth=0.8)
        ax.set_xticks(np.arange(len(df)))
        ax.set_xticklabels([LABELS.get(k, k) for k in df["key"]], rotation=38, ha="right", fontsize=8)
        ax.set_title(title, fontsize=11)
    fig.suptitle("Wildfire Prediction Validation Across F2, MCC, Brier Skill Score, and PR-AUC "
                 "(all on the 287-window 2023/24 test)\n"
                 "Persistence tops the rank metrics (degenerate — copies already-burning fire) "
                 "but its BSS is negative; only fcnhead and MLP have BSS > 0",
                 fontsize=12.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_metrics_287.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(df.to_string(index=False))
    print("\nwrote figures/fig_metrics_287.{png,pdf}")


if __name__ == "__main__":
    main()
