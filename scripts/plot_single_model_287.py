"""Single-model comparison, 2023/24 287-window in-distribution eval.
Lift@5000 (pixel) + Lift@30km (cluster). SINGLE models only (no ensemble).
Colors come from the canonical plot_paper_style palette — keep consistent with
every other figure. Reads results/eval/single_model_287.csv; add baseline rows
(climatology/fwi_oracle/logreg/ecmwf_s2s/persistence) there once computed.

Usage: python3 scripts/plot_single_model_287.py
"""
from __future__ import annotations
import os, sys, json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_paper_style import COLORS, SHORT, BAR_ORDER, apply_style  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "eval", "single_model_287.csv")
EVAL = os.path.join(ROOT, "results", "eval")
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
N_BASELINE_WIN = 435  # SE approximation for baseline CSVs (same as fig2)


def _baseline_rows():
    """climatology / fwi_oracle / persistence / logreg / ecmwf_s2s, read from the
    SAME leak-free sources as plot_fig2_aggregate_bars (values the user confirmed
    correct). Baselines are ~test-set-insensitive (static clim / deterministic FWI
    / near-zero S2S skill), so reusing the 435-win numbers is sound."""
    rows = []
    bl = pd.read_csv(os.path.join(EVAL, "baselines_per_window_leakfree.csv"))

    def from_csv(name, key):
        r = bl[(bl.baseline == name) & (bl.k == 5000)].iloc[0]
        def ci(m, sd):
            se = (sd / np.sqrt(N_BASELINE_WIN)) if sd and not np.isnan(sd) else 0.0
            return m - 1.96 * se, m + 1.96 * se
        l5lo, l5hi = ci(r.lift_k, r.lift_k_std)
        l3lo, l3hi = ci(r.lift_coarse, r.lift_coarse_std)
        return dict(key=key, lift_5000=r.lift_k, l5_lo=l5lo, l5_hi=l5hi,
                    lift_30km=r.lift_coarse, l30_lo=l3lo, l30_hi=l3hi)

    for name, key in (("climatology", "climatology"), ("fwi_oracle", "fwi_oracle"),
                      ("persistence", "persistence")):
        try:
            rows.append(from_csv(name, key))
        except Exception as e:
            print(f"  [skip {key}] {e}")
    # MLP is now in single_model_287.csv on the correct 287-window dump basis
    # (5.87/5.74), NOT the old 435-win mlp_FULL reuse — so it is not added here.
    return pd.DataFrame(rows)


def main():
    apply_style()
    # Fixed canonical 9-model order (no sort) — identical across Fig 4/5/6/8.
    df = pd.concat([pd.read_csv(CSV), _baseline_rows()], ignore_index=True)
    df = df.set_index("key").reindex(BAR_ORDER).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4))
    YMAX = {"lift_5000": 11.0, "lift_30km": 9.5}  # cap; persistence clipped
    panels = [("lift_5000", "Lift@5000  (pixel-scale)", "Lift@5000"),
              ("lift_30km", "Lift@30 km  (cluster-scale)", "Lift@30 km")]
    for ax, (col, title, ylab) in zip(axes, panels):
        ym = YMAX[col]
        for i, r in df.iterrows():
            k = r["key"]
            m = r[col]
            if pd.isna(m):
                ax.text(i, 0.1, "n/a", ha="center", va="bottom", fontsize=7.5, color="#999")
                continue
            degen = (k == "persistence")
            ax.bar(i, min(m, ym), color=COLORS.get(k, "#888"), edgecolor="black", linewidth=0.4,
                   alpha=0.5 if degen else 0.92, hatch="//" if degen else None)
            if m > ym:
                ax.annotate(f"{m:.1f} ↑\n(degenerate)", xy=(i, ym * 0.93),
                            ha="center", va="top", fontsize=6.8, color="#333")
            else:
                ax.text(i, m + 0.12, f"{m:.2f}", ha="center", va="bottom", fontsize=7.5)
        ax.axhline(1.0, color="#999", linewidth=0.7, linestyle=":")
        ax.set_xticks(np.arange(len(df)))
        ax.set_xticklabels([SHORT.get(k, k) for k in df["key"]], rotation=35, ha="right",
                           fontsize=8)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_ylim(0, ym)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_single_model_287.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(df.to_string(index=False))
    print("\nwrote figures/fig_single_model_287.{png,pdf}")


if __name__ == "__main__":
    main()
