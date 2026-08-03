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
from plot_paper_style import COLORS, LABELS, apply_style  # noqa: E402

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
    # logreg (no 30km)
    try:
        lr = pd.read_csv(os.path.join(EVAL, "benchmark_logreg.csv"))
        lrr = lr[(lr.k == 5000) & (lr.lift_k.notna())].iloc[-1]
        rows.append(dict(key="logreg", lift_5000=float(lrr.lift_k), l5_lo=float(lrr.lift_k),
                         l5_hi=float(lrr.lift_k), lift_30km=np.nan, l30_lo=np.nan, l30_hi=np.nan))
    except Exception as e:
        print(f"  [skip logreg] {e}")
    # MLP — no 287-win dump exists; reuse leak-free 435-win per-window (same
    # source as fig2). Learned model but treated like the other reused numbers.
    try:
        pw = json.load(open(os.path.join(EVAL, "mlp_FULL.json")))["per_window"]
        def bc(a):
            a = np.asarray([x for x in a if x is not None], float)
            r = np.random.default_rng(0); m = a[r.integers(0, len(a), size=(2000, len(a)))].mean(1)
            return a.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)
        l5 = bc([w.get("lift_k") for w in pw]); l3 = bc([w.get("lift_coarse") for w in pw])
        rows.append(dict(key="mlp", lift_5000=l5[0], l5_lo=l5[1], l5_hi=l5[2],
                         lift_30km=l3[0], l30_lo=l3[1], l30_hi=l3[2]))
    except Exception as e:
        print(f"  [skip mlp] {e}")
    # ECMWF S2S
    try:
        e = json.load(open(os.path.join(EVAL, "baseline_ecmwf_s2s.json")))
        rows.append(dict(key="ecmwf_s2s",
                         lift_5000=e["lift_5000"]["mean"], l5_lo=e["lift_5000"]["ci_lo"], l5_hi=e["lift_5000"]["ci_hi"],
                         lift_30km=e["lift_30km"]["mean"], l30_lo=e["lift_30km"]["ci_lo"], l30_hi=e["lift_30km"]["ci_hi"]))
    except Exception as e:
        print(f"  [skip ecmwf] {e}")
    return pd.DataFrame(rows)


def main():
    apply_style()
    df = pd.concat([pd.read_csv(CSV), _baseline_rows()], ignore_index=True)
    df = df.sort_values("lift_5000").reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    YMAX = {"lift_5000": 11.0, "lift_30km": 9.5}  # cap; persistence clipped
    panels = [("lift_5000", "l5_lo", "l5_hi", "Lift@5000  (pixel-scale)", "Lift@5000"),
              ("lift_30km", "l30_lo", "l30_hi", "Lift@30 km  (cluster-scale)", "Lift@30 km")]
    for ax, (col, lo, hi, title, ylab) in zip(axes, panels):
        ym = YMAX[col]
        for i, r in df.iterrows():
            k = r["key"]
            m = r[col]
            if pd.isna(m):
                ax.text(i, 0.1, "n/a", ha="center", va="bottom", fontsize=7.5, color="#999")
                continue
            degen = (k == "persistence")
            ax.bar(i, m, color=COLORS.get(k, "#888"), edgecolor="black", linewidth=0.4,
                   alpha=0.5 if degen else 0.92, hatch="//" if degen else None)
            clipped = m > ym
            if (not pd.isna(r[lo])) and (not clipped):
                ax.errorbar(i, m, yerr=[[m - r[lo]], [r[hi] - m]], fmt="none", ecolor="black",
                            elinewidth=0.9, capsize=3, capthick=0.9)
            if clipped:
                ax.annotate(f"{m:.1f}  ↑\n(degenerate,\nnovel = 0)", xy=(i, ym * 0.93),
                            ha="center", va="top", fontsize=6.8, color="#333")
            else:
                ax.text(i, m + 0.12, f"{m:.2f}", ha="center", va="bottom", fontsize=7.5)
        ax.axhline(1.0, color="#999", linewidth=0.7, linestyle=":")
        ax.set_xticks(np.arange(len(df)))
        ax.set_xticklabels([LABELS.get(k, k) for k in df["key"]], rotation=35, ha="right")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_ylim(0, ym)

    fig.suptitle("Single-model prediction performance — pixel & 30 km Lift "
                 "(DL: 2023/24 287-window in-dist; baselines: leak-free 2022-25)",
                 fontsize=11.5, y=1.01)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_single_model_287.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(df.to_string(index=False))
    print("\nwrote figures/fig_single_model_287.{png,pdf}")


if __name__ == "__main__":
    main()
