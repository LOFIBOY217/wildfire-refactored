#!/usr/bin/env python3
"""
Plot val_lift_k vs SGD step from the lift_trajectory CSV
produced by --mid_epoch_val_every.

Usage:
  python scripts/plot_lift_trajectory.py \
    --csv outputs/v3_9ch_enc14_2000_lift_traj_lift_trajectory.csv

Tests the hypothesis that val Lift@K peaks at ~5-10k SGD updates
then declines. See docs/CALIBRATION_VS_RANK_HYPOTHESIS.md.
"""
import argparse
import csv
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None,
                    help="Output PNG path (default: alongside CSV)")
    args = ap.parse_args()

    if args.out is None:
        args.out = args.csv.replace(".csv", ".png")

    rows = []
    with open(args.csv) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: float(v) if k not in ("epoch",) else int(v)
                         for k, v in row.items()})

    if not rows:
        print("No data in CSV.")
        return

    print(f"  {len(rows)} eval points")
    print(f"  step range: {int(rows[0]['global_step'])} - {int(rows[-1]['global_step'])}")
    print()
    print(f"  {'step':>8} {'epoch':>5} {'train_loss':>10} {'val_lift':>8} {'roc_auc':>8} {'brier':>8}")
    for r in rows:
        print(f"  {int(r['global_step']):>8} {int(r['epoch']):>5} "
              f"{r['train_loss']:>10.4f} {r['val_lift_k']:>8.3f} "
              f"{r['val_roc_auc']:>8.4f} {r['val_brier']:>8.4f}")

    # Find peak
    peak = max(rows, key=lambda r: r['val_lift_k'])
    print()
    print(f"  PEAK val_lift = {peak['val_lift_k']:.3f}x at step {int(peak['global_step'])} "
          f"(epoch {int(peak['epoch'])}, batch {int(peak['batch'])})")

    # Quick analysis
    final = rows[-1]
    if peak['global_step'] < final['global_step']:
        delta = (peak['val_lift_k'] - final['val_lift_k']) / final['val_lift_k'] * 100
        print(f"  HYPOTHESIS SUPPORTED: peak ({peak['val_lift_k']:.2f}x) is "
              f"{delta:+.1f}% above final ({final['val_lift_k']:.2f}x)")
    else:
        print(f"  HYPOTHESIS NOT SUPPORTED: val_lift still rising at end")

    # Try to plot if matplotlib available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps = [r['global_step'] for r in rows]
        lifts = [r['val_lift_k'] for r in rows]
        rocs = [r['val_roc_auc'] for r in rows]
        briers = [r['val_brier'] for r in rows]
        train_losses = [r['train_loss'] for r in rows]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(steps, lifts, "o-", color="C3")
        axes[0, 0].axvline(peak['global_step'], color="grey", linestyle="--",
                           label=f"peak step {int(peak['global_step'])}")
        axes[0, 0].set_xlabel("SGD step"); axes[0, 0].set_ylabel("val Lift@K")
        axes[0, 0].set_title("Val Lift@K (PRIMARY HYPOTHESIS METRIC)")
        axes[0, 0].legend(); axes[0, 0].grid(True)

        axes[0, 1].plot(steps, rocs, "o-", color="C0")
        axes[0, 1].set_xlabel("SGD step"); axes[0, 1].set_ylabel("val ROC-AUC")
        axes[0, 1].set_title("ROC-AUC (predicted to be stable)")
        axes[0, 1].grid(True)

        axes[1, 0].plot(steps, briers, "o-", color="C2")
        axes[1, 0].set_xlabel("SGD step"); axes[1, 0].set_ylabel("val Brier")
        axes[1, 0].set_title("Brier (predicted to improve monotonically)")
        axes[1, 0].grid(True)

        axes[1, 1].plot(steps, train_losses, "o-", color="C1")
        axes[1, 1].set_xlabel("SGD step"); axes[1, 1].set_ylabel("train loss")
        axes[1, 1].set_title("Train loss (decreases monotonically)")
        axes[1, 1].grid(True)

        fig.suptitle(os.path.basename(args.csv))
        fig.tight_layout()
        fig.savefig(args.out, dpi=120)
        print(f"  → wrote {args.out}")
    except ImportError:
        print("  (matplotlib not available; CSV analysis only)")


if __name__ == "__main__":
    main()
