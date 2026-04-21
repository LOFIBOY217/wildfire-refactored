#!/usr/bin/env python3
"""
Compare two fire-label stacks (old CWFIS-only vs new NBAC+NFDB).

Reports:
    - total positive pixel count per scheme
    - per-year positive pixel count
    - per-year "how many MORE fires (pixels) did new label add over old"
    - pixel-level overlap: (old ∩ new) / (old ∪ new) = IoU
    - pixel-level NEW-only: pixels in new but not old (drift correction)
    - pixel-level OLD-only: pixels in old but not new (CWFIS false positives + tiny fires)

Usage:
    python scripts/compare_labels.py \\
        --old data/fire_labels/fire_labels_cwfis_2000-05-01_2025-12-21_2281x2709_r14.npy \\
        --new data/fire_labels/fire_labels_nbac_nfdb_2000-05-01_2025-12-21_2281x2709_r14.npy \\
        --out data/audit/label_comparison.csv
"""

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def build_date_list(start: str, end: str):
    sd = date.fromisoformat(start)
    ed = date.fromisoformat(end)
    dates = []
    cur = sd
    while cur <= ed:
        dates.append(cur)
        cur += timedelta(days=1)
    return dates


def load_with_provenance(npy_path: Path):
    stack = np.load(npy_path, mmap_mode="r")
    json_path = npy_path.with_suffix(".json")
    provenance = {}
    if json_path.exists():
        with open(json_path) as f:
            provenance = json.load(f)
    return stack, provenance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, type=Path,
                    help="CWFIS-only label .npy")
    ap.add_argument("--new", required=True, type=Path,
                    help="NBAC+NFDB label .npy")
    ap.add_argument("--out", default="data/audit/label_comparison.csv")
    args = ap.parse_args()

    print(f"Loading old label: {args.old}")
    old_stack, old_prov = load_with_provenance(args.old)
    print(f"  shape={old_stack.shape}  dtype={old_stack.dtype}")
    print(f"  provenance: scheme={old_prov.get('scheme')}  "
          f"raw_pos={old_prov.get('raw_positive'):,}  "
          f"dilated_pos={old_prov.get('dilated_positive'):,}")

    print(f"\nLoading new label: {args.new}")
    new_stack, new_prov = load_with_provenance(args.new)
    print(f"  shape={new_stack.shape}  dtype={new_stack.dtype}")
    print(f"  provenance: scheme={new_prov.get('scheme')}  "
          f"raw_pos={new_prov.get('raw_positive'):,}  "
          f"dilated_pos={new_prov.get('dilated_positive'):,}")

    if old_stack.shape != new_stack.shape:
        raise SystemExit(
            f"Shape mismatch: old={old_stack.shape} new={new_stack.shape}")

    T, H, W = old_stack.shape
    npix_per_frame = H * W

    # Reconstruct date list from provenance
    dr = old_prov.get("date_range") or new_prov.get("date_range")
    dates = build_date_list(dr[0], dr[1])
    if len(dates) != T:
        print(f"  [warn] date_range→T={len(dates)} but stack T={T}; "
              f"truncating dates to stack")
        dates = dates[:T]

    # ------------------------------------------------------------
    # Per-year metrics (memory-friendly: loop day-by-day)
    # ------------------------------------------------------------
    years = pd.DatetimeIndex([pd.Timestamp(d) for d in dates]).year
    unique_years = sorted(set(years))
    rows = []
    print(f"\nComputing per-year overlap ({len(unique_years)} years)...")
    for yr in unique_years:
        idx = np.where(years == yr)[0]
        old_sum, new_sum = 0, 0
        both, new_only, old_only = 0, 0, 0
        for t in idx:
            old_f = np.asarray(old_stack[t]).astype(bool)
            new_f = np.asarray(new_stack[t]).astype(bool)
            old_sum += int(old_f.sum())
            new_sum += int(new_f.sum())
            both += int((old_f & new_f).sum())
            new_only += int((new_f & ~old_f).sum())
            old_only += int((old_f & ~new_f).sum())

        union = old_sum + new_sum - both
        iou = both / union if union > 0 else 0.0
        rows.append({
            "year": int(yr),
            "old_pos_pix": int(old_sum),
            "new_pos_pix": int(new_sum),
            "delta_abs": int(new_sum - old_sum),
            "delta_pct": round(100 * (new_sum - old_sum) / max(old_sum, 1), 1),
            "both": int(both),
            "new_only": int(new_only),
            "old_only": int(old_only),
            "iou": round(iou, 3),
        })
        print(f"  {yr}: old={old_sum:>10,}  new={new_sum:>10,}  "
              f"Δ={new_sum - old_sum:+12,} ({100 * (new_sum - old_sum) / max(old_sum, 1):+6.1f}%)  "
              f"IoU={iou:.3f}  new_only={new_only:,}")

    df = pd.DataFrame(rows).set_index("year")

    print("\n=== SUMMARY ===")
    totals = df.sum()
    print(f"Old label total positive-pixel-days: {int(totals['old_pos_pix']):,}")
    print(f"New label total positive-pixel-days: {int(totals['new_pos_pix']):,}")
    print(f"  → Delta: {int(totals['new_pos_pix'] - totals['old_pos_pix']):+,} "
          f"({100 * (totals['new_pos_pix'] - totals['old_pos_pix']) / max(totals['old_pos_pix'], 1):+.1f}%)")
    total_union = totals['old_pos_pix'] + totals['new_pos_pix'] - totals['both']
    print(f"  Overall IoU: {totals['both'] / max(total_union, 1):.3f}")

    print(f"\nEarly era (2000-2011):")
    early = df.loc[[y for y in df.index if y <= 2011]]
    print(f"  mean Δ%: {early['delta_pct'].mean():+.1f}%")
    print(f"  mean IoU: {early['iou'].mean():.3f}")
    print(f"  total new_only: {int(early['new_only'].sum()):,}")

    print(f"\nLate era (2012-now):")
    late = df.loc[[y for y in df.index if y >= 2012]]
    print(f"  mean Δ%: {late['delta_pct'].mean():+.1f}%")
    print(f"  mean IoU: {late['iou'].mean():.3f}")
    print(f"  total new_only: {int(late['new_only'].sum()):,}")

    # Interpretation hint
    print(f"\n=== INTERPRETATION ===")
    if early['delta_pct'].mean() > 50 and late['delta_pct'].mean() < 20:
        print(f"  ✓ Pattern matches drift hypothesis: early years get "
              f"MUCH more positive labels from NBAC+NFDB than CWFIS; "
              f"late years converge.")
    else:
        print(f"  Pattern unclear — inspect per-year table.")

    # Save CSV
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
