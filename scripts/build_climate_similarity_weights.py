"""
Build per-year sample weights based on climate similarity to the
validation period.

Motivation: instead of a generic "recency" weight (newer = better),
weight each training year by how climatologically similar it is to the
validation period. Fire activity has strong multi-year cycles (ENSO,
PDO, drought) — matching climate state matters more than calendar
proximity.

This script uses the **ONI (Oceanic Niño Index)** fire-season (Apr-Oct)
mean per year. Hard-coded values are publicly available from NOAA NCEI
(https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php).

Output:
  CSV with columns: year, oni_fire_season_mean, abs_diff_from_val,
                    similarity_weight_unnormalized, weight (normalised mean=1)

Usage:
  python -m scripts.build_climate_similarity_weights \\
      --train_years 2014 2015 2016 2017 2018 2019 2020 2021 \\
      --val_years 2022 2023 2024 2025 \\
      --scale auto \\
      --output data/climate_indices/oni_similarity_12y_to_val2022_2025.csv
"""
from __future__ import annotations
import argparse
import csv
import os
from pathlib import Path

# ONI Apr-Oct mean per year, derived from NOAA monthly ERSSTv5 ONI table
# (3-month running mean, fire-season subset). Updated 2026-05-03.
ONI_FIRE_SEASON_MEAN = {
    1981:  0.21, 1982:  1.00, 1983:  0.61, 1984: -0.27, 1985: -0.50,
    1986:  0.07, 1987:  1.41, 1988: -0.69, 1989: -0.34, 1990:  0.34,
    1991:  0.66, 1992:  0.34, 1993:  0.36, 1994:  0.36, 1995: -0.37,
    1996: -0.26, 1997:  1.91, 1998: -0.86, 1999: -0.93, 2000: -0.61,
    2001: -0.16, 2002:  0.79, 2003:  0.30, 2004:  0.50, 2005: -0.06,
    2006:  0.26, 2007: -0.53, 2008: -0.43, 2009:  0.69, 2010: -0.94,
    2011: -0.49, 2012:  0.20, 2013: -0.20, 2014:  0.04, 2015:  1.78,
    2016:  0.04, 2017: -0.27, 2018:  0.10, 2019:  0.31, 2020: -0.74,
    2021: -0.46, 2022: -0.93, 2023:  1.07, 2024:  0.51, 2025: -0.05,
}


def fire_season_mean(year):
    if year not in ONI_FIRE_SEASON_MEAN:
        raise KeyError(f"No ONI value for year {year}")
    return ONI_FIRE_SEASON_MEAN[year]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_years", type=int, nargs="+", required=True)
    ap.add_argument("--val_years", type=int, nargs="+", required=True)
    ap.add_argument("--scale", type=str, default="auto",
                    help="Distance scale for the exp similarity weight. "
                         "'auto' = std(ONI[train_years]); else float.")
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()

    val_oni = [fire_season_mean(y) for y in args.val_years]
    train_oni = [fire_season_mean(y) for y in args.train_years]

    val_mean = sum(val_oni) / len(val_oni)

    if args.scale == "auto":
        # std of training year ONI values
        if len(train_oni) <= 1:
            scale = 1.0
        else:
            mu = sum(train_oni) / len(train_oni)
            var = sum((x - mu) ** 2 for x in train_oni) / (len(train_oni) - 1)
            scale = max(var ** 0.5, 1e-3)
    else:
        scale = float(args.scale)

    print(f"Val years      : {args.val_years}")
    print(f"  ONI per year : {[round(x, 2) for x in val_oni]}")
    print(f"  Mean         : {val_mean:.3f}")
    print(f"Train years    : {args.train_years}")
    print(f"  ONI per year : {[round(x, 2) for x in train_oni]}")
    print(f"Distance scale : {scale:.3f}")
    print()

    # Compute raw similarity (Gaussian on |diff|)
    raw_w = []
    for y, o in zip(args.train_years, train_oni):
        diff = abs(o - val_mean)
        w = pow(2.71828182845904523536, -(diff / scale))   # exp(-d/scale)
        raw_w.append(w)

    mean_w = sum(raw_w) / len(raw_w)
    norm_w = [w / mean_w for w in raw_w]

    # Write CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("year   oni    |diff|   raw_w   norm_w")
    with open(out_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["year", "oni_fire_season_mean",
                     "abs_diff_from_val_mean",
                     "similarity_weight_unnormalized",
                     "weight"])
        for y, o, w_raw, w_n in zip(args.train_years, train_oni, raw_w, norm_w):
            d = abs(o - val_mean)
            print(f"{y}   {o:+.2f}   {d:.3f}   {w_raw:.3f}   {w_n:.3f}")
            wr.writerow([y, f"{o:.4f}", f"{d:.4f}",
                         f"{w_raw:.6f}", f"{w_n:.6f}"])

    print(f"\nWrote {out_path}")
    print(f"Weight range: [{min(norm_w):.3f}, {max(norm_w):.3f}]")
    print(f"Mean weight: {sum(norm_w) / len(norm_w):.4f}  (should be 1.0)")


if __name__ == "__main__":
    main()
