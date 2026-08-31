#!/usr/bin/env python3
"""
Unified inference entry point.
==============================

One command turns a trained V3 checkpoint + one or more forecast issue dates
into the full deliverable set, without any training code:

    1. Fire-probability GeoTIFFs  (delegates to forecast_v3_to_tif.forecast)
    2. A top-K ranking CSV of the highest-risk pixels (lon/lat + prob)
    3. A Canada-wide probability map PNG (optionally paired with observed fire)
    4. Verification metrics JSON (only when an evaluation window is given)

Steps 3-4 use ONE chosen lead day (--lead); step 1 still writes every lead
(14..45) to disk. Nothing here re-implements prediction, observed-fire
rasterization, or metric math -- it orchestrates existing modules:

    forecast_v3_to_tif.forecast          -> probability GeoTIFFs
    data_ops.observed_fire.build_observed_window -> observed truth raster
    evaluation.metrics.compute_all_metrics       -> Lift / Lift@30km / BSS ...

Two modes, selected by whether --eval_window is passed:

    * Forecast-only (no truth): probability tif + map + ranking.
      Use this for genuine future dates -- Lift is undefined without truth.
    * Verification (--eval_window N): also builds the observed-fire raster over
      [target, target + N days], writes a side-by-side map, and scores metrics.

Truth source is EXPLICIT (--truth_source), never inferred from the year:
    nbac_nfdb (default, = training target) | ciffc (current season) | cwfis.

Metrics are computed over the full EPSG:3978 grid the probability tif lives on
(same convention as the Canada-map figures); no confidence intervals are
produced, per project figure policy.

Examples
--------
Forecast-only (future date):
    python -m src.forecasting.run_inference \
        --ckpt checkpoints/v3_9ch_enc21_12y_2014/best_model.pt \
        --s2s_cache data/s2s_processed/s2s_decoder_cache.dat \
        --issue_dates 2026-05-15 --lead 30 \
        --out_dir outputs/inf_20260515

Verification against NBAC+NFDB (history):
    python -m src.forecasting.run_inference \
        --ckpt checkpoints/v3_9ch_enc21_12y_2014/best_model.pt \
        --s2s_cache data/s2s_processed/s2s_decoder_cache.dat \
        --issue_dates 2023-08-15 --lead 30 --eval_window 14 \
        --out_dir outputs/inf_20230815

Verification against CIFFC (current season):
    python -m src.forecasting.run_inference \
        --ckpt ... --s2s_cache ... \
        --issue_dates 2026-05-15 --lead 30 --eval_window 14 \
        --truth_source ciffc --out_dir outputs/inf_20260515_ciffc
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from datetime import date, timedelta
from types import SimpleNamespace
from typing import Optional

import numpy as np
import rasterio

from src.config import load_config
from src.forecasting.forecast_v3_to_tif import forecast
from src.data_ops.observed_fire import build_observed_window
from src.evaluation.metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def _find_prob_tif(out_dir: str, issue: date, lead: int) -> str:
    """Locate the probability tif forecast() wrote for (issue, lead)."""
    day = issue.strftime("%Y%m%d")
    patt = os.path.join(out_dir, day, f"fire_prob_lead{lead:02d}d_*.tif")
    hits = sorted(glob.glob(patt))
    if not hits:
        raise FileNotFoundError(
            f"No probability tif matching {patt}. Did forecast() run for "
            f"issue={day} lead={lead}? Valid leads are 14..45.")
    return hits[0]


def _read_raster(path: str):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
        nodata = src.nodata
    return arr, profile, transform, crs, bounds, nodata


# ---------------------------------------------------------------------------
# Step 2: ranking CSV
# ---------------------------------------------------------------------------
def write_ranking_csv(prob: np.ndarray, transform, crs, topk: int,
                      out_csv: str, nodata: Optional[float] = None) -> int:
    """Write the top-K highest-probability pixels as ranked rows.

    Columns: rank, row, col, x_3978, y_3978, lon, lat, prob.
    Pixel coordinates are cell centres. Returns the number of rows written.
    """
    flat = prob.ravel().astype(np.float64)
    valid = np.isfinite(flat)
    if nodata is not None:
        valid &= (flat != nodata)
    flat = np.where(valid, flat, -np.inf)

    n_valid = int(valid.sum())
    k = min(topk, n_valid)
    if k <= 0:
        raise ValueError("No valid pixels to rank.")

    # top-k indices, then sort those descending by prob
    part = np.argpartition(flat, -k)[-k:]
    order = part[np.argsort(flat[part])[::-1]]

    rows, cols = np.unravel_index(order, prob.shape)
    xs, ys = rasterio.transform.xy(transform, rows.tolist(), cols.tolist())
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    lons = lats = None
    try:
        from pyproj import Transformer
        tr = Transformer.from_crs(crs.to_wkt(), "EPSG:4326", always_xy=True)
        lons, lats = tr.transform(xs, ys)
    except Exception as e:  # pyproj missing or CRS unusable -> still emit x/y
        print(f"  [ranking] lon/lat unavailable ({e}); writing 3978 coords only")

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "row", "col", "x_3978", "y_3978",
                    "lon", "lat", "prob"])
        for i in range(k):
            lon = "" if lons is None else f"{lons[i]:.5f}"
            lat = "" if lats is None else f"{lats[i]:.5f}"
            w.writerow([i + 1, int(rows[i]), int(cols[i]),
                        f"{xs[i]:.1f}", f"{ys[i]:.1f}", lon, lat,
                        f"{float(prob[rows[i], cols[i]]):.6f}"])
    print(f"  [ranking] wrote top-{k} pixels -> {out_csv}")
    return k


# ---------------------------------------------------------------------------
# Step 3: Canada map
# ---------------------------------------------------------------------------
# Sequential prob colormap, matching the Canada-map figures (fig3). This is a
# heatmap ramp, not a per-model hue, so it is defined locally rather than
# imported from the paper-figure style module.
_PROB_CMAP = [
    (0.00, (1.00, 1.00, 1.00, 0.00)),
    (0.05, (1.00, 0.94, 0.70, 0.35)),
    (0.30, (0.99, 0.68, 0.30, 0.75)),
    (0.65, (0.86, 0.20, 0.10, 0.92)),
    (1.00, (0.45, 0.00, 0.05, 1.00)),
]


def _load_provinces(shp_path: Optional[str], crs):
    if not shp_path or not os.path.exists(shp_path):
        return None
    try:
        import geopandas as gpd
        gdf = gpd.read_file(shp_path)
        if "admin" in gdf.columns:
            gdf = gdf[gdf["admin"] == "Canada"]
        return gdf.to_crs(crs)
    except Exception as e:
        print(f"  [map] province overlay skipped ({e})")
        return None


def _draw_panel(ax, arr, bounds, provinces, *, vmin, vmax, cmap, title,
                binary=False):
    import numpy as _np
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    show = _np.where(_np.isfinite(arr), arr, _np.nan)
    if binary:
        show = _np.where(show > 0, 1.0, _np.nan)
    ax.imshow(show, extent=extent, origin="upper", cmap=cmap,
              vmin=vmin, vmax=vmax, interpolation="nearest")
    if provinces is not None:
        provinces.boundary.plot(ax=ax, linewidth=0.4, edgecolor="#555555")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def render_map(prob, prob_bounds, prob_crs, out_png, *, title_pred,
               provinces_shp=None, observed=None, obs_bounds=None,
               title_obs=None):
    """Single-panel probability map, or two-panel (predicted | observed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    prob_cmap = LinearSegmentedColormap.from_list("prob_red", _PROB_CMAP)
    obs_cmap = LinearSegmentedColormap.from_list(
        "obs_red", [(0.0, (0.6, 0.0, 0.05, 1.0)), (1.0, (0.6, 0.0, 0.05, 1.0))])

    provinces = _load_provinces(provinces_shp, prob_crs)
    vmax = float(np.nanpercentile(prob[np.isfinite(prob)], 99.5)) or 1.0

    two = observed is not None
    fig, axes = plt.subplots(1, 2 if two else 1,
                             figsize=(12 if two else 6.5, 6))
    ax_list = axes if two else [axes]

    _draw_panel(ax_list[0], prob, prob_bounds, provinces,
                vmin=0.0, vmax=vmax, cmap=prob_cmap, title=title_pred)
    if two:
        _draw_panel(ax_list[1], observed, obs_bounds or prob_bounds, provinces,
                    vmin=0.0, vmax=1.0, cmap=obs_cmap,
                    title=title_obs or "Observed fire", binary=True)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [map] wrote {out_png}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _resolve_truth_paths(cfg: dict, args) -> dict:
    paths = cfg.get("paths", {})
    return {
        "nbac_path": args.nbac_path,
        "nfdb_path": args.nfdb_path,
        "hotspot_csv": args.hotspot_csv or paths.get("hotspot_csv"),
        "ciffc_path": args.ciffc_path or paths.get("ciffc_csv"),
    }


def run_inference(args) -> None:
    cfg = load_config(args.config)
    issue_dates = [_parse_date(s) for s in args.issue_dates]

    # ---- Step 1: probability GeoTIFFs (delegate, all leads) -------------
    print("=== Step 1: forecast -> probability GeoTIFFs ===")
    fc_args = SimpleNamespace(
        config=args.config, ckpt=args.ckpt, s2s_cache=args.s2s_cache,
        issue_dates=args.issue_dates, out_dir=args.out_dir,
        pred_batch_size=args.pred_batch_size)
    forecast(fc_args)

    truth_paths = _resolve_truth_paths(cfg, args)

    for issue in issue_dates:
        tag = issue.strftime("%Y%m%d")
        target = issue + timedelta(days=args.lead)
        lead_str = f"lead{args.lead:02d}d"
        print(f"\n=== Post-processing issue={tag} {lead_str} "
              f"target={target.isoformat()} ===")

        try:
            prob_tif = _find_prob_tif(args.out_dir, issue, args.lead)
        except FileNotFoundError as e:
            # forecast() legitimately skips issue dates outside the S2S cache
            # (see its "[skip] issue_date ... not in S2S cache" path); don't
            # crash the whole run -- warn and move to the next date.
            print(f"  [skip] no probability tif for {tag} lead {args.lead}; "
                  f"forecast likely skipped this date. ({e})")
            continue
        prob, profile, transform, crs, bounds, nodata = _read_raster(prob_tif)

        # ---- Step 2: ranking CSV ---------------------------------------
        if not args.skip_ranking:
            out_csv = os.path.join(args.out_dir,
                                   f"ranking_{tag}_{lead_str}.csv")
            write_ranking_csv(prob, transform, crs, args.topk, out_csv, nodata)

        # ---- Step 4 (optional): observed truth + metrics ---------------
        observed = None
        obs_bounds = None
        if args.eval_window is not None:
            win_end = target + timedelta(days=args.eval_window)
            obs, prov = build_observed_window(
                target.isoformat(), win_end.isoformat(),
                source=args.truth_source, profile=profile,
                dilate_radius=args.dilate_radius, **truth_paths)
            observed = obs.astype(np.float32)
            obs_bounds = bounds

            # write observed raster
            out_prof = profile.copy()
            out_prof.update(dtype="uint8", count=1, nodata=0)
            actual_tif = os.path.join(
                args.out_dir, tag,
                f"fire_actual_{target.strftime('%Y%m%d')}_"
                f"{win_end.strftime('%Y%m%d')}.tif")
            with rasterio.open(actual_tif, "w", **out_prof) as dst:
                dst.write(obs.astype(np.uint8), 1)
            print(f"  [truth] wrote {actual_tif}")

            if not args.skip_metrics:
                m = compute_all_metrics(
                    prob, observed, k_values=(args.topk,), coarsen_factor=15)
                m = {k: (float(v) if isinstance(v, (np.floating, np.integer))
                         else v) for k, v in m.items()}
                m["_meta"] = {
                    "issue_date": issue.isoformat(),
                    "lead": args.lead,
                    "target_date": target.isoformat(),
                    "eval_window": args.eval_window,
                    "truth_source": args.truth_source,
                    "topk": args.topk,
                    "prob_tif": os.path.relpath(prob_tif, args.out_dir),
                    **prov,
                }
                out_json = os.path.join(args.out_dir,
                                        f"metrics_{tag}_{lead_str}.json")
                with open(out_json, "w") as f:
                    json.dump(m, f, indent=2, default=str)
                print(f"  [metrics] Lift@{args.topk}={m.get('lift_k', 0):.2f}  "
                      f"Lift@30km={m.get('lift_coarse', 0):.2f}  "
                      f"BSS={m.get('bss', 0):.3f} -> {out_json}")

        # ---- Step 3: map ------------------------------------------------
        if not args.skip_map:
            out_png = os.path.join(args.out_dir, f"map_{tag}_{lead_str}.png")
            title_pred = (f"Forecast {issue.isoformat()}, lead {args.lead}d "
                          f"-> {target.isoformat()}")
            title_obs = None
            if observed is not None:
                win_end = target + timedelta(days=args.eval_window)
                title_obs = (f"Observed fire {target.isoformat()} to "
                             f"{win_end.isoformat()} ({args.truth_source})")
            render_map(prob, bounds, crs, out_png, title_pred=title_pred,
                       provinces_shp=args.provinces_shp,
                       observed=observed, obs_bounds=obs_bounds,
                       title_obs=title_obs)

    print(f"\n[run_inference] DONE. Output dir: {args.out_dir}")


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Unified inference: probability tif + ranking + map + "
                    "(optional) verification metrics.")
    # forecast passthrough
    ap.add_argument("--config", default="configs/paths_narval.yaml")
    ap.add_argument("--ckpt", required=True, help="V3 checkpoint (best_model.pt)")
    ap.add_argument("--s2s_cache", default=None,
                    help="s2s_decoder_cache.dat (required if decoder=s2s_legacy)")
    ap.add_argument("--issue_dates", nargs="+", required=True,
                    help="One or more YYYY-MM-DD issue dates")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pred_batch_size", type=int, default=512)
    # product selection
    ap.add_argument("--lead", type=int, default=30,
                    help="Lead day for map/ranking/metrics (14..45; default 30)")
    ap.add_argument("--topk", type=int, default=5000,
                    help="K for the ranking CSV and prec@K / Lift@K")
    # verification
    ap.add_argument("--eval_window", type=int, default=None,
                    help="If set, build observed truth over [target, target+N] "
                         "days and compute metrics; omit for forecast-only")
    ap.add_argument("--truth_source", default="nbac_nfdb",
                    choices=("nbac_nfdb", "ciffc", "cwfis"),
                    help="Observed-fire source (explicit; not year-inferred)")
    ap.add_argument("--dilate_radius", type=int, default=14)
    # truth paths (config supplies ciffc/hotspot; nbac/nfdb default like build_fire_labels)
    ap.add_argument("--nbac_path",
                    default="data/burn_scars_raw/NBAC_1972to2024_shp.zip")
    ap.add_argument("--nfdb_path", default="data/nfdb/NFDB_point.zip")
    ap.add_argument("--hotspot_csv", default=None, help="override config")
    ap.add_argument("--ciffc_path", default=None, help="override config")
    # map
    ap.add_argument("--provinces_shp",
                    default="results/maps/ne_50m_admin_1/"
                            "ne_50m_admin_1_states_provinces.shp",
                    help="Natural Earth admin_1 shp for province overlay "
                         "(skipped if missing)")
    # toggles
    ap.add_argument("--skip_map", action="store_true")
    ap.add_argument("--skip_ranking", action="store_true")
    ap.add_argument("--skip_metrics", action="store_true")
    return ap


def main():
    args = _build_arg_parser().parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
