"""Figure 8: Canada-wide fire probability vs observed fire, three periods.

Three rows, one forecast issue date each. Left column = predicted ignition
probability at 30-day lead (the three lead-day maps are near-identical, so a
single representative lead is shown); right column = observed fires (NBAC+NFDB)
over the corresponding verification window. This replaces the earlier
single-date / three-lead layout, whose three lead panels were visually
indistinguishable.

All rasters EPSG:3978 (Canada Lambert). Province boundaries from Natural Earth
admin_1, reprojected on the fly.

Usage:
    python3 scripts/plot_fig3_canada_map.py
"""

from __future__ import annotations

import os
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap

TRANS_RED = LinearSegmentedColormap.from_list(
    "trans_red",
    [
        (0.00, (1.00, 1.00, 1.00, 0.00)),
        (0.05, (1.00, 0.94, 0.70, 0.35)),
        (0.30, (0.99, 0.68, 0.30, 0.75)),
        (0.65, (0.86, 0.20, 0.10, 0.92)),
        (1.00, (0.45, 0.00, 0.05, 1.00)),
    ],
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_paper_style import apply_style  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPS_DIR = os.path.join(ROOT, "results", "maps")
OUT_DIR = os.path.join(ROOT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# (prob tif, actual tif, predicted-title, observed-title)
CASES = [
    ("fcnhead_prob_20220815_lead30d.tif", "fire_actual_20220829_20220929.tif",
     "Forecast 2022-08-15,  lead 30 d  →  2022-09-14", "Observed fire 2022-08-29 to 09-29"),
    ("fcnhead_prob_20230515_lead30d.tif", "fire_actual_20230529_20230629.tif",
     "Forecast 2023-05-15,  lead 30 d  →  2023-06-14", "Observed fire 2023-05-29 to 06-29"),
    ("fcnhead_prob_20230815_lead30d.tif", "fire_actual_20230829_20230929.tif",
     "Forecast 2023-08-15,  lead 30 d  →  2023-09-14", "Observed fire 2023-08-29 to 09-29"),
]
SHP = os.path.join(MAPS_DIR, "ne_50m_admin_1",
                   "ne_50m_admin_1_states_provinces.shp")
RASTER_CRS = "EPSG:3978"


def _read(path):
    with rasterio.open(path) as src:
        arr = src.read(1)
        bounds = src.bounds
        nodata = src.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr.astype(np.float32))
    return arr, bounds


def _load_canada_provinces():
    gdf = gpd.read_file(SHP)
    return gdf[gdf["admin"] == "Canada"].to_crs(RASTER_CRS)


def _draw_panel(ax, arr, bounds, provinces, *, vmin, vmax, cmap,
                title, is_binary=False, view_bounds=None):
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    provinces.plot(ax=ax, facecolor="#ECECEC", edgecolor="#2C3E50",
                   linewidth=0.5, alpha=1.0, zorder=1)
    mt = 0.5 if is_binary else max(vmax * 0.02, 1e-4)
    masked = np.ma.masked_where(~np.isfinite(arr) | (arr < mt), arr)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=extent, origin="upper", aspect="equal",
                   interpolation="nearest", zorder=2)
    try:
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path as MplPath
        verts, codes = [], []
        for geom in provinces.geometry:
            polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
            for poly in polys:
                ext = np.asarray(poly.exterior.coords)
                verts.extend(ext); codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(ext) - 1))
                for interior in poly.interiors:
                    int_ = np.asarray(interior.coords)
                    verts.extend(int_); codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(int_) - 1))
        clip_patch = PathPatch(MplPath(verts, codes), transform=ax.transData,
                               facecolor="none", edgecolor="none")
        ax.add_patch(clip_patch)
        im.set_clip_path(clip_patch)
    except Exception as e:
        print(f"  [warn] clip failed: {e}")
    provinces.boundary.plot(ax=ax, color="#2C3E50", linewidth=0.5,
                            alpha=0.85, zorder=3)
    if view_bounds is not None:
        ax.set_xlim(view_bounds[0], view_bounds[2])
        ax.set_ylim(view_bounds[1], view_bounds[3])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9, pad=3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor("white")


def main():
    apply_style()
    provinces = _load_canada_provinces()

    prob_arrs, vmax = [], 0.0
    for pf, _, _, _ in CASES:
        arr, bounds = _read(os.path.join(MAPS_DIR, pf))
        fin = arr[np.isfinite(arr)]
        if fin.size:
            vmax = max(vmax, float(np.percentile(fin, 99.5)))
        prob_arrs.append((arr, bounds))

    cb = provinces.total_bounds
    pad_x = (cb[2] - cb[0]) * 0.02
    pad_y = (cb[3] - cb[1]) * 0.02
    view = (cb[0] - pad_x, cb[1] - pad_y, cb[2] + pad_x, cb[3] + pad_y)

    fig, axes = plt.subplots(3, 2, figsize=(10.5, 13.2),
                             gridspec_kw={"hspace": 0.10, "wspace": 0.04})
    for i, (pf, af, ptitle, atitle) in enumerate(CASES):
        parr, pbounds = prob_arrs[i]
        _draw_panel(axes[i, 0], parr, pbounds, provinces, vmin=0, vmax=vmax,
                    cmap=TRANS_RED, title=ptitle, view_bounds=view)
        aarr, abounds = _read(os.path.join(MAPS_DIR, af))
        _draw_panel(axes[i, 1], aarr, abounds, provinces, vmin=0, vmax=1,
                    cmap=TRANS_RED, title=atitle, is_binary=True, view_bounds=view)

    cbar_ax = fig.add_axes([0.93, 0.35, 0.013, 0.30])
    sm = plt.cm.ScalarMappable(cmap=TRANS_RED, norm=plt.Normalize(vmin=0, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Predicted fire probability", fontsize=9)

    fig.suptitle(
        "Patch Transformer (conv-stem + FCN head, SOTA) forecasts versus observed fire, three periods\n"
        "Left: predicted probability at 30-day lead.   Right: observed fires (NBAC + NFDB).",
        fontsize=12, y=0.995,
    )
    pdf = os.path.join(OUT_DIR, "fig3_canada_map_multidate.pdf")
    png = os.path.join(OUT_DIR, "fig3_canada_map_multidate.png")
    fig.savefig(pdf, dpi=200, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {pdf}")
    print(f"  wrote {png}")


if __name__ == "__main__":
    main()
