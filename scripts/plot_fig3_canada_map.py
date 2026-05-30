"""Figure 3: Canada-wide fire probability map vs actual fires.

2x2 panel for issue date 2023-05-15:
    [pred lead 14d → 2023-05-29] | [pred lead 30d → 2023-06-14]
    [pred lead 45d → 2023-06-29] | [actual fires 2023-05-29 to 2023-06-29]

All rasters in EPSG:3978 (Canada Lambert). Province boundaries from
Natural Earth admin_1, reprojected on the fly.

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


# Custom colormap: low values fully transparent, ramps up to dark red.
# Solves the "black-pixel salt-and-pepper" look of inferno on a light map.
TRANS_RED = LinearSegmentedColormap.from_list(
    "trans_red",
    [
        (0.00, (1.00, 1.00, 1.00, 0.00)),   # fully transparent
        (0.05, (1.00, 0.94, 0.70, 0.35)),   # pale yellow
        (0.30, (0.99, 0.68, 0.30, 0.75)),   # orange
        (0.65, (0.86, 0.20, 0.10, 0.92)),   # red
        (1.00, (0.45, 0.00, 0.05, 1.00)),   # dark red
    ],
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_paper_style import apply_style  # noqa: E402


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPS_DIR = os.path.join(ROOT, "results", "maps")
OUT_DIR = os.path.join(ROOT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

PROB_TIFS = [
    ("sota_prob_20230515_lead14d.tif", "Lead 14 d  →  2023-05-29"),
    ("sota_prob_20230515_lead30d.tif", "Lead 30 d  →  2023-06-14"),
    ("sota_prob_20230515_lead45d.tif", "Lead 45 d  →  2023-06-29"),
]
ACTUAL_TIF = "fire_actual_20230529_20230629.tif"
SHP = os.path.join(MAPS_DIR, "ne_50m_admin_1",
                   "ne_50m_admin_1_states_provinces.shp")
RASTER_CRS = "EPSG:3978"


def _read(path: str):
    with rasterio.open(path) as src:
        arr = src.read(1)
        bounds = src.bounds
        nodata = src.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr.astype(np.float32))
    return arr, bounds


def _load_canada_provinces():
    """Read Natural Earth admin_1 → keep Canada (+ optional US AK/north
    states for visual context) → reproject to EPSG:3978."""
    gdf = gpd.read_file(SHP)
    canada = gdf[gdf["admin"] == "Canada"].to_crs(RASTER_CRS)
    return canada


def _draw_panel(ax, arr, bounds, provinces, *, vmin, vmax, cmap,
                title, is_binary=False, mask_threshold=None,
                view_bounds=None):
    """White-background map + provinces fill + raster overlay clipped
    to Canada. `view_bounds` = (minx, miny, maxx, maxy) for tight crop."""
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)

    # 1) base map fill
    provinces.plot(ax=ax, facecolor="#ECECEC", edgecolor="#2C3E50",
                   linewidth=0.5, alpha=1.0, zorder=1)

    # 2) raster overlay, mask near-zero values
    if is_binary:
        mt = 0.5
    elif mask_threshold is not None:
        mt = mask_threshold
    else:
        mt = max(vmax * 0.02, 1e-4)
    masked = np.ma.masked_where(~np.isfinite(arr) | (arr < mt), arr)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=extent, origin="upper", aspect="equal",
                   interpolation="nearest", zorder=2)

    # 3) clip the raster to the union of all Canada provinces
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

    # 4) re-draw province outlines on TOP of raster so borders stay visible
    provinces.boundary.plot(ax=ax, color="#2C3E50", linewidth=0.5,
                            alpha=0.85, zorder=3)

    if view_bounds is not None:
        ax.set_xlim(view_bounds[0], view_bounds[2])
        ax.set_ylim(view_bounds[1], view_bounds[3])
    else:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, pad=4)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor("white")


def main():
    apply_style()
    provinces = _load_canada_provinces()

    # ---- read everything ----
    panels = []
    vmax = 0.0
    for fname, ttl in PROB_TIFS:
        arr, bounds = _read(os.path.join(MAPS_DIR, fname))
        # robust upper limit for color scale (99.5th percentile, ignoring nan)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            vmax = max(vmax, float(np.percentile(finite, 99.5)))
        panels.append((arr, bounds, ttl))
    actual_arr, actual_bounds = _read(os.path.join(MAPS_DIR, ACTUAL_TIF))

    # Tight Canada bbox in EPSG:3978 from province polygons
    cb = provinces.total_bounds   # (minx, miny, maxx, maxy)
    # Add 3% padding
    pad_x = (cb[2] - cb[0]) * 0.02
    pad_y = (cb[3] - cb[1]) * 0.02
    view = (cb[0] - pad_x, cb[1] - pad_y, cb[2] + pad_x, cb[3] + pad_y)

    # ---- figure layout ----
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.5),
                             gridspec_kw={"hspace": 0.12, "wspace": 0.04})
    axes = axes.flatten()

    for ax, (arr, bounds, ttl) in zip(axes[:3], panels):
        _draw_panel(ax, arr, bounds, provinces,
                    vmin=0, vmax=vmax, cmap=TRANS_RED,
                    title=ttl, view_bounds=view)

    _draw_panel(axes[3], actual_arr, actual_bounds, provinces,
                vmin=0, vmax=1, cmap=TRANS_RED,
                title="Observed fires 2023-05-29 to 2023-06-29  (NBAC + NFDB)",
                is_binary=True, view_bounds=view)

    # shared colorbar for prob panels (3 panels, top half)
    cbar_ax = fig.add_axes([0.94, 0.56, 0.014, 0.30])
    sm = plt.cm.ScalarMappable(cmap=TRANS_RED,
                               norm=plt.Normalize(vmin=0, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Predicted fire probability", fontsize=9)

    fig.suptitle(
        "Forecast issued 2023-05-15  —  Patch Transformer (V3 SOTA, 8.5 M params)",
        fontsize=12, y=0.96,
    )
    pdf = os.path.join(OUT_DIR, "fig3_canada_map_20230515.pdf")
    png = os.path.join(OUT_DIR, "fig3_canada_map_20230515.png")
    fig.savefig(pdf, dpi=200, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {pdf}")
    print(f"  wrote {png}")


if __name__ == "__main__":
    main()
