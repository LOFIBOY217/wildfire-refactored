"""
Reproject ECMWF S2S Fire Danger NetCDF (1° lat/lon, all leadtimes per
file) to per-issue, per-leadday GeoTIFFs aligned to the EPSG:3978 2 km
grid.

Output naming convention:
  data/ecmwf_s2s_fire_epsg3978/{var}/issue_YYYYMM/lead_DDD.tif

Where:
  YYYYMM = SEAS5 issue year-month
  DDD    = lead day index (001..215)

Each file is a single-band GeoTIFF on the canonical EPSG:3978 grid
(2 709 × 2 281 pixels at 2 km), so it can be loaded directly with
rasterio and compared to NBAC + NFDB labels.

Usage:
  python -m src.data_ops.processing.ecmwf_s2s_to_epsg3978 \\
      --input_dir data/ecmwf_s2s_fire \\
      --output_dir data/ecmwf_s2s_fire_epsg3978 \\
      --reference data/fwi_data/fwi_20250615.tif \\
      --variables fwinx
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np


def reproject_one_file(nc_path, ref_profile, ref_transform, ref_crs,
                       ref_h, ref_w, output_root, variable, overwrite=False):
    """
    Process one ECMWF S2S NetCDF file → per-leadday GeoTIFFs.

    Args:
        nc_path: Path to s2s_<var>_YYYYMM.nc
        ref_profile: rasterio profile from reference FWI tif
        output_root: data/ecmwf_s2s_fire_epsg3978
        variable: "fwinx" / "ffmc" / etc.
    """
    import xarray as xr
    import rasterio
    from rasterio.warp import reproject, Resampling

    # Auto-detect engine: cfgrib for .grib, netcdf4 for .nc
    if str(nc_path).endswith(".grib") or str(nc_path).endswith(".grb"):
        ds = xr.open_dataset(nc_path, engine="cfgrib",
                             backend_kwargs={"indexpath": ""})
    else:
        ds = xr.open_dataset(nc_path)

    # cems-fire-seasonal GRIB layout from EWDS:
    #   dims = (number=51 members, step=215 lead days, latitude, longitude)
    #   Take ensemble MEAN over `number` dim → (step, lat, lon).
    var_name = list(ds.data_vars)[0]   # 'fwinx' / 'ffmc' / etc.
    arr = ds[var_name]
    if "number" in arr.dims:
        arr = arr.mean(dim="number", skipna=True)   # ensemble mean
    if "forecast_reference_time" in arr.dims and arr.sizes["forecast_reference_time"] == 1:
        arr = arr.isel(forecast_reference_time=0)
    # After this: arr has dims (step, latitude, longitude)

    # Determine source lat/lon
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    # SEAS5 longitude is in [0, 360]; convert to [-180, 180] for reproject
    if lon.max() > 180:
        # roll the longitude axis so values become continuous in [-180, 180]
        lon_shifted = ((lon + 180) % 360) - 180
        # find the order that puts the data right side
        sort_idx = np.argsort(lon_shifted)
        lon = lon_shifted[sort_idx]
        arr = arr.isel({arr.dims[-1]: sort_idx})
    if lat.ndim != 1 or lon.ndim != 1:
        raise RuntimeError(f"Unexpected lat/lon shape: {lat.shape} / {lon.shape}")
    n_lat, n_lon = len(lat), len(lon)

    # Source affine (lat/lon, EPSG:4326)
    res_lat = abs(lat[1] - lat[0])
    res_lon = abs(lon[1] - lon[0])
    src_transform = rasterio.transform.from_origin(
        lon[0] - res_lon / 2, lat[0] + res_lat / 2, res_lon, res_lat
    )
    src_crs = "EPSG:4326"

    # Issue tag from filename: s2s_fwinx_202307.grib → 202307
    issue_tag = nc_path.stem.split("_")[-1]
    n_lead = arr.shape[0] if arr.ndim == 3 else 1
    arr_np = arr.values  # materialise once (small after ensemble-mean)

    issue_dir = output_root / variable / f"issue_{issue_tag}"
    issue_dir.mkdir(parents=True, exist_ok=True)

    n_done = 0
    for lead_idx in range(n_lead):
        outfile = issue_dir / f"lead_{lead_idx + 1:03d}.tif"
        if outfile.exists() and not overwrite:
            n_done += 1
            continue

        # SEAS5 ensemble mean for this lead day (already meaned over `number`)
        if arr_np.ndim == 3:
            src_data = arr_np[lead_idx].astype(np.float32)
        else:
            src_data = arr_np.astype(np.float32)

        # Replace NaN with sentinel
        src_data = np.nan_to_num(src_data, nan=-9999.0)

        dst_data = np.full((ref_h, ref_w), -9999.0, dtype=np.float32)
        reproject(
            source=src_data, destination=dst_data,
            src_transform=src_transform, src_crs=src_crs,
            dst_transform=ref_transform, dst_crs=ref_crs,
            resampling=Resampling.bilinear,
            src_nodata=-9999.0, dst_nodata=-9999.0,
        )

        out_profile = ref_profile.copy()
        out_profile.update(dtype=rasterio.float32, count=1, nodata=-9999.0,
                           compress="DEFLATE", tiled=True,
                           blockxsize=256, blockysize=256)
        with rasterio.open(outfile, "w", **out_profile) as dst:
            dst.write(dst_data, 1)
        n_done += 1

    ds.close()
    return n_done, n_lead


def main():
    ap = argparse.ArgumentParser(
        description="Reproject ECMWF S2S Fire Danger NetCDF to EPSG:3978 GeoTIFFs"
    )
    ap.add_argument("--input_dir", type=str, default="data/ecmwf_s2s_fire")
    ap.add_argument("--output_dir", type=str,
                    default="data/ecmwf_s2s_fire_epsg3978")
    ap.add_argument("--reference", type=str,
                    default="data/fwi_data/fwi_20250615.tif",
                    help="Reference FWI tif defining the EPSG:3978 grid")
    ap.add_argument("--variables", type=str, nargs="+", default=["fwinx"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    try:
        import rasterio
        import xarray  # noqa: F401
    except ImportError as e:
        print(f"[ERROR] missing dep: {e}. Run: pip install rasterio xarray netCDF4")
        sys.exit(1)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read reference grid
    with rasterio.open(args.reference) as ref:
        ref_profile = ref.profile
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_h, ref_w = ref.height, ref.width
    print(f"[REF] {args.reference}")
    print(f"      CRS={ref_crs}  shape=({ref_h},{ref_w})  transform={ref_transform}")

    # Process each variable
    for var in args.variables:
        var_input = input_dir / var
        # accept both .grib (preferred, default from EWDS) and legacy .nc
        nc_files = sorted(list(var_input.glob("s2s_*.grib")) +
                          list(var_input.glob("s2s_*.nc")))
        if not nc_files:
            print(f"[WARN] no NetCDF files in {var_input}")
            continue
        print(f"\n[VAR] {var}: {len(nc_files)} input files")
        t0 = time.time()
        for i, nc in enumerate(nc_files, 1):
            try:
                n_done, n_lead = reproject_one_file(
                    nc, ref_profile, ref_transform, ref_crs,
                    ref_h, ref_w, output_dir, var, args.overwrite
                )
                print(f"  [{i}/{len(nc_files)}] {nc.name}: "
                      f"{n_done}/{n_lead} leads written")
            except Exception as e:
                print(f"  [{i}/{len(nc_files)}] {nc.name}: ERROR {e}")
        elapsed = time.time() - t0
        print(f"[DONE] {var} in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
