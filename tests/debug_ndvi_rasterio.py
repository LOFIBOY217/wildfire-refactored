"""Test: can rasterio open MODIS HDF4 directly (bypassing pyhdf)?"""
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from rasterio.crs import CRS

files = sorted(glob.glob("data/ndvi_raw/2023/*.hdf"))
print(f"Files: {len(files)}")
hdf_path = files[len(files) // 2]
print(f"Testing: {hdf_path}")

# Method 1: rasterio subdataset
with rasterio.open(hdf_path) as src:
    print(f"\nMain dataset:")
    print(f"  subdatasets: {src.subdatasets[:5]}")

# Find NDVI subdataset
ndvi_ds = None
for sd in src.subdatasets:
    if "NDVI" in sd and "1 km" in sd:
        ndvi_ds = sd
        break
    # Also try without space
    if "NDVI" in sd:
        ndvi_ds = sd

if ndvi_ds is None:
    print("No NDVI subdataset found!")
    print(f"All subdatasets: {src.subdatasets}")
    exit(1)

print(f"\nOpening subdataset: {ndvi_ds}")
with rasterio.open(ndvi_ds) as sub:
    data = sub.read(1).astype(np.float32)
    print(f"  Shape: {data.shape}")
    print(f"  CRS: {sub.crs}")
    print(f"  Transform: {sub.transform}")
    print(f"  Range: [{data.min()}, {data.max()}]")

    # Scale + mask
    data[data <= -2000] = np.nan
    data = data * 0.0001
    valid = np.isfinite(data)
    print(f"  Valid: {valid.sum()}/{data.size} ({100*valid.mean():.1f}%)")

    # Reproject to FWI grid using rasterio's built-in CRS
    FWI_BOUNDS = (-2378164.0, -707617.0, 3039835.0, 3854382.0)
    dst_tf = from_bounds(*FWI_BOUNDS, 2709, 2281)
    dst_crs = CRS.from_epsg(3978)
    dst = np.full((2281, 2709), np.nan, dtype=np.float32)

    reproject(
        source=data,
        destination=dst,
        src_transform=sub.transform,
        src_crs=sub.crs,        # rasterio reads CRS from HDF4 directly!
        dst_transform=dst_tf,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    valid_dst = np.isfinite(dst)
    print(f"\n  Reprojected valid: {valid_dst.sum()}/{dst.size} ({100*valid_dst.mean():.1f}%)")
    if valid_dst.sum() > 0:
        print(f"  Range: [{dst[valid_dst].min():.3f}, {dst[valid_dst].max():.3f}]")
        print("  SUCCESS!")
    else:
        print("  FAILED: all NaN")
