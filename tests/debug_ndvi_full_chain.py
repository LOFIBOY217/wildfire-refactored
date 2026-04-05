"""Full chain debug: HDF4 raw → read SDS → metadata bounds → reproject → FWI grid.
Tests each step independently to find where the data becomes NaN."""

import sys
import numpy as np

# Step 1: Can we find and open an HDF4 file?
print("=== STEP 1: Find HDF4 file ===")
import glob
files = sorted(glob.glob("data/ndvi_raw/2023/*.hdf"))
print(f"Files: {len(files)}")
if not files:
    print("FAIL: no files")
    sys.exit(1)
hdf_path = files[len(files) // 2]
print(f"Using: {hdf_path}")

# Step 2: Read raw SDS data
print("\n=== STEP 2: Read raw SDS ===")
from pyhdf.SD import SD, SDC
hdf = SD(str(hdf_path), SDC.READ)
ndvi_raw = hdf.select("1 km 16 days NDVI").get().astype(np.float32)
qa_raw = hdf.select("1 km 16 days pixel reliability").get().astype(np.float32)
attrs = hdf.attributes()
meta = attrs.get("StructMetadata.0", "")
hdf.end()

print(f"NDVI shape: {ndvi_raw.shape}")
print(f"NDVI range: [{ndvi_raw.min()}, {ndvi_raw.max()}]")
print(f"NDVI non-fill: {(ndvi_raw != -3000).sum()} / {ndvi_raw.size}")
print(f"QA shape: {qa_raw.shape}")
print(f"QA values: {np.unique(qa_raw)[:10]}")

# Step 3: Apply QA + scale
print("\n=== STEP 3: QA mask + scale ===")
fill_val = -3000  # MODIS NDVI fill
bad = (qa_raw > 1) | (ndvi_raw == fill_val)
ndvi_raw[bad] = np.nan
ndvi_scaled = np.where(np.isfinite(ndvi_raw), ndvi_raw * 0.0001, np.nan)
ndvi_scaled = np.clip(ndvi_scaled, -1.0, 1.0)
valid = np.isfinite(ndvi_scaled)
print(f"Valid pixels after QA: {valid.sum()} / {ndvi_scaled.size} ({100*valid.mean():.1f}%)")
if valid.sum() > 0:
    print(f"NDVI range (valid): [{ndvi_scaled[valid].min():.3f}, {ndvi_scaled[valid].max():.3f}]")
else:
    print("FAIL: zero valid pixels after QA")

# Step 4: Parse metadata bounds
print("\n=== STEP 4: Metadata bounds ===")
import re
ul = re.search(r"UpperLeftPointMtrs=\(([-\d.]+),([-\d.]+)\)", meta)
lr = re.search(r"LowerRightMtrs=\(([-\d.]+),([-\d.]+)\)", meta)
if ul and lr:
    left, top = float(ul.group(1)), float(ul.group(2))
    right, bottom = float(lr.group(1)), float(lr.group(2))
    print(f"Sinusoidal bounds: left={left} top={top} right={right} bottom={bottom}")
else:
    print("FAIL: cannot parse bounds")
    sys.exit(1)

# Step 5: Build source transform
print("\n=== STEP 5: Source transform ===")
from rasterio.transform import from_bounds
from rasterio.crs import CRS
h, w = ndvi_scaled.shape
src_tf = from_bounds(left, bottom, right, top, w, h)
print(f"Source transform: {src_tf}")

# Step 6: Define sinusoidal CRS
print("\n=== STEP 6: Sinusoidal CRS ===")
try:
    sin_crs = CRS.from_proj4("+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs")
    print(f"Sinusoidal CRS: {sin_crs}")
    print(f"CRS valid: {sin_crs.is_valid}")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

# Step 7: Define FWI target
print("\n=== STEP 7: FWI target CRS ===")
FWI_CRS = "EPSG:3978"
FWI_WIDTH = 2709
FWI_HEIGHT = 2281
FWI_BOUNDS = (-2378164.0, -707617.0, 3039835.0, 3854382.0)
dst_tf = from_bounds(*FWI_BOUNDS, FWI_WIDTH, FWI_HEIGHT)
try:
    dst_crs = CRS.from_string(FWI_CRS)
    print(f"FWI CRS: {dst_crs}")
    print(f"CRS valid: {dst_crs.is_valid}")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

# Step 8: Reproject
print("\n=== STEP 8: Reproject ===")
from rasterio.warp import reproject, Resampling
dst_data = np.full((FWI_HEIGHT, FWI_WIDTH), np.nan, dtype=np.float32)
try:
    reproject(
        source=ndvi_scaled,
        destination=dst_data,
        src_transform=src_tf,
        src_crs=sin_crs,
        dst_transform=dst_tf,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    valid_dst = np.isfinite(dst_data)
    print(f"Output valid: {valid_dst.sum()} / {dst_data.size} ({100*valid_dst.mean():.1f}%)")
    if valid_dst.sum() > 0:
        print(f"Output range: [{dst_data[valid_dst].min():.3f}, {dst_data[valid_dst].max():.3f}]")
        print("SUCCESS: reproject produced valid data")
    else:
        print("FAIL: reproject produced all NaN")
        # Try with EPSG:4326 as test
        print("\n  Trying WGS84 as sanity check...")
        dst_wgs = np.full((180, 360), np.nan, dtype=np.float32)
        dst_wgs_tf = from_bounds(-180, -90, 180, 90, 360, 180)
        reproject(
            source=ndvi_scaled,
            destination=dst_wgs,
            src_transform=src_tf,
            src_crs=sin_crs,
            dst_transform=dst_wgs_tf,
            dst_crs=CRS.from_epsg(4326),
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        v = np.isfinite(dst_wgs).sum()
        print(f"  WGS84 valid: {v} / {dst_wgs.size}")
except Exception as e:
    print(f"FAIL: reproject error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== DONE ===")
