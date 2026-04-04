"""Quick test: which NetCDF reader works for GLM granules on this system?"""
import s3fs
import tempfile
import numpy as np

s3 = s3fs.S3FileSystem(anon=True)
files = s3.glob("noaa-goes16/GLM-L2-LCFA/2023/182/00/*.nc")
print(f"Files in hour 00: {len(files)}")

with s3.open(files[0], "rb") as f:
    raw = f.read()
print(f"Raw size: {len(raw)} bytes")

# Method 1: xarray + scipy
try:
    import xarray as xr
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=True) as tmp:
        tmp.write(raw)
        tmp.flush()
        ds = xr.open_dataset(tmp.name, engine="scipy")
        if "flash_lat" in ds:
            print(f"[xarray+scipy] OK: {len(ds.flash_lat)} flashes")
        ds.close()
except Exception as e:
    print(f"[xarray+scipy] FAIL: {e}")

# Method 2: h5py
try:
    import h5py
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=True) as tmp:
        tmp.write(raw)
        tmp.flush()
        with h5py.File(tmp.name, "r") as f:
            if "flash_lat" in f:
                print(f"[h5py] OK: {len(f['flash_lat'][:])} flashes")
except Exception as e:
    print(f"[h5py] FAIL: {e}")

# Method 3: netCDF4 (known to segfault on some systems)
try:
    import netCDF4 as nc4
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=True) as tmp:
        tmp.write(raw)
        tmp.flush()
        with nc4.Dataset(tmp.name, "r") as ds:
            if "flash_lat" in ds.variables:
                print(f"[netCDF4] OK: {len(ds['flash_lat'][:].data)} flashes")
except Exception as e:
    print(f"[netCDF4] FAIL: {e}")

print("Done")
