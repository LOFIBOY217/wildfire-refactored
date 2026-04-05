"""Debug: check if MODIS HDF4 metadata contains UpperLeftPointMtrs."""
import glob
import re
from pathlib import Path
from pyhdf.SD import SD, SDC

files = sorted(glob.glob("data/ndvi_raw/2023/*.hdf"))
print(f"Found {len(files)} HDF4 files")

if not files:
    exit(1)

hdf_path = files[len(files) // 2]
print(f"Testing: {hdf_path}")

hdf = SD(str(hdf_path), SDC.READ)
attrs = hdf.attributes()
meta = attrs.get("StructMetadata.0", "")
hdf.end()

print(f"StructMetadata.0 length: {len(meta)}")

if not meta:
    print("NO StructMetadata.0 found!")
    print(f"Available attributes: {list(attrs.keys())[:10]}")
    exit(1)

ul = re.search(r"UpperLeftPointMtrs=\(([-\d.]+),([-\d.]+)\)", meta)
lr = re.search(r"LowerRightMtrs=\(([-\d.]+),([-\d.]+)\)", meta)

if ul and lr:
    left, top = float(ul.group(1)), float(ul.group(2))
    right, bottom = float(lr.group(1)), float(lr.group(2))
    print(f"Bounds: left={left} top={top} right={right} bottom={bottom}")
    print("METADATA PARSING OK")
else:
    print("REGEX FAILED")
    # Show relevant section
    for line in meta.split("\n"):
        if "LeftPoint" in line or "RightPoint" in line or "Mtrs" in line:
            print(f"  {line.strip()}")
    if not any("LeftPoint" in l for l in meta.split("\n")):
        print("No LeftPoint/RightPoint in metadata at all")
        print(f"First 300 chars: {meta[:300]}")
