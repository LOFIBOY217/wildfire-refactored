#!/bin/bash
#SBATCH --job-name=wf-resamp-uvc
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/jiaqi217/logs/resample_uvc_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/resample_uvc_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# Resample u10/v10/CAPE from ERA5 daily (WGS84) → FWI grid (EPSG:3978)
# Only 2018-2025 (needed for training), skips existing files.
# Source: data/era5_daily/{u10,v10,cape}_YYYYMMDD.tif (0.25° WGS84)
# Output: data/era5_{u10,v10,cape}/{var}_YYYYMMDD.tif (EPSG:3978 2709x2281)
# ----------------------------------------------------------------

set -euo pipefail

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}

[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1

source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored

echo "============================================="
echo "  Resample u10/v10/CAPE → FWI grid (2018-2025 only)"
echo "  Node: $(hostname)  Time: $(date)"
echo "============================================="

python3 -u -c "
import glob, os, re, numpy as np, rasterio, time
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling

FWI_CRS = 'EPSG:3978'
FWI_W, FWI_H = 2709, 2281
FWI_BOUNDS = (-2378164.0, -707617.0, 3039835.0, 3854382.0)
dst_tf = from_bounds(*FWI_BOUNDS, FWI_W, FWI_H)
dst_crs = CRS.from_string(FWI_CRS)

profile = dict(driver='GTiff', dtype='float32', width=FWI_W, height=FWI_H,
               count=1, crs=dst_crs, transform=dst_tf, nodata=np.nan, compress='lzw')

for var in ['u10', 'v10', 'cape']:
    outdir = f'data/era5_{var}'
    os.makedirs(outdir, exist_ok=True)

    # Only 2018-2025
    src_files = []
    for year in range(2018, 2026):
        src_files.extend(sorted(glob.glob(f'data/era5_daily/{var}_{year}*.tif')))

    print(f'\n=== {var}: {len(src_files)} source TIFs (2018-2025) ===')

    done = skip = fail = 0
    t0 = time.time()
    for i, sf in enumerate(src_files):
        bn = os.path.basename(sf)
        out = os.path.join(outdir, bn)
        if os.path.exists(out) and os.path.getsize(out) > 1000:
            skip += 1
            continue
        try:
            with rasterio.open(sf) as src:
                data = src.read(1).astype(np.float32)
                dst_data = np.full((FWI_H, FWI_W), np.nan, dtype=np.float32)
                reproject(data, dst_data, src_transform=src.transform, src_crs=src.crs,
                          dst_transform=dst_tf, dst_crs=dst_crs, resampling=Resampling.bilinear,
                          src_nodata=np.nan, dst_nodata=np.nan)
            with rasterio.open(out, 'w', **profile) as dst:
                dst.write(dst_data, 1)
            done += 1
        except Exception as e:
            print(f'  [{i}] {bn}: {e}')
            fail += 1

        if (done + skip) % 500 == 0 and done > 0:
            elapsed = time.time() - t0
            rate = done / elapsed
            remaining = (len(src_files) - i - 1) / rate if rate > 0 else 0
            print(f'  {var}: {done} done, {skip} skip, {fail} fail / {i+1}  '
                  f'({elapsed:.0f}s, ~{remaining/60:.0f}m left)')

    elapsed = time.time() - t0
    print(f'  {var} DONE: {done} processed, {skip} skipped, {fail} failed  ({elapsed:.0f}s)')

    # Verify
    n_out = len(glob.glob(os.path.join(outdir, f'{var}_201*.tif'))) + \
            len(glob.glob(os.path.join(outdir, f'{var}_202*.tif')))
    print(f'  {var} FWI-grid TIFs (2018-2025): {n_out}')

print('\n=== ALL DONE ===')
" 2>&1

echo ""
echo "Done: $(date)"
