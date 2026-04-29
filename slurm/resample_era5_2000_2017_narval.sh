#!/bin/bash

# DEPRECATED (2026-04-18): use slurm/process_era5_narval.sh instead.
# This hardcoded-year script is preserved for git history/reference only.
# Example: START_YEAR=2009 END_YEAR=2017 sbatch slurm/process_era5_narval.sh

#SBATCH --job-name=wf-resamp-all
#SBATCH --time=14:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/jiaqi217/logs/resample_era5_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/resample_era5_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# Resample ERA5 daily TIFs (WGS84) → FWI grid (EPSG:3978)
# for 2000-2017 data. 2018+ already done.
#
# Usage (one variable at a time for parallelism):
#   RESAMP_VAR=2t sbatch slurm/resample_era5_2000_2017_narval.sh
#   RESAMP_VAR=2d sbatch slurm/resample_era5_2000_2017_narval.sh
#   RESAMP_VAR=tcw sbatch slurm/resample_era5_2000_2017_narval.sh
#   ... etc for sm20, st20, v10, cape
#
# Or submit all 7 at once:
#   for v in 2t 2d tcw sm20 st20 v10 cape; do RESAMP_VAR=$v sbatch slurm/resample_era5_2000_2017_narval.sh; done
# ----------------------------------------------------------------

set -euo pipefail

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}

[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1

source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored

export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

VAR=${RESAMP_VAR:?Must set RESAMP_VAR (e.g. 2t, 2d, tcw, sm20, st20, v10, cape)}

# Output directory depends on variable
case $VAR in
    2t|2d|tcw|sm20|st20)
        OUTDIR="data/ecmwf_observation/$VAR"
        ;;
    v10|cape|u10)
        OUTDIR="data/era5_$VAR"
        ;;
    *)
        echo "Unknown variable: $VAR"; exit 1
        ;;
esac

mkdir -p "$OUTDIR"

echo "============================================="
echo "  Resample $VAR → FWI grid (2000-2017 only)"
echo "  Source: data/era5_daily/${VAR}_*.tif"
echo "  Output: $OUTDIR/${VAR}_*.tif"
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

var = '${VAR}'
outdir = '${OUTDIR}'

# Only process 2000-2017 (skip 2018+, already done)
src_files = []
for year in range(2000, 2018):
    src_files.extend(sorted(glob.glob(f'data/era5_daily/{var}_{year}*.tif')))

total = len(src_files)
print(f'=== {var}: {total} source TIFs (2000-2017) ===')

n_ok, n_skip, n_fail = 0, 0, 0
t0 = time.time()

for i, src_path in enumerate(src_files):
    fname = os.path.basename(src_path)
    out_path = os.path.join(outdir, fname)

    if os.path.exists(out_path):
        n_skip += 1
        continue

    try:
        with rasterio.open(src_path) as src:
            data = src.read(1).astype(np.float32)
            src_crs = src.crs
            src_tf = src.transform

        dst = np.full((FWI_H, FWI_W), np.nan, dtype=np.float32)
        reproject(
            source=data, destination=dst,
            src_transform=src_tf, src_crs=src_crs,
            dst_transform=dst_tf, dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )

        with rasterio.open(out_path, 'w', **profile) as out:
            out.write(dst, 1)
        n_ok += 1

    except Exception as e:
        print(f'  FAIL: {fname}: {e}')
        n_fail += 1

    if (i + 1) % 200 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (total - i - 1) / rate
        print(f'  [{i+1}/{total}] ok={n_ok} skip={n_skip} fail={n_fail} '
              f'({elapsed:.0f}s, ~{eta/60:.0f}m left)')

elapsed = time.time() - t0
print(f'Done: {n_ok} resampled, {n_skip} skipped, {n_fail} failed ({elapsed:.0f}s)')
"

echo "=== Resample $VAR complete: $(date) ==="
