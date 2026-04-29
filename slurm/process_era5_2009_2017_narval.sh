#!/bin/bash

# DEPRECATED (2026-04-18): use slurm/process_era5_narval.sh instead.
# This hardcoded-year script is preserved for git history/reference only.
# Example: START_YEAR=2009 END_YEAR=2017 sbatch slurm/process_era5_narval.sh

#SBATCH --job-name=wf-era5-process
#SBATCH --time=18:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/jiaqi217/logs/era5_2009_2017_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/era5_2009_2017_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# Process ERA5 daily GRIBs (2009-2017) → EPSG:3978 TIFs
#
# Pipeline:
#   data/era5_on_fwi_grid/era5_sfc_YYYY_MM_DD.grib  (downloaded, WGS84)
#   ↓ era5_to_daily.py (extract per-variable, daily avg)
#   data/era5_daily/{2t,2d,tcw,sm20,st20,u10,v10,cape,tp}_YYYYMMDD.tif (WGS84)
#   ↓ resample (warp to EPSG:3978 / 2km Canada Lambert)
#   data/ecmwf_observation/{2t,2d,tcw,sm20,st20}/  + data/era5_{u10,v10,cape}/
#
# After this completes, all 2000-2025 ERA5 TIFs are ready for
# 9ch cache build (rebuild_cache_2000_2025_narval.sh).
# ----------------------------------------------------------------

set -uo pipefail

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}

[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1 eccodes/2.31.0

# venv's cfgrib package needs ecCodes C library on LD_LIBRARY_PATH.
# `module load eccodes` sets up build envs but not LD_LIBRARY_PATH.
# Without this, cfgrib import fails with "Cannot find the ecCodes library".
export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcc12/eccodes/2.31.0/lib64:${LD_LIBRARY_PATH:-}

source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

echo "============================================="
echo "  ERA5 2009-2017 GRIB → EPSG:3978 TIF pipeline"
echo "  Node: $(hostname)  Time: $(date)"
echo "============================================="

# ─── STEP 1: GRIB → WGS84 daily TIFs ─────────────────────────
echo ""
echo "=== STEP 1/2: extracting per-variable WGS84 TIFs ==="
mkdir -p data/era5_daily

python3 -u -m src.data_ops.processing.era5_to_daily \
    --input-dir data/era5_on_fwi_grid \
    --output-dir data/era5_daily

STEP1_EXIT=$?
if [ $STEP1_EXIT -ne 0 ]; then
    echo "STEP 1 FAILED (exit=$STEP1_EXIT)"
    exit $STEP1_EXIT
fi

# ─── STEP 2: WGS84 → EPSG:3978 (resample) for 2009-2017 ─────
echo ""
echo "=== STEP 2/2: resampling WGS84 → EPSG:3978 (2009-2017 only) ==="

for VAR in 2t 2d tcw sm20 st20 u10 v10 cape; do
    case $VAR in
        2t|2d|tcw|sm20|st20)  OUTDIR="data/ecmwf_observation/$VAR" ;;
        u10|v10|cape)         OUTDIR="data/era5_$VAR" ;;
    esac
    mkdir -p "$OUTDIR"
    echo ""
    echo "--- Resampling $VAR → $OUTDIR ---"

    python3 -u -c "
import glob, os, numpy as np, rasterio, time
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling

FWI_CRS = 'EPSG:3978'
FWI_W, FWI_H = 2709, 2281
FWI_BOUNDS = (-2378164.0, -707617.0, 3039835.0, 3854382.0)
dst_tf = from_bounds(*FWI_BOUNDS, FWI_W, FWI_H)
dst_crs = CRS.from_string(FWI_CRS)

# Use nan as nodata sentinel — project-wide de facto standard.
# Audit 2026-04-18 (12 channel sources):
#   nan: FWI/FFMC/DC/u10/v10/CAPE/NDVI/population/2000-2008 ERA5 obs (9/12)
#   -9999: 2018+ ecmwf_observation/{2t,2d,tcw,sm20,st20} (2/12, outlier)
#   9999 sentinel: burn_scars uint16 (1/12, source of historical bug)
#
# Decision: nan for new 2009-2017 files because
#   1. Matches 2000-2008 (keeps fresh time series consistent)
#   2. IEEE 754 standard — propagates loudly through math
#   3. Loader (train_v3._load_static_channel) handles both via
#      arr[~np.isfinite(arr)] = 0 + arr[arr==src.nodata] = 0
#   4. 2018+ is the outlier; future cleanup should standardize to nan
var = '${VAR}'
outdir = '${OUTDIR}'

DST_NODATA = float('nan')  # project standard
profile = dict(driver='GTiff', dtype='float32', width=FWI_W, height=FWI_H,
               count=1, crs=dst_crs, transform=dst_tf,
               nodata=DST_NODATA, compress='lzw')

src_files = []
for year in range(2009, 2018):
    src_files.extend(sorted(glob.glob(f'data/era5_daily/{var}_{year}*.tif')))

total = len(src_files)
print(f'  {var}: {total} source TIFs (2009-2017)')

n_ok, n_skip, n_fail = 0, 0, 0
t0 = time.time()
for i, src_path in enumerate(src_files):
    fname = os.path.basename(src_path)
    out_path = os.path.join(outdir, fname)
    if os.path.exists(out_path):
        n_skip += 1; continue
    try:
        with rasterio.open(src_path) as src:
            data = src.read(1).astype(np.float32)
            src_crs = src.crs
            src_tf = src.transform
            src_nd = src.nodata  # capture before src closes
        # Fill with nan initially (project standard, see header comment).
        # IMPORTANT: do NOT pass src_nodata/dst_nodata to reproject. ERA5
        # reanalysis has no real missing data — passing src_nodata=nan makes
        # reproject refuse to extrapolate at Canada Lambert edges beyond
        # WGS84 source bbox [-141..-52, 41..83], leaving 24% of pixels nan.
        # Old pre-existing 2000-2008/2018+ files were produced WITHOUT these
        # args, yielding fully populated rasters via bilinear extension.
        # This bug was caught 2026-04-18 via audit_channel_stats.py (2009-
        # 2017 had 24% nan, mean +3°C due to cold pixels excluded).
        dst = np.full((FWI_H, FWI_W), np.nan, dtype=np.float32)
        reproject(source=data, destination=dst,
                  src_transform=src_tf, src_crs=src_crs,
                  dst_transform=dst_tf, dst_crs=dst_crs,
                  resampling=Resampling.bilinear)
        with rasterio.open(out_path, 'w', **profile) as out:
            out.write(dst, 1)
        n_ok += 1
    except Exception as e:
        print(f'  FAIL {fname}: {e}'); n_fail += 1
    if (i+1) % 500 == 0:
        elapsed = time.time() - t0
        eta = (total - i - 1) / max(1, (i+1)/elapsed)
        print(f'  [{var}: {i+1}/{total}] ok={n_ok} skip={n_skip} fail={n_fail} ({elapsed:.0f}s, ~{eta/60:.0f}m left)')

print(f'  {var} done: {n_ok} resampled, {n_skip} skipped, {n_fail} failed ({time.time()-t0:.0f}s)')
"
done

STEP2_EXIT=$?
echo ""
echo "============================================="
echo "  Pipeline complete: $(date)  step1=$STEP1_EXIT  step2=$STEP2_EXIT"
echo "============================================="
exit $STEP2_EXIT
