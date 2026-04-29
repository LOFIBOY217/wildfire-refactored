#!/bin/bash
#SBATCH --job-name=wf-era5-proc
#SBATCH --time=18:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/jiaqi217/logs/era5_proc_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/era5_proc_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ================================================================
#  UNIFIED ERA5 Processing Pipeline
#  (replaces: process_era5_2009_2017, resample_era5_2000_2017,
#             process_swvl2_2000_2017, process_swvl2_2025,
#             process_tp_2023_2025, extract_era5_tp_only)
#
#  Pipeline: GRIB → WGS84 daily TIF → EPSG:3978 Canada Lambert TIF
#
#  Required env vars:
#    START_YEAR   e.g. 2009
#    END_YEAR     e.g. 2017
#
#  Optional env vars:
#    VARS         comma-separated variable list
#                 default: "2t,2d,tcw,sm20,st20,u10,v10,cape"
#                 for swvl2:    VARS=swvl2
#                 for precip:   VARS=tp
#    GRIB_DIR     where to find *.grib daily files
#                 default: "data/era5_on_fwi_grid"
#                 for swvl2:    GRIB_DIR=data/era5_deep_soil
#    WGS84_DIR    where STEP 1 writes WGS84 TIFs
#                 default: "data/era5_daily"
#    SKIP_STAGE1  set to 1 if WGS84 TIFs already exist
#                 (e.g., re-running only resample after a bug fix)
#
#  Examples:
#    # Main ERA5 vars, 2009-2017
#    START_YEAR=2009 END_YEAR=2017 sbatch slurm/process_era5_narval.sh
#
#    # swvl2 (deep soil moisture), 2000-2017
#    START_YEAR=2000 END_YEAR=2017 VARS=swvl2 GRIB_DIR=data/era5_deep_soil \
#      sbatch slurm/process_era5_narval.sh
#
#    # Only resample (WGS84 TIFs already exist)
#    START_YEAR=2009 END_YEAR=2017 SKIP_STAGE1=1 sbatch slurm/process_era5_narval.sh
#
#  BUGS PREVENTED BY THIS SCRIPT (see docs/DATA_CONVENTIONS.md):
#    - Edge-nan bug: reproject() called WITHOUT src_nodata/dst_nodata
#      (ERA5 has no real nans; passing them prevents bilinear edge
#       extension, leaving 24% nan pixels at Canada Lambert edges)
#    - Mixed nodata convention: always writes nan (project standard)
#    - LD_LIBRARY_PATH: eccodes lib exported for cfgrib venv wrapper
# ================================================================

set -uo pipefail

: "${START_YEAR:?Must set START_YEAR (e.g. 2009)}"
: "${END_YEAR:?Must set END_YEAR (e.g. 2017)}"
VARS=${VARS:-"2t,2d,tcw,sm20,st20,u10,v10,cape"}
GRIB_DIR=${GRIB_DIR:-"data/era5_on_fwi_grid"}
WGS84_DIR=${WGS84_DIR:-"data/era5_daily"}
SKIP_STAGE1=${SKIP_STAGE1:-0}

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}

[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1 eccodes/2.31.0
# cfgrib venv wrapper needs the C library path explicit
export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcc12/eccodes/2.31.0/lib64:${LD_LIBRARY_PATH:-}

source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

echo "=============================================================="
echo "  UNIFIED ERA5 Processing"
echo "  Year range: $START_YEAR..$END_YEAR"
echo "  Variables : $VARS"
echo "  GRIB dir  : $GRIB_DIR"
echo "  WGS84 dir : $WGS84_DIR"
echo "  Skip stage1: $SKIP_STAGE1"
echo "  Node: $(hostname)  Time: $(date)"
echo "=============================================================="

# ─── STAGE 1: GRIB → WGS84 TIF (multi-variable extraction) ─────
if [ "$SKIP_STAGE1" != "1" ]; then
    echo ""
    echo "=== STAGE 1/2: GRIB → WGS84 TIFs ==="
    mkdir -p "$WGS84_DIR"
    python3 -u -m src.data_ops.processing.era5_to_daily \
        --input-dir "$GRIB_DIR" \
        --output-dir "$WGS84_DIR"
    STAGE1_EXIT=$?
    if [ $STAGE1_EXIT -ne 0 ]; then
        echo "STAGE 1 FAILED (exit=$STAGE1_EXIT)"
        exit $STAGE1_EXIT
    fi
else
    echo ""
    echo "=== STAGE 1 SKIPPED (SKIP_STAGE1=1) ==="
fi

# ─── STAGE 2: WGS84 TIF → EPSG:3978 (Canada Lambert) ────────────
echo ""
echo "=== STAGE 2/2: resampling to EPSG:3978 (Canada Lambert 2km) ==="

IFS=',' read -ra VAR_LIST <<< "$VARS"

for VAR in "${VAR_LIST[@]}"; do
    # Output directory depends on variable category (historical convention)
    case $VAR in
        2t|2d|tcw|sm20|st20)  OUTDIR="data/ecmwf_observation/$VAR" ;;
        u10|v10|cape)         OUTDIR="data/era5_$VAR" ;;
        tp)                   OUTDIR="data/era5_precip" ;;
        swvl2)                OUTDIR="data/era5_deep_soil" ;;
        *)                    OUTDIR="data/era5_$VAR" ;;
    esac
    mkdir -p "$OUTDIR"
    echo ""
    echo "--- Resampling $VAR → $OUTDIR ($START_YEAR-$END_YEAR) ---"

    START_YEAR="$START_YEAR" END_YEAR="$END_YEAR" VAR="$VAR" \
    WGS84_DIR="$WGS84_DIR" OUTDIR="$OUTDIR" python3 -u <<'PYEOF'
import glob, os, numpy as np, rasterio, time
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling

# Canada Lambert (EPSG:3978) target grid — matches docs/DATA_CONVENTIONS.md
FWI_CRS = 'EPSG:3978'
FWI_W, FWI_H = 2709, 2281
FWI_BOUNDS = (-2378164.0, -707617.0, 3039835.0, 3854382.0)
dst_tf = from_bounds(*FWI_BOUNDS, FWI_W, FWI_H)
dst_crs = CRS.from_string(FWI_CRS)

# Project standard: nan for all new data. See docs/DATA_CONVENTIONS.md sec 3.
profile = dict(driver='GTiff', dtype='float32', width=FWI_W, height=FWI_H,
               count=1, crs=dst_crs, transform=dst_tf,
               nodata=float('nan'), compress='lzw')

var       = os.environ['VAR']
wgs84_dir = os.environ['WGS84_DIR']
outdir    = os.environ['OUTDIR']
yrs       = range(int(os.environ['START_YEAR']), int(os.environ['END_YEAR']) + 1)

src_files = []
for year in yrs:
    src_files.extend(sorted(glob.glob(f'{wgs84_dir}/{var}_{year}*.tif')))

total = len(src_files)
print(f'  {var}: {total} source WGS84 TIFs')

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
            src_crs_ = src.crs
            src_tf_  = src.transform
        # IMPORTANT: do NOT pass src_nodata/dst_nodata. ERA5 has no real
        # nans; passing them makes reproject refuse to bilinear-extend
        # at Canada Lambert edges, leaving 24% of pixels as nan (seen
        # 2026-04-18 as critical bug on 2009-2017 batch). Without these
        # args, reproject extends the WGS84 source to fill full target.
        dst = np.full((FWI_H, FWI_W), np.nan, dtype=np.float32)
        reproject(source=data, destination=dst,
                  src_transform=src_tf_, src_crs=src_crs_,
                  dst_transform=dst_tf, dst_crs=dst_crs,
                  resampling=Resampling.bilinear)
        with rasterio.open(out_path, 'w', **profile) as out:
            out.write(dst, 1)
        n_ok += 1
    except Exception as e:
        print(f'  FAIL {fname}: {e}')
        n_fail += 1
    if (i + 1) % 500 == 0:
        elapsed = time.time() - t0
        eta = (total - i - 1) / max(1, (i + 1) / elapsed)
        print(f'  [{var}: {i+1}/{total}] ok={n_ok} skip={n_skip} '
              f'fail={n_fail} ({elapsed:.0f}s, ~{eta/60:.0f}m left)')

print(f'  {var} done: resampled={n_ok} skipped={n_skip} failed={n_fail} '
      f'({time.time()-t0:.0f}s)')
PYEOF
done

PY_EXIT=$?
echo ""
echo "=============================================================="
echo "  Pipeline complete: $(date) exit=$PY_EXIT"
echo "=============================================================="
exit $PY_EXIT
