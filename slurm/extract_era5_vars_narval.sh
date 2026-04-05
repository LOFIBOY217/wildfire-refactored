#!/bin/bash
#SBATCH --job-name=wf-era5-ext
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/jiaqi217/logs/extract_era5_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/extract_era5_%j.err

# ----------------------------------------------------------------
# Extract u10/v10/tp/cape from existing ERA5 GRIBs → daily TIFs
# Then resample to FWI grid (EPSG:3978)
#
# GRIBs already exist in data/era5_on_fwi_grid/ (6338 files)
# Only new variables (u10/v10/tp/cape) will be extracted,
# existing 2t/2d TIFs are automatically skipped.
# ----------------------------------------------------------------

set -euo pipefail

if [ -z "${SCRATCH:-}" ]; then
    export SCRATCH=/scratch/jiaqi217
fi

module load gcc/12.3 python/3.11.5 proj/9.4.1 eccodes/2.31.0 2>/dev/null || true
source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored

echo "============================================="
echo "  ERA5 Variable Extraction (u10/v10/tp/cape)"
echo "  Node: $(hostname)  Time: $(date)"
echo "============================================="

# Step 1: Extract daily averages from GRIBs
# era5_to_daily.py reads VARIABLE_MAP which now includes u10/v10/tp/cape
# Existing 2t/2d/tcw/sm20/st20 TIFs are skipped automatically
echo ""
echo "=== Step 1: GRIB → daily TIFs (0.25° WGS84) ==="
python3 -u -m src.data_ops.processing.era5_to_daily \
    --config configs/paths_narval.yaml \
    --input-dir data/era5_on_fwi_grid \
    --output-dir data/era5_daily 2>&1
echo "era5_to_daily exit: $?"

# Step 2: Resample new variables to FWI grid
echo ""
echo "=== Step 2: Resample to FWI grid (EPSG:3978) ==="

for var in u10 v10 tp cape; do
    echo ""
    echo "--- Resampling $var ---"
    # Count source TIFs
    n_src=$(ls data/era5_daily/${var}_*.tif 2>/dev/null | wc -l)
    echo "  Source TIFs: $n_src"

    if [ "$n_src" -eq 0 ]; then
        echo "  [SKIP] No source TIFs for $var"
        continue
    fi

    # Create output directory
    outdir="data/era5_${var}"
    mkdir -p "$outdir"

    # Resample each TIF
    python3 -u -m src.data_ops.processing.resample_to_fwi_grid \
        --input-dir data/era5_daily \
        --output-dir "$outdir" \
        --prefix "$var" \
        --config configs/paths_narval.yaml 2>&1 || {
        echo "  [WARN] resample_to_fwi_grid failed for $var, trying manual reproject"
        # Fallback: manual reproject with rasterio
        python3 -u -c "
import glob, os, numpy as np, rasterio
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

src_files = sorted(glob.glob('data/era5_daily/${var}_*.tif'))
done = skip = 0
for sf in src_files:
    bn = os.path.basename(sf)
    out = '$outdir/' + bn
    if os.path.exists(out):
        skip += 1
        continue
    with rasterio.open(sf) as src:
        data = src.read(1).astype(np.float32)
        dst_data = np.full((FWI_H, FWI_W), np.nan, dtype=np.float32)
        reproject(data, dst_data, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=dst_tf, dst_crs=dst_crs, resampling=Resampling.bilinear,
                  src_nodata=np.nan, dst_nodata=np.nan)
    with rasterio.open(out, 'w', **profile) as dst:
        dst.write(dst_data, 1)
    done += 1
    if done % 200 == 0:
        print(f'  {var}: {done}/{len(src_files)} done')
print(f'  {var}: done={done} skip={skip}')
" 2>&1
    }
done

# Step 3: Summary
echo ""
echo "=== Summary ==="
for var in u10 v10 tp cape; do
    n=$(ls data/era5_${var}/${var}_*.tif 2>/dev/null | wc -l)
    echo "  $var: $n TIFs on FWI grid"
done

echo ""
echo "Done: $(date)"
