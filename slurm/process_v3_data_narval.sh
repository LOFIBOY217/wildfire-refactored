#!/bin/bash
#SBATCH --job-name=wf-v3-proc
#SBATCH --time=8:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/jiaqi217/logs/process_v3_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/process_v3_%j.err

# ----------------------------------------------------------------
# Process all V3 raw data → FWI grid TIFs
# Runs on compute node (large memory for SRTM merge, NBAC rasterize)
# ----------------------------------------------------------------

set -euo pipefail

if [ -z "${SCRATCH:-}" ]; then
    export SCRATCH=/scratch/jiaqi217
fi

module load gcc/12.3 python/3.11.5 proj/9.4.1 eccodes/2.31.0 2>/dev/null || true
source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored

echo "============================================="
echo "  V3 Data Processing (compute node)"
echo "  Node: $(hostname)  Time: $(date)"
echo "============================================="

# 1. NBAC burn scars (needs geopandas + rasterize, ~10 min)
echo ""
echo "=== [1/4] NBAC Burn Scars ==="
python3 -u -m src.data_ops.processing.process_nbac_burn_scars \
    --config configs/paths_narval.yaml \
    --start_year 2018 --end_year 2024 2>&1
echo "NBAC exit: $?"

# 2. SRTM slope/aspect (needs merge 1374 tiles, ~30 min)
echo ""
echo "=== [2/4] SRTM Slope ==="
python3 -u -m src.data_ops.processing.process_srtm_slope \
    --config configs/paths_narval.yaml 2>&1
echo "SRTM exit: $?"

# 3. NDVI processing for all years (2018-2024)
echo ""
echo "=== [3/4] NDVI Processing (2018-2024) ==="
python3 -u -m src.data_ops.processing.process_modis_ndvi \
    --config configs/paths_narval.yaml \
    --start_year 2018 --end_year 2024 --overwrite 2>&1
echo "NDVI exit: $?"

# 4. GLM lightning reproject (if any raw TIFs exist)
echo ""
echo "=== [4/4] GLM Lightning Reproject ==="
python3 -u -m src.data_ops.processing.resample_glm_to_fwi_grid \
    --config configs/paths_narval.yaml 2>&1
echo "GLM exit: $?"

# 5. Validation
echo ""
echo "=== VALIDATION ==="
python3 -u -m src.data_ops.validation.check_v3_data \
    --config configs/paths_narval.yaml 2>&1

echo ""
echo "Done: $(date)"
