#!/bin/bash
#SBATCH --job-name=wf-proc-cdem
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/jiaqi217/logs/proc_cdem_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/proc_cdem_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# Process CDEM tiles → slope_cdem.tif + aspect_cdem.tif on FWI grid
# Then cross-validate against SRTM slope in overlap zone
# ----------------------------------------------------------------

set -euo pipefail

if [ -z "${SCRATCH:-}" ]; then
    export SCRATCH=/scratch/jiaqi217
fi

module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1 2>/dev/null || true
source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored

echo "============================================="
echo "  CDEM Processing + Cross-Validation"
echo "  Node: $(hostname)  Time: $(date)"
echo "============================================="

# Step 1: Rename existing SRTM slope.tif → slope_srtm.tif (if not done)
TERRAIN_DIR=$SCRATCH/wildfire-refactored/data/terrain
if [ -f "$TERRAIN_DIR/slope.tif" ] && [ ! -f "$TERRAIN_DIR/slope_srtm.tif" ]; then
    echo "Renaming SRTM slope.tif → slope_srtm.tif"
    cp "$TERRAIN_DIR/slope.tif" "$TERRAIN_DIR/slope_srtm.tif"
    [ -f "$TERRAIN_DIR/aspect.tif" ] && cp "$TERRAIN_DIR/aspect.tif" "$TERRAIN_DIR/aspect_srtm.tif"
fi

# Step 2: Process CDEM tiles → slope_cdem.tif
echo ""
echo "--- CDEM Processing ---"
python3 -u -m src.data_ops.processing.process_cdem_slope \
    --config configs/paths_narval.yaml

# Step 3: Cross-validate SRTM vs CDEM
echo ""
echo "--- Cross-Validation ---"
python3 -u -m src.data_ops.validation.cross_validate_slope \
    --config configs/paths_narval.yaml

echo ""
echo "Done: $(date)"
