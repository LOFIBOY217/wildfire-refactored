#!/bin/bash
#SBATCH --job-name=wf-dl-v3
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/jiaqi217/logs/download_v3_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/download_v3_%j.err

# ----------------------------------------------------------------
# Download all V3 data sources on a compute node
# NOTE: This must be submitted from a login node with internet
#       access (narval1/narval3). Compute nodes may not have internet.
#       If that's the case, run this script directly on login node:
#         bash slurm/download_all_v3_data.sh
# ----------------------------------------------------------------

set -uo pipefail

if [ -z "${SCRATCH:-}" ]; then
    export SCRATCH=/scratch/jiaqi217
fi

cd $SCRATCH/wildfire-refactored
source $SCRATCH/venv-wildfire/bin/activate

export EARTHDATA_USERNAME=jiaqihuang
export EARTHDATA_PASSWORD=sitXuh-hecvox-tesju0

echo "============================================="
echo "  V3 Data Download"
echo "  Node: $(hostname)  Time: $(date)"
echo "============================================="

# ----------------------------------------------------------------
# 1. Deep Soil Moisture (ERA5 swvl2) — CDS API, ~3h
# ----------------------------------------------------------------
echo ""
echo "=== [1/6] Deep Soil Moisture (ERA5 swvl2) ==="
python3 -u -m src.data_ops.download.download_era5_deep_soil \
    2018-01-01 2025-10-31 \
    --config configs/paths_narval.yaml \
    2>&1 | tee $SCRATCH/logs/dl_deep_soil_slurm.log
echo "Deep soil exit: $?"

# ----------------------------------------------------------------
# 2. Lightning (GOES GLM) — AWS S3, ~6-12h
# ----------------------------------------------------------------
echo ""
echo "=== [2/6] Lightning (GOES GLM) ==="
python3 -u -m src.data_ops.download.download_goes_glm \
    --start 20180501 --end 20241031 --workers 4 \
    --config configs/paths_narval.yaml \
    2>&1 | tee $SCRATCH/logs/dl_lightning_slurm.log
echo "Lightning exit: $?"

# ----------------------------------------------------------------
# 3. NDVI (MODIS MOD13A2) — NASA Earthdata, ~2-4h
# ----------------------------------------------------------------
echo ""
echo "=== [3/6] NDVI (MODIS) ==="
python3 -u -m src.data_ops.download.download_modis_ndvi \
    --config configs/paths_narval.yaml \
    --start_year 2018 --end_year 2024 \
    --months 4 5 6 7 8 9 10 \
    2>&1 | tee $SCRATCH/logs/dl_ndvi_slurm.log
echo "NDVI exit: $?"

# ----------------------------------------------------------------
# 4. Terrain / Slope (SRTM) — NASA Earthdata, ~30min
# ----------------------------------------------------------------
echo ""
echo "=== [4/6] Terrain (SRTM slope) ==="
python3 -u -m src.data_ops.download.download_srtm_slope \
    --config configs/paths_narval.yaml \
    2>&1 | tee $SCRATCH/logs/dl_terrain_slurm.log
echo "Terrain exit: $?"

# ----------------------------------------------------------------
# 5. Burn Scars (NBAC) — NRCan open data, ~30min
# ----------------------------------------------------------------
echo ""
echo "=== [5/6] Burn Scars (NBAC) ==="
python3 -u -m src.data_ops.download.download_nbac_burn_scars \
    --config configs/paths_narval.yaml \
    --start_year 2000 --end_year 2024 \
    2>&1 | tee $SCRATCH/logs/dl_burn_scars_slurm.log
echo "Burn scars exit: $?"

# ----------------------------------------------------------------
# 6. Verify all data
# ----------------------------------------------------------------
echo ""
echo "============================================="
echo "  DATA VERIFICATION"
echo "============================================="

echo "Population:" && ls -lh data/population_density.tif 2>/dev/null || echo "  MISSING"
echo "Deep Soil TIFs:" && ls data/era5_deep_soil/deep_soil_*.tif 2>/dev/null | wc -l
echo "Lightning TIFs:" && ls data/lightning/lightning_*.tif 2>/dev/null | wc -l
echo "NDVI TIFs:" && ls data/ndvi_data/ndvi_*.tif 2>/dev/null | wc -l
echo "Terrain:" && ls -lh data/terrain/slope.tif 2>/dev/null || echo "  MISSING"
echo "Burn Scars:" && ls data/burn_scars/years_since_burn_*.tif 2>/dev/null | wc -l

echo ""
echo "Done: $(date)"
