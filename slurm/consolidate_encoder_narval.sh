#!/bin/bash
#SBATCH --job-name=wf-consolidate
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/jiaqi217/logs/consolidate_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/consolidate_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# Consolidate all encoder TIF files into a single contiguous memmap.
# Run ONCE, output reused by all future cache builds.
#
# 16ch: ~1.1 TB output, ~3-4h runtime
# After this, train_v3.py STEP 6 can use --consolidated flag
# to read frames by index instead of 27000+ rasterio.open() calls.
# ----------------------------------------------------------------

set -uo pipefail

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}

[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1

source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PYTHONUNBUFFERED=1

N=${CONSOLIDATE_CHANNELS:-16}

if [ "$N" == "16" ]; then
    CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,deep_soil,precip_def,u10,v10,CAPE,NDVI,population,slope,burn_age,burn_count"
    OUT="$SCRATCH/meteo_cache/encoder_consolidated_16ch.dat"
elif [ "$N" == "13" ]; then
    CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,deep_soil,precip_def,NDVI,population,slope,burn_age,burn_count"
    OUT="$SCRATCH/meteo_cache/encoder_consolidated_13ch.dat"
elif [ "$N" == "9" ]; then
    CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,population,slope,burn_age"
    OUT="$SCRATCH/meteo_cache/encoder_consolidated_9ch.dat"
else
    echo "Unknown CONSOLIDATE_CHANNELS=$N"; exit 1
fi

echo "============================================="
echo "  Consolidate Encoder Data → Single Memmap"
echo "  Channels: $CHANNELS ($N ch)"
echo "  Output: $OUT"
echo "  Node: $(hostname)  Time: $(date)"
echo "============================================="

python -u -m src.data_ops.processing.consolidate_encoder_data \
    --config configs/paths_narval.yaml \
    --channels "$CHANNELS" \
    --data_start 2018-05-01 \
    --data_end 2025-12-31 \
    --out-file "$OUT"

echo ""
echo "Done: $(date)"
ls -lh "$OUT" "$OUT.dates.npy" 2>/dev/null
