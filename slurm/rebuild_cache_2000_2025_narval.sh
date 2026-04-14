#!/bin/bash
#SBATCH --job-name=wf-cache-2000
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-12:00:00
#SBATCH --output=/scratch/jiaqi217/logs/cache_2000_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/cache_2000_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# Rebuild encoder cache with extended 2000-2025 training data.
#
# Expected: T≈9400 days (vs 2791 before), 3.4× more training data.
# With bug-fixed code (nodata mask, year-1 burn_age, annual fire_clim).
#
# Channel sets:
#   CACHE_CHANNELS=9:  FWI, 2t, fire_clim, 2d, tcw, sm20, population, slope, burn_age
#   CACHE_CHANNELS=13: + deep_soil, precip_def, NDVI, burn_count
#   CACHE_CHANNELS=16: + u10, v10, CAPE
#
# Usage:
#   CACHE_CHANNELS=9 sbatch slurm/rebuild_cache_2000_2025_narval.sh
# ----------------------------------------------------------------

set -uo pipefail

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}

[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1

source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

N=${CACHE_CHANNELS:-9}

if [ "$N" == "16" ]; then
    CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,deep_soil,precip_def,u10,v10,CAPE,NDVI,population,slope,burn_age,burn_count"
    CACHE_DIR="$SCRATCH/meteo_cache/v3_full_2000"
elif [ "$N" == "13" ]; then
    CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,deep_soil,precip_def,NDVI,population,slope,burn_age,burn_count"
    CACHE_DIR="$SCRATCH/meteo_cache/v3_13ch_2000"
elif [ "$N" == "9" ]; then
    CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,population,slope,burn_age"
    CACHE_DIR="$SCRATCH/meteo_cache/v3_9ch_2000"
else
    echo "Unknown CACHE_CHANNELS=$N"; exit 1
fi

mkdir -p "$CACHE_DIR"

echo "============================================="
echo "  Rebuild V3 Cache — 2000-2025 training data (${N}ch)"
echo "  channels: $CHANNELS"
echo "  output: $CACHE_DIR"
echo "  Node: $(hostname)  Time: $(date)"
echo "============================================="

# Run train_v3 with --epochs 0 to only build cache
python3 -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name cache_build_2000_${N}ch \
    --data_start 2000-05-01 \
    --pred_start 2022-05-01 \
    --pred_end 2025-10-31 \
    --channels "$CHANNELS" \
    --decoder random \
    --batch_size 1024 \
    --epochs 0 \
    --lr 1e-4 \
    --d_model 256 --nhead 8 --enc_layers 4 --dec_layers 4 --patch_size 16 \
    --cache_dir "$CACHE_DIR" \
    --chunk_patches 2000 --num_workers 8 \
    --skip_forecast \
    --overwrite

echo "=== Done: $(date) ==="
ls -lh "$CACHE_DIR"
