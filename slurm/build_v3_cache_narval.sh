#!/bin/bash
#SBATCH --job-name=wf-cache
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --output=/scratch/jiaqi217/logs/build_cache_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/build_cache_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# Pre-build meteo memmap cache for V3 training/eval.
# No GPU needed — pure CPU data processing.
#
# Usage:
#   sbatch slurm/build_v3_cache_narval.sh                    # default 13ch
#   CACHE_CHANNELS=16 sbatch slurm/build_v3_cache_narval.sh  # 16ch
#
# Output: $SCRATCH/meteo_cache/v3_{N}ch/meteo_v3_p16_C{N}_T*_pf.dat
# ----------------------------------------------------------------

set -euo pipefail

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}

[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1

source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PYTHONUNBUFFERED=1

N=${CACHE_CHANNELS:-13}

if [ "$N" == "16" ]; then
    CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,deep_soil,precip_def,u10,v10,CAPE,NDVI,population,slope,burn_age,burn_count"
    CACHE_DIR="$SCRATCH/meteo_cache/v3_full"
elif [ "$N" == "13" ]; then
    CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,deep_soil,precip_def,NDVI,population,slope,burn_age,burn_count"
    CACHE_DIR="$SCRATCH/meteo_cache/v3_13ch"
elif [ "$N" == "9" ]; then
    CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,population,slope,burn_age"
    CACHE_DIR="$SCRATCH/meteo_cache/v3_9ch"
else
    echo "Unknown CACHE_CHANNELS=$N"; exit 1
fi

mkdir -p "$CACHE_DIR"

echo "============================================="
echo "  V3 Cache Builder — ${N}ch"
echo "  channels: $CHANNELS"
echo "  output: $CACHE_DIR"
echo "  Node: $(hostname)  Time: $(date)"
echo "============================================="

# Run train_v3 with --epochs 0 --skip_forecast to only build cache
python3 -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name cache_build_${N}ch \
    --data_start 2018-05-01 \
    --pred_start 2022-05-01 \
    --pred_end 2025-10-31 \
    --channels "$CHANNELS" \
    --decoder random \
    --batch_size 1024 \
    --epochs 0 \
    --lr 1e-4 \
    --d_model 256 \
    --nhead 8 \
    --enc_layers 4 \
    --dec_layers 4 \
    --patch_size 16 \
    --cache_dir "$CACHE_DIR" \
    --chunk_patches 2000 \
    --num_workers 4 \
    --skip_forecast \
    --overwrite

echo "Done: $(date)"
echo "Cache: $(ls -lh $CACHE_DIR/*pf.dat 2>/dev/null)"
