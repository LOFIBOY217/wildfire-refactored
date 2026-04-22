#!/bin/bash
#SBATCH --job-name=wf-cache-2000
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=1-12:00:00
# Mem raised 128G → 256G after 9ch build OOM'd at transpose-end (2026-04-20).
# Bigger channel sets (13/16) have proportionally bigger peak mem at transpose.
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
DATA_START="${DATA_START:-2000-05-01}"   # train start
PRED_START="${PRED_START:-2022-05-01}"   # train/val cutoff
PRED_END="${PRED_END:-2025-10-31}"        # val end

# Cache dir tag: short label distinguishing date range
# - default 2000-05-01 → 2025-10-31 → "2000" (Phase 2/3 22y)
# - DATA_START=2018-05-01 → "4y_2018"     (Phase 2 isolated label-effect)
case "$DATA_START" in
    2000-05-01) TAG="2000" ;;
    2018-05-01) TAG="4y_2018" ;;
    *)          TAG="custom_${DATA_START//-/}" ;;
esac

if [ "$N" == "16" ]; then
    CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,deep_soil,precip_def,u10,v10,CAPE,NDVI,population,slope,burn_age,burn_count"
    CACHE_DIR="$SCRATCH/meteo_cache/v3_full_${TAG}"
elif [ "$N" == "13" ]; then
    CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,deep_soil,precip_def,NDVI,population,slope,burn_age,burn_count"
    CACHE_DIR="$SCRATCH/meteo_cache/v3_13ch_${TAG}"
elif [ "$N" == "9" ]; then
    CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,population,slope,burn_age"
    CACHE_DIR="$SCRATCH/meteo_cache/v3_9ch_${TAG}"
else
    echo "Unknown CACHE_CHANNELS=$N"; exit 1
fi

mkdir -p "$CACHE_DIR"

echo "============================================="
echo "  Rebuild V3 Cache (${N}ch)"
echo "  date range:  $DATA_START → $PRED_END  (TAG=$TAG)"
echo "  channels:    $CHANNELS"
echo "  output:      $CACHE_DIR"
echo "  Node: $(hostname)  Time: $(date)"
echo "============================================="

# Run train_v3 with --epochs 0 to only build cache.
#
# IMPORTANT: --lead_end 45 (NOT the default 46) is required so that the
# cache key date-range matches what training will produce. Training uses
# --decoder s2s_legacy, which train_v3.py hard-clamps lead_end to 45
# (line ~992). Without this flag, cache aligns to pred_end + 46 + 5 days
# = 2025-12-21 (T=9333) while training aligns to + 45 + 5 = 2025-12-20
# (T=9332). That single-day mismatch makes training's cache lookup fail
# and triggers a full 30h cache rebuild inside the training job.
python3 -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name cache_build_${TAG}_${N}ch \
    --data_start "$DATA_START" \
    --pred_start "$PRED_START" \
    --pred_end "$PRED_END" \
    --lead_end 45 \
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
