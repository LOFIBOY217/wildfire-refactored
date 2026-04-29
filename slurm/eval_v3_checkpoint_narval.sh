#!/bin/bash
#SBATCH --job-name=wf-eval
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=500G
#SBATCH --output=/scratch/jiaqi217/logs/eval_v3_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/eval_v3_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# Standalone V3 checkpoint evaluator.
# Loads a single checkpoint, runs full validation, exits.
#
# Usage:
#   CKPT=checkpoints/v3_full/epoch_03.pt \
#     RUN_NAME=v3_full \
#     CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,deep_soil,precip_def,u10,v10,CAPE,NDVI,population,slope,burn_age,burn_count" \
#     sbatch slurm/eval_v3_checkpoint_narval.sh
#
# Set FULL_VAL=1 for 811-window full eval (slower, ~6h).
# Default is 20-window sample (fast, ~2min).
# ----------------------------------------------------------------

set -uo pipefail

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}

[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 proj/9.4.1 eccodes/2.31.0

cd "$SCRATCH/wildfire-refactored"
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

source $SCRATCH/venv-wildfire/bin/activate
PYTHON=python

# Required env vars
: "${CKPT:?Must set CKPT=path/to/checkpoint.pt}"
: "${RUN_NAME:?Must set RUN_NAME (e.g. v3_full)}"
: "${CHANNELS:?Must set CHANNELS=... comma-separated}"

# Default optional args
DECODER=${DECODER:-s2s_legacy}
IN_DAYS=${IN_DAYS:-7}
FULL_VAL_FLAG=""
[ "${FULL_VAL:-0}" = "1" ] && FULL_VAL_FLAG="--full_val"
# Optional: skip cluster_eval (default: run it).
# cluster_eval does a 2nd forward-pass over all windows → doubles runtime.
# Set SKIP_CLUSTER=1 if Lift@30km from pixel eval is sufficient.
CLUSTER_FLAG="--cluster_eval"
[ "${SKIP_CLUSTER:-0}" = "1" ] && CLUSTER_FLAG=""

echo "============================================="
echo "  V3 Checkpoint Eval"
echo "  Checkpoint: $CKPT"
echo "  Run name  : $RUN_NAME"
echo "  Channels  : $CHANNELS"
echo "  Decoder   : $DECODER"
echo "  in_days   : $IN_DAYS"
echo "  Full val  : ${FULL_VAL:-0}"
echo "  Node      : $(hostname)"
echo "  Time      : $(date)"
echo "============================================="

LOCAL_CACHE=$SLURM_TMPDIR/cache
mkdir -p "$LOCAL_CACHE"
cp $SCRATCH/meteo_cache/s2s_decoder_cache.dat "$LOCAL_CACHE/" 2>/dev/null || true
cp $SCRATCH/meteo_cache/s2s_decoder_cache.dat.dates.npy "$LOCAL_CACHE/" 2>/dev/null || true

$PYTHON -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name "$RUN_NAME" \
    --data_start 2018-05-01 \
    --pred_start 2022-05-01 \
    --pred_end 2025-10-31 \
    --channels "$CHANNELS" \
    --in_days "$IN_DAYS" \
    --decoder "$DECODER" \
    --s2s_cache "$LOCAL_CACHE/s2s_decoder_cache.dat" \
    --s2s_max_issue_lag 3 \
    --batch_size 1024 \
    --epochs 0 \
    --d_model 256 \
    --nhead 8 \
    --enc_layers 4 \
    --dec_layers 4 \
    --patch_size 16 \
    --val_lift_k 5000 \
    --val_lift_sample_wins 20 \
    --fire_season_only \
    $CLUSTER_FLAG \
    --decoder_ctx \
    --cache_dir "${CACHE_DIR:-$SCRATCH/meteo_cache/v3_full}" \
    --chunk_patches 2000 \
    --num_workers 2 \
    --skip_forecast \
    --eval_checkpoint "$CKPT" \
    $FULL_VAL_FLAG

echo ""
echo "=== Eval complete ==="
