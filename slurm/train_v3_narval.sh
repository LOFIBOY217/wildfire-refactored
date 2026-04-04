#!/bin/bash
#SBATCH --job-name=wf-v3
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --output=/scratch/jiaqi217/logs/train_v3_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/train_v3_%j.err

# ----------------------------------------------------------------
# V3 Training: 10-channel, Focal Loss, Hard Negative Mining
# ----------------------------------------------------------------

set -euo pipefail

# Guard: ensure SCRATCH is set (non-login shells may not source profile)
if [ -z "${SCRATCH:-}" ]; then
    export SCRATCH=/scratch/jiaqi217
fi

# Module loads
module load gcc/12.3 python/3.11.5 proj/9.4.1 2>/dev/null || true

# Virtual environment
VENV="$SCRATCH/venv-wildfire"
if [ -d "$VENV" ]; then
    source "$VENV/bin/activate"
else
    echo "[ERROR] venv not found: $VENV"
    exit 1
fi

cd "$SCRATCH/wildfire-refactored"

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
RUN_NAME="${1:-v3_focal_10ch}"
DECODER="${2:-oracle}"
LOSS_FN="${3:-focal}"

echo "============================================="
echo "  V3 Training: $RUN_NAME"
echo "  Decoder: $DECODER  Loss: $LOSS_FN"
echo "  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "============================================="

# Use separate cache dir for V3 to avoid conflicts with V2
CACHE_DIR="$SCRATCH/meteo_cache/v3"
mkdir -p "$CACHE_DIR"

python -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name "$RUN_NAME" \
    --data_start 2018-05-01 \
    --pred_start 2022-05-01 \
    --pred_end 2025-10-31 \
    --decoder "$DECODER" \
    --channels "FWI,2t,fire_clim,lightning,NDVI,population,deep_soil,precip_def,slope,burn_age" \
    --loss_fn "$LOSS_FN" \
    --focal_alpha 0.25 \
    --focal_gamma 2.0 \
    --hard_neg_fraction 0.5 \
    --neg_ratio 20 \
    --neg_buffer 2 \
    --batch_size 4096 \
    --epochs 8 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --dropout 0.1 \
    --d_model 256 \
    --nhead 8 \
    --enc_layers 4 \
    --dec_layers 4 \
    --patch_size 16 \
    --dilate_radius 14 \
    --val_lift_k 5000 \
    --val_lift_sample_wins 20 \
    --cluster_eval \
    --fire_season_only \
    --load_train_to_ram \
    --cache_dir "$CACHE_DIR" \
    --chunk_patches 2000 \
    --num_workers 4 \
    --log_interval 200 \
    --skip_forecast

TRAIN_EXIT=$?

echo ""
echo "Training exit code: $TRAIN_EXIT"
echo "Done: $(date)"

exit $TRAIN_EXIT
