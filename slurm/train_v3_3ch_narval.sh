#!/bin/bash
#SBATCH --job-name=wf-v3-3ch
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=128G
#SBATCH --output=/scratch/jiaqi217/logs/train_v3_3ch_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/train_v3_3ch_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# V3 Training: 3-channel baseline (FWI + 2t + fire_clim)
# Validates Focal Loss + Hard Negative Mining effect vs V2 BCE
# ----------------------------------------------------------------

set -euo pipefail

if [ -z "${SCRATCH:-}" ]; then
    export SCRATCH=/scratch/jiaqi217
fi

module load gcc/12.3 python/3.11.5 proj/9.4.1 2>/dev/null || true

VENV="$SCRATCH/venv-wildfire"
if [ -d "$VENV" ]; then
    source "$VENV/bin/activate"
else
    echo "[ERROR] venv not found: $VENV"
    exit 1
fi

cd "$SCRATCH/wildfire-refactored"

echo "============================================="
echo "  V3 3-Channel Baseline: Focal Loss + Hard Neg"
echo "  Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "============================================="

CACHE_DIR="$SCRATCH/meteo_cache/v3_3ch"
mkdir -p "$CACHE_DIR"

python3 -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name v3_focal_3ch \
    --data_start 2018-05-01 \
    --pred_start 2022-05-01 \
    --pred_end 2025-10-31 \
    --decoder oracle \
    --channels "FWI,2t,fire_clim" \
    --loss_fn focal \
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
echo "Training exit: $TRAIN_EXIT"
exit $TRAIN_EXIT
