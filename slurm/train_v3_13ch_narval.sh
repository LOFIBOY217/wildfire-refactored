#!/bin/bash
#SBATCH --job-name=wf-v3-13ch
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=400G
#SBATCH --output=/scratch/jiaqi217/logs/train_v3_13ch_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/train_v3_13ch_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca
#SBATCH --account=def-inghaw

# ----------------------------------------------------------------
# V3 13-Channel Training (everything ready NOW, no u10/v10/CAPE):
#
# Weather/atmosphere (4):   FWI, 2t, 2d, tcw
# Surface/soil (3):         sm20, deep_soil (swvl2), precip_def (tp 30d deficit)
# Vegetation (1):           NDVI
# Spatial prior (3):        fire_clim, population, slope
# Fuel history (2):         burn_age, burn_count
#
# Decoder: s2s_legacy (6-ch ECMWF S2S forecast) + decoder_ctx
# Total encoder channels: 13  →  enc_dim = 256 × 13 = 3328
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

CACHE_DIR="$SCRATCH/meteo_cache/v3_13ch"
mkdir -p "$CACHE_DIR"

CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,deep_soil,precip_def,NDVI,population,slope,burn_age,burn_count"

echo "============================================="
echo "  V3 13-Channel Training (s2s_legacy decoder)"
echo "  channels: $CHANNELS"
echo "  Node: $(hostname)  Time: $(date)"
echo "============================================="

python3 -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name v3_13ch \
    --data_start 2018-05-01 \
    --pred_start 2022-05-01 \
    --pred_end 2025-10-31 \
    --channels "$CHANNELS" \
    --decoder s2s_legacy \
    --s2s_cache "$SCRATCH/meteo_cache/s2s_decoder_cache.dat" \
    --s2s_max_issue_lag 3 \
    --loss_fn focal \
    --focal_alpha 0.25 \
    --focal_gamma 2.0 \
    --hard_neg_fraction 0.5 \
    --neg_ratio 20 \
    --neg_buffer 2 \
    --batch_size 1024 \
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
    --fire_season_only \
    --cluster_eval \
    --decoder_ctx \
    --load_train_to_ram \
    --cache_dir "$CACHE_DIR" \
    --chunk_patches 2000 \
    --num_workers 4 \
    --log_interval 200 \
    --skip_forecast

TRAIN_EXIT=$?
echo "Exit: $TRAIN_EXIT"
exit $TRAIN_EXIT
