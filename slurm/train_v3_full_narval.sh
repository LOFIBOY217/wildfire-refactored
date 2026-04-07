#!/bin/bash
#SBATCH --job-name=wf-v3-full
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=256G
#SBATCH --output=/scratch/jiaqi217/logs/train_v3_full_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/train_v3_full_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca
#SBATCH --account=def-inghaw

# ----------------------------------------------------------------
# V3 Full Training (all available channels):
#
# Weather/atmosphere (5):
#   FWI      — composite fire weather index
#   2t       — 2m temperature
#   2d       — 2m dewpoint (→ VPD proxy)
#   tcw      — total column water vapour
#   CAPE     — convective available potential energy (thunderstorm/lightning proxy)
#
# Surface/soil (3):
#   sm20     — 0-20cm soil moisture
#   deep_soil — deep soil moisture (swvl2)
#   precip_def — 30-day rolling precipitation deficit
#
# Wind (2):
#   u10      — 10m eastward wind
#   v10      — 10m northward wind
#
# Vegetation (1):
#   NDVI     — normalized difference vegetation index (16-day interp)
#
# Spatial prior (3):
#   fire_clim   — historical fire frequency (static)
#   population  — population density (static)
#   slope       — terrain slope from CDEM (static)
#
# Fuel history (2):
#   burn_age   — years since last burn
#   burn_count — cumulative burn frequency
#
# Total: 16 channels
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

CACHE_DIR="$SCRATCH/meteo_cache/v3_full"
mkdir -p "$CACHE_DIR"

CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,deep_soil,precip_def,u10,v10,CAPE,NDVI,population,slope,burn_age,burn_count"

echo "============================================="
echo "  V3 Full Training (16 channels)"
echo "  channels: $CHANNELS"
echo "  Node: $(hostname)  Time: $(date)"
echo "============================================="

python3 -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name v3_full \
    --data_start 2018-05-01 \
    --pred_start 2022-05-01 \
    --pred_end 2025-10-31 \
    --channels "$CHANNELS" \
    --decoder random \
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
