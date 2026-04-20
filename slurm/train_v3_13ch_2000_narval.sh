#!/bin/bash
#SBATCH --job-name=wf-v3-13ch-2000
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=500G
#SBATCH --output=/scratch/jiaqi217/logs/train_v3_13ch_2000_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/train_v3_13ch_2000_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca
#SBATCH --account=def-inghaw

# ----------------------------------------------------------------
#   V3 13ch x enc{ENC} x 2000-2025 EXTENDED TRAINING
# ----------------------------------------------------------------
# Channel set: 9ch + deep_soil + precip_def + NDVI + burn_count
# CHANNELS = FWI,2t,fire_clim,2d,tcw,sm20,deep_soil,precip_def,
#            NDVI,population,slope,burn_age,burn_count
#
# Compares to 9ch SOTA (enc21 = 6.47x Lift@5000) — does adding
# deep_soil + precip + veg + fire density bump the ceiling?
#
# Usage:
#   ENC=14 sbatch slurm/train_v3_13ch_2000_narval.sh
#   ENC=21 sbatch slurm/train_v3_13ch_2000_narval.sh
#   ENC=28 sbatch slurm/train_v3_13ch_2000_narval.sh
#   ENC=35 sbatch slurm/train_v3_13ch_2000_narval.sh
#
# PREREQUISITES:
#   - tp 2000-2025 fully reprojected (data/era5_precip/tp_*.tif complete)
#     run: START_YEAR=2000 END_YEAR=2025 VARS=tp \
#          GRIB_DIR=data/era5_precip sbatch slurm/process_era5_narval.sh
#   - 13ch cache built at $SCRATCH/meteo_cache/v3_13ch_2000
#     run: CACHE_CHANNELS=13 sbatch slurm/rebuild_cache_2000_2025_narval.sh
# ----------------------------------------------------------------

set -uo pipefail
ENC=${ENC:?Must set ENC (one of 14, 21, 28, 35)}

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 proj/9.4.1 eccodes/2.31.0
cd "$SCRATCH/wildfire-refactored"
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

source slurm/lib_copy_cache.sh
copy_venv $SCRATCH/venv-wildfire
$PYTHON -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())" || exit 1

LOCAL_CACHE=$SLURM_TMPDIR/cache
mkdir -p "$LOCAL_CACHE"
copy_s2s_cache "$SCRATCH/meteo_cache" "$LOCAL_CACHE"

CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,deep_soil,precip_def,NDVI,population,slope,burn_age,burn_count"
CACHE_DIR_2000="$SCRATCH/meteo_cache/v3_13ch_2000"
RUN_NAME="v3_13ch_enc${ENC}_2000"

if [ ! -d "$CACHE_DIR_2000" ] || [ -z "$(ls -A "$CACHE_DIR_2000" 2>/dev/null)" ]; then
    echo "ERROR: $CACHE_DIR_2000 is empty or missing."
    echo "Run cache build first: CACHE_CHANNELS=13 sbatch slurm/rebuild_cache_2000_2025_narval.sh"
    exit 1
fi

echo "============================================="
echo "  V3 13ch x enc${ENC} x 2000-2025"
echo "  Run name: $RUN_NAME"
echo "  Cache: $CACHE_DIR_2000"
echo "============================================="

$PYTHON -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name "$RUN_NAME" \
    --data_start 2000-05-01 --pred_start 2022-05-01 --pred_end 2025-10-31 \
    --channels "$CHANNELS" --in_days "$ENC" \
    --decoder s2s_legacy --s2s_cache "$LOCAL_CACHE/s2s_decoder_cache.dat" --s2s_max_issue_lag 3 \
    --loss_fn focal --focal_alpha 0.25 --focal_gamma 2.0 \
    --hard_neg_fraction 0.5 --neg_ratio 20 --neg_buffer 2 \
    --batch_size 4096 --epochs 4 --lr 1e-4 --weight_decay 0.01 --dropout 0.2 \
    --d_model 256 --nhead 8 --enc_layers 4 --dec_layers 4 --patch_size 16 \
    --dilate_radius 14 --val_lift_k 5000 --val_lift_sample_wins 20 \
    --fire_season_only --cluster_eval --decoder_ctx --load_train_to_ram \
    --cache_dir "$CACHE_DIR_2000" --chunk_patches 2000 --num_workers 4 \
    --log_interval 200 --skip_forecast

PY_EXIT=$?
echo "=== Done: $(date) exit=$PY_EXIT ==="
exit $PY_EXIT
