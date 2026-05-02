#!/bin/bash
#SBATCH --job-name=wf-16ch-range-master
#SBATCH --gpus-per-node=1
#SBATCH --time=0-16:00:00
#SBATCH --mem=600G
#SBATCH --output=/scratch/jiaqi217/logs/train_v3_16ch_range_master_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/train_v3_16ch_range_master_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca
#SBATCH --account=def-inghaw

# ----------------------------------------------------------------
# V3 16ch x enc{ENC} x arbitrary RANGE  (master-cache reuse)
# ----------------------------------------------------------------
# 16ch = 13ch + {u10, v10, CAPE}  (wind components + convective potential)
# Reuses the v3_full_2000 master cache by time-slicing.
#
# Required env:
#   ENC          (e.g. 14, 21, 28, 35)
#   DATA_START   (e.g. 2000-05-01 → 22y range)
#   RANGE_TAG    (e.g. 22y_2000)
#
# Usage:
#   ENC=21 DATA_START=2000-05-01 RANGE_TAG=22y_2000 \
#     sbatch slurm/train_v3_16ch_range_master_narval.sh
# ----------------------------------------------------------------

set -uo pipefail
ENC=${ENC:?Must set ENC}
DATA_START=${DATA_START:?Must set DATA_START (YYYY-MM-DD)}
RANGE_TAG=${RANGE_TAG:?Must set RANGE_TAG (e.g. 22y_2000)}
PRED_START=${PRED_START:-2022-05-01}
PRED_END=${PRED_END:-2025-10-31}

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 proj/9.4.1 eccodes/2.31.0
cd "$SCRATCH/wildfire-refactored"
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

export WANDB_MODE=offline
export WANDB_ENTITY=jiaaqii-huang-university-of-toronto
export WANDB_DIR=$SCRATCH/wandb_offline

source slurm/lib_copy_cache.sh
copy_venv $SCRATCH/venv-wildfire
$PYTHON -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())" || exit 1

LOCAL_CACHE=$SLURM_TMPDIR/cache
mkdir -p "$LOCAL_CACHE"
copy_s2s_cache "$SCRATCH/meteo_cache" "$LOCAL_CACHE"

CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,deep_soil,precip_def,u10,v10,CAPE,NDVI,population,slope,burn_age,burn_count"
MASTER_CACHE_LUSTRE="$SCRATCH/meteo_cache/v3_full_2000"
MASTER_DATA_START="2000-05-01"
RUN_NAME="v3_16ch_enc${ENC}_${RANGE_TAG}"

if [ ! -d "$MASTER_CACHE_LUSTRE" ] || [ -z "$(ls -A "$MASTER_CACHE_LUSTRE" 2>/dev/null)" ]; then
    echo "ERROR: master cache $MASTER_CACHE_LUSTRE missing"; exit 1
fi

LOCAL_METEO="$LOCAL_CACHE/meteo"
mkdir -p "$LOCAL_METEO"
echo "=== copy 16ch master meteo to local SSD (~1.7 TB) ==="
t0=$SECONDS
for f in "$MASTER_CACHE_LUSTRE"/*; do
    [ -f "$f" ] || continue
    fname=$(basename "$f")
    sz=$(du -h "$f" | cut -f1)
    echo "  copy $fname ($sz)"
    cp "$f" "$LOCAL_METEO/" || { echo "FATAL"; exit 1; }
done
echo "  done in $((SECONDS - t0))s"

if [ ! -d "$SCRATCH/wildfire-refactored/data/fire_clim_annual_nbac" ]; then
    echo "ERROR: data/fire_clim_annual_nbac missing"; exit 1
fi

echo "============================================="
echo "  V3 16ch x enc${ENC} x RANGE_TAG=${RANGE_TAG}"
echo "  data_start = $DATA_START  pred_start = $PRED_START  pred_end = $PRED_END"
echo "  master     = $LOCAL_METEO  (start=$MASTER_DATA_START)"
echo "  Run name   = $RUN_NAME"
echo "============================================="

$PYTHON -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name "$RUN_NAME" \
    --data_start "$DATA_START" --pred_start "$PRED_START" --pred_end "$PRED_END" \
    --channels "$CHANNELS" --in_days "$ENC" \
    --decoder s2s_legacy --s2s_cache "$LOCAL_CACHE/s2s_decoder_cache.dat" --s2s_max_issue_lag 3 \
    --loss_fn focal --focal_alpha 0.25 --focal_gamma 2.0 \
    --hard_neg_fraction 0.5 --neg_ratio 20 --neg_buffer 2 \
    --batch_size 4096 --epochs 4 --lr 1e-4 --weight_decay 0.01 --dropout 0.2 \
    --d_model 256 --nhead 8 --enc_layers 4 --dec_layers 4 --patch_size 16 \
    --dilate_radius 14 --val_lift_k 5000 --val_lift_sample_wins 20 \
    --fire_season_only --cluster_eval --decoder_ctx --load_train_to_ram \
    --master_cache_dir "$LOCAL_METEO" --master_data_start "$MASTER_DATA_START" \
    --chunk_patches 2000 --num_workers 4 \
    --log_interval 200 --skip_forecast \
    --label_fusion --nfdb_min_size_ha 1.0 \
    --fire_clim_dir data/fire_clim_annual_nbac \
    --wandb_project wildfire-s2s \
    --wandb_tags "16ch,enc${ENC},${RANGE_TAG},master_cache" \
    --save_per_window_json "$SCRATCH/wildfire-refactored/outputs/${RUN_NAME}_per_window.json"

PY_EXIT=$?
echo "=== Done: $(date) exit=$PY_EXIT ==="
exit $PY_EXIT
