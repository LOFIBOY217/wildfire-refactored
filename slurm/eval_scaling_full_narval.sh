#!/bin/bash
#SBATCH --job-name=wf-scaling-eval
#SBATCH --gpus-per-node=1
#SBATCH --time=0-14:00:00
#SBATCH --mem=400G
#SBATCH --output=/scratch/jiaqi217/logs/scaling_eval_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/scaling_eval_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# Plan A: full-window (≈435-win) eval for data-scaling checkpoints,
# to replace the noisy 20-window sample Lift values in Fig 4.
#
# Each scaling point shares the SAME val windows (2022→2025) but was
# trained on a different number of years (data_start → 2022). Models
# trained via master-cache time-slicing (10y_2016/14y_2012/16y_2010)
# must be evaluated the same way: --master_cache_dir v3_9ch_2000 +
# --master_data_start 2000-05-01 + --data_start <range>.
#
# We read the cache straight from Lustre (no SSD copy) because eval is
# a single forward pass, not multi-epoch training.
#
# Usage (submit one per range):
#   RANGE_TAG=4y_2018  DATA_START=2018-05-01 sbatch slurm/eval_scaling_full_narval.sh
#   RANGE_TAG=10y_2016 DATA_START=2016-05-01 sbatch slurm/eval_scaling_full_narval.sh
#   RANGE_TAG=14y_2012 DATA_START=2012-05-01 sbatch slurm/eval_scaling_full_narval.sh
#   RANGE_TAG=16y_2010 DATA_START=2010-05-01 sbatch slurm/eval_scaling_full_narval.sh
#   RANGE_TAG=2000     DATA_START=2000-05-01 sbatch slurm/eval_scaling_full_narval.sh
# ----------------------------------------------------------------

set -uo pipefail
RANGE_TAG=${RANGE_TAG:?Must set RANGE_TAG (e.g. 10y_2016)}
DATA_START=${DATA_START:?Must set DATA_START (e.g. 2016-05-01)}
# Master-cache slices must stop within the master cache length
# (v3_9ch_2000 T=9332 ≈ 2025-11-17). Default pred_end 2025-09-23 keeps
# t_offset+T <= 9332 for all start years; late-2025 windows are ~empty
# (NBAC 2025 not released) so dropping them is harmless.
PRED_END=${PRED_END:-2025-09-23}

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 proj/9.4.1 eccodes/2.31.0
cd "$SCRATCH/wildfire-refactored"
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

source slurm/lib_copy_cache.sh
copy_venv $SCRATCH/venv-wildfire
cuda_probe || exit 1

LOCAL_CACHE=$SLURM_TMPDIR/cache
mkdir -p "$LOCAL_CACHE"
copy_s2s_cache "$SCRATCH/meteo_cache" "$LOCAL_CACHE"

CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,population,slope,burn_age"
RUN_NAME="v3_9ch_enc21_${RANGE_TAG}"
CKPT="$SCRATCH/wildfire-refactored/checkpoints/${RUN_NAME}/best_model.pt"
OUT_JSON="$SCRATCH/wildfire-refactored/outputs/${RUN_NAME}_FULL_per_window.json"
[ -f "$CKPT" ] || { echo "ERROR: checkpoint missing: $CKPT"; exit 1; }

# Pick cache: own cache if it exists, else master-slice from v3_9ch_2000.
MASTER_ARGS=""
if [ -d "$SCRATCH/meteo_cache/${RUN_NAME#v3_9ch_enc21_}" ]; then : ; fi
OWN_CACHE="$SCRATCH/meteo_cache/v3_9ch_${RANGE_TAG}"
MASTER_CACHE="$SCRATCH/meteo_cache/v3_9ch_2000"
if [ -d "$OWN_CACHE" ] && [ -n "$(ls -A "$OWN_CACHE" 2>/dev/null)" ]; then
    SRC_CACHE="$OWN_CACHE"; USE_MASTER=0
    echo "  source cache: OWN $SRC_CACHE"
else
    SRC_CACHE="$MASTER_CACHE"
    USE_MASTER=$([ "$DATA_START" = "2000-05-01" ] && echo 0 || echo 1)
    echo "  source cache: MASTER $SRC_CACHE (time-slice=$USE_MASTER)"
fi

# Copy cache to local SSD — reading the 1.2 TB master cache straight
# from Lustre timed out at 9 h (random per-window reads). The SOTA eval
# that COMPLETED in 4.4 h copied its cache to SSD first; do the same.
LOCAL_METEO="$LOCAL_CACHE/meteo"
mkdir -p "$LOCAL_METEO"
echo "=== copy cache to local SSD: $SRC_CACHE ==="
t0=$SECONDS
for f in "$SRC_CACHE"/*; do
    [ -f "$f" ] || continue
    cp "$f" "$LOCAL_METEO/" || { echo "FATAL: cache copy failed"; exit 1; }
done
echo "  done in $((SECONDS - t0))s"
CACHE_DIR="$LOCAL_METEO"
if [ "$USE_MASTER" = "1" ]; then
    MASTER_ARGS="--master_cache_dir $LOCAL_METEO --master_data_start 2000-05-01"
fi

echo "============================================="
echo "  SCALING FULL EVAL  —  $RUN_NAME"
echo "  data_start : $DATA_START   ckpt: $CKPT"
echo "  cache      : $CACHE_DIR  $MASTER_ARGS"
echo "  out        : $OUT_JSON"
echo "============================================="

$PYTHON -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name "${RUN_NAME}_eval_full" \
    --eval_checkpoint "$CKPT" \
    --epochs 0 \
    --data_start "$DATA_START" --pred_start 2022-05-01 --pred_end "$PRED_END" \
    --channels "$CHANNELS" --in_days 21 \
    --decoder s2s_legacy --s2s_cache "$LOCAL_CACHE/s2s_decoder_cache.dat" --s2s_max_issue_lag 3 \
    --loss_fn focal --focal_alpha 0.25 --focal_gamma 2.0 \
    --hard_neg_fraction 0.5 --neg_ratio 20 --neg_buffer 2 \
    --batch_size 4096 --lr 1e-4 --weight_decay 0.01 --dropout 0.2 \
    --d_model 256 --patch_size 16 \
    --dilate_radius 14 --val_lift_k 5000 --val_lift_sample_wins 9999 \
    --fire_season_only --cluster_eval --decoder_ctx \
    --cache_dir "$CACHE_DIR" $MASTER_ARGS --chunk_patches 2000 --num_workers 4 \
    --log_interval 200 --skip_forecast \
    --label_fusion --nfdb_min_size_ha 1.0 \
    --fire_clim_dir data/fire_clim_annual_nbac \
    --save_per_window_json "$OUT_JSON"

PY_EXIT=$?
echo "=== Done $(date) exit=$PY_EXIT ==="
ls -lh "$OUT_JSON" || true
exit $PY_EXIT
