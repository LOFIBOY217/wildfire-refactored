#!/bin/bash
#SBATCH --job-name=wf-encsweep-eval
#SBATCH --gpus-per-node=1
#SBATCH --time=0-04:00:00
#SBATCH --mem=400G
#SBATCH --output=/scratch/jiaqi217/logs/encsweep_eval_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/encsweep_eval_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# Fig 4c: full-window (~435-win) eval for the 12y encoder-length
# sweep checkpoints. The encsweep TRAINING jobs only run a 20-window
# Lift sample (--val_lift_sample_wins 20) and may not even write a
# per-window JSON if training NaN-stops early. So we re-evaluate each
# checkpoint here with the SAME full-window protocol used for the
# data-scaling points (eval_scaling_full_narval.sh).
#
# All 8 points share the SAME val windows (2022->2025); only --in_days
# differs. Each checkpoint was trained on the 12y cache
# (v3_9ch_12y_2014, frame 0 = 2014-05-01). We read it straight from
# Lustre (SKIP_COPY) with a late --data_start so only ~4y of frames
# are touched (Lustre IO stays small, no 14h timeout). Frame indexing
# uses --master_data_start 2014-05-01 so the late data_start lines up.
#
# Usage (submit one per encoder length):
#   ENC=7  sbatch slurm/eval_encsweep_full_narval.sh
#   ENC=10 sbatch slurm/eval_encsweep_full_narval.sh
#   ENC=14 sbatch slurm/eval_encsweep_full_narval.sh
#   ENC=28 sbatch slurm/eval_encsweep_full_narval.sh
#   ENC=35 sbatch slurm/eval_encsweep_full_narval.sh
#   ENC=42 sbatch slurm/eval_encsweep_full_narval.sh
#   ENC=56 sbatch slurm/eval_encsweep_full_narval.sh
# (enc21 == SOTA, already has its FULL per-window JSON.)
# Output: outputs/v3_9ch_enc${ENC}_12y_2014_FULL_per_window.json
# ----------------------------------------------------------------

set -uo pipefail
ENC=${ENC:?Must set ENC (e.g. 7, 10, 14, 28, 35, 42, 56)}
DATA_START=${DATA_START:-2021-05-01}
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
RUN_NAME="v3_9ch_enc${ENC}_12y_2014"
CKPT="$SCRATCH/wildfire-refactored/checkpoints/${RUN_NAME}/best_model.pt"
OUT_JSON="$SCRATCH/wildfire-refactored/outputs/${RUN_NAME}_FULL_per_window.json"
[ -f "$CKPT" ] || { echo "ERROR: checkpoint missing: $CKPT"; exit 1; }

# 12y master cache; read straight from Lustre with a late data_start.
MASTER_CACHE="$SCRATCH/meteo_cache/v3_9ch_12y_2014"
[ -d "$MASTER_CACHE" ] || { echo "ERROR: master cache missing: $MASTER_CACHE"; exit 1; }
CACHE_DIR="$MASTER_CACHE"
echo "=== SKIP_COPY: reading 12y cache from Lustre directly: $CACHE_DIR ==="

echo "============================================="
echo "  ENCSWEEP FULL EVAL  —  $RUN_NAME  (in_days=$ENC)"
echo "  data_start : $DATA_START   ckpt: $CKPT"
echo "  cache      : $CACHE_DIR  (master_data_start 2014-05-01)"
echo "  out        : $OUT_JSON"
echo "============================================="

$PYTHON -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name "${RUN_NAME}_eval_full" \
    --eval_checkpoint "$CKPT" \
    --epochs 0 \
    --data_start "$DATA_START" --pred_start 2022-05-01 --pred_end "$PRED_END" \
    --channels "$CHANNELS" --in_days "$ENC" \
    --decoder s2s_legacy --s2s_cache "$LOCAL_CACHE/s2s_decoder_cache.dat" --s2s_max_issue_lag 3 \
    --loss_fn focal --focal_alpha 0.25 --focal_gamma 2.0 \
    --hard_neg_fraction 0.5 --neg_ratio 20 --neg_buffer 2 \
    --batch_size 4096 --lr 1e-4 --weight_decay 0.01 --dropout 0.2 \
    --d_model 256 --patch_size 16 \
    --dilate_radius 14 --val_lift_k 5000 --val_lift_sample_wins 9999 \
    --fire_season_only --cluster_eval --decoder_ctx \
    --cache_dir "$CACHE_DIR" \
    --master_cache_dir "$CACHE_DIR" --master_data_start 2014-05-01 \
    --chunk_patches 2000 --num_workers 4 \
    --log_interval 200 --skip_forecast \
    --label_fusion --nfdb_min_size_ha 1.0 \
    --fire_clim_dir data/fire_clim_annual_nbac \
    --save_per_window_json "$OUT_JSON"

PY_EXIT=$?
echo "=== Done $(date) exit=$PY_EXIT ==="
ls -lh "$OUT_JSON" || true
exit $PY_EXIT
