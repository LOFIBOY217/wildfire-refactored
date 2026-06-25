#!/bin/bash
#SBATCH --job-name=wf-dmodel-eval
#SBATCH --gpus-per-node=1
#SBATCH --time=0-06:00:00
#SBATCH --mem=400G
#SBATCH --output=/scratch/jiaqi217/logs/dmodel_eval_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/dmodel_eval_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# Fig 4b: full-window (~435-win) eval for the d_model sweep checkpoints
# (climsim variants). The dmodel-sweep TRAINING jobs only logged a
# 20-window Lift sample; this re-evaluates each checkpoint with the SAME
# full-window protocol as eval_scaling_full / eval_encsweep_full.
#
# d_model and layer count are coupled (matches the training scripts):
#   d_model=128 -> enc/dec_layers=4   (under-parameterized probe)
#   d_model=384 -> enc/dec_layers=6
#   d_model=512 -> enc/dec_layers=6
# nhead=8 throughout. climate_similarity_csv is a TRAIN-time sample
# weight only and is intentionally omitted at eval.
#
# Cache is read straight from Lustre (SKIP_COPY) with a late --data_start
# so only ~4y of frames are touched; frame indexing uses --master_data_start
# matched to the range (2014 for 12y, 2000 for 22y).
#
# Usage (submit one per point; 4 points total):
#   D_MODEL=128 RANGE=12y_2014 sbatch slurm/eval_dmodel_full_narval.sh
#   D_MODEL=384 RANGE=12y_2014 sbatch slurm/eval_dmodel_full_narval.sh
#   D_MODEL=384 RANGE=2000     sbatch slurm/eval_dmodel_full_narval.sh
#   D_MODEL=512 RANGE=2000     sbatch slurm/eval_dmodel_full_narval.sh
# (d_model=256 == SOTA, already has its FULL per-window JSON.)
# Output: outputs/v3_9ch_enc21_${RANGE}_climsim_dm${D_MODEL}_FULL_per_window.json
# ----------------------------------------------------------------

set -uo pipefail
D_MODEL=${D_MODEL:?Must set D_MODEL (128, 384, 512)}
RANGE=${RANGE:?Must set RANGE (12y_2014 or 2000)}
DATA_START=${DATA_START:-2021-05-01}
PRED_END=${PRED_END:-2025-09-23}

# Layers scale with d_model, matching the training recipe.
if [ "$D_MODEL" = "128" ]; then
    ENC_L=4; DEC_L=4
else
    ENC_L=6; DEC_L=6
fi
NHEAD=8

# Cache + master_data_start per data range.
if [ "$RANGE" = "2000" ]; then
    MASTER_CACHE="$SCRATCH/meteo_cache/v3_9ch_2000"; MASTER_DS="2000-05-01"
else
    MASTER_CACHE="$SCRATCH/meteo_cache/v3_9ch_12y_2014"; MASTER_DS="2014-05-01"
fi

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
RUN_NAME="v3_9ch_enc21_${RANGE}_climsim_dm${D_MODEL}"
CKPT="$SCRATCH/wildfire-refactored/checkpoints/${RUN_NAME}/best_model.pt"
OUT_JSON="$SCRATCH/wildfire-refactored/outputs/${RUN_NAME}_FULL_per_window.json"
[ -f "$CKPT" ] || { echo "ERROR: checkpoint missing: $CKPT"; exit 1; }
[ -d "$MASTER_CACHE" ] || { echo "ERROR: master cache missing: $MASTER_CACHE"; exit 1; }
CACHE_DIR="$MASTER_CACHE"
echo "=== SKIP_COPY: reading cache from Lustre directly: $CACHE_DIR ==="

echo "============================================="
echo "  DMODEL FULL EVAL  —  $RUN_NAME"
echo "  d_model=$D_MODEL  enc/dec_layers=$ENC_L/$DEC_L  nhead=$NHEAD"
echo "  data_start=$DATA_START  master_data_start=$MASTER_DS"
echo "  ckpt: $CKPT"
echo "  out : $OUT_JSON"
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
    --d_model "$D_MODEL" --nhead "$NHEAD" --enc_layers "$ENC_L" --dec_layers "$DEC_L" --patch_size 16 \
    --dilate_radius 14 --val_lift_k 5000 --val_lift_sample_wins 9999 \
    --fire_season_only --cluster_eval --decoder_ctx \
    --cache_dir "$CACHE_DIR" \
    --master_cache_dir "$CACHE_DIR" --master_data_start "$MASTER_DS" \
    --chunk_patches 2000 --num_workers 4 \
    --log_interval 200 --skip_forecast \
    --label_fusion --nfdb_min_size_ha 1.0 \
    --fire_clim_dir data/fire_clim_annual_nbac \
    --save_per_window_json "$OUT_JSON"

PY_EXIT=$?
echo "=== Done $(date) exit=$PY_EXIT ==="
ls -lh "$OUT_JSON" || true
exit $PY_EXIT
