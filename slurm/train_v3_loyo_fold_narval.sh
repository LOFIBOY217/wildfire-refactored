#!/bin/bash
#SBATCH --job-name=wf-loyo
#SBATCH --gpus-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=480G
#SBATCH --output=/scratch/jiaqi217/logs/loyo_fold_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/loyo_fold_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca
#SBATCH --account=def-inghaw

# ----------------------------------------------------------------
# Forward-chaining LOYO fold — one held-out year.
#
# For each VAL_YEAR Y in {2020..2024}:
#   train = 2014-05-01 .. (Y-1)-12-31   (strictly past — no leakage)
#   val   = Y-05-01 .. Y-10-31          (held-out fire season)
#
# Together the 5 folds give an honest leave-one-year-out estimate of
# Lift@5000 / Lift@30km without the 2022-05-01 single-split assumption.
#
# Usage:
#   VAL_YEAR=2020 sbatch slurm/train_v3_loyo_fold_narval.sh
#   ... (one sbatch per year; orchestrator: scripts/submit_loyo_all.sh)
#
# Reuses the v3_9ch_2000 master cache to avoid 36h per-fold rebuild.
# ----------------------------------------------------------------

set -uo pipefail
VAL_YEAR=${VAL_YEAR:?Must set VAL_YEAR (2020..2024)}
ENC=${ENC:-21}
DATA_START=${DATA_START:-2014-05-01}
PRED_START="${VAL_YEAR}-05-01"
PRED_END="${VAL_YEAR}-10-31"
RANGE_TAG="loyo_val${VAL_YEAR}"
RUN_NAME="v3_9ch_enc${ENC}_${RANGE_TAG}"

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

CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,population,slope,burn_age"
MASTER_CACHE_LUSTRE="$SCRATCH/meteo_cache/v3_9ch_2000"
MASTER_DATA_START="2000-05-01"

if [ ! -d "$MASTER_CACHE_LUSTRE" ] || [ -z "$(ls -A "$MASTER_CACHE_LUSTRE" 2>/dev/null)" ]; then
    echo "ERROR: master cache $MASTER_CACHE_LUSTRE missing"; exit 1
fi

LOCAL_METEO="$LOCAL_CACHE/meteo"
mkdir -p "$LOCAL_METEO"
echo "=== copy 9ch master meteo to local SSD (~960 GB) ==="
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
echo "  LOYO fold: VAL_YEAR=$VAL_YEAR (held-out fire season)"
echo "  train  : $DATA_START .. $((VAL_YEAR - 1))-12-31"
echo "  val    : $PRED_START .. $PRED_END"
echo "  master : $LOCAL_METEO  (start=$MASTER_DATA_START)"
echo "  run    : $RUN_NAME"
echo "============================================="

# Smaller training ranges (Y=2020 → 6y) easily fit RAM; default on.
LOAD_TRAIN_TO_RAM=${LOAD_TRAIN_TO_RAM:-1}
RAM_FLAG=""
[ "$LOAD_TRAIN_TO_RAM" = "1" ] && RAM_FLAG="--load_train_to_ram"
echo "  load_train_to_ram = $LOAD_TRAIN_TO_RAM  (flag: '$RAM_FLAG')"

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
    --fire_season_only --cluster_eval --decoder_ctx $RAM_FLAG \
    --master_cache_dir "$LOCAL_METEO" --master_data_start "$MASTER_DATA_START" \
    --chunk_patches 2000 --num_workers 4 \
    --log_interval 200 --skip_forecast \
    --label_fusion --nfdb_min_size_ha 1.0 \
    --fire_clim_dir data/fire_clim_annual_nbac \
    --wandb_project wildfire-s2s \
    --wandb_tags "9ch,enc${ENC},loyo,val${VAL_YEAR}" \
    --save_per_window_json "$SCRATCH/wildfire-refactored/outputs/loyo/${RUN_NAME}_per_window.json"

PY_EXIT=$?
echo "=== Done: $(date) VAL_YEAR=$VAL_YEAR exit=$PY_EXIT ==="
exit $PY_EXIT
