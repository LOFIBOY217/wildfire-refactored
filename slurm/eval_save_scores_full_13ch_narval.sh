#!/bin/bash
#SBATCH --job-name=wf-save-13ch
#SBATCH --gpus-per-node=1
#SBATCH --time=0-08:00:00
#SBATCH --mem=400G
#SBATCH --output=/scratch/jiaqi217/logs/save_scores_full_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/save_scores_full_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# 13ch FULL eval — save per-pixel scores on ALL val windows.
# Mirror of eval_save_scores_full_narval.sh but for 13ch cache + ckpt.
#
# 13 channels = 9ch + {deep_soil, precip_def, NDVI, burn_count}
#
# Usage:
#   RANGE=4y ENC=14 sbatch slurm/eval_save_scores_full_13ch_narval.sh
#   ...
# ----------------------------------------------------------------

set -uo pipefail
ENC=${ENC:?Must set ENC}
RANGE=${RANGE:-4y}

case "$RANGE" in
    22y) DATA_START=2000-05-01; CACHE_LUSTRE=v3_13ch_2000;     RUN_TAG=2000 ;;
    12y) DATA_START=2014-05-01; CACHE_LUSTRE=v3_13ch_12y_2014; RUN_TAG=12y_2014 ;;
    4y)  DATA_START=2018-05-01; CACHE_LUSTRE=v3_13ch_4y_2018;  RUN_TAG=4y_2018 ;;
    *) echo "ERROR: unknown RANGE=$RANGE"; exit 1 ;;
esac

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

CACHE_DIR_LUSTRE="$SCRATCH/meteo_cache/$CACHE_LUSTRE"
LOCAL_METEO="$LOCAL_CACHE/meteo"
mkdir -p "$LOCAL_METEO"
echo "=== copy 13ch meteo to local SSD ($CACHE_LUSTRE) ==="
t0=$SECONDS
for f in "$CACHE_DIR_LUSTRE"/*; do
    [ -f "$f" ] || continue
    fname=$(basename "$f")
    sz=$(du -h "$f" | cut -f1)
    echo "  copy $fname ($sz)"
    cp "$f" "$LOCAL_METEO/" || { echo "FATAL"; exit 1; }
done
echo "  done in $((SECONDS - t0))s"

CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,deep_soil,precip_def,NDVI,population,slope,burn_age,burn_count"
SUFFIX=${SUFFIX:-}
RUN_NAME="v3_13ch_enc${ENC}_${RUN_TAG}${SUFFIX}"
CKPT="$SCRATCH/wildfire-refactored/checkpoints/$RUN_NAME/best_model.pt"
SCORES_DIR="$SCRATCH/wildfire-refactored/outputs/window_scores_full/$RUN_NAME"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: ckpt missing: $CKPT"; exit 1
fi
mkdir -p "$SCORES_DIR"

echo "============================================="
echo "  FULL EVAL 13ch: $RUN_NAME"
echo "  ckpt: $CKPT"
echo "  out : $SCORES_DIR"
echo "============================================="

$PYTHON -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name "${RUN_NAME}_eval_full" \
    --eval_checkpoint "$CKPT" \
    --save_window_scores_dir "$SCORES_DIR" \
    --full_val \
    --data_start "$DATA_START" --pred_start 2022-05-01 --pred_end 2025-10-31 \
    --channels "$CHANNELS" --in_days "$ENC" \
    --decoder s2s_legacy --s2s_cache "$LOCAL_CACHE/s2s_decoder_cache.dat" --s2s_max_issue_lag 3 \
    --loss_fn focal --focal_alpha 0.25 --focal_gamma 2.0 \
    --batch_size 4096 --epochs 0 \
    --d_model 256 --nhead 8 --enc_layers 4 --dec_layers 4 --patch_size 16 \
    --dilate_radius 14 --val_lift_k 5000 --val_lift_sample_wins 20 \
    --fire_season_only --cluster_eval --decoder_ctx \
    --cache_dir "$LOCAL_METEO" --chunk_patches 2000 --num_workers 4 \
    --skip_forecast \
    --label_fusion --nfdb_min_size_ha 1.0 \
    --fire_clim_dir data/fire_clim_annual_nbac

echo "=== done $(date) ==="
ls "$SCORES_DIR/" | wc -l
echo "saved $(ls $SCORES_DIR/ | wc -l) window score files"
