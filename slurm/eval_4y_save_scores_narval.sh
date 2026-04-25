#!/bin/bash
#SBATCH --job-name=wf-eval-save-scores
#SBATCH --gpus-per-node=1
#SBATCH --time=0-08:00:00
#SBATCH --mem=400G
#SBATCH --output=/scratch/jiaqi217/logs/eval_save_scores_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/eval_save_scores_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# EVAL-ONLY: load 4y best_model.pt × {ENC} and dump per-window
# per-pixel scores so we can compute novel-ignition / per-region /
# per-month lift offline without re-running model inference.
#
# Usage:
#   ENC=14 sbatch slurm/eval_4y_save_scores_narval.sh
#   ENC=21 sbatch slurm/eval_4y_save_scores_narval.sh
#   ENC=28 sbatch slurm/eval_4y_save_scores_narval.sh
#   ENC=35 sbatch slurm/eval_4y_save_scores_narval.sh
#
# Outputs:
#   outputs/window_scores/v3_9ch_enc{ENC}_4y_2018/window_NNNN_DATE.npz
#   (one file per val window, ~2-5 MB each, total 20 files)
# ----------------------------------------------------------------

set -uo pipefail
ENC=${ENC:?Must set ENC (14, 21, 28, 35)}

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

# Copy 4y meteo cache to local SSD (4y total ~320GB, ~10min)
CACHE_DIR_LUSTRE="$SCRATCH/meteo_cache/v3_9ch_4y_2018"
LOCAL_METEO="$LOCAL_CACHE/meteo"
mkdir -p "$LOCAL_METEO"
echo "=== copy meteo to local SSD ==="
t0=$SECONDS
for f in "$CACHE_DIR_LUSTRE"/*; do
    [ -f "$f" ] || continue
    fname=$(basename "$f")
    sz=$(du -h "$f" | cut -f1)
    echo "  copy $fname ($sz)"
    cp "$f" "$LOCAL_METEO/" || { echo "FATAL: copy failed"; exit 1; }
done
echo "  done in $((SECONDS - t0))s"

CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,population,slope,burn_age"
RUN_NAME="v3_9ch_enc${ENC}_4y_2018"
CKPT="$SCRATCH/wildfire-refactored/checkpoints/$RUN_NAME/best_model.pt"
SCORES_DIR="$SCRATCH/wildfire-refactored/outputs/window_scores/$RUN_NAME"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT"
    exit 1
fi

mkdir -p "$SCORES_DIR"
echo "============================================="
echo "  EVAL-ONLY: $RUN_NAME"
echo "  ckpt: $CKPT"
echo "  out : $SCORES_DIR"
echo "============================================="

$PYTHON -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name "${RUN_NAME}_eval" \
    --eval_checkpoint "$CKPT" \
    --save_window_scores_dir "$SCORES_DIR" \
    --data_start 2018-05-01 --pred_start 2022-05-01 --pred_end 2025-10-31 \
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
ls -lh "$SCORES_DIR/"
