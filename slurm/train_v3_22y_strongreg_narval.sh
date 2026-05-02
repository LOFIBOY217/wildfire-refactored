#!/bin/bash
#SBATCH --job-name=wf-22y-reg
#SBATCH --gpus-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=750G
#SBATCH --output=/scratch/jiaqi217/logs/train_22y_reg_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/train_22y_reg_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# DIAGNOSTIC: 22y x 4 enc with STRONG REGULARIZATION
#
# Hypothesis: 22y_ep1 (1 epoch) experiment failed → not pure overfit.
# But 22y might still benefit from stronger regularization (allows
# more epochs without diverging on val).
#
# Recipe difference from default:
#   --weight_decay 0.01 → 0.05  (5x stronger L2)
#   --dropout 0.2 → 0.4         (2x dropout)
#   --epochs 4 (same)
#
# RUN_NAME: v3_9ch_enc${ENC}_2000_strongreg
# Output: /checkpoints/v3_9ch_enc${ENC}_2000_strongreg/best_model.pt
#
# Usage:
#   ENC=14 sbatch slurm/train_v3_22y_strongreg_narval.sh
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

CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,population,slope,burn_age"
CACHE_DIR_LUSTRE="$SCRATCH/meteo_cache/v3_9ch_2000"
RUN_NAME="v3_9ch_enc${ENC}_2000_strongreg"

# 2026-05-02: copy 22y meteo to SSD (Lustre fancy-indexing 100x slower).
LOCAL_METEO="$LOCAL_CACHE/meteo"
mkdir -p "$LOCAL_METEO"
echo "=== copy 22y meteo to local SSD (~750 GB) ==="
for f in "$CACHE_DIR_LUSTRE"/*; do
    [ -f "$f" ] || continue
    cp "$f" "$LOCAL_METEO/" || { echo "FATAL"; exit 1; }
done
TRAIN_CACHE_DIR="$LOCAL_METEO"

if [ ! -d "$TRAIN_CACHE_DIR" ] || [ -z "$(ls -A "$TRAIN_CACHE_DIR" 2>/dev/null)" ]; then
    echo "ERROR: $TRAIN_CACHE_DIR is empty or missing"; exit 1
fi

echo "============================================="
echo "  V3 9ch enc${ENC} 22y with STRONG REGULARIZATION"
echo "  Run: $RUN_NAME"
echo "  weight_decay=0.05 (vs 0.01)"
echo "  dropout=0.4 (vs 0.2)"
echo "  epochs=4 (unchanged)"
echo "============================================="

$PYTHON -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name "$RUN_NAME" \
    --data_start 2000-05-01 --pred_start 2022-05-01 --pred_end 2025-10-31 \
    --channels "$CHANNELS" --in_days "$ENC" \
    --decoder s2s_legacy --s2s_cache "$LOCAL_CACHE/s2s_decoder_cache.dat" --s2s_max_issue_lag 3 \
    --loss_fn focal --focal_alpha 0.25 --focal_gamma 2.0 \
    --hard_neg_fraction 0.5 --neg_ratio 20 --neg_buffer 2 \
    --batch_size 4096 --epochs 4 --lr 1e-4 \
    --weight_decay 0.05 --dropout 0.4 \
    --d_model 256 --nhead 8 --enc_layers 4 --dec_layers 4 --patch_size 16 \
    --dilate_radius 14 --val_lift_k 5000 --val_lift_sample_wins 20 \
    --fire_season_only --cluster_eval --decoder_ctx --load_train_to_ram \
    --cache_dir "$TRAIN_CACHE_DIR" --chunk_patches 2000 --num_workers 4 \
    --log_interval 200 --skip_forecast \
    --label_fusion --nfdb_min_size_ha 1.0 \
    --fire_clim_dir data/fire_clim_annual_nbac \
    --save_per_window_json "$SCRATCH/wildfire-refactored/outputs/${RUN_NAME}_per_window.json"

PY_EXIT=$?
echo "=== Done $(date) exit=$PY_EXIT ==="
exit $PY_EXIT
