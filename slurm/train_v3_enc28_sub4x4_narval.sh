#!/bin/bash
#SBATCH --job-name=wf-v3-sub4
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=500G
#SBATCH --output=/scratch/jiaqi217/logs/train_v3_enc28_sub4x4_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/train_v3_enc28_sub4x4_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca
#SBATCH --account=def-inghaw

# ----------------------------------------------------------------
# Best combo: enc28 + subpatch4x4 decoder (128 dim) + anti-overfit
#
# Why: enc28=6.43x is best encoder. s2s_legacy=9dim too compressed.
# subpatch4x4=128dim (14x more signal) and fits on SSD (309GB).
# full-patch 2048dim too slow (4.9TB Lustre random IO).
# ----------------------------------------------------------------

set -uo pipefail
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

# Copy subpatch4x4 cache to SSD (309GB, fits on most nodes)
LOCAL_CACHE=$SLURM_TMPDIR/cache
mkdir -p "$LOCAL_CACHE"

S2S_SUB4_SCRATCH="$SCRATCH/meteo_cache/s2s_subpatch4x4_cache.dat"
S2S_SUB4_LOCAL="$LOCAL_CACHE/s2s_subpatch4x4_cache.dat"

SSD_AVAIL_KB=$(df --output=avail $SLURM_TMPDIR | tail -1)
SSD_AVAIL_GB=$((SSD_AVAIL_KB / 1048576))
echo "SSD available: ${SSD_AVAIL_GB}GB"

if [ "$SSD_AVAIL_GB" -ge 400 ]; then
    echo "Copying subpatch4x4 cache (309G) to SSD..."
    cp "$S2S_SUB4_SCRATCH" "$S2S_SUB4_LOCAL" && echo "  OK" || {
        echo "  WARN: copy failed, using Lustre"
        S2S_SUB4_LOCAL="$S2S_SUB4_SCRATCH"
    }
    cp "${S2S_SUB4_SCRATCH}.dates.npy" "${S2S_SUB4_LOCAL}.dates.npy" 2>/dev/null || true
else
    echo "SSD too small (${SSD_AVAIL_GB}GB), using Lustre"
    S2S_SUB4_LOCAL="$S2S_SUB4_SCRATCH"
fi

CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,population,slope,burn_age"

echo "============================================="
echo "  V3 enc28 + subpatch4x4 decoder (128 dim)"
echo "============================================="

$PYTHON -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name v3_enc28_sub4x4 \
    --data_start 2018-05-01 --pred_start 2022-05-01 --pred_end 2025-10-31 \
    --channels "$CHANNELS" --in_days 28 \
    --decoder s2s --s2s_full_cache "$S2S_SUB4_LOCAL" --dec_dim 128 \
    --loss_fn focal --focal_alpha 0.25 --focal_gamma 2.0 \
    --hard_neg_fraction 0.5 --neg_ratio 20 --neg_buffer 2 \
    --batch_size 4096 --epochs 4 --lr 1e-4 --weight_decay 0.01 --dropout 0.2 \
    --d_model 256 --nhead 8 --enc_layers 4 --dec_layers 4 --patch_size 16 \
    --dilate_radius 14 --val_lift_k 5000 --val_lift_sample_wins 20 \
    --fire_season_only --cluster_eval --decoder_ctx --load_train_to_ram \
    --cache_dir "$SCRATCH/meteo_cache/v3_9ch" --chunk_patches 2000 --num_workers 4 \
    --log_interval 200 --skip_forecast

PY_EXIT=$?
echo "=== Done: $(date) exit=$PY_EXIT ==="
exit $PY_EXIT
