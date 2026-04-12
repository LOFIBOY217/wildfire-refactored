#!/bin/bash
#SBATCH --job-name=wf-v3-fp14
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=500G
#SBATCH --output=/scratch/jiaqi217/logs/train_v3_fp_enc14_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/train_v3_fp_enc14_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca
#SBATCH --account=def-inghaw

# ----------------------------------------------------------------
# Full-patch S2S decoder (2048 dim) + 14-day encoder + anti-overfit
#
# Key change: decoder uses ALL 16×16×8=2048 dims from S2S forecast
# instead of patch-mean 9 dims. Preserves spatial gradients (temp
# differences, wind patterns within each patch).
#
# 9ch encoder + full-patch decoder + decoder_ctx
# bs=4096, dropout=0.2, epochs=4
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

LOCAL_CACHE=$SLURM_TMPDIR/cache
mkdir -p "$LOCAL_CACHE"

# S2S full-patch cache is 4.9TB — too large for SSD, read from Lustre
S2S_FULL_CACHE="$SCRATCH/wildfire-refactored/data/s2s_full_patch_cache.dat"

CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,population,slope,burn_age"

echo "============================================="
echo "  V3 Full-Patch S2S Decoder (2048 dim) + Enc14"
echo "  channels: $CHANNELS"
echo "  decoder: s2s (full-patch 2048 dim)"
echo "  s2s_full_cache: $S2S_FULL_CACHE"
echo "============================================="

$PYTHON -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name v3_fullpatch_enc14 \
    --data_start 2018-05-01 \
    --pred_start 2022-05-01 \
    --pred_end 2025-10-31 \
    --channels "$CHANNELS" \
    --in_days 14 \
    --decoder s2s \
    --s2s_full_cache "$S2S_FULL_CACHE" \
    --dec_dim 2048 \
    --loss_fn focal \
    --focal_alpha 0.25 \
    --focal_gamma 2.0 \
    --hard_neg_fraction 0.5 \
    --neg_ratio 20 \
    --neg_buffer 2 \
    --batch_size 4096 \
    --epochs 4 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --dropout 0.2 \
    --d_model 256 \
    --nhead 8 \
    --enc_layers 4 \
    --dec_layers 4 \
    --patch_size 16 \
    --dilate_radius 14 \
    --val_lift_k 5000 \
    --val_lift_sample_wins 20 \
    --fire_season_only \
    --cluster_eval \
    --decoder_ctx \
    --load_train_to_ram \
    --cache_dir "$SCRATCH/meteo_cache/v3_9ch" \
    --chunk_patches 2000 \
    --num_workers 4 \
    --log_interval 200 \
    --skip_forecast

echo "=== Done: $(date) ==="
