#!/bin/bash
#SBATCH --job-name=wf-v3-9s2s
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=256G
#SBATCH --output=/scratch/jiaqi217/logs/train_v3_9ch_s2s_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/train_v3_9ch_s2s_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca
#SBATCH --account=def-inghaw

# ----------------------------------------------------------------
# V3 9-Channel Training — upgraded with S2S decoder
#
# Same 9 channels as Plan C, but with:
#   - s2s_legacy decoder (was: random)
#   - decoder_ctx (patch-mean spatial + lead/season)
#   - dual-path embedding
#   - batch_size 1024 (was: 4096)
#
# Reuses existing 288GB memmap on scratch (no rebuild needed!)
# ----------------------------------------------------------------

set -euo pipefail

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}

[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 proj/9.4.1 eccodes/2.31.0

cd "$SCRATCH/wildfire-refactored"
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

source slurm/lib_copy_cache.sh
copy_venv $SCRATCH/venv-wildfire

ts "=== PREFLIGHT ==="
ts "Node: $(hostname)"
$PYTHON -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())" || exit 1
ts "=== PREFLIGHT OK ==="

# Copy S2S decoder cache to SSD
LOCAL_CACHE=$SLURM_TMPDIR/cache
mkdir -p "$LOCAL_CACHE"
copy_s2s_cache "$SCRATCH/meteo_cache" "$LOCAL_CACHE"

# Reuse existing 9ch memmap from scratch (288GB, already built)
SCRATCH_CACHE="$SCRATCH/meteo_cache/v3_9ch"

CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,population,slope,burn_age"

ts "============================================="
ts "  V3 9ch + S2S Legacy Decoder"
ts "  channels: $CHANNELS"
ts "  Reusing memmap: $SCRATCH_CACHE"
ts "============================================="

$PYTHON -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name v3_9ch_s2s \
    --data_start 2018-05-01 \
    --pred_start 2022-05-01 \
    --pred_end 2025-10-31 \
    --channels "$CHANNELS" \
    --decoder s2s_legacy \
    --s2s_cache "$LOCAL_CACHE/s2s_decoder_cache.dat" \
    --s2s_max_issue_lag 3 \
    --loss_fn focal \
    --focal_alpha 0.25 \
    --focal_gamma 2.0 \
    --hard_neg_fraction 0.5 \
    --neg_ratio 20 \
    --neg_buffer 2 \
    --batch_size 1024 \
    --epochs 8 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --dropout 0.1 \
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
    --cache_dir "$SCRATCH_CACHE" \
    --chunk_patches 2000 \
    --num_workers 4 \
    --log_interval 200 \
    --skip_forecast

ts "Exit: $?"
