#!/bin/bash
#SBATCH --job-name=wf-v3-8bce
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=400G
#SBATCH --output=/scratch/jiaqi217/logs/train_v3_8ch_bce_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/train_v3_8ch_bce_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca
#SBATCH --account=def-inghaw

# ----------------------------------------------------------------
# FAIR COMPARISON: V3 code + V2 channels (8ch) + BCE loss
#
# Purpose: isolate "fire_clim leakage" effect.
# - Same 8 channels as V2: FWI, 2t, 2d, FFMC, DMC, DC, BUI, fire_clim
# - V3 code: annual rolling fire_clim (leak-free), burn_age fix, etc.
# - BCE loss (same as V2), NOT focal
# - NO hard neg mining, NO decoder_ctx (match V2 as closely as possible)
# - s2s_legacy decoder (same as V2)
#
# If V2 got 7.35x because of fire_clim leakage, this run should be
# significantly lower. If it still gets ~7x, then V3's lower scores
# are caused by architecture changes (focal/hard_neg/decoder_ctx).
# ----------------------------------------------------------------

set -uo pipefail

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}

[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 proj/9.4.1 eccodes/2.31.0

cd "$SCRATCH/wildfire-refactored"
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

source $SCRATCH/venv-wildfire/bin/activate
PYTHON=python

$PYTHON -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())" || exit 1

# S2S cache to SSD
LOCAL_CACHE=$SLURM_TMPDIR/cache
mkdir -p "$LOCAL_CACHE"
cp $SCRATCH/meteo_cache/s2s_decoder_cache.dat "$LOCAL_CACHE/"
cp $SCRATCH/meteo_cache/s2s_decoder_cache.dat.dates.npy "$LOCAL_CACHE/"

# V2 channels: FWI, 2t, 2d, FFMC, DMC, DC, BUI, fire_clim
CHANNELS="FWI,2t,fire_clim,2d,FFMC,DMC,DC,BUI"

echo "============================================="
echo "  V3 Code + V2 Channels (8ch BCE) — Fair Comparison"
echo "  channels: $CHANNELS"
echo "============================================="

$PYTHON -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name v3_8ch_bce \
    --data_start 2018-05-01 \
    --pred_start 2022-05-01 \
    --pred_end 2025-10-31 \
    --channels "$CHANNELS" \
    --decoder s2s_legacy \
    --s2s_cache "$LOCAL_CACHE/s2s_decoder_cache.dat" \
    --s2s_max_issue_lag 3 \
    --loss_fn bce \
    --neg_ratio 20 \
    --neg_buffer 0 \
    --hard_neg_fraction 0.0 \
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
    --load_train_to_ram \
    --cache_dir "$LOCAL_CACHE" \
    --chunk_patches 2000 \
    --num_workers 4 \
    --log_interval 200 \
    --skip_forecast

echo "=== Done: $(date) ==="
