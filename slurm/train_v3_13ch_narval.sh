#!/bin/bash
#SBATCH --job-name=wf-v3-13ch
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=400G
#SBATCH --output=/scratch/jiaqi217/logs/train_v3_13ch_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/train_v3_13ch_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca
#SBATCH --account=def-inghaw

# ----------------------------------------------------------------
# V3 13-Channel Training (everything ready NOW, no u10/v10/CAPE):
#
# ENCODER (13ch → enc_dim = 256×13 = 3328):
#   Weather/atmosphere (4):   FWI, 2t, 2d, tcw
#   Surface/soil (3):         sm20, deep_soil (swvl2), precip_def (30d)
#   Vegetation (1):           NDVI
#   Spatial prior (3):        fire_clim, population, slope
#   Fuel history (2):         burn_age, burn_count
#
# DECODER (s2s_legacy → dec_dim = 9 + decoder_ctx):
#   S2S ECMWF forecast (6ch): 2t, 2d, tcw, sm20, st20, VPD
#   + issue_age, is_fallback, is_missing (3 metadata dims)
#   + decoder_ctx: 5 static patches + 4 lead/season encoding
#
# SSD: venv + S2S cache + meteo cache all on local NVMe
# ----------------------------------------------------------------

set -uo pipefail
# NOTE: not using -e so that copy-back runs even if training crashes

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}

[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 proj/9.4.1 eccodes/2.31.0

cd "$SCRATCH/wildfire-refactored"
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

# ---- SSD setup ----
source slurm/lib_copy_cache.sh

copy_venv $SCRATCH/venv-wildfire

ts "=== PREFLIGHT ==="
ts "Node: $(hostname)"
$PYTHON -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())" || exit 1
ts "=== PREFLIGHT OK ==="

# Copy S2S decoder cache to local SSD (15G, needed for s2s_legacy decoder)
LOCAL_CACHE=$SLURM_TMPDIR/cache
mkdir -p "$LOCAL_CACHE"
copy_s2s_cache "$SCRATCH/meteo_cache" "$LOCAL_CACHE"

# Check local disk space (13ch memmap ~446GB, need room)
ts "=== LOCAL DISK ==="
df -h $SLURM_TMPDIR
ts "================="

CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,deep_soil,precip_def,NDVI,population,slope,burn_age,burn_count"

ts "============================================="
ts "  V3 13-Channel Training (s2s_legacy decoder)"
ts "  channels: $CHANNELS"
ts "============================================="

# Use local SSD for cache_dir → meteo memmap built on NVMe (fast I/O)
$PYTHON -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name v3_13ch \
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
    --cache_dir "$LOCAL_CACHE" \
    --chunk_patches 2000 \
    --num_workers 4 \
    --log_interval 200 \
    --skip_forecast

TRAIN_EXIT=$?

# Copy meteo cache back to scratch for future resume/eval
SCRATCH_CACHE="$SCRATCH/meteo_cache/v3_13ch"
mkdir -p "$SCRATCH_CACHE"

ts "=== COPYING CACHE BACK TO SCRATCH ==="
# Stats (tiny files)
cp "$LOCAL_CACHE"/meteo_v3_p16_C13_stats.npy "$SCRATCH_CACHE/" 2>/dev/null || true
cp "$LOCAL_CACHE"/*_stats.npy "$SCRATCH_CACHE/" 2>/dev/null || true
cp "$LOCAL_CACHE"/fire_*.npy "$SCRATCH_CACHE/" 2>/dev/null || true
cp "$LOCAL_CACHE"/norm_stats*.npy "$SCRATCH_CACHE/" 2>/dev/null || true

# Meteo memmap (large, ~600GB — essential for eval/resume without rebuilding)
PF_DAT=$(ls "$LOCAL_CACHE"/meteo_v3_*_pf.dat 2>/dev/null | head -1)
if [ -n "$PF_DAT" ]; then
    PF_SIZE=$(du -h "$PF_DAT" | cut -f1)
    ts "  Copying memmap $PF_SIZE → scratch (this takes ~10-30min)..."
    timeout 3600 cp "$PF_DAT" "$SCRATCH_CACHE/" && \
        ts "  Memmap copied OK" || \
        ts "  WARNING: memmap copy failed/timed out (eval will rebuild)"
else
    ts "  No pf.dat found on SSD"
fi
ts "=== COPY BACK COMPLETE ==="

ts "Exit: $TRAIN_EXIT"
exit $TRAIN_EXIT
