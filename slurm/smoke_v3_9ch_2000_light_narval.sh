#!/bin/bash
#SBATCH --job-name=wf-smoke-9ch-light
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/jiaqi217/logs/smoke_9ch_light_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/smoke_9ch_light_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
#   LIGHTWEIGHT SMOKE (D variant): v3_9ch_2000 cache validation
# ----------------------------------------------------------------
# Goal: validate 1TB cache loads + model runs end-to-end, with minimal
# resources so it fits backfill / fast queues.
#
# Tricks vs heavy smoke:
#   - data_start=2024-05-01 → only 1 year of train data (vs 22y)
#   - pred_end=2025-07-31 → only 3 months val
#   - val_lift_sample_wins=3 (vs 20 prod)
#   - batch_size=1024 (vs 4096)
#   - no --load_train_to_ram (rely on memmap for 64G mem)
#   - num_workers=2
#
# This WON'T produce a production-quality model, but it confirms:
#   1. Cache memmap loads correctly
#   2. Training forward+backward passes execute
#   3. Val pipeline produces a Lift number (maybe ~2-4x on tiny train)
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

CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,population,slope,burn_age"
CACHE_DIR_2000="$SCRATCH/meteo_cache/v3_9ch_2000"

if [ ! -d "$CACHE_DIR_2000" ] || [ -z "$(ls -A "$CACHE_DIR_2000" 2>/dev/null)" ]; then
    echo "ERROR: $CACHE_DIR_2000 missing — cannot run smoke"
    exit 1
fi

# Copy s2s cache to local SSD (required for s2s_legacy decoder)
LOCAL=$SLURM_TMPDIR/cache
mkdir -p "$LOCAL"
cp "$SCRATCH/meteo_cache/s2s_decoder_cache.dat" "$LOCAL/" || exit 1
cp "$SCRATCH/meteo_cache/s2s_decoder_cache.dat.dates.npy" "$LOCAL/" 2>/dev/null || true

echo "============================================="
echo "  LIGHT SMOKE 9ch x enc21 (2024-05 train / 2025-05..07 val)"
echo "  Cache: $CACHE_DIR_2000"
echo "============================================="

$PYTHON -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name smoke_9ch_light \
    --data_start 2024-05-01 --pred_start 2025-05-01 --pred_end 2025-07-31 \
    --channels "$CHANNELS" --in_days 21 \
    --decoder s2s_legacy --s2s_cache "$LOCAL/s2s_decoder_cache.dat" --s2s_max_issue_lag 3 \
    --loss_fn focal --focal_alpha 0.25 --focal_gamma 2.0 \
    --hard_neg_fraction 0.5 --neg_ratio 20 --neg_buffer 2 \
    --batch_size 1024 --epochs 1 --lr 1e-4 --weight_decay 0.01 --dropout 0.2 \
    --d_model 256 --nhead 8 --enc_layers 4 --dec_layers 4 --patch_size 16 \
    --dilate_radius 14 --val_lift_k 5000 --val_lift_sample_wins 3 \
    --fire_season_only --decoder_ctx \
    --cache_dir "$CACHE_DIR_2000" --chunk_patches 1000 --num_workers 2 \
    --log_interval 50 --skip_forecast

EXIT=$?
echo ""
if [ $EXIT -eq 0 ]; then
    echo "=== LIGHT SMOKE PASSED ✓ ==="
else
    echo "=== LIGHT SMOKE FAILED (exit=$EXIT) ==="
fi
exit $EXIT
