#!/bin/bash
#SBATCH --job-name=wf-v3-enc28-2000
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=500G
#SBATCH --output=/scratch/jiaqi217/logs/train_v3_9ch_enc28_2000_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/train_v3_9ch_enc28_2000_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca
#SBATCH --account=def-inghaw

# ----------------------------------------------------------------
#   V3 9ch × enc28 × 2000-2025 EXTENDED TRAINING
# ----------------------------------------------------------------
# Goal: test scaling law by extending train set from 4 years (2018-2022)
# to 22 years (2000-2022). 5.5x more training data.
#
# Baseline for comparison:
#   enc28 × 2018-2022 (4y train) → Lift@5000 = 6.43x (ep1, 20-win)
#   enc35 × 2018-2022 (4y train) → Lift@5000 = 6.56x (ep1, current SOTA)
#
# Target: beat 6.56x with same enc28 config + 5.5x more data.
#
# PREREQUISITES (verified before submitting):
#   ✅ fire_clim_upto_{2000..2025}.tif exist (26 files)
#   ✅ burn_age/burn_count_{2000..2024}.tif exist (25 files)
#   ✅ FWI/2t/2d/tcw/sm20 for 2000-2008 + 2018-2025 complete
#   ⚠️ ERA5 2010-2017 must be resampled before submitting this
#   ✅ NDVI 2000-2025 (not in 9ch but ready)
#   ⚠️ Cache built at $CACHE_DIR_2000 via rebuild_cache_2000_2025_narval.sh
#
# Differences from enc28 baseline script:
#   --data_start 2000-05-01  (was 2018-05-01)
#   --cache_dir $SCRATCH/meteo_cache/v3_9ch_2000  (NEW — don't overwrite old)
#   --run_name v3_9ch_enc28_2000
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
copy_s2s_cache "$SCRATCH/meteo_cache" "$LOCAL_CACHE"

CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,population,slope,burn_age"
CACHE_DIR_2000="$SCRATCH/meteo_cache/v3_9ch_2000"

# Sanity check: cache must exist
if [ ! -d "$CACHE_DIR_2000" ] || [ -z "$(ls -A "$CACHE_DIR_2000" 2>/dev/null)" ]; then
    echo "ERROR: $CACHE_DIR_2000 is empty or missing."
    echo "Run cache build first: CACHE_CHANNELS=9 sbatch slurm/rebuild_cache_2000_2025_narval.sh"
    exit 1
fi

echo "============================================="
echo "  V3 9ch × enc28 × 2000-2025 (5.5× data extension)"
echo "  Cache: $CACHE_DIR_2000"
echo "============================================="

$PYTHON -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name v3_9ch_enc28_2000 \
    --data_start 2000-05-01 --pred_start 2022-05-01 --pred_end 2025-10-31 \
    --channels "$CHANNELS" --in_days 28 \
    --decoder s2s_legacy --s2s_cache "$LOCAL_CACHE/s2s_decoder_cache.dat" --s2s_max_issue_lag 3 \
    --loss_fn focal --focal_alpha 0.25 --focal_gamma 2.0 \
    --hard_neg_fraction 0.5 --neg_ratio 20 --neg_buffer 2 \
    --batch_size 4096 --epochs 4 --lr 1e-4 --weight_decay 0.01 --dropout 0.2 \
    --d_model 256 --nhead 8 --enc_layers 4 --dec_layers 4 --patch_size 16 \
    --dilate_radius 14 --val_lift_k 5000 --val_lift_sample_wins 20 \
    --fire_season_only --cluster_eval --decoder_ctx --load_train_to_ram \
    --cache_dir "$CACHE_DIR_2000" --chunk_patches 2000 --num_workers 4 \
    --log_interval 200 --skip_forecast

PY_EXIT=$?
echo "=== Done: $(date) exit=$PY_EXIT ==="
exit $PY_EXIT
