#!/bin/bash
#SBATCH --job-name=wf-smoke
#SBATCH --gpus-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=200G
#SBATCH --output=/scratch/jiaqi217/logs/smoke_test_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/smoke_test_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# Smoke test: exercise the full train→val→cluster_eval path in
# ~10 minutes with a tiny date range and 2 val windows.
#
# Run this BEFORE submitting long training jobs. Catches:
#   - shape mismatches (via model.forward assertions + val probe)
#   - missing decoder_ctx in val path
#   - sentinel value bugs (via data quality guards)
#   - module import/dependency issues
#   - cache build & SSD setup failures
#
# Usage: sbatch slurm/smoke_test_v3_narval.sh
# ----------------------------------------------------------------

set -uo pipefail
# NOTE: not using -e so cleanup always runs

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}

[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 proj/9.4.1 eccodes/2.31.0

cd "$SCRATCH/wildfire-refactored"
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

source $SCRATCH/venv-wildfire/bin/activate
PYTHON=python

echo "=== SMOKE TEST — $(date) ==="
echo "Node: $(hostname)"
echo "Git HEAD: $(git rev-parse --short HEAD) $(git log -1 --format='%s')"

# Tiny date range: 3 months train, 1 month val
# Using fire season (July-Sept) to ensure some positive samples
CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,deep_soil,precip_def,NDVI,population,slope,burn_age,burn_count"

LOCAL_CACHE=$SLURM_TMPDIR/smoke_cache
mkdir -p "$LOCAL_CACHE"

# Copy S2S cache to SSD
cp $SCRATCH/meteo_cache/s2s_decoder_cache.dat "$LOCAL_CACHE/" || {
    echo "FAIL: copy S2S cache"; exit 1
}
cp $SCRATCH/meteo_cache/s2s_decoder_cache.dat.dates.npy "$LOCAL_CACHE/" || true

$PYTHON -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name smoke_test \
    --data_start 2021-06-01 \
    --pred_start 2021-08-01 \
    --pred_end 2021-09-30 \
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
    --batch_size 256 \
    --epochs 1 \
    --lr 1e-4 \
    --d_model 256 \
    --nhead 8 \
    --enc_layers 4 \
    --dec_layers 4 \
    --patch_size 16 \
    --val_lift_k 1000 \
    --val_lift_sample_wins 2 \
    --fire_season_only \
    --cluster_eval \
    --decoder_ctx \
    --load_train_to_ram \
    --cache_dir "$LOCAL_CACHE" \
    --chunk_patches 2000 \
    --num_workers 2 \
    --log_interval 10 \
    --skip_forecast

EXIT1=$?

if [ $EXIT1 -eq 0 ]; then
    echo "=== SMOKE TEST 1 (s2s_legacy) PASSED ✓ ==="
else
    echo "=== SMOKE TEST 1 (s2s_legacy) FAILED (exit=$EXIT1) ==="
    exit $EXIT1
fi

# Smoke test 2: full-patch S2S decoder (2048 dim) + enc21
echo ""
echo "=== SMOKE TEST 2: full-patch decoder + enc21 ==="
S2S_FULL_CACHE="$SCRATCH/wildfire-refactored/data/s2s_full_patch_cache.dat"

if [ -f "$S2S_FULL_CACHE" ]; then
    $PYTHON -u -m src.training.train_v3 \
        --config configs/paths_narval.yaml \
        --run_name smoke_test_fp \
        --data_start 2021-04-01 \
        --pred_start 2021-08-01 \
        --pred_end 2021-09-30 \
        --channels "$CHANNELS" \
        --in_days 21 \
        --decoder s2s \
        --s2s_full_cache "$S2S_FULL_CACHE" \
        --dec_dim 2048 \
        --loss_fn focal \
        --focal_alpha 0.25 \
        --focal_gamma 2.0 \
        --hard_neg_fraction 0.5 \
        --neg_ratio 20 \
        --neg_buffer 2 \
        --batch_size 256 \
        --epochs 1 \
        --lr 1e-4 \
        --d_model 256 \
        --nhead 8 \
        --enc_layers 4 \
        --dec_layers 4 \
        --patch_size 16 \
        --val_lift_k 1000 \
        --val_lift_sample_wins 2 \
        --fire_season_only \
        --cluster_eval \
        --decoder_ctx \
        --load_train_to_ram \
        --cache_dir "$LOCAL_CACHE" \
        --chunk_patches 2000 \
        --num_workers 2 \
        --log_interval 10 \
        --skip_forecast
    EXIT2=$?
else
    echo "  [SKIP] s2s_full_patch_cache.dat not found"
    EXIT2=0
fi

echo ""
if [ $EXIT2 -eq 0 ]; then
    echo "=== SMOKE TEST 2 (full-patch + enc21) PASSED ✓ ==="
else
    echo "=== SMOKE TEST 2 (full-patch + enc21) FAILED (exit=$EXIT2) ==="
fi

if [ $EXIT1 -eq 0 ] && [ $EXIT2 -eq 0 ]; then
    echo "=== ALL SMOKE TESTS PASSED ✓ ==="
    exit 0
else
    echo "=== SOME SMOKE TESTS FAILED ==="
    exit 1
fi
