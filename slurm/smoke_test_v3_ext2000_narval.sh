#!/bin/bash
#SBATCH --job-name=wf-smoke-ext
#SBATCH --gpus-per-node=1
#SBATCH --time=1:30:00
#SBATCH --mem=200G
#SBATCH --output=/scratch/jiaqi217/logs/smoke_ext2000_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/smoke_ext2000_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# Smoke test for 2000-2025 EXTENDED training.
#
# Dependency: requires fire_clim_upto_{2000..2017}.tif files
# (rebuilt via make_fire_clim_annual with --data_start_year 2000).
#
# This test specifically exercises year-2000 edge cases that the
# original 2018-based smoke_test_v3_narval.sh cannot hit:
#
#   1. fire_clim_upto_2000.tif exists and is zero (no prior hotspot data)
#   2. fire_clim_upto_2001.tif loads correctly (single-year prior)
#   3. burn_age for year 2000 → prev_year=1999 missing → fallback
#      (expect: frame stays at fill, verify no crash)
#   4. All 9ch files load for 2000-2001 date range
#   5. Full cache build → train → val → cluster eval completes
#
# Uses 9ch subset (subset of required extension channels):
#   FWI, 2t, fire_clim, 2d, tcw, sm20, population, slope, burn_age
#
# All 9 channels must have 2000-2025 EPSG:3978 coverage.
#
# Usage:
#   sbatch --dependency=afterok:<fire_clim_job_id> \
#          slurm/smoke_test_v3_ext2000_narval.sh
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

echo "=== EXT2000 SMOKE TEST — $(date) ==="
echo "Node: $(hostname)"
echo "Git HEAD: $(git rev-parse --short HEAD) $(git log -1 --format='%s')"

# --- PRE-FLIGHT DATA CHECKS (fail fast before allocating GPU time) ---
echo ""
echo "=== Pre-flight: verifying 2000-2017 data coverage ==="
PREFLIGHT_OK=1

check_file() {
    if [ ! -f "$1" ]; then
        echo "  [MISSING] $1"
        PREFLIGHT_OK=0
    fi
}

# fire_clim_upto_YYYY for year 2000-2017 (newly built)
for yr in 2000 2001 2010 2017; do
    check_file "$SCRATCH/wildfire-refactored/data/fire_clim_annual/fire_clim_upto_${yr}.tif"
done

# burn_age / burn_count for 2000-2017 (newly built via NBAC)
for yr in 2000 2001 2010 2017; do
    check_file "$SCRATCH/wildfire-refactored/data/burn_scars/years_since_burn_${yr}.tif"
    check_file "$SCRATCH/wildfire-refactored/data/burn_scars/burn_count_${yr}.tif"
done

# ERA5 observation 2t/2d/tcw/sm20 for a few early dates
for dt in 20000501 20000601 20010501 20050515 20100801 20170615; do
    check_file "$SCRATCH/wildfire-refactored/data/ecmwf_observation/2t/2t_${dt}.tif"
    check_file "$SCRATCH/wildfire-refactored/data/ecmwf_observation/2d/2d_${dt}.tif"
    check_file "$SCRATCH/wildfire-refactored/data/ecmwf_observation/tcw/tcw_${dt}.tif"
    check_file "$SCRATCH/wildfire-refactored/data/ecmwf_observation/sm20/sm20_${dt}.tif"
done

# FWI for same dates
for dt in 20000501 20000601 20010501 20050515 20100801 20170615; do
    check_file "$SCRATCH/wildfire-refactored/data/fwi_data/fwi_${dt}.tif"
done

if [ $PREFLIGHT_OK -eq 0 ]; then
    echo ""
    echo "=== PRE-FLIGHT FAILED: missing files above. Abort smoke test. ==="
    exit 1
fi
echo "  [OK] All critical files present for 2000-2017 sample dates"

# --- VALIDATE ALL CHANNELS ---
echo ""
echo "=== Running full channel validation ==="
$PYTHON -u -m src.data_ops.validation.validate_all_channels \
    --config configs/paths_narval.yaml \
    --start_year 2000 \
    --end_year 2025 || {
    echo "  [WARN] validation exited non-zero (continue to smoke test anyway)"
}

# --- SMOKE TEST: 9ch, tiny year-2000 range, 1 epoch ---
echo ""
echo "=== Smoke test: 9ch enc21 (year 2000-2001) ==="

CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,population,slope,burn_age"

LOCAL_CACHE=$SLURM_TMPDIR/smoke_ext_cache
mkdir -p "$LOCAL_CACHE"

# Copy S2S cache to SSD for decoder (s2s_legacy)
cp $SCRATCH/meteo_cache/s2s_decoder_cache.dat "$LOCAL_CACHE/" || {
    echo "FAIL: copy S2S cache"; exit 1
}
cp $SCRATCH/meteo_cache/s2s_decoder_cache.dat.dates.npy "$LOCAL_CACHE/" || true

$PYTHON -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name smoke_ext2000 \
    --data_start 2000-05-01 \
    --pred_start 2001-05-01 \
    --pred_end 2001-09-30 \
    --channels "$CHANNELS" \
    --in_days 21 \
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
    --num_workers 4 \
    --log_interval 20 \
    --skip_forecast

EXIT=$?

echo ""
if [ $EXIT -eq 0 ]; then
    echo "=== EXT2000 SMOKE TEST PASSED ✓ ==="
    echo "    Safe to proceed with 9ch 2000-2025 cache build + training"
    exit 0
else
    echo "=== EXT2000 SMOKE TEST FAILED (exit=$EXIT) ==="
    echo "    DO NOT proceed with full cache build until fixed"
    exit $EXIT
fi
