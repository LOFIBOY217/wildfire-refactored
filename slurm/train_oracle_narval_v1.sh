#!/bin/bash
#SBATCH --job-name=wf-oracle
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/jiaqi217/logs/train_oracle_%j.out
#SBATCH --error=/scratch/jiaqi217/logs/train_oracle_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# Ensure $SCRATCH is set (may be unset if submitted from non-login SSH session)
export SCRATCH=${SCRATCH:-/scratch/jiaqi217}

# Ensure module command is available (may be missing in non-login shells)
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh

module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 proj/9.4.1 eccodes/2.31.0

mkdir -p /scratch/jiaqi217/logs

SCRATCH_CACHE=/scratch/jiaqi217/meteo_cache
LOCAL_CACHE=$SLURM_TMPDIR/cache
# Dedicated oracle cache — keeps T2740+ pf.dat separate from T2427 in main SCRATCH_CACHE
# This avoids the fuzzy-match crash: ls T*_pf.dat | head -1 picking T2427 (too small for pred_end=2025-10-31)
ORACLE_SCRATCH=$SCRATCH_CACHE/oracle_2025
ORACLE_LOCAL=$SLURM_TMPDIR/cache_oracle
DATA_START=2018-01-01

cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

source slurm/lib_copy_cache.sh
copy_venv $SCRATCH/venv-wildfire

ts "=== PREFLIGHT ==="
ts "Node     : $(hostname)"
$PYTHON -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())" || exit 1
$PYTHON -c "import rasterio; print('rasterio:', rasterio.__version__)" || exit 1
$PYTHON -c "import sklearn; print('sklearn:', sklearn.__version__)" || exit 1
ts "=== PREFLIGHT OK ==="

# Copy core meteo caches (STEP 1-5) to LOCAL_CACHE — always needed
copy_meteo_caches $SCRATCH_CACHE $LOCAL_CACHE 3600 $DATA_START

# Use ORACLE-specific cache dir to avoid fuzzy-matching T2427 (too small for 2025-10-31).
# Strategy (in order):
#   1. ORACLE_SCRATCH already has pf.dat → copy to ORACLE_LOCAL, skip STEP 6
#   2. SCRATCH_CACHE has large pf.dat (T≥2700, covers 2025-10-31) → seed ORACLE_SCRATCH, skip STEP 6
#   3. Neither → build in ORACLE_SCRATCH (STEP 6 ~9h, persists for future runs)
get_pf_T() { basename "$1" | grep -oP 'T\K[0-9]+(?=_[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{4})'; }
mkdir -p $ORACLE_SCRATCH $ORACLE_LOCAL
PF_IN_ORACLE=$(ls $ORACLE_SCRATCH/meteo_p16_C8_T*_${DATA_START}_*_pf.dat 2>/dev/null | head -1)

if [ -n "$PF_IN_ORACLE" ]; then
    ts "Oracle pf.dat found in ORACLE_SCRATCH (T=$(get_pf_T $PF_IN_ORACLE)) → copying to ORACLE_LOCAL..."
    cp -r $LOCAL_CACHE/. $ORACLE_LOCAL/
    cp "$PF_IN_ORACLE" "$ORACLE_LOCAL/"
    CACHE_DIR="$ORACLE_LOCAL"
    ts "Oracle pf.dat ready → using ORACLE_LOCAL (STEP 6 skipped)"
else
    # Check if SCRATCH_CACHE has a large pf.dat (T≥2700) usable for pred_end=2025-10-31
    SEED_PF=""
    for f in $(ls $SCRATCH_CACHE/meteo_p16_C8_T*_${DATA_START}_*_pf.dat 2>/dev/null | sort -t'T' -k2 -n); do
        T_val=$(get_pf_T "$f")
        if [ -n "$T_val" ] && [ "$T_val" -ge 2700 ]; then
            SEED_PF="$f"
        fi
    done
    if [ -n "$SEED_PF" ]; then
        ts "Seeding ORACLE_SCRATCH from SCRATCH_CACHE: $(basename $SEED_PF) (T=$(get_pf_T $SEED_PF))..."
        cp "$SEED_PF" "$ORACLE_SCRATCH/" && ts "Seed done."
        cp -r $LOCAL_CACHE/. $ORACLE_LOCAL/
        cp "$SEED_PF" "$ORACLE_LOCAL/"
        CACHE_DIR="$ORACLE_LOCAL"
        ts "Oracle seeded from SCRATCH_CACHE → using ORACLE_LOCAL (STEP 6 skipped)"
    else
        CACHE_DIR="$ORACLE_SCRATCH"
        ts "No suitable pf.dat found → building in ORACLE_SCRATCH (STEP 6 runs once, then persists)"
    fi
fi

ts "=== STARTING TRAINING (oracle decoder — light regularization) ==="
$PYTHON src/training/train_s2s_hotspot_cwfis_v2.py \
  --config configs/paths_narval.yaml \
  --run_name oracle_narval_v1 \
  --decoder oracle \
  --pred_end 2025-10-31 \
  --num_workers 16 \
  --batch_size 8192 \
  --epochs 8 \
  --lr 1e-4 \
  --lr_min 1e-6 \
  --dropout 0.1 \
  --weight_decay 0.01 \
  --log_interval 1000 \
  --cache_dir $CACHE_DIR \
  --skip_forecast

TRAIN_EXIT=$?
ts "=== TRAINING FINISHED (exit code: $TRAIN_EXIT) ==="

# Copy oracle pf.dat back to ORACLE_SCRATCH if not already there (for future resumes)
PF_ORACLE_LOCAL=$(ls $ORACLE_LOCAL/meteo_p16_C8_T*_${DATA_START}_*_pf.dat 2>/dev/null | head -1)
PF_ORACLE_SCRATCH=$(ls $ORACLE_SCRATCH/meteo_p16_C8_T*_${DATA_START}_*_pf.dat 2>/dev/null | head -1)
if [ -n "$PF_ORACLE_LOCAL" ] && [ -z "$PF_ORACLE_SCRATCH" ]; then
    ts "Copying oracle pf.dat to ORACLE_SCRATCH for future resume jobs..."
    cp "$PF_ORACLE_LOCAL" "$ORACLE_SCRATCH/" && ts "oracle pf.dat cached to ORACLE_SCRATCH OK" &
fi

exit $TRAIN_EXIT
