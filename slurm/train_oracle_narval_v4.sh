#!/bin/bash
#SBATCH --job-name=wf-oracle-v4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=498G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/jiaqi217/logs/train_oracle_v4_%j.out
#SBATCH --error=/scratch/jiaqi217/logs/train_oracle_v4_%j.err
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

# Copy core meteo caches (STEP 1-5) to LOCAL_CACHE
copy_meteo_caches $SCRATCH_CACHE $LOCAL_CACHE 3600 $DATA_START

# Reuse oracle pf.dat (T2922, data_start=2018-01-01) from ORACLE_SCRATCH
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

# Key insight: batch_size=1024 gives 8x more gradient updates/epoch than batch_size=8192.
# With oracle (IO-bound training), epoch time stays ~45-60 min regardless of batch size.
# This replicates Trillium's training dynamics (batch=256 gave 19.09x at ep1).
# batch=1024 → ~32K updates/epoch (vs Trillium's 130K @ batch=256, 4x closer).
# Expected peak: ep4-6 (vs ep3 with batch=8192, vs ep1 with batch=256).
ts "=== STARTING TRAINING (oracle v4 — batch=1024, 8x more grad updates/epoch) ==="
$PYTHON src/training/train_s2s_hotspot_cwfis_v2.py \
  --config configs/paths_narval.yaml \
  --run_name oracle_narval_v4 \
  --decoder oracle \
  --pred_end 2025-10-31 \
  --num_workers 8 \
  --batch_size 1024 \
  --epochs 16 \
  --lr 1e-4 \
  --lr_min 1e-6 \
  --dropout 0.0 \
  --weight_decay 0.0 \
  --log_interval 1000 \
  --cache_dir $CACHE_DIR \
  --skip_forecast

TRAIN_EXIT=$?
ts "=== TRAINING FINISHED (exit code: $TRAIN_EXIT) ==="
exit $TRAIN_EXIT
