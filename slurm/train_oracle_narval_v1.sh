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
DATA_START=2018-05-01

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

# pf.dat T2427 (2018-05-01→2024-12-21) covers pred_end=2024-10-31 (T_needed≈2374)
# → copy_meteo_caches will copy it to LOCAL_CACHE → STEP 6 skipped
copy_meteo_caches $SCRATCH_CACHE $LOCAL_CACHE 3600 $DATA_START

# Use LOCAL_CACHE if a matching pf.dat already exists in SCRATCH_CACHE
PF_IN_SCRATCH=$(ls $SCRATCH_CACHE/meteo_p16_C8_T*_${DATA_START}_*_pf.dat 2>/dev/null | head -1)
if [ -n "$PF_IN_SCRATCH" ]; then
    CACHE_DIR="$LOCAL_CACHE"
    ts "pf.dat found in SCRATCH_CACHE → using LOCAL_CACHE (STEP 6 skipped)"
else
    CACHE_DIR="$SCRATCH_CACHE"
    ts "pf.dat not in SCRATCH_CACHE → building there (STEP 6 runs once, then persists)"
fi

ts "=== STARTING TRAINING (oracle decoder — light regularization) ==="
$PYTHON src/training/train_s2s_hotspot_cwfis_v2.py \
  --config configs/paths_narval.yaml \
  --run_name oracle_narval_v1 \
  --decoder oracle \
  --pred_end 2024-10-31 \
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

# Copy pf.dat back to SCRATCH_CACHE if not already there (for future resumes)
PF_LOCAL=$(ls $LOCAL_CACHE/meteo_p16_C8_T*_${DATA_START}_*_pf.dat 2>/dev/null | head -1)
PF_SCRATCH=$(ls $SCRATCH_CACHE/meteo_p16_C8_T*_${DATA_START}_*_pf.dat 2>/dev/null | head -1)
if [ -n "$PF_LOCAL" ] && [ -z "$PF_SCRATCH" ]; then
    ts "Copying pf.dat to SCRATCH_CACHE for future resume jobs..."
    cp "$PF_LOCAL" "$SCRATCH_CACHE/" && ts "pf.dat cached to SCRATCH_CACHE OK" &
fi

exit $TRAIN_EXIT
