#!/bin/bash
#SBATCH --job-name=eval-oracle-v3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=498G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/jiaqi217/logs/eval_oracle_v3_%j.out
#SBATCH --error=/scratch/jiaqi217/logs/eval_oracle_v3_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=END,FAIL
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
ts "=== PREFLIGHT OK ==="

# Copy core meteo caches (STEP 1-5) to LOCAL_CACHE
copy_meteo_caches $SCRATCH_CACHE $LOCAL_CACHE 3600 $DATA_START

# Set up oracle cache dir (T2922, data_start=2018-01-01)
get_pf_T() { basename "$1" | grep -oP 'T\K[0-9]+(?=_[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{4})'; }
mkdir -p $ORACLE_LOCAL
PF_IN_ORACLE=$(ls $ORACLE_SCRATCH/meteo_p16_C8_T*_${DATA_START}_*_pf.dat 2>/dev/null | head -1)

if [ -n "$PF_IN_ORACLE" ]; then
    ts "Oracle pf.dat found (T=$(get_pf_T $PF_IN_ORACLE)) → copying to ORACLE_LOCAL..."
    cp -r $LOCAL_CACHE/. $ORACLE_LOCAL/
    cp "$PF_IN_ORACLE" "$ORACLE_LOCAL/"
    CACHE_DIR="$ORACLE_LOCAL"
    ts "Oracle pf.dat ready on local SSD"
else
    ts "WARNING: No oracle pf.dat found in ORACLE_SCRATCH. Falling back to ORACLE_SCRATCH."
    CACHE_DIR="$ORACLE_SCRATCH"
fi

ts "=== STARTING FULL EVAL (oracle_narval_v3 — all val windows, all epoch checkpoints) ==="
$PYTHON src/training/train_s2s_hotspot_cwfis_v2.py \
  --config configs/paths_narval.yaml \
  --run_name oracle_narval_v3 \
  --decoder oracle \
  --pred_end 2025-10-31 \
  --num_workers 8 \
  --batch_size 8192 \
  --cache_dir $CACHE_DIR \
  --skip_forecast \
  --eval_epochs \
  --eval_n_windows 9999

EVAL_EXIT=$?
ts "=== EVAL FINISHED (exit code: $EVAL_EXIT) ==="
exit $EVAL_EXIT
