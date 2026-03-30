#!/bin/bash
#SBATCH --job-name=wf-dec-random
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=17:00:00
#SBATCH --output=/scratch/jiaqi217/logs/train_dec_random_%j.out
#SBATCH --error=/scratch/jiaqi217/logs/train_dec_random_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 proj/9.4.1 eccodes/2.31.0

mkdir -p /scratch/jiaqi217/logs

SCRATCH_CACHE=/scratch/jiaqi217/meteo_cache
LOCAL_CACHE=$SLURM_TMPDIR/cache

cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

# Load shared copy library
source slurm/lib_copy_cache.sh

# Copy venv to local SSD (saves ~3h preflight)
copy_venv $SCRATCH/venv-wildfire
# PYTHON is set by copy_venv (local SSD or Lustre fallback)

# Preflight
ts "=== PREFLIGHT ==="
ts "Node     : $(hostname)"
ts "Python   : $(which python)"
$PYTHON -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())" || exit 1
$PYTHON -c "import rasterio; print('rasterio:', rasterio.__version__)" || exit 1
$PYTHON -c "import sklearn; print('sklearn:', sklearn.__version__)" || exit 1
ts "=== PREFLIGHT OK ==="

# Copy caches to local SSD (timeout 1h per large file)
DATA_START=2018-05-01
copy_meteo_caches $SCRATCH_CACHE $LOCAL_CACHE 3600 $DATA_START

# Train
ts "=== STARTING TRAINING ==="
$PYTHON src/training/train_s2s_hotspot_cwfis_v2.py \
  --config configs/paths_narval.yaml \
  --run_name s2s_decoder_random \
  --decoder random \
  --pred_end 2025-10-31 \
  --num_workers 16 \
  --batch_size 8192 \
  --epochs 6 \
  --lr 1e-4 \
  --lr_min 1e-6 \
  --log_interval 1000 \
  --cache_dir $LOCAL_CACHE \
  --skip_forecast

ts "=== TRAINING FINISHED (exit code: $?) ==="
