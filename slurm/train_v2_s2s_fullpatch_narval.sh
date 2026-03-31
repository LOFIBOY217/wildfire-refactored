#!/bin/bash
#SBATCH --job-name=wf-s2s-fp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/jiaqi217/logs/train_s2s_fp_%j.out
#SBATCH --error=/scratch/jiaqi217/logs/train_s2s_fp_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 proj/9.4.1 eccodes/2.31.0

mkdir -p /scratch/jiaqi217/logs

SCRATCH=/scratch/jiaqi217
PROJECT=$SCRATCH/wildfire-refactored
LOCAL_CACHE=$SLURM_TMPDIR/cache

cd $PROJECT
export PYTHONPATH=$PROJECT:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

source $SCRATCH/venv-wildfire/bin/activate

source slurm/lib_copy_cache.sh

ts "=== PREFLIGHT ==="
ts "Node     : $(hostname)"
ts "Python   : $(which python)"
$PYTHON -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())" || exit 1
$PYTHON -c "import rasterio; print('rasterio:', rasterio.__version__)" || exit 1
ts "=== PREFLIGHT OK ==="

# Copy meteo cache (encoder) to local SSD
DATA_START=2018-05-01
copy_meteo_caches $SCRATCH/meteo_cache $LOCAL_CACHE 3600 $DATA_START

# Copy S2S full-patch cache to local SSD (5.27TB — skip if too large)
S2S_FP_CACHE=$PROJECT/data/s2s_full_patch_cache.dat
S2S_FP_DATES=$PROJECT/data/s2s_full_patch_cache.dat.dates.npy

ts "=== STARTING TRAINING (S2S full-patch decoder, dec_dim=2048) ==="
$PYTHON src/training/train_s2s_hotspot_cwfis_v2.py \
  --config configs/paths_narval.yaml \
  --run_name s2s_fullpatch_v1 \
  --decoder s2s \
  --s2s_full_cache $S2S_FP_CACHE \
  --pred_end 2025-10-31 \
  --s2s_max_issue_lag 3 \
  --num_workers 8 \
  --batch_size 8192 \
  --epochs 8 \
  --lr 1e-4 \
  --lr_min 1e-6 \
  --dropout 0.1 \
  --weight_decay 0.01 \
  --log_interval 1000 \
  --cache_dir $LOCAL_CACHE \
  --skip_forecast

ts "=== TRAINING FINISHED (exit code: $?) ==="
