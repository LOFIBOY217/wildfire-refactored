#!/bin/bash
#SBATCH --job-name=wf-s2s-v4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=17:00:00
#SBATCH --output=/scratch/jiaqi217/logs/train_s2s_v4_%j.out
#SBATCH --error=/scratch/jiaqi217/logs/train_s2s_v4_%j.err
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

source slurm/lib_copy_cache.sh

copy_venv $SCRATCH/venv-wildfire

ts "=== PREFLIGHT ==="
ts "Node     : $(hostname)"
$PYTHON -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())" || exit 1
ts "=== PREFLIGHT OK ==="

DATA_START=2018-05-01
copy_meteo_caches $SCRATCH_CACHE $LOCAL_CACHE 3600 $DATA_START
copy_s2s_cache $SCRATCH_CACHE $LOCAL_CACHE 1800

ts "=== RESUMING TRAINING (s2s_legacy v4 — light regularization) ==="
$PYTHON src/training/train_s2s_hotspot_cwfis_v2.py \
  --config configs/paths_narval.yaml \
  --run_name s2s_decoder_s2s_v4 \
  --decoder s2s_legacy \
  --s2s_cache $LOCAL_CACHE/s2s_decoder_cache.dat \
  --pred_end 2025-10-31 \
  --s2s_max_issue_lag 3 \
  --num_workers 8 \
  --batch_size 8192 \
  --epochs 8 \
  --lr 1e-4 \
  --lr_min 1e-6 \
  --dropout 0.1 \
  --weight_decay 0.01 \
  --lead_end 45 \
  --log_interval 1000 \
  --cache_dir $LOCAL_CACHE \
  --skip_forecast \
  --resume

ts "=== TRAINING FINISHED (exit code: $?) ==="
