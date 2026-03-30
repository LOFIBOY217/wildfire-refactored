#!/bin/bash
#SBATCH --job-name=wf-s2s-1g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/jiaqi217/logs/train_s2s_1g_%j.out
#SBATCH --error=/scratch/jiaqi217/logs/train_s2s_1g_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 proj/9.4.1
source $SCRATCH/venv-wildfire/bin/activate

mkdir -p /scratch/jiaqi217/logs

PYTHON=$SCRATCH/venv-wildfire/bin/python

cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

S2S_CACHE=/scratch/jiaqi217/wildfire-refactored/data/s2s_decoder_cache/s2s_decoder_cache.dat

echo "=== PREFLIGHT ==="
echo "Node     : $(hostname)"
$PYTHON -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())" || exit 1
$PYTHON -c "import rasterio; print('rasterio:', rasterio.__version__)" || exit 1
ls -lh $S2S_CACHE || { echo "ERROR: S2S cache not found"; exit 1; }
echo "=== PREFLIGHT OK ==="

echo "=== STARTING TRAINING (S2S decoder, 1GPU, fire_season_only) ==="
$PYTHON src/training/train_s2s_hotspot_cwfis_v2.py \
  --config configs/paths_trillium.yaml \
  --run_name s2s_decoder_s2s_v3_1gpu \
  --decoder s2s \
  --s2s_cache $S2S_CACHE \
  --pred_end 2025-10-31 \
  --num_workers 12 \
  --batch_size 1024 \
  --epochs 12 \
  --lr 1e-4 \
  --lr_min 1e-6 \
  --dropout 0.2 \
  --weight_decay 0.05 \
  --label_smoothing 0.05 \
  --neg_buffer 2 \
  --fire_season_only \
  --skip_forecast

echo "=== TRAINING FINISHED (exit code: $?) ==="
