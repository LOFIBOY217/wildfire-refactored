#!/bin/bash
#SBATCH --job-name=wildfire-v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=compute_full_node
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/jiaqi217/logs/train_v2_%j.out
#SBATCH --error=/scratch/jiaqi217/logs/train_v2_%j.err

module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 proj/9.4.1
source $SCRATCH/venv-wildfire/bin/activate

mkdir -p /scratch/jiaqi217/logs

PYTHON=$SCRATCH/venv-wildfire/bin/python

cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/proj/9.4.1/share/proj

# Preflight checks
echo "=== PREFLIGHT ==="
echo "Node     : $(hostname)"
echo "Python   : $(which python)"
$PYTHON -c "import torch;    print('torch    :', torch.__version__, '| CUDA:', torch.cuda.is_available())" || exit 1
$PYTHON -c "import rasterio; print('rasterio :', rasterio.__version__)" || exit 1
$PYTHON -c "import scipy;    print('scipy    :', scipy.__version__)"    || exit 1
$PYTHON -c "import numpy;    print('numpy    :', numpy.__version__)"    || exit 1
echo "=== PREFLIGHT OK ==="

$PYTHON src/training/train_s2s_hotspot_cwfis_v2.py \
  --config configs/paths_trillium.yaml \
  --num_workers 12 \
  --batch_size 512 \
  --epochs 10
