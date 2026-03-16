#!/bin/bash
#SBATCH --job-name=wildfire-v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=compute_full_node
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/jiaqi217/logs/train_v2_%j.out
#SBATCH --error=/scratch/jiaqi217/logs/train_v2_%j.err

module load StdEnv/2023 gcc/12.3 cuda/12.2

# Fix missing libcpupower.so.0 symlink on compute nodes
mkdir -p $SCRATCH/lib
ln -sf /usr/lib64/libcpupower.so.0.0.1 $SCRATCH/lib/libcpupower.so.0
export LD_LIBRARY_PATH=$SCRATCH/lib:$LD_LIBRARY_PATH

mkdir -p /scratch/jiaqi217/logs

PYTHON=$SCRATCH/miniforge3/envs/wildfore-r/bin/python

cd $SCRATCH/wildfire-refactored

# Preflight checks
echo "=== PREFLIGHT ==="
echo "Node     : $(hostname)"
echo "SCRATCH  : $SCRATCH"
echo "LD_PATH  : $LD_LIBRARY_PATH"
echo "libcpu   : $(ls -la $SCRATCH/lib/libcpupower.so.0 2>/dev/null || echo MISSING)"
$PYTHON -c "import torch;    print('torch    :', torch.__version__, '| CUDA:', torch.cuda.is_available())" || exit 1
$PYTHON -c "import rasterio; print('rasterio :', rasterio.__version__)" || exit 1
$PYTHON -c "import scipy;    print('scipy    :', scipy.__version__)"    || exit 1
$PYTHON -c "import numpy;    print('numpy    :', numpy.__version__)"    || exit 1
echo "=== PREFLIGHT OK ==="

CUDA_VISIBLE_DEVICES=0 $PYTHON src/training/train_s2s_hotspot_cwfis_v2.py \
  --config configs/paths_trillium.yaml \
  --num_workers 12 \
  --batch_size 512 \
  --epochs 10
