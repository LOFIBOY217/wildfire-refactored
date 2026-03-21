#!/bin/bash
#SBATCH --job-name=wildfire-v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpubase_bygpu_b3
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=320G
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/jiaqi217/logs/train_v2_%j.out
#SBATCH --error=/scratch/jiaqi217/logs/train_v2_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 proj/9.4.1
source $SCRATCH/venv-wildfire/bin/activate

mkdir -p /scratch/jiaqi217/logs

PYTHON=$SCRATCH/venv-wildfire/bin/python

cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj

# Preflight checks
echo "=== PREFLIGHT ==="
echo "Node     : $(hostname)"
echo "Python   : $(which python)"
$PYTHON -c "import torch;    print('torch    :', torch.__version__, '| CUDA:', torch.cuda.is_available())" || exit 1
$PYTHON -c "import rasterio; print('rasterio :', rasterio.__version__)" || exit 1
$PYTHON -c "import scipy;    print('scipy    :', scipy.__version__)"    || exit 1
$PYTHON -c "import numpy;    print('numpy    :', numpy.__version__)"    || exit 1
echo "=== PREFLIGHT OK ==="

# RAM check
echo "=== RAM CHECK ==="
free -h
TOTAL_RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
NEEDED_GB=260
if [ "$TOTAL_RAM_GB" -lt "$NEEDED_GB" ]; then
  echo "ERROR: Not enough RAM. Available=${TOTAL_RAM_GB}GB, needed=${NEEDED_GB}GB"
  exit 1
fi
echo "RAM OK: ${TOTAL_RAM_GB}GB available (need ~260GB for --load_to_ram full dataset)"
echo "==========================="

$PYTHON src/training/train_s2s_hotspot_cwfis_v2.py \
  --config configs/paths_narval.yaml \
  --num_workers 16 \
  --batch_size 512 \
  --epochs 30 \
  --lr 1e-4 \
  --lr_min 1e-6 \
  --load_to_ram \
  --skip_forecast
