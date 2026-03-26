#!/bin/bash
#SBATCH --job-name=s2s-cache
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/jiaqi217/logs/s2s_cache_%j.out
#SBATCH --error=/scratch/jiaqi217/logs/s2s_cache_%j.err
#SBATCH --account=def-inghaw_c
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1
source $SCRATCH/venv-wildfire/bin/activate

mkdir -p /scratch/jiaqi217/logs
mkdir -p /scratch/jiaqi217/meteo_cache

PYTHON=$SCRATCH/venv-wildfire/bin/python

cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

echo "=== PREFLIGHT ==="
echo "Node     : $(hostname)"
echo "Python   : $PYTHON"
$PYTHON -c "import rasterio; print('rasterio :', rasterio.__version__)" || exit 1
$PYTHON -c "import numpy;    print('numpy    :', numpy.__version__)"    || exit 1
echo "=== PREFLIGHT OK ==="

echo "=== RAM CHECK ==="
free -h
echo "==========================="

$PYTHON -m src.data_ops.processing.build_s2s_decoder_cache \
  --s2s-dir /scratch/jiaqi217/wildfire-refactored/data/s2s_processed \
  --out-file /scratch/jiaqi217/meteo_cache/s2s_decoder_cache.dat \
  --reference /scratch/jiaqi217/wildfire-refactored/data/fwi_data/fwi_20250615.tif \
  --workers 8
