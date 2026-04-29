#!/bin/bash
#SBATCH --job-name=meteo-cache
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/jiaqi217/logs/meteo_cache_%j.out
#SBATCH --error=/scratch/jiaqi217/logs/meteo_cache_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1 eccodes/2.31.0
source $SCRATCH/venv-wildfire/bin/activate

mkdir -p /scratch/jiaqi217/logs

PYTHON=$SCRATCH/venv-wildfire/bin/python
SCRATCH_CACHE=/scratch/jiaqi217/meteo_cache
LOCAL_CACHE=$SLURM_TMPDIR/cache

cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

source slurm/lib_copy_cache.sh

ts "=== PREFLIGHT ==="
ts "Node     : $(hostname)"
ts "Python   : $(which python)"
$PYTHON -c "import rasterio; print('rasterio:', rasterio.__version__)" || exit 1
$PYTHON -c "import numpy;    print('numpy   :', numpy.__version__)"    || exit 1
$PYTHON -c "import scipy;    print('scipy   :', scipy.__version__)"    || exit 1
$PYTHON -c "import pyproj;   print('pyproj  :', pyproj.__version__)"   || exit 1
ts "=== PREFLIGHT OK ==="

ts "=== RAM CHECK ==="
free -h
echo "==========================="

# Build cache — reads TIFs from Lustre, writes output to local SSD
ts "=== SLURM_TMPDIR ==="
df -h $SLURM_TMPDIR
mkdir -p $LOCAL_CACHE
echo "==========================="

ts "=== STARTING METEO CACHE BUILD ==="
$PYTHON scripts/build_meteo_cache.py \
  --config configs/paths_narval.yaml \
  --cache-dir $LOCAL_CACHE \
  --data-start 2018-01-01 \
  --pred-end 2025-12-31 \
  --lead-end 45 \
  --patch-size 16 \
  --dilate-radius 14 \
  --chunk-patches 500

EXIT_CODE=$?
ts "=== BUILD FINISHED (exit code: $EXIT_CODE) ==="

# Copy results back to scratch
copy_back "$LOCAL_CACHE" "$SCRATCH_CACHE"

exit $EXIT_CODE
