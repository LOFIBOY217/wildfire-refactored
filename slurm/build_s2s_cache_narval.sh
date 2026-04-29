#!/bin/bash
#SBATCH --job-name=s2s-cache
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/jiaqi217/logs/s2s_cache_%j.out
#SBATCH --error=/scratch/jiaqi217/logs/s2s_cache_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1
source $SCRATCH/venv-wildfire/bin/activate

mkdir -p /scratch/jiaqi217/logs
mkdir -p /scratch/jiaqi217/meteo_cache

PYTHON=$SCRATCH/venv-wildfire/bin/python
SCRATCH_CACHE=/scratch/jiaqi217/meteo_cache
LOCAL_OUT=$SLURM_TMPDIR/s2s_cache

cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

source slurm/lib_copy_cache.sh

ts "=== PREFLIGHT ==="
ts "Node     : $(hostname)"
ts "Python   : $PYTHON"
$PYTHON -c "import rasterio; print('rasterio:', rasterio.__version__)" || exit 1
$PYTHON -c "import numpy;    print('numpy   :', numpy.__version__)"    || exit 1
ts "=== PREFLIGHT OK ==="

ts "=== RAM CHECK ==="
free -h
echo "==========================="

# Resume: copy existing partial cache to local SSD
mkdir -p $LOCAL_OUT
if [ -f "$SCRATCH_CACHE/s2s_decoder_cache.dat" ]; then
    ts "Copying existing partial cache to local SSD for resume..."
    copy_with_timeout "$SCRATCH_CACHE/s2s_decoder_cache.dat" "$LOCAL_OUT" 1800
    [ -f "$SCRATCH_CACHE/s2s_decoder_cache.dates.npy" ] && \
        cp "$SCRATCH_CACHE/s2s_decoder_cache.dates.npy" "$LOCAL_OUT/"
fi

ts "=== SLURM_TMPDIR ==="
df -h $SLURM_TMPDIR
echo "==========================="

# Build cache (reads TIFs from Lustre, writes to local SSD)
ts "=== STARTING S2S CACHE BUILD ==="
$PYTHON -m src.data_ops.processing.build_s2s_decoder_cache \
  --s2s-dir /scratch/jiaqi217/wildfire-refactored/data/s2s_processed \
  --out-file $LOCAL_OUT/s2s_decoder_cache.dat \
  --reference /scratch/jiaqi217/wildfire-refactored/data/fwi_data/fwi_20250615.tif \
  --workers 8

EXIT_CODE=$?
ts "=== BUILD FINISHED (exit code: $EXIT_CODE) ==="

# Copy results back to scratch
copy_back "$LOCAL_OUT" "$SCRATCH_CACHE"

exit $EXIT_CODE
