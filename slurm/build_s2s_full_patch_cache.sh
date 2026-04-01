#!/bin/bash
#SBATCH --job-name=wf-s2s-cache
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/jiaqi217/logs/build_s2s_cache_%j.out
#SBATCH --error=/scratch/jiaqi217/logs/build_s2s_cache_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1 eccodes/2.31.0

mkdir -p /scratch/jiaqi217/logs

SCRATCH=/scratch/jiaqi217
PROJECT=$SCRATCH/wildfire-refactored

cd $PROJECT
export PYTHONPATH=$PROJECT:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

# eccodes shared library (Python package needs this at runtime)
export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcc12/eccodes/2.31.0/lib64:$LD_LIBRARY_PATH

source $SCRATCH/venv-wildfire/bin/activate

echo "=== BUILD S2S FULL-PATCH CACHE ==="
echo "Node     : $(hostname)"
echo "Python   : $(which python)"
echo "Start    : $(date)"
echo ""

python $PROJECT/src/data_ops/processing/build_s2s_full_patch_cache.py \
    --s2s-dir      $PROJECT/data/s2s_processed \
    --out-file     $PROJECT/data/s2s_full_patch_cache.dat \
    --reference    $PROJECT/data/fwi_data/fwi_20250615.tif \
    --fire-clim    $PROJECT/data/fire_climatology.tif \
    --ffmc-dir     $PROJECT/data/ffmc_data \
    --dmc-dir      $PROJECT/data/dmc_data \
    --dc-dir       $PROJECT/data/dc_data \
    --norm-stats   $PROJECT/checkpoints/s2s_hotspot_cwfis_v2/norm_stats.npy \
    --workers      16

echo ""
echo "=== DONE: $(date) ==="
