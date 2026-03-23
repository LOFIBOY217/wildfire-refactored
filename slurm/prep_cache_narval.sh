#!/bin/bash
#SBATCH --job-name=wildfire-prep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=cpularge_bycore_b4
#SBATCH --mem=480G
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/jiaqi217/logs/prep_cache_%j.out
#SBATCH --error=/scratch/jiaqi217/logs/prep_cache_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 proj/9.4.1 eccodes/2.31.0
source $SCRATCH/venv-wildfire/bin/activate

mkdir -p /scratch/jiaqi217/logs
mkdir -p /scratch/jiaqi217/meteo_cache

PYTHON=$SCRATCH/venv-wildfire/bin/python

cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

echo "=== PREP CACHE JOB ==="
echo "Node     : $(hostname)"
echo "Python   : $(which python)"
echo "Goal     : Build meteo memmap cache (full dataset, no fire_season_only)"
echo "Cache dir: /scratch/jiaqi217/meteo_cache/"
echo "Started  : $(date)"
echo "========================"

$PYTHON src/training/train_s2s_hotspot_cwfis_v2.py \
  --config configs/paths_narval.yaml \
  --num_workers 32 \
  --cache_dir /scratch/jiaqi217/meteo_cache \
  --chunk_patches 8000 \
  --prep_only \
  --skip_forecast

echo "=== PREP DONE: $(date) ==="
echo "Cache files:"
ls -lh /scratch/jiaqi217/meteo_cache/
