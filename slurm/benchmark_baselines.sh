#!/bin/bash
#SBATCH --job-name=wf-benchmark
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/jiaqi217/logs/benchmark_%j.out
#SBATCH --error=/scratch/jiaqi217/logs/benchmark_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1
source $SCRATCH/venv-wildfire/bin/activate

mkdir -p /scratch/jiaqi217/logs

PYTHON=$SCRATCH/venv-wildfire/bin/python

cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

echo "=== PREFLIGHT ==="
echo "Node: $(hostname)"
$PYTHON -c "import rasterio; print('rasterio:', rasterio.__version__)" || exit 1
$PYTHON -c "import sklearn; print('sklearn:', sklearn.__version__)" || exit 1
echo "=== PREFLIGHT OK ==="

$PYTHON -m src.evaluation.benchmark_baselines \
  --config configs/paths_narval.yaml \
  --baseline fwi_oracle climatology \
  --eval_mode per_leadday \
  --pred_start 2022-05-01 \
  --pred_end 2025-10-31 \
  --k_values 1000 2500 5000 10000 25000 \
  --n_sample_wins 20 \
  --dilate_radius 14 \
  --fire_season_only \
  --output_csv outputs/benchmark_baselines_per_leadday.csv

echo "=== BENCHMARK FINISHED (exit code: $?) ==="
