#!/bin/bash
#SBATCH --job-name=wf-proc-tif
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/jiaqi217/logs/process_core_tif_%j.out
#SBATCH --error=/scratch/jiaqi217/logs/process_core_tif_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1 eccodes/2.31.0
export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcc12/eccodes/2.31.0/lib64:$LD_LIBRARY_PATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj

SCRATCH=/scratch/jiaqi217
PROJECT=$SCRATCH/wildfire-refactored

source $SCRATCH/venv-wildfire/bin/activate
cd $PROJECT
export PYTHONPATH=$PROJECT:$PYTHONPATH
export PYTHONUNBUFFERED=1

echo "=== PROCESS S2S CORE TIFs ==="
echo "Node   : $(hostname)"
echo "Start  : $(date)"
echo ""

# Step 1: core GRIBs (data/s2s_forecast/) → data/s2s_processed/
echo "[1/2] Processing core S2S GRIBs → TIFs ..."
python -m src.data_ops.processing.process_s2s_to_tif \
    --s2s-dir  $PROJECT/data/s2s_forecast \
    --out-dir  $PROJECT/data/s2s_processed \
    --reference $PROJECT/data/fwi_data/fwi_20250615.tif \
    --workers  16

echo ""
echo "[2/2] Processing ext GRIBs (10u/10v/tp) → TIFs ..."
python -m src.data_ops.processing.process_s2s_to_tif \
    --ext-only \
    --ext-dir  $PROJECT/data/ecmwf_observation \
    --s2s-dir  $PROJECT/data/s2s_forecast \
    --out-dir  $PROJECT/data/s2s_processed \
    --reference $PROJECT/data/fwi_data/fwi_20250615.tif \
    --workers  16

echo ""
echo "=== DONE: $(date) ==="
