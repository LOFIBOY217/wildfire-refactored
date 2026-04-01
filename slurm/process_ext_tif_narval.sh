#!/bin/bash
#SBATCH --job-name=wf-ext-tif
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/jiaqi217/logs/process_ext_tif_%j.out
#SBATCH --error=/scratch/jiaqi217/logs/process_ext_tif_%j.err
#SBATCH --account=def-inghaw

module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1 eccodes/2.31.0
export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcc12/eccodes/2.31.0/lib64:$LD_LIBRARY_PATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj

source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH

echo "Start: $(date)"
python -m src.data_ops.processing.process_s2s_to_tif \
    --config configs/paths_narval.yaml \
    --ext-only \
    --ext-dir data/ecmwf_observation \
    --s2s-dir data/s2s_processed \
    --overwrite \
    --workers 16
echo "Done: $(date)"
