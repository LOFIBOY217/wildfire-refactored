#!/bin/bash
#SBATCH --job-name=fire-clim
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=compute_full_node
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/jiaqi217/logs/fire_clim_%j.out
#SBATCH --error=/scratch/jiaqi217/logs/fire_clim_%j.err

mkdir -p /scratch/jiaqi217/logs

PYTHON=$SCRATCH/miniforge3/envs/wildfore-r/bin/python
cd $SCRATCH/wildfire-refactored

$PYTHON -m src.data_ops.processing.make_fire_climatology \
    --config configs/paths_trillium.yaml \
    --reference data/fwi_data/fwi_20250615.tif
