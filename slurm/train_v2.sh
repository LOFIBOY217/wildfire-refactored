#!/bin/bash
#SBATCH --job-name=wildfire-v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=compute_full_node
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/jiaqi217/logs/train_v2_%j.out
#SBATCH --error=/scratch/jiaqi217/logs/train_v2_%j.err

module load StdEnv/2023 cuda/12.6

mkdir -p /scratch/jiaqi217/logs

PYTHON=$SCRATCH/miniforge3/envs/wildfore-r/bin/python

cd $SCRATCH/wildfire-refactored

CUDA_VISIBLE_DEVICES=0 $PYTHON src/training/train_s2s_hotspot_cwfis_v2.py \
  --config configs/paths_trillium.yaml \
  --num_workers 12 \
  --batch_size 512 \
  --epochs 10
