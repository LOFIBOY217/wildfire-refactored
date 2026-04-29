#!/bin/bash
#SBATCH --job-name=eval-epochs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/jiaqi217/logs/eval_epochs_%j.out
#SBATCH --error=/scratch/jiaqi217/logs/eval_epochs_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 proj/9.4.1
source $SCRATCH/venv-wildfire/bin/activate

mkdir -p /scratch/jiaqi217/logs

PYTHON=$SCRATCH/venv-wildfire/bin/python

cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/proj/9.4.1/share/proj

echo "=== EVAL EPOCHS ==="
echo "Node: $(hostname)"
echo "Evaluating epoch_01.pt ... epoch_05.pt on val set"
echo "==========================="

$PYTHON src/training/train_s2s_hotspot_cwfis_v2.py \
  --config configs/paths_trillium.yaml \
  --num_workers 4 \
  --batch_size 512 \
  --eval_epochs \
  --eval_n_windows 20
