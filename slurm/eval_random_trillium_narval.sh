#!/bin/bash
#SBATCH --job-name=eval-random-tri
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/jiaqi217/logs/eval_random_tri_%j.out
#SBATCH --error=/scratch/jiaqi217/logs/eval_random_tri_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# Evaluates Trillium job 389031 (random_decoder_reg_1gpu, 12 epochs, heavy reg)
# Checkpoint migrated from Trillium to Narval on 2026-04-02

# Ensure $SCRATCH is set (may be unset if submitted from non-login SSH session)
export SCRATCH=${SCRATCH:-/scratch/jiaqi217}

# Ensure module command is available (may be missing in non-login shells)
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh

module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 proj/9.4.1 eccodes/2.31.0

mkdir -p /scratch/jiaqi217/logs

SCRATCH_CACHE=/scratch/jiaqi217/meteo_cache
LOCAL_CACHE=$SLURM_TMPDIR/cache

cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

source slurm/lib_copy_cache.sh
copy_venv $SCRATCH/venv-wildfire

ts "=== PREFLIGHT ==="
ts "Node: $(hostname)"
$PYTHON -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())" || exit 1
ts "=== PREFLIGHT OK ==="

DATA_START=2018-01-01
copy_meteo_caches $SCRATCH_CACHE $LOCAL_CACHE 3600 $DATA_START

ts "=== FULL EVAL (random_decoder_reg_1gpu — Trillium ep1-12, all 1285 val windows) ==="
$PYTHON src/training/train_s2s_hotspot_cwfis_v2.py \
  --config configs/paths_narval.yaml \
  --run_name random_decoder_reg_1gpu \
  --decoder random \
  --pred_end 2025-10-31 \
  --num_workers 8 \
  --batch_size 8192 \
  --cache_dir $LOCAL_CACHE \
  --skip_forecast \
  --eval_epochs \
  --eval_n_windows 9999

ts "=== EVAL FINISHED (exit code: $?) ==="
