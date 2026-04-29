#!/bin/bash
#SBATCH --job-name=wf-null-input
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=498G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/jiaqi217/logs/train_null_input_%j.out
#SBATCH --error=/scratch/jiaqi217/logs/train_null_input_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# Ensure $SCRATCH is set (may be unset if submitted from non-login SSH session)
export SCRATCH=${SCRATCH:-/scratch/jiaqi217}

# Ensure module command is available (may be missing in non-login shells)
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh

module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 proj/9.4.1 eccodes/2.31.0

mkdir -p /scratch/jiaqi217/logs

SCRATCH_CACHE=/scratch/jiaqi217/meteo_cache
LOCAL_CACHE=$SLURM_TMPDIR/cache
DATA_START=2018-05-01

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

copy_meteo_caches $SCRATCH_CACHE $LOCAL_CACHE 3600 $DATA_START

# NULL INPUT BASELINE (Input Randomization Test)
# ─────────────────────────────────────────────
# Both encoder AND decoder are replaced with i.i.d. N(0,1) random noise.
# This is a diagnostic experiment:
#
#   "Input Randomization Test" (Adebayo et al., NeurIPS 2018)
#
# The model receives ZERO real information from weather or fire history.
# Any Lift above 1.0x reveals spatial priors memorized in model weights.
#
# Expected outcomes:
#   Lift ≈ 1.0x → model correctly requires inputs to discriminate patches
#               → confirms fire_clim in encoder is the main driver of ~8x
#   Lift ≈ 8x  → model has learned a fixed spatial heatmap in its weights
#               → current ~8x performance is NOT from reading inputs at all
#
# The result tells us: is the 8x Lift real (from reading inputs) or spurious
# (from weight memorization)? This is the most important diagnostic we can run.
ts "=== STARTING NULL INPUT BASELINE (enc=random, dec=random) ==="
$PYTHON src/training/train_s2s_hotspot_cwfis_v2.py \
  --config configs/paths_narval.yaml \
  --run_name null_input_baseline \
  --decoder random \
  --random_encoder \
  --pred_end 2025-10-31 \
  --num_workers 8 \
  --batch_size 8192 \
  --epochs 4 \
  --lr 1e-4 \
  --lr_min 1e-6 \
  --dropout 0.0 \
  --weight_decay 0.0 \
  --log_interval 1000 \
  --cache_dir $LOCAL_CACHE \
  --skip_forecast

TRAIN_EXIT=$?
ts "=== FINISHED (exit code: $TRAIN_EXIT) ==="
exit $TRAIN_EXIT
