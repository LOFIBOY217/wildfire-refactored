#!/bin/bash
#SBATCH --job-name=wf-s2s-v4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=17:00:00
#SBATCH --output=/scratch/jiaqi217/logs/train_s2s_v4_%j.out
#SBATCH --error=/scratch/jiaqi217/logs/train_s2s_v4_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

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
ts "Node     : $(hostname)"
$PYTHON -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())" || exit 1
ts "=== PREFLIGHT OK ==="

DATA_START=2018-01-01
copy_meteo_caches $SCRATCH_CACHE $LOCAL_CACHE 3600 $DATA_START
copy_s2s_cache $SCRATCH_CACHE $LOCAL_CACHE 1800

# Decide cache_dir: if pf.dat already in SCRATCH_CACHE, copy_meteo_caches put it in
# LOCAL_CACHE → use LOCAL_CACHE (fast local SSD reads during training).
# If not in SCRATCH_CACHE, build directly in SCRATCH_CACHE so it persists for future
# resume jobs (avoids re-running STEP 6 every time).
PF_IN_SCRATCH=$(ls $SCRATCH_CACHE/meteo_p16_C8_T*_${DATA_START}_*_pf.dat 2>/dev/null | head -1)
if [ -n "$PF_IN_SCRATCH" ]; then
    CACHE_DIR="$LOCAL_CACHE"
    ts "pf.dat found in SCRATCH_CACHE → using LOCAL_CACHE (STEP 6 skipped)"
else
    CACHE_DIR="$SCRATCH_CACHE"
    ts "pf.dat not in SCRATCH_CACHE → building there (STEP 6 runs once, then persists)"
fi

ts "=== RESUMING TRAINING (s2s_legacy v4 — light regularization) ==="
$PYTHON src/training/train_s2s_hotspot_cwfis_v2.py \
  --config configs/paths_narval.yaml \
  --run_name s2s_decoder_s2s_v4 \
  --decoder s2s_legacy \
  --s2s_cache $LOCAL_CACHE/s2s_decoder_cache.dat \
  --pred_end 2025-10-31 \
  --s2s_max_issue_lag 3 \
  --num_workers 8 \
  --batch_size 8192 \
  --epochs 8 \
  --lr 1e-4 \
  --lr_min 1e-6 \
  --dropout 0.1 \
  --weight_decay 0.01 \
  --lead_end 45 \
  --log_interval 1000 \
  --cache_dir $CACHE_DIR \
  --skip_forecast \
  --resume

TRAIN_EXIT=$?
ts "=== TRAINING FINISHED (exit code: $TRAIN_EXIT) ==="

# Copy pf.dat back to SCRATCH_CACHE if LOCAL has a larger T than what's already there
# (T2427 may already exist in SCRATCH_CACHE, but T2791 is bigger → should replace)
get_pf_T() { basename "$1" | grep -oP 'T\K[0-9]+(?=_[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{4})'; }
PF_LOCAL=$(ls $LOCAL_CACHE/meteo_p16_C8_T*_${DATA_START}_*_pf.dat 2>/dev/null | head -1)
if [ -n "$PF_LOCAL" ]; then
    T_local=$(get_pf_T "$PF_LOCAL")
    PF_SCRATCH_BEST=$(ls $SCRATCH_CACHE/meteo_p16_C8_T*_${DATA_START}_*_pf.dat 2>/dev/null \
        | sort -t'T' -k2 -n | tail -1)
    T_scratch=0
    [ -n "$PF_SCRATCH_BEST" ] && T_scratch=$(get_pf_T "$PF_SCRATCH_BEST")
    if [ -n "$T_local" ] && [ "$T_local" -gt "${T_scratch:-0}" ]; then
        ts "Caching larger pf.dat (T=$T_local > T_scratch=$T_scratch) to SCRATCH_CACHE (~274GB, background)..."
        cp "$PF_LOCAL" "$SCRATCH_CACHE/" && ts "pf.dat T=$T_local cached to SCRATCH_CACHE OK" &
    else
        ts "pf.dat copy skipped: T_local=$T_local <= T_scratch=$T_scratch already in SCRATCH_CACHE"
    fi
fi

exit $TRAIN_EXIT
