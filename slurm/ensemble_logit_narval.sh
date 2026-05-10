#!/bin/bash
#SBATCH --job-name=wf-ens-logit
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-03:00:00
#SBATCH --output=/scratch/jiaqi217/logs/ens_logit_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/ens_logit_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# Logit-mean ensemble across the same 10 ckpts as the prob-mean run.
# Hypothesis: logit-mean preserves per-ckpt high-confidence peaks → should
# recover Lift@30km that prob-mean smoothed away (4.37x → ?).

set -uo pipefail
export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5
cd "$SCRATCH/wildfire-refactored"
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PYTHONUNBUFFERED=1

source slurm/lib_copy_cache.sh
copy_venv $SCRATCH/venv-wildfire

SCORES_ROOT="$SCRATCH/wildfire-refactored/outputs/window_scores_full"

# Same 10 ckpts as the prob-mean run (9.57x Lift@5000, 4.37x Lift@30km).
DIRS=(
    "$SCORES_ROOT/v3_9ch_enc21_12y_2014"
    "$SCORES_ROOT/v3_9ch_enc21_12y_2014_climsim"
    "$SCORES_ROOT/v3_9ch_enc21_12y_2014_climblend_a0.3"
    "$SCORES_ROOT/v3_9ch_enc21_12y_2014_climblend_a0.5"
    "$SCORES_ROOT/v3_9ch_enc28_12y_2014"
    "$SCORES_ROOT/v3_9ch_enc35_12y_2014"
    "$SCORES_ROOT/v3_13ch_enc14_12y_2014"
    "$SCORES_ROOT/v3_13ch_enc21_12y_2014"
    "$SCORES_ROOT/v3_13ch_enc28_12y_2014"
    "$SCORES_ROOT/v3_13ch_enc35_12y_2014"
)

# Filter to existing dirs (some ckpts may not have full eval yet)
EXISTING=()
for d in "${DIRS[@]}"; do
    if [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ]; then
        EXISTING+=("$d")
    else
        echo "[skip missing] $d"
    fi
done
echo "Using ${#EXISTING[@]} ckpt dirs"

OUT="$SCRATCH/wildfire-refactored/outputs/ensemble_logit_10ckpt.json"
$PYTHON -u -m scripts.ensemble_ckpts_lift \
    --score_dirs "${EXISTING[@]}" \
    --pred_start 2022-05-01 --pred_end 2025-10-31 \
    --ensemble_mode logit_mean \
    --k 5000 \
    --output "$OUT"

echo "=== Done $(date) exit=$? ==="
