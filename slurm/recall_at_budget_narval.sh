#!/bin/bash
#SBATCH --job-name=wf-recall-budget
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=/scratch/jiaqi217/logs/recall_budget_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/recall_budget_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# Recall@budget — operational metric (advisor-requested)
# ----------------------------------------------------------------
# For each val window: rank Canada land pixels by predicted prob,
# select top X% (X ∈ {0.1, 0.5, 1, 5, 10}), count fraction of
# CONNECTED FIRE EVENTS captured. Reports per-window recall + 95% CI.
#
# Required env:
#   RUN     run name (must match a saved scores dir under
#           outputs/window_scores_full/<RUN>)
#   TAG     short identifier for the output files (e.g. enc21_12y)
#
# Usage:
#   RUN=v3_9ch_enc21_12y_2014 TAG=enc21_12y \
#     sbatch slurm/recall_at_budget_narval.sh
# ----------------------------------------------------------------

set -uo pipefail
RUN=${RUN:?Must set RUN, e.g. v3_9ch_enc21_12y_2014}
TAG=${TAG:?Must set TAG, e.g. enc21_12y}

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1
cd "$SCRATCH/wildfire-refactored"
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

source slurm/lib_copy_cache.sh
copy_venv $SCRATCH/venv-wildfire

SCORES_DIR="$SCRATCH/wildfire-refactored/outputs/window_scores_full/${RUN}"
LABEL_NPY="$SCRATCH/wildfire-refactored/data/fire_labels/fire_labels_nbac_nfdb_2000-05-01_2025-12-21_2281x2709_r14.npy"
OUT_PREFIX="$SCRATCH/wildfire-refactored/outputs/recall_at_budget_${TAG}"

if [ ! -d "$SCORES_DIR" ]; then
    echo "ERROR: scores dir missing: $SCORES_DIR"; exit 1
fi
if [ ! -f "$LABEL_NPY" ]; then
    echo "ERROR: label npy missing: $LABEL_NPY"; exit 1
fi
NWIN=$(ls "$SCORES_DIR" | wc -l)

echo "============================================="
echo "  Recall@budget"
echo "  RUN        : $RUN"
echo "  scores dir : $SCORES_DIR  ($NWIN files)"
echo "  label npy  : $LABEL_NPY"
echo "  output     : ${OUT_PREFIX}_*.json"
echo "============================================="

$PYTHON -u -m scripts.recall_at_budget \
    --scores_dir "$SCORES_DIR" \
    --label_npy "$LABEL_NPY" \
    --label_data_start 2000-05-01 \
    --pred_start 2022-05-01 --pred_end 2025-10-31 \
    --lead_start 14 --lead_end 46 \
    --budgets 0.001 0.005 0.01 0.05 0.10 \
    --output_prefix "$OUT_PREFIX"

PY_EXIT=$?
echo "=== done $(date) exit=$PY_EXIT ==="
exit $PY_EXIT
