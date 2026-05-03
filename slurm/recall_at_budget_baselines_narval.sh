#!/bin/bash
#SBATCH --job-name=wf-recall-bl
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=/scratch/jiaqi217/logs/recall_baseline_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/recall_baseline_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# Run Recall@budget for one baseline on the SAME val windows + labels as
# the model evaluation.
# Usage:
#   METHOD=climatology  TAG=climatology  sbatch slurm/recall_at_budget_baselines_narval.sh
#   METHOD=persistence  TAG=persistence  sbatch slurm/recall_at_budget_baselines_narval.sh
#   METHOD=ecmwf_s2s    TAG=ecmwf_s2s    sbatch slurm/recall_at_budget_baselines_narval.sh

set -uo pipefail
METHOD=${METHOD:?must set METHOD}
TAG=${TAG:?must set TAG}

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1 eccodes/2.31.0
cd "$SCRATCH/wildfire-refactored"
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

source slurm/lib_copy_cache.sh
copy_venv $SCRATCH/venv-wildfire

SCORES_DIR="$SCRATCH/wildfire-refactored/outputs/window_scores_full/v3_9ch_enc21_12y_2014"
LABEL_NPY="$SCRATCH/wildfire-refactored/data/fire_labels/fire_labels_nbac_nfdb_2000-05-01_2025-12-21_2281x2709_r14.npy"
OUT_PREFIX="$SCRATCH/wildfire-refactored/outputs/recall_at_budget_${TAG}"

echo "=== Recall@budget BASELINE: $METHOD ==="

$PYTHON -u -m scripts.recall_at_budget_baselines \
    --method "$METHOD" \
    --scores_dir "$SCORES_DIR" \
    --label_npy "$LABEL_NPY" \
    --label_data_start 2000-05-01 \
    --fire_clim_dir "$SCRATCH/wildfire-refactored/data/fire_clim_annual_nbac" \
    --ecmwf_dir "$SCRATCH/wildfire-refactored/data/ecmwf_s2s_fire_epsg3978/fwinx" \
    --pred_start 2022-05-01 --pred_end 2025-10-31 \
    --lead_start 14 --lead_end 46 \
    --budgets 0.001 0.005 0.01 0.05 0.10 \
    --patch_size 16 --n_rows 142 --n_cols 169 \
    --output_prefix "$OUT_PREFIX"

PY_EXIT=$?
echo "=== done $(date) exit=$PY_EXIT ==="
exit $PY_EXIT
