#!/bin/bash
#SBATCH --job-name=wf-recall-boreal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=/scratch/jiaqi217/logs/recall_boreal_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/recall_boreal_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# Recall@budget restricted to boreal-belt (top X% of climatology).
# Tests dynamic forecast skill within already-known high-fire regions.
# Usage:
#   METHOD=model       sbatch slurm/recall_at_budget_boreal_narval.sh
#   METHOD=climatology sbatch slurm/recall_at_budget_boreal_narval.sh
#   METHOD=persistence sbatch slurm/recall_at_budget_boreal_narval.sh
#   METHOD=ecmwf_s2s   sbatch slurm/recall_at_budget_boreal_narval.sh

set -uo pipefail
METHOD=${METHOD:?must set METHOD (model|climatology|persistence|ecmwf_s2s)}

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1 eccodes/2.31.0
cd "$SCRATCH/wildfire-refactored"
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

source slurm/lib_copy_cache.sh
copy_venv $SCRATCH/venv-wildfire

MASK_NPY="$SCRATCH/wildfire-refactored/data/masks/boreal_belt_top30pct.npy"
SCORES_DIR="$SCRATCH/wildfire-refactored/outputs/window_scores_full/v3_9ch_enc21_12y_2014"
LABEL_NPY="$SCRATCH/wildfire-refactored/data/fire_labels/fire_labels_nbac_nfdb_2000-05-01_2025-12-21_2281x2709_r14.npy"
OUT_PREFIX="$SCRATCH/wildfire-refactored/outputs/recall_boreal_${METHOD}"

# Build mask if missing
if [ ! -f "$MASK_NPY" ]; then
    echo "Building boreal mask..."
    $PYTHON -u -m scripts.build_boreal_mask \
        --climatology_tif "$SCRATCH/wildfire-refactored/data/fire_clim_annual_nbac/fire_clim_upto_2022.tif" \
        --top_frac 0.30 \
        --output "$MASK_NPY"
fi

if [ "$METHOD" = "model" ]; then
    $PYTHON -u -m scripts.recall_at_budget \
        --scores_dir "$SCORES_DIR" \
        --pred_start 2022-05-01 --pred_end 2025-10-31 \
        --lead_start 14 --lead_end 46 \
        --budgets 0.001 0.005 0.01 0.05 0.10 \
        --patch_size 16 --n_rows 142 --n_cols 169 \
        --restrict_mask_npy "$MASK_NPY" \
        --output_prefix "$OUT_PREFIX"
else
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
        --restrict_mask_npy "$MASK_NPY" \
        --output_prefix "$OUT_PREFIX"
fi

PY_EXIT=$?
echo "=== done $(date) exit=$PY_EXIT ==="
exit $PY_EXIT
