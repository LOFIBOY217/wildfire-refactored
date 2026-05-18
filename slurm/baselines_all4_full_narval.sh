#!/bin/bash
#SBATCH --job-name=wf-baselines-all4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/jiaqi217/logs/baselines_all4_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/baselines_all4_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# Full evaluation of all 4 stateless baselines on the SOTA val
# window set (2022-05-01 → 2025-09-30, ~583 windows).
#
# Produces two CSVs:
#   outputs/baselines_per_window.csv   — per-window aggregated metrics
#                                        (Lift@K, Lift@30km, F1/F2/MCC, BSS, ...)
#   outputs/baselines_per_leadday.csv  — per-lead-day breakdown
#                                        (for §6 Lift-vs-lead-day figure)
#
# Window/lead config matches the 12y_2014 SOTA model exactly:
#   in_days=21, lead_start=14, lead_end=46 (33-day decoder window)
#   dilate_radius=14, fire_season_only, label_fusion via fire_clim path
#
# Why both modes?
#   per_window  → headline numbers for §6 baselines table
#   per_leadday → lift-decay-vs-lead curve (climatology + persistence +
#                 fwi_threshold give flat baselines; fwi_oracle gives
#                 the ceiling at each lead)
# ----------------------------------------------------------------

set -uo pipefail
export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1
source $SCRATCH/venv-wildfire/bin/activate

mkdir -p /scratch/jiaqi217/logs
mkdir -p $SCRATCH/wildfire-refactored/outputs

PYTHON=$SCRATCH/venv-wildfire/bin/python

cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

echo "=== PREFLIGHT ==="
echo "Node: $(hostname)"
$PYTHON -c "import rasterio; print('rasterio:', rasterio.__version__)" || exit 1
$PYTHON -c "import sklearn; print('sklearn:', sklearn.__version__)" || exit 1
echo "=== PREFLIGHT OK ==="

# ----------------------------------------------------------------
# Shared args (match SOTA model)
# ----------------------------------------------------------------
COMMON_ARGS=(
    --config configs/paths_narval.yaml
    --baseline climatology persistence fwi_threshold fwi_oracle
    --pred_start 2022-05-01
    --pred_end   2025-09-30
    --in_days 21
    --lead_start 14
    --lead_end   46
    --patch_size 16
    --dilate_radius 14
    --k_values 1000 2500 5000 10000 25000
    --n_sample_wins 1000   # >> 583 → effectively full eval
    --fire_season_only
)

# ----------------------------------------------------------------
# 1. per_window eval (headline §6 baselines table)
# ----------------------------------------------------------------
echo "============================================="
echo " 1/2  per-window eval  (all 4 baselines)"
echo "============================================="
$PYTHON -m src.evaluation.benchmark_baselines \
    "${COMMON_ARGS[@]}" \
    --eval_mode per_window \
    --output_csv outputs/baselines_per_window.csv
PW_EXIT=$?
echo "[per_window] exit=$PW_EXIT"

# ----------------------------------------------------------------
# 2. per_leadday eval (Lift-vs-lead-day figure)
# ----------------------------------------------------------------
echo "============================================="
echo " 2/2  per-leadday eval  (all 4 baselines)"
echo "============================================="
$PYTHON -m src.evaluation.benchmark_baselines \
    "${COMMON_ARGS[@]}" \
    --eval_mode per_leadday \
    --output_csv outputs/baselines_per_leadday.csv
PL_EXIT=$?
echo "[per_leadday] exit=$PL_EXIT"

echo "=== DONE $(date)   per_window=$PW_EXIT  per_leadday=$PL_EXIT ==="
exit $(( PW_EXIT | PL_EXIT ))
