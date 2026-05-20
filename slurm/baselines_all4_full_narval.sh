#!/bin/bash
#SBATCH --job-name=wf-baselines-all4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/jiaqi217/logs/baselines_all4_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/baselines_all4_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# Full evaluation of all 4 stateless baselines on the SOTA val
# window set, using the SAME labels + climatology as the model:
#   * NBAC+NFDB fused labels (--fire_label_npy), NOT legacy CWFIS
#   * leak-free climatology fire_clim_upto_2022.tif (only pre-2022
#     data → leak-free for the entire 2022-2025 val period)
#
# Run ONE eval mode per job (EVAL_MODE env), so the fast per_window
# result is not lost if the slower per_leadday run times out.
#
#   EVAL_MODE=per_window  sbatch slurm/baselines_all4_full_narval.sh
#   EVAL_MODE=per_leadday sbatch slurm/baselines_all4_full_narval.sh
#
# Outputs:
#   outputs/baselines_per_window.csv   — §6 baselines table headline
#   outputs/baselines_per_leadday.csv  — flat baseline curves for the
#                                        lift-vs-lead-day figure
# ----------------------------------------------------------------

set -uo pipefail
EVAL_MODE=${EVAL_MODE:-per_window}

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

# --- NBAC+NFDB labels + leak-free clim (match the model) ----------
FIRE_LABEL_NPY="data/fire_labels/fire_labels_nbac_nfdb_2000-05-01_2025-12-21_2281x2709_r14.npy"
CLIM_TIF="data/fire_clim_annual_nbac/fire_clim_upto_2022.tif"

echo "=== PREFLIGHT ==="
echo "Node: $(hostname)   EVAL_MODE=$EVAL_MODE"
$PYTHON -c "import rasterio; print('rasterio:', rasterio.__version__)" || exit 1
[ -f "$FIRE_LABEL_NPY" ] || { echo "ERROR: label npy missing: $FIRE_LABEL_NPY"; exit 1; }
[ -f "$CLIM_TIF" ]       || { echo "ERROR: clim tif missing: $CLIM_TIF"; exit 1; }
echo "=== PREFLIGHT OK ==="

$PYTHON -m src.evaluation.benchmark_baselines \
    --config configs/paths_narval.yaml \
    --baseline climatology persistence fwi_threshold fwi_oracle \
    --eval_mode "$EVAL_MODE" \
    --fire_label_npy "$FIRE_LABEL_NPY" \
    --climatology_tif "$CLIM_TIF" \
    --pred_start 2022-05-01 \
    --pred_end   2025-09-23 \
    --in_days 21 \
    --lead_start 14 \
    --lead_end   46 \
    --patch_size 16 \
    --dilate_radius 14 \
    --k_values 1000 2500 5000 10000 25000 \
    --n_sample_wins 1000 \
    --fire_season_only \
    --output_csv "outputs/baselines_${EVAL_MODE}.csv"

PY_EXIT=$?
echo "=== DONE $(date)  EVAL_MODE=$EVAL_MODE exit=$PY_EXIT ==="
exit $PY_EXIT
