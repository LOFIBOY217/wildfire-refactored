#!/bin/bash
#SBATCH --job-name=wf-baseline
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/jiaqi217/logs/baseline_full_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/baseline_full_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# Full evaluation of Climatology + FWI Oracle baselines
# Uses ALL val windows (n_sample_wins=9999) for stable metrics
# ----------------------------------------------------------------

set -euo pipefail

if [ -z "${SCRATCH:-}" ]; then
    export SCRATCH=/scratch/jiaqi217
fi

module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1 2>/dev/null || true
source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored

echo "============================================="
echo "  Baseline Full Evaluation"
echo "  Node: $(hostname)  Time: $(date)"
echo "============================================="

python3 -u -m src.evaluation.benchmark_baselines \
    --config configs/paths_narval.yaml \
    --baseline fwi_oracle climatology \
    --eval_mode per_window \
    --k_values 1000 2500 5000 10000 25000 \
    --n_sample_wins 9999 \
    --pred_start 2022-05-01 \
    --pred_end 2024-10-31 \
    --dilate_radius 14 \
    --climatology_tif "${CLIMATOLOGY_TIF:-data/fire_clim_annual_nbac/fire_clim_upto_2024.tif}" \
    ${FIRE_LABEL_NPY:+--fire_label_npy "$FIRE_LABEL_NPY"}

# 2026-04-21 Plan A: pred_end=2024-10-31 because NBAC only ships through 2024.
# Defaults to NBAC-based fire_clim and (when FIRE_LABEL_NPY env is set)
# pre-built NBAC+NFDB label .npy. Overridable via env vars to A/B against
# legacy CWFIS sources.
#
# Usage examples:
#   # Plan A (default): NBAC-based clim, NBAC+NFDB labels
#   FIRE_LABEL_NPY=data/fire_labels/fire_labels_nbac_nfdb_2000-05-01_2025-12-21_2281x2709_r14.npy \
#       sbatch slurm/eval_baselines_full_narval.sh
#
#   # Old comparison: CWFIS clim and CWFIS labels (apples-to-apples with old SOTA)
#   CLIMATOLOGY_TIF=data/fire_clim_annual/fire_clim_upto_2022.tif \
#   FIRE_LABEL_NPY=data/fire_labels/fire_labels_cwfis_2000-05-01_2025-12-21_2281x2709_r14.npy \
#       sbatch slurm/eval_baselines_full_narval.sh

echo ""
echo "Done: $(date)"
