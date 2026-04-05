#!/bin/bash
#SBATCH --job-name=wf-baseline
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/jiaqi217/logs/baseline_full_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/baseline_full_%j.err
#SBATCH --mail-type=END,FAIL
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
    --eval_mode per_leadday \
    --k_values 1000 2500 5000 10000 25000 \
    --n_sample_wins 9999 \
    --pred_start 2022-05-01 \
    --pred_end 2025-10-31 \
    --dilate_radius 14

echo ""
echo "Done: $(date)"
