#!/bin/bash
# ----------------------------------------------------------------
# Submit all 5 forward-chaining LOYO folds (VAL_YEAR=2020..2024).
#
# Each fold trains on 2014-05-01 .. (Y-1)-12-31 and evaluates on
# Y's fire season. Together they replace the single 2022-05-01
# train/val split with a proper rolling-origin evaluation.
#
# Usage:
#   bash scripts/submit_loyo_all.sh
# ----------------------------------------------------------------

set -euo pipefail

mkdir -p /scratch/jiaqi217/wildfire-refactored/outputs/loyo

for Y in 2020 2021 2022 2023 2024; do
    echo "=== submit LOYO fold VAL_YEAR=$Y ==="
    VAL_YEAR=$Y sbatch slurm/train_v3_loyo_fold_narval.sh
done

echo
echo "Submitted 5 LOYO folds. Track via:"
echo "  squeue -u jiaqi217 -n wf-loyo"
echo "After all complete, aggregate via:"
echo "  python3 scripts/aggregate_loyo.py"
