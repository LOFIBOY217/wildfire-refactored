#!/bin/bash
#SBATCH --job-name=wf-s2s-fwi
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/jiaqi217/logs/s2s_fwi_baseline_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/s2s_fwi_baseline_%j.err

# ----------------------------------------------------------------
# Step 1: Compute FWI from S2S forecast weather (not observed)
# Step 2: Evaluate S2S FWI as a baseline predictor
#
# This answers: "how well does S2S-predicted FWI rank fire locations?"
# Compare with FWI Oracle (uses observed weather) and Climatology.
# ----------------------------------------------------------------

set -euo pipefail

if [ -z "${SCRATCH:-}" ]; then
    export SCRATCH=/scratch/jiaqi217
fi

module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1 2>/dev/null || true
source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored

echo "============================================="
echo "  S2S FWI Baseline Evaluation"
echo "  Node: $(hostname)  Time: $(date)"
echo "============================================="

# Step 1: Compute S2S FWI from forecast weather
echo ""
echo "=== Step 1: Compute FWI from S2S forecasts ==="
python3 -u -m src.data_ops.processing.compute_s2s_fwi \
    --config configs/paths_narval.yaml \
    --s2s_dir data/s2s_processed \
    --output_dir data/s2s_fwi \
    --lead_start 14 --lead_end 45 2>&1
echo "compute_s2s_fwi exit: $?"

# Step 2: Evaluate as baseline
# S2S FWI evaluation needs a separate script or we use the existing
# evaluate_topk_cwfis.py on the S2S FWI output TIFs
echo ""
echo "=== Step 2: S2S FWI data summary ==="
echo "S2S FWI files:" && find data/s2s_fwi -name '*.tif' 2>/dev/null | wc -l

echo ""
echo "Done: $(date)"
