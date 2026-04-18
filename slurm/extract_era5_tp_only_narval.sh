#!/bin/bash

# DEPRECATED (2026-04-18): use slurm/process_era5_narval.sh instead.
# This hardcoded-year script is preserved for git history/reference only.
# Example: START_YEAR=2009 END_YEAR=2017 sbatch slurm/process_era5_narval.sh

#SBATCH --job-name=wf-era5-tp
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/jiaqi217/logs/extract_era5_tp_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/extract_era5_tp_%j.err
# No dependency — resubmitting standalone after previous timeout

# ----------------------------------------------------------------
# Re-extract tp (total precipitation) from ERA5 GRIBs using fixed
# era5_to_daily.py that handles tp's extra 'step' dimension.
# Depends on wf-era5-ext (58873913) finishing first.
# ----------------------------------------------------------------

set -euo pipefail

if [ -z "${SCRATCH:-}" ]; then
    export SCRATCH=/scratch/jiaqi217
fi

module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1 eccodes/2.31.0
export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcc12/eccodes/2.31.0/lib64:${LD_LIBRARY_PATH:-}
source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored

echo "============================================="
echo "  ERA5 tp-only Extraction (fixed code)"
echo "  Node: $(hostname)  Time: $(date)"
echo "============================================="

# Step 1: Remove old broken tp TIFs (if any exist from failed runs)
echo "Removing old tp TIFs from era5_daily..."
rm -f data/era5_daily/tp_*.tif
echo "  Done"

# Step 2: Re-extract tp from GRIBs with fixed era5_to_daily.py
echo ""
echo "=== Step 1: GRIB -> daily TIFs ==="
python3 -u -m src.data_ops.processing.era5_to_daily \
    --config configs/paths_narval.yaml \
    --input-dir data/era5_on_fwi_grid \
    --output-dir data/era5_daily 2>&1
echo "era5_to_daily exit: $?"

# Step 3: Resample tp to FWI grid
echo ""
echo "=== Step 2: Resample tp to FWI grid ==="
mkdir -p data/era5_tp

python3 -u -m src.data_ops.processing.resample_to_fwi_grid \
    --input-dir data/era5_daily \
    --output-dir data/era5_tp \
    --variable tp \
    --config configs/paths_narval.yaml 2>&1

echo ""
echo "=== Summary ==="
n=$(ls data/era5_tp/tp_*.tif 2>/dev/null | wc -l)
echo "  tp: $n TIFs on FWI grid"

echo ""
echo "Done: $(date)"
