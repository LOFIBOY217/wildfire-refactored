#!/bin/bash
#SBATCH --job-name=wf-ecmwf-s2s-baseline
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=/scratch/jiaqi217/logs/ecmwf_s2s_baseline_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/ecmwf_s2s_baseline_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# ECMWF S2S Fire Danger BASELINE — full pipeline
# ----------------------------------------------------------------
# Runs in 3 stages:
#   1. (--convert-only): reproject already-downloaded NetCDF → EPSG:3978 GeoTIFF
#   2. Evaluate: compute Lift@5000 / Lift@30km vs NBAC + NFDB labels
#
# Download itself MUST be done on a login node (compute nodes have
# no internet). This SLURM job assumes the NetCDF files are already at
#   $SCRATCH/wildfire-refactored/data/ecmwf_s2s_fire/<var>/s2s_*.nc
# Run the download separately on tri-login03 (or your login node) via:
#   python -m src.data_ops.download.download_ecmwf_s2s_fire \
#       --start_year 2022 --end_year 2025 \
#       --output_dir data/ecmwf_s2s_fire --workers 2
# ----------------------------------------------------------------

set -uo pipefail
export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1
cd "$SCRATCH/wildfire-refactored"
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

source slurm/lib_copy_cache.sh
copy_venv $SCRATCH/venv-wildfire

INPUT_DIR="$SCRATCH/wildfire-refactored/data/ecmwf_s2s_fire"
OUTPUT_DIR="$SCRATCH/wildfire-refactored/data/ecmwf_s2s_fire_epsg3978"
REFERENCE="$SCRATCH/wildfire-refactored/data/fwi_data/fwi_20250615.tif"
LABEL_NPY="$SCRATCH/wildfire-refactored/data/fire_labels/fire_labels_nbac_nfdb_2000-05-01_2025-12-21_2281x2709_r14.npy"

if [ ! -d "$INPUT_DIR/fwinx" ] || [ -z "$(ls -A "$INPUT_DIR/fwinx" 2>/dev/null)" ]; then
    echo "ERROR: $INPUT_DIR/fwinx is empty."
    echo "Run on login node first:"
    echo "  python -m src.data_ops.download.download_ecmwf_s2s_fire \\"
    echo "    --start_year 2022 --end_year 2025 --workers 2"
    exit 1
fi
if [ ! -f "$REFERENCE" ]; then
    echo "ERROR: reference $REFERENCE missing"; exit 1
fi
if [ ! -f "$LABEL_NPY" ]; then
    echo "ERROR: label $LABEL_NPY missing"; exit 1
fi

# Stage 1: reproject NC → EPSG:3978 TIFs
echo "============================================="
echo "  STAGE 1 — Reproject NetCDF → EPSG:3978 TIFs"
echo "============================================="
$PYTHON -u -m src.data_ops.processing.ecmwf_s2s_to_epsg3978 \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --reference "$REFERENCE" \
    --variables fwinx
PY_EXIT=$?
[ "$PY_EXIT" -ne 0 ] && { echo "Stage 1 FAILED"; exit $PY_EXIT; }

# Stage 2: evaluate
echo "============================================="
echo "  STAGE 2 — Evaluate baseline vs NBAC+NFDB labels"
echo "============================================="
$PYTHON -u -m scripts.eval_ecmwf_s2s_baseline \
    --s2s_dir "$OUTPUT_DIR/fwinx" \
    --label_npy "$LABEL_NPY" \
    --pred_start 2022-05-01 --pred_end 2025-10-31 \
    --lead_start 14 --lead_end 46 \
    --val_lift_k 5000 \
    --output_prefix "$SCRATCH/wildfire-refactored/outputs/baseline_ecmwf_s2s"
PY_EXIT=$?

echo "=== Done $(date) exit=$PY_EXIT ==="
exit $PY_EXIT
