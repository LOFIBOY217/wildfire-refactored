#!/bin/bash
#SBATCH --job-name=wf-cmp-labels
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=/scratch/jiaqi217/logs/cmp_labels_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/cmp_labels_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# Quantify CWFIS drift to answer:
#   "At the 12y (2014-2025) training scale, is CWFIS still bad enough to
#    justify the NBAC+NFDB switch — or would CWFIS have been acceptable?"
#
# Two complementary views:
#   1) analyze_label_drift.py  — raw-source year-by-year counts (CWFIS
#      hotspots vs NFDB reported fires vs NBAC burned area). Doesn't need
#      pre-built label .npy. This is the cleanest drift signal.
#   2) compare_labels.py       — pixel-level IoU of dilated CWFIS-only
#      vs NBAC+NFDB label stacks. Only runs if both .npy exist.

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

AUDIT_DIR="$SCRATCH/wildfire-refactored/data/audit"
mkdir -p "$AUDIT_DIR"

# ----------------------------------------------------------------
# 1) Raw-source drift (always runs)
# ----------------------------------------------------------------
echo "=================================================="
echo "  PART 1: raw-source drift (CWFIS / NFDB / NBAC)"
echo "=================================================="

$PYTHON -u scripts/analyze_label_drift.py \
    --hotspot data/hotspot/hotspot_2000_2025.csv \
    --nfdb data/nfdb/NFDB_point.zip \
    --nbac data/burn_scars_raw/NBAC_1972to2024_shp.zip \
    --out "$AUDIT_DIR/label_drift_full.csv" \
    --start 2000 --end 2024

# ----------------------------------------------------------------
# 2) Pixel-level IoU on dilated stacks (only if both .npy exist)
# ----------------------------------------------------------------
LABELS_DIR="$SCRATCH/wildfire-refactored/data/fire_labels"
OLD="$LABELS_DIR/fire_labels_cwfis_2000-05-01_2025-12-21_2281x2709_r14.npy"
NEW="$LABELS_DIR/fire_labels_nbac_nfdb_2000-05-01_2025-12-21_2281x2709_r14.npy"

if [ -f "$OLD" ] && [ -f "$NEW" ]; then
    echo
    echo "=================================================="
    echo "  PART 2: pixel-level IoU (dilated label stacks)"
    echo "=================================================="
    $PYTHON -u scripts/compare_labels.py \
        --old "$OLD" --new "$NEW" \
        --out "$AUDIT_DIR/label_comparison_full.csv"
else
    echo
    echo "=================================================="
    echo "  PART 2: skipped — CWFIS-only .npy not found"
    echo "  Build it via:"
    echo "    SCHEME=cwfis sbatch slurm/build_fire_labels_narval.sh"
    echo "  then re-run this script for IoU view."
    echo "=================================================="
fi

echo
echo "=== Done $(date) ==="
echo "Drift table:    $AUDIT_DIR/label_drift_full.csv"
[ -f "$AUDIT_DIR/label_comparison_full.csv" ] && \
    echo "IoU table:      $AUDIT_DIR/label_comparison_full.csv"
