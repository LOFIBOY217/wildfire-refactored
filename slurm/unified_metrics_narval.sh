#!/bin/bash
#SBATCH --job-name=wf-unified
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --output=/scratch/jiaqi217/logs/unified_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/unified_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# Run scripts/compute_unified_metrics.py on a single full-eval scores dir.
#
# Usage:
#   RUN=v3_9ch_enc14_4y_2018  TAG=enc14_4y  sbatch slurm/unified_metrics_narval.sh
#   RUN=v3_9ch_enc21_4y_2018  TAG=enc21_4y  sbatch slurm/unified_metrics_narval.sh
#   ...
#
# Why CPU-only: pure post-processing (read npy + numpy). Previous v2 ran 2h
# walltime, processed only 125/604 windows → ~62 win/h → 604 win needs ~10h.
# Set walltime 12h for safety.
# ----------------------------------------------------------------

set -uo pipefail
RUN=${RUN:?Must set RUN, e.g. v3_9ch_enc14_4y_2018}
TAG=${TAG:?Must set TAG, e.g. enc14_4y}

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1
cd "$SCRATCH/wildfire-refactored"
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

source slurm/lib_copy_cache.sh
copy_venv $SCRATCH/venv-wildfire

SCORES_DIR="outputs/window_scores_full/${RUN}"
LABEL_NPY="data/fire_labels/fire_labels_nbac_nfdb_2000-05-01_2025-12-21_2281x2709_r14.npy"
CLIM_TIF="data/fire_clim_annual_nbac/fire_clim_upto_2022.tif"
OUT_PREFIX="outputs/unified_${TAG}"

if [ ! -d "$SCORES_DIR" ]; then
    echo "ERROR: scores dir missing: $SCORES_DIR"; exit 1
fi
NWIN=$(ls "$SCORES_DIR" | wc -l)
echo "============================================="
echo "  UNIFIED METRICS: $RUN"
echo "  scores dir : $SCORES_DIR ($NWIN windows)"
echo "  label      : $LABEL_NPY"
echo "  climatology: $CLIM_TIF"
echo "  out prefix : $OUT_PREFIX"
echo "============================================="

$PYTHON -u -m scripts.compute_unified_metrics \
    --scores_dir "$SCORES_DIR" \
    --fire_label_npy "$LABEL_NPY" \
    --climatology_tif "$CLIM_TIF" \
    --output_prefix "$OUT_PREFIX"

echo "=== done $(date) ==="
ls -la "${OUT_PREFIX}"*.csv 2>/dev/null
