#!/bin/bash
#SBATCH --job-name=wf-build-labels
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/jiaqi217/logs/build_labels_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/build_labels_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# Build fire-label .npy files for training (post LABEL_DECISION_2026_04_21).
#
# Memory: full 25-year label stack is ~60 GB uint8, plus dilation needs
# another ~60 GB scratch → request 200G. Login node OOMs at this size.
#
# Required env:
#   SCHEME = cwfis | nbac_nfdb
#
# Optional env:
#   START = 2000-05-01
#   END   = 2025-12-21
#   DIL_R = 14
#   NFDB_MIN_HA = 1.0  (nbac_nfdb only)
#
# Usage:
#   SCHEME=cwfis sbatch slurm/build_fire_labels_narval.sh
#   SCHEME=nbac_nfdb sbatch slurm/build_fire_labels_narval.sh
# ----------------------------------------------------------------

set -uo pipefail

SCHEME="${SCHEME:?Must set SCHEME=cwfis or nbac_nfdb}"
START="${START:-2000-05-01}"
END="${END:-2025-12-21}"
DIL_R="${DIL_R:-14}"
NFDB_MIN_HA="${NFDB_MIN_HA:-1.0}"

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1

source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

echo "=============================================="
echo "  Building fire labels: $SCHEME"
echo "  Range: $START .. $END"
echo "  Dilation radius: $DIL_R"
echo "  NFDB min size ha: $NFDB_MIN_HA"
echo "  Node: $(hostname)  Time: $(date)"
echo "=============================================="

EXTRA_ARGS=""
if [ "$SCHEME" = "nbac_nfdb" ]; then
    EXTRA_ARGS="--nfdb_min_size_ha $NFDB_MIN_HA --exclude_prescribed"
fi

python3 -u scripts/build_fire_labels.py \
    --scheme "$SCHEME" \
    --start "$START" --end "$END" \
    --dilate_radius "$DIL_R" \
    --output_dir data/fire_labels \
    --overwrite \
    $EXTRA_ARGS

EXIT=$?
echo ""
echo "=== Done: $(date) exit=$EXIT ==="
exit $EXIT
