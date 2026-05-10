#!/bin/bash
#SBATCH --job-name=wf-full-metric
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=/scratch/jiaqi217/logs/full_metric_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/full_metric_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# Computes ALL paper metrics for one ckpt or baseline.
# Usage:
#   SOURCE=model RUN=v3_9ch_enc21_12y_2014_climsim sbatch slurm/full_metric_card_narval.sh
#   SOURCE=climatology REF_RUN=v3_9ch_enc21_12y_2014 sbatch slurm/full_metric_card_narval.sh
#   SOURCE=persistence REF_RUN=v3_9ch_enc21_12y_2014 sbatch slurm/full_metric_card_narval.sh
#   SOURCE=ecmwf_s2s   REF_RUN=v3_9ch_enc21_12y_2014 sbatch slurm/full_metric_card_narval.sh

set -uo pipefail
SOURCE=${SOURCE:?Must set SOURCE (model|climatology|persistence|ecmwf_s2s)}
RUN=${RUN:-}
REF_RUN=${REF_RUN:-v3_9ch_enc21_12y_2014_climsim}

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1 eccodes/2.31.0
cd "$SCRATCH/wildfire-refactored"
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

source slurm/lib_copy_cache.sh
copy_venv $SCRATCH/venv-wildfire

LABEL_NPY="$SCRATCH/wildfire-refactored/data/fire_labels/fire_labels_nbac_nfdb_2000-05-01_2025-12-21_2281x2709_r14.npy"

case "$SOURCE" in
    model)
        [ -z "$RUN" ] && { echo "ERROR: SOURCE=model needs RUN="; exit 1; }
        SCORES_DIR="$SCRATCH/wildfire-refactored/outputs/window_scores_full/$RUN"
        OUT="$SCRATCH/wildfire-refactored/outputs/metric_card_${RUN}.json"
        $PYTHON -u -m scripts.compute_full_metric_card \
            --source model --scores_dir "$SCORES_DIR" --output "$OUT"
        ;;
    climatology)
        REF_DIR="$SCRATCH/wildfire-refactored/outputs/window_scores_full/$REF_RUN"
        OUT="$SCRATCH/wildfire-refactored/outputs/metric_card_climatology.json"
        $PYTHON -u -m scripts.compute_full_metric_card \
            --source climatology --reference_scores_dir "$REF_DIR" --output "$OUT"
        ;;
    persistence)
        REF_DIR="$SCRATCH/wildfire-refactored/outputs/window_scores_full/$REF_RUN"
        OUT="$SCRATCH/wildfire-refactored/outputs/metric_card_persistence.json"
        $PYTHON -u -m scripts.compute_full_metric_card \
            --source persistence --reference_scores_dir "$REF_DIR" \
            --label_npy "$LABEL_NPY" --output "$OUT"
        ;;
    ecmwf_s2s)
        REF_DIR="$SCRATCH/wildfire-refactored/outputs/window_scores_full/$REF_RUN"
        OUT="$SCRATCH/wildfire-refactored/outputs/metric_card_ecmwf_s2s.json"
        $PYTHON -u -m scripts.compute_full_metric_card \
            --source ecmwf_s2s --reference_scores_dir "$REF_DIR" \
            --output "$OUT"
        ;;
esac

PY_EXIT=$?
echo "=== Done $(date) exit=$PY_EXIT ==="
exit $PY_EXIT
