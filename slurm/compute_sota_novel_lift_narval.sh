#!/bin/bash
#SBATCH --job-name=wf-sota-novel-lift
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --output=/scratch/jiaqi217/logs/sota_novel_lift_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/sota_novel_lift_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# Compute novel-ignition Lift (7d/30d/90d lookback) for the SOTA
# checkpoint from its full-window per-window score dump, so the
# cross-model novel comparison uses REAL SOTA numbers (not the
# 4y_2018 20-window proxy).
#
# Inputs : outputs/v3_9ch_enc21_12y_2014_FULL_window_scores/  (583 npz)
#          data/fire_labels/fire_labels_nbac_nfdb_*.npy
# Output : outputs/model_novel_lift_SOTA_full.csv
# ----------------------------------------------------------------

set -uo pipefail
export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1
source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PYTHONUNBUFFERED=1

# RUN_NAME selects which model's score dump to process. Defaults to SOTA.
#   RUN_NAME=v3_9ch_enc21_12y_2014        → SOTA            (out tag: SOTA)
#   RUN_NAME=baseline_convlstm_12y_2014_9ch → ConvLSTM
#   RUN_NAME=baseline_mlp_12y_2014_9ch      → MLP
RUN_NAME=${RUN_NAME:-v3_9ch_enc21_12y_2014}
OUT_TAG=${OUT_TAG:-$RUN_NAME}

SCORES_DIR="outputs/${RUN_NAME}_FULL_window_scores"
LABEL_NPY="data/fire_labels/fire_labels_nbac_nfdb_2000-05-01_2025-12-21_2281x2709_r14.npy"
OUT_CSV="outputs/model_novel_lift_${OUT_TAG}_full.csv"

[ -d "$SCORES_DIR" ] || { echo "ERROR: scores dir missing: $SCORES_DIR"; exit 1; }
[ -f "$LABEL_NPY" ]  || { echo "ERROR: label npy missing: $LABEL_NPY"; exit 1; }
echo "  scores: $(ls $SCORES_DIR | wc -l) npz files"

python3 -u -m scripts.compute_lift_from_scores \
    --scores_dir "$SCORES_DIR" \
    --fire_label_npy "$LABEL_NPY" \
    --label_start_date 2000-05-01 \
    --patch_size 16 \
    --lookback_days_list 7 30 90 \
    --k_values 1000 5000 10000 \
    --run_name "$OUT_TAG" \
    --output_csv "$OUT_CSV"

PY_EXIT=$?
echo "=== Done $(date) exit=$PY_EXIT ==="
ls -lh "$OUT_CSV" 2>/dev/null && head -5 "$OUT_CSV" 2>/dev/null
exit $PY_EXIT
