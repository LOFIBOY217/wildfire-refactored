#!/bin/bash
#SBATCH --job-name=wf-ens-novel-lift
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-03:00:00
#SBATCH --output=/scratch/jiaqi217/logs/ens_novel_lift_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/ens_novel_lift_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# Ensemble (prob-mean) novel-ignition Lift from the 10 member score
# dumps already on disk (no GPU inference needed). Member list matches
# ensemble_logit_10ckpt.json ckpt_dirs. CPU-only.
#
# Output: outputs/model_novel_lift_ensemble_full.csv
# ----------------------------------------------------------------

set -uo pipefail
export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1
source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PYTHONUNBUFFERED=1

WS=outputs/window_scores_full
LABEL_NPY="data/fire_labels/fire_labels_nbac_nfdb_2000-05-01_2025-12-21_2281x2709_r14.npy"

# 10 members = the same set used for the 10-ckpt ensemble Lift JSON.
MEMBERS=(
    "$WS/v3_9ch_enc21_12y_2014"
    "$WS/v3_9ch_enc21_12y_2014_climsim"
    "$WS/v3_9ch_enc21_12y_2014_climblend_a0.3"
    "$WS/v3_9ch_enc21_12y_2014_climblend_a0.5"
    "$WS/v3_9ch_enc28_12y_2014"
    "$WS/v3_9ch_enc35_12y_2014"
    "$WS/v3_13ch_enc14_12y_2014"
    "$WS/v3_13ch_enc21_12y_2014"
    "$WS/v3_13ch_enc28_12y_2014"
    "$WS/v3_13ch_enc35_12y_2014"
)

echo "=== ensemble members present ==="
for d in "${MEMBERS[@]}"; do
    n=$(ls "$d"/window_*.npz 2>/dev/null | wc -l)
    echo "  $d : $n npz"
done

python3 -u -m scripts.compute_ensemble_novel_lift \
    --scores_dirs "${MEMBERS[@]}" \
    --fire_label_npy "$LABEL_NPY" \
    --label_start_date 2000-05-01 \
    --patch_size 16 \
    --lookback_days_list 7 30 90 \
    --k_values 1000 5000 10000 \
    --run_name ensemble_prob_10ckpt \
    --output_csv outputs/model_novel_lift_ensemble_full.csv

PY_EXIT=$?
echo "=== Done $(date) exit=$PY_EXIT ==="
exit $PY_EXIT
