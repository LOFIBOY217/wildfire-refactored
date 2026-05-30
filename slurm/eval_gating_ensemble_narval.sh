#!/bin/bash
#SBATCH --job-name=wf-gate-ensemble
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-03:00:00
#SBATCH --output=/scratch/jiaqi217/logs/gate_ensemble_%j.log
#SBATCH --account=def-inghaw
set -uo pipefail
export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1
source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PYTHONUNBUFFERED=1
WS=outputs/window_scores_full
# 10 existing ensemble members + per_pixel gating (11 total)
MEMBERS=(
  "$WS/v3_9ch_enc21_12y_2014" "$WS/v3_9ch_enc21_12y_2014_climsim"
  "$WS/v3_9ch_enc21_12y_2014_climblend_a0.3" "$WS/v3_9ch_enc21_12y_2014_climblend_a0.5"
  "$WS/v3_9ch_enc28_12y_2014" "$WS/v3_9ch_enc35_12y_2014"
  "$WS/v3_13ch_enc14_12y_2014" "$WS/v3_13ch_enc21_12y_2014"
  "$WS/v3_13ch_enc28_12y_2014" "$WS/v3_13ch_enc35_12y_2014"
  "$WS/v3_9ch_enc21_12y_2014_gate_per_pixel"
)
for MODE in prob_mean logit_mean; do
  echo "===== 11-member (+gating) $MODE ====="
  python3 -u -m scripts.ensemble_ckpts_lift --score_dirs "${MEMBERS[@]}" \
    --ensemble_mode $MODE --k 5000 --pred_end 2025-09-23 \
    --output outputs/ensemble_11_gating_${MODE}.json
done
echo "=== done $(date) ==="
