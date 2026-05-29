#!/bin/bash
#SBATCH --job-name=wf-ens-full-metrics
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-03:00:00
#SBATCH --output=/scratch/jiaqi217/logs/ens_full_metrics_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/ens_full_metrics_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca
set -uo pipefail
export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1
source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PYTHONUNBUFFERED=1
WS=outputs/window_scores_full
python3 -u -m scripts.compute_ensemble_full_metrics \
    --scores_dirs \
        "$WS/v3_9ch_enc21_12y_2014" "$WS/v3_9ch_enc21_12y_2014_climsim" \
        "$WS/v3_9ch_enc21_12y_2014_climblend_a0.3" "$WS/v3_9ch_enc21_12y_2014_climblend_a0.5" \
        "$WS/v3_9ch_enc28_12y_2014" "$WS/v3_9ch_enc35_12y_2014" \
        "$WS/v3_13ch_enc14_12y_2014" "$WS/v3_13ch_enc21_12y_2014" \
        "$WS/v3_13ch_enc28_12y_2014" "$WS/v3_13ch_enc35_12y_2014" \
    --k 5000 --run_name ensemble_prob_10ckpt \
    --output_json outputs/ensemble_prob_FULL_per_window.json
echo "=== done $(date) exit=$? ==="
