#!/bin/bash
#SBATCH --job-name=wf-ensemble-eval
#SBATCH --cpus-per-task=4
#SBATCH --mem=400G
#SBATCH --time=0-02:00:00
#SBATCH --output=/scratch/jiaqi217/logs/ensemble_eval_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/ensemble_eval_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# Persistence × climatology ensemble across alpha = {0, 0.25, 0.5, 0.75, 1.0}
# under both total and novel-30d labels. Tests if recency + spatial prior
# have synergy (relevant to whether 10ch retrain with fire_recent_density
# is worth doing).

set -uo pipefail
export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1
source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

LABEL_NPY="$SCRATCH/wildfire-refactored/data/fire_labels/fire_labels_nbac_nfdb_2000-05-01_2025-12-21_2281x2709_r14.npy"
CLIM_TIF="$SCRATCH/wildfire-refactored/data/fire_clim_annual_nbac/fire_clim_upto_2022.tif"

python3 -u -m scripts.evaluate_novel_ensemble \
    --config configs/paths_narval.yaml \
    --pred_start 2022-05-01 --pred_end 2024-10-31 \
    --in_days 7 --lead_start 14 --lead_end 45 \
    --patch_size 16 --dilate_radius 14 \
    --fire_label_npy "$LABEL_NPY" \
    --climatology_tif "$CLIM_TIF" \
    --alphas 0.0 0.25 0.5 0.75 1.0 \
    --lookback_days 30 \
    --k 5000 \
    --n_sample_wins 20 \
    --output_csv "$SCRATCH/wildfire-refactored/outputs/benchmark_ensemble.csv"

echo "=== done $(date) ==="
