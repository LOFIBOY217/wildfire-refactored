#!/bin/bash
#SBATCH --job-name=wf-extra-baselines
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=0-08:00:00
#SBATCH --output=/scratch/jiaqi217/logs/extra_baselines_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/extra_baselines_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# Run persistence + fwi_threshold baselines on NEW NBAC labels.
# (climatology + fwi_oracle already in benchmark_baselines.csv from prior runs.)

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

echo "=== Extra baselines (persistence, fwi_threshold) ==="
python3 -u -m src.evaluation.benchmark_baselines \
    --config configs/paths_narval.yaml \
    --baseline persistence fwi_threshold \
    --eval_mode per_window \
    --pred_start 2022-05-01 --pred_end 2024-10-31 \
    --in_days 7 --lead_start 14 --lead_end 45 \
    --patch_size 16 --dilate_radius 14 \
    --fire_label_npy "$LABEL_NPY" \
    --climatology_tif "$CLIM_TIF" \
    --n_sample_wins 20 \
    --output_csv "$SCRATCH/wildfire-refactored/outputs/benchmark_extra.csv"

echo "=== done $(date) ==="
