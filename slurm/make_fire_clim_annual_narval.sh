#!/bin/bash
#SBATCH --job-name=wf-fire-clim
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/jiaqi217/logs/fire_clim_annual_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/fire_clim_annual_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# Generate annual rolling fire climatology TIFs for train_v3.
#
# For each target year Y (2018-2025):
#   fire_clim_upto_Y.tif = hotspots from [2017, Y-1]
#
# Requires: hotspot CSV includes 2017 data
#   (run slurm/download_hotspot_2017_narval.sh first)
#
# Output: data/fire_clim_annual/fire_clim_upto_*.tif
# ----------------------------------------------------------------

set -euo pipefail

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}

[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1

source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored

echo "============================================="
echo "  Annual Rolling Fire Climatology Generation"
echo "  Node: $(hostname)  Time: $(date)"
echo "============================================="

python3 -u -m src.data_ops.processing.make_fire_clim_annual \
    --config configs/paths_narval.yaml \
    --start_year 2018 \
    --end_year 2025 \
    --data_start_year 2017 \
    --months 5-10 \
    --output_dir data/fire_clim_annual

echo "Done: $(date)"
