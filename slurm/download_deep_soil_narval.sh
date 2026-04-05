#!/bin/bash
#SBATCH --job-name=wf-dl-soil
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/jiaqi217/logs/dl_soil_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/dl_soil_%j.err

set -euo pipefail
if [ -z "${SCRATCH:-}" ]; then export SCRATCH=/scratch/jiaqi217; fi
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1 2>/dev/null || true
source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored

python3 -u -m src.data_ops.download.download_era5_deep_soil \
    2018-01-01 2025-10-31 --config configs/paths_narval.yaml
