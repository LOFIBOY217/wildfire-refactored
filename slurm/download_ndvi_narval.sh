#!/bin/bash
#SBATCH --job-name=wf-dl-ndvi
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/jiaqi217/logs/dl_ndvi_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/dl_ndvi_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# Download MODIS NDVI HDF4 files for all years (2018-2024)
# Skips years that already have >500 files

set -euo pipefail
if [ -z "${SCRATCH:-}" ]; then export SCRATCH=/scratch/jiaqi217; fi
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1 2>/dev/null || true
source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored

export EARTHDATA_USERNAME=jiaqihuang
export EARTHDATA_PASSWORD=sitXuh-hecvox-tesju0

python3 -u -m src.data_ops.download.download_modis_ndvi \
    --config configs/paths_narval.yaml \
    --start_year 2018 --end_year 2024 \
    --batch_size 50
