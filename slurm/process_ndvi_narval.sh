#!/bin/bash
#SBATCH --job-name=wf-proc-ndvi
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/jiaqi217/logs/proc_ndvi_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/proc_ndvi_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# Process MODIS HDF4 → daily NDVI TIFs on FWI grid (2018-2024)
# Uses sinusoidal projection fix (per-tile reproject)

set -euo pipefail
if [ -z "${SCRATCH:-}" ]; then export SCRATCH=/scratch/jiaqi217; fi
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1 2>/dev/null || true
# Ensure pyproj can find PROJ data
export PROJ_DATA=${EBROOTPROJ:-/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/proj/9.4.1}/share/proj
source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored

python3 -u -m src.data_ops.processing.process_modis_ndvi \
    --config configs/paths_narval.yaml \
    --start_year 2018 --end_year 2024 \
    --overwrite
