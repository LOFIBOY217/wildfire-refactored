#!/bin/bash
#SBATCH --job-name=wf-ndvi-dl-2025
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/jiaqi217/logs/ndvi_dl_2025_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/ndvi_dl_2025_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# Download MODIS MOD13A2 NDVI HDF4 granules for 2025 (Canada bbox)
# earthaccess reads credentials from ~/.netrc (urs.earthdata.nasa.gov)

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}

[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1

source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored

# earthaccess strategy="environment" needs these; fall back to netrc if not set
EARTHDATA_LOGIN=$(awk '/urs.earthdata.nasa.gov/{found=1} found && /login/{print $2; exit}' ~/.netrc)
EARTHDATA_PASS=$(awk '/urs.earthdata.nasa.gov/{found=1} found && /password/{print $2; exit}' ~/.netrc)
export EARTHDATA_USERNAME="$EARTHDATA_LOGIN"
export EARTHDATA_PASSWORD="$EARTHDATA_PASS"

echo "============================================="
echo "  NDVI 2025 Download (MODIS MOD13A2)"
echo "  Node: $(hostname)  Time: $(date)"
echo "  User: $EARTHDATA_USERNAME"
echo "============================================="

python3 -u -m src.data_ops.download.download_modis_ndvi \
    --config configs/paths_narval.yaml \
    --start_year 2025 \
    --end_year 2025 \
    --months 1 2 3 4 5 6 7 8 9 10 11 12

echo "Done: $(date)"
