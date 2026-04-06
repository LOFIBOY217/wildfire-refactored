#!/bin/bash
#SBATCH --job-name=wf-ndvi-proc-2025
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/jiaqi217/logs/ndvi_proc_2025_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/ndvi_proc_2025_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# Process 2025 MODIS MOD13A2 HDF4 → daily NDVI TIFs on FWI grid (EPSG:3978)
# Input:  data/ndvi_raw/2025/*.hdf
# Output: data/ndvi_data/ndvi_20250101.tif ... ndvi_20251031.tif

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}

[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1

source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored

echo "============================================="
echo "  NDVI 2025 Processing (HDF4 → daily TIF)"
echo "  Node: $(hostname)  Time: $(date)"
echo "============================================="

python3 -u -m src.data_ops.processing.process_modis_ndvi \
    --config configs/paths_narval.yaml \
    --start_year 2025 \
    --end_year 2025

EXIT=$?

echo ""
echo "--- NDVI TIF summary ---"
python3 -c "
import glob, os
files = sorted(glob.glob('data/ndvi_data/ndvi_2025*.tif'))
print(f'2025 TIFs: {len(files)}')
if files:
    print(f'First: {os.path.basename(files[0])}')
    print(f'Last : {os.path.basename(files[-1])}')
total = len(glob.glob('data/ndvi_data/ndvi_*.tif'))
print(f'Total NDVI TIFs: {total}')
" 2>&1

echo "Done: $(date)"
exit $EXIT
