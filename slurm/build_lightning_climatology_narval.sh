#!/bin/bash
#SBATCH --job-name=wf-lightning-clim
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
#SBATCH --output=/scratch/jiaqi217/logs/lightning_clim_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/lightning_clim_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# Build lightning_climatology.tif from GLM raw daily TIFs (~30 min CPU).

set -uo pipefail
export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1
cd "$SCRATCH/wildfire-refactored"
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

source slurm/lib_copy_cache.sh
copy_venv $SCRATCH/venv-wildfire

$PYTHON -u -m scripts.build_lightning_climatology \
    --input_dir "$SCRATCH/wildfire-refactored/data/lightning_raw" \
    --reference "$SCRATCH/wildfire-refactored/data/fwi_data/fwi_20250615.tif" \
    --output "$SCRATCH/wildfire-refactored/data/lightning_climatology.tif"

PY_EXIT=$?
echo "=== done $(date) exit=$PY_EXIT ==="
exit $PY_EXIT
