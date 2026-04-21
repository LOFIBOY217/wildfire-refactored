#!/bin/bash
#SBATCH --job-name=wf-fire-clim-nbac
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/jiaqi217/logs/fire_clim_nbac_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/fire_clim_nbac_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# Build NBAC+NFDB-based fire climatology files (replaces CWFIS-based
# fire_clim_upto_*.tif that were polluted by CWFIS 350x drift).
#
# Output: data/fire_clim_annual_nbac/fire_clim_nbac_upto_{2000..2025}.tif
# ~26 files, each 2281x2709 float32 ~24 MB → ~624 MB total
# Build time: ~30-90 min (per-polygon rasterize × 25 years)
# ----------------------------------------------------------------

set -uo pipefail

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1
source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

echo "=============================================="
echo "  Building fire_clim_nbac_upto_*.tif"
echo "  Years: 2000..2025"
echo "  Node: $(hostname)  Time: $(date)"
echo "=============================================="

python3 -u -m src.data_ops.processing.make_fire_clim_nbac \
    --config configs/paths_narval.yaml \
    --start_year 2000 --end_year 2025 \
    --months 5-10 \
    --nfdb_min_size_ha 1.0 \
    --output_dir data/fire_clim_annual_nbac

echo ""
echo "=== Done: $(date) ==="
ls -lh /scratch/jiaqi217/wildfire-refactored/data/fire_clim_annual_nbac/ | head -30
