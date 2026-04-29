#!/bin/bash
#SBATCH --job-name=wf-prebuild-12y-dilated
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=0-04:00:00
#SBATCH --output=/scratch/jiaqi217/logs/prebuild_12y_dilated_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/prebuild_12y_dilated_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# Pre-build the 12y NBAC+NFDB dilated label file and place it in the
# 12y meteo cache dir on Lustre. When the 4 12y training jobs run, the
# slurm script copies the cache dir (including this file) to local SSD,
# and train_v3 finds the file at its expected cache_key path → skips
# the ~1-2h dilation per job (saves ~4-6 CPU-hours total).
#
# Output filename matches what train_v3.py:1411-1415 expects:
#   fire_dilated_r14_nbac_nfdb_2014-05-01_2025-12-20_2281x2709.npy

set -uo pipefail
export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1
source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

CACHE_12Y=$SCRATCH/meteo_cache/v3_9ch_12y_2014
mkdir -p "$CACHE_12Y"

echo "=== build 12y NBAC+NFDB dilated label ==="
echo "  date range: 2014-05-01 to 2025-12-20"
echo "  output dir: $SCRATCH/wildfire-refactored/data/fire_labels/"

# build_fire_labels.py produces: fire_labels_nbac_nfdb_{start}_{end}_{H}x{W}_r{r}.npy
python3 -u scripts/build_fire_labels.py \
    --scheme nbac_nfdb \
    --start 2014-05-01 \
    --end 2025-12-20 \
    --dilate_radius 14 \
    --output_dir "$SCRATCH/wildfire-refactored/data/fire_labels/" \
    --nfdb_min_size_ha 1.0

# Now copy to the train_v3-expected cache key path
SRC="$SCRATCH/wildfire-refactored/data/fire_labels/fire_labels_nbac_nfdb_2014-05-01_2025-12-20_2281x2709_r14.npy"
DST="$CACHE_12Y/fire_dilated_r14_nbac_nfdb_2014-05-01_2025-12-20_2281x2709.npy"

if [ ! -f "$SRC" ]; then
    echo "ERROR: build_fire_labels.py did not produce: $SRC"
    exit 1
fi

echo "=== copy to 12y cache key path ==="
echo "  src: $SRC"
echo "  dst: $DST"
cp "$SRC" "$DST"
ls -lh "$DST"

echo "=== done $(date) ==="
