#!/bin/bash
#SBATCH --job-name=wf-s2s-compress
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/jiaqi217/logs/s2s_compress_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/s2s_compress_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# Build 3 compressed S2S decoder caches from full-patch cache (4.9TB)
#   1. multi_stat  (24 dims, ~40G)
#   2. subpatch_4x4 (128 dims, ~260G)
#   3. pca (128 dims, ~260G)
# ----------------------------------------------------------------

set -euo pipefail

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}

[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5

source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PYTHONUNBUFFERED=1

FULL_CACHE=$SCRATCH/wildfire-refactored/data/s2s_full_patch_cache.dat
OUT_DIR=$SCRATCH/meteo_cache

mkdir -p "$OUT_DIR"

echo "============================================="
echo "  Build Compressed S2S Decoder Caches"
echo "  Node: $(hostname)  Time: $(date)"
echo "  Full-patch cache: $FULL_CACHE"
echo "============================================="

# 1. multi_stat (mean/std/max per channel → 24 dims)
echo ""
echo ">>> MODE 1: multi_stat (24 dims)"
python -u -m src.data_ops.processing.build_s2s_compressed_caches \
    --full-cache "$FULL_CACHE" \
    --out-file "$OUT_DIR/s2s_multistat_cache.dat" \
    --mode multi_stat

# 2. subpatch_4x4 (4×4 sub-block means → 128 dims)
echo ""
echo ">>> MODE 2: subpatch_4x4 (128 dims)"
python -u -m src.data_ops.processing.build_s2s_compressed_caches \
    --full-cache "$FULL_CACHE" \
    --out-file "$OUT_DIR/s2s_subpatch4x4_cache.dat" \
    --mode subpatch_4x4

# 3. pca (128 components from 2048 dims)
echo ""
echo ">>> MODE 3: pca (128 dims)"
python -u -m src.data_ops.processing.build_s2s_compressed_caches \
    --full-cache "$FULL_CACHE" \
    --out-file "$OUT_DIR/s2s_pca128_cache.dat" \
    --mode pca \
    --pca-components 128 \
    --pca-samples 1000000

echo ""
echo "============================================="
echo "  ALL DONE: $(date)"
echo "============================================="
ls -lh "$OUT_DIR"/s2s_*cache*
