#!/bin/bash
#SBATCH --job-name=wf-prep-16ch
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/jiaqi217/logs/prep_16ch_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/prep_16ch_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# 16ch prep: Build NBAC+NFDB dilated label + fire_patched.dat
# for 4y / 12y / 22y 16ch caches.
#
# Existing caches (built without --label_fusion) have:
#   - meteo (huge, keep!)
#   - fire_dilated CWFIS (ignore)
#   - fire_patched buggy CWFIS (delete first)
#
# This prep adds:
#   - fire_dilated_r14_nbac_nfdb_*.npy
#   - fire_patched_v3_r14_nbac_nfdb_*.dat (with fusion_tag)
#
# Uses --epochs 0 + --prep_only so no training, just cache building.
# Does NOT use --overwrite so meteo cache is preserved.
#
# Usage:
#   RANGE=4y  sbatch slurm/prep_16ch_nbac_nfdb_narval.sh
#   RANGE=12y sbatch slurm/prep_16ch_nbac_nfdb_narval.sh
#   RANGE=22y sbatch slurm/prep_16ch_nbac_nfdb_narval.sh
# ----------------------------------------------------------------

set -uo pipefail
RANGE=${RANGE:?Must set RANGE (4y / 12y / 22y)}

case "$RANGE" in
    22y)
        DATA_START=2000-05-01
        CACHE_DIR=/scratch/jiaqi217/meteo_cache/v3_full_2000
        ;;
    12y)
        DATA_START=2014-05-01
        CACHE_DIR=/scratch/jiaqi217/meteo_cache/v3_full_12y_2014
        ;;
    4y)
        DATA_START=2018-05-01
        CACHE_DIR=/scratch/jiaqi217/meteo_cache/v3_full_4y_2018
        ;;
    *) echo "ERROR: unknown RANGE=$RANGE"; exit 1 ;;
esac

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1 eccodes/2.31.0
cd "$SCRATCH/wildfire-refactored"
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

source slurm/lib_copy_cache.sh
copy_venv $SCRATCH/venv-wildfire

# Clean up any old buggy fire_patched.dat (no fusion_tag) — would not be
# loaded by post-fix code (filename mismatch) but takes disk space
buggy_pat="$CACHE_DIR/fire_patched_v3_r14_${DATA_START}_2025-12-20_*.dat"
if ls $buggy_pat >/dev/null 2>&1; then
    echo "Deleting buggy (no-fusion-tag) fire_patched.dat:"
    ls -la $buggy_pat
    rm -v $buggy_pat
fi

CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,deep_soil,precip_def,u10,v10,CAPE,NDVI,population,slope,burn_age,burn_count"

echo "============================================="
echo "  16ch NBAC+NFDB prep for $RANGE"
echo "  Cache: $CACHE_DIR"
echo "  Data start: $DATA_START"
echo "============================================="

if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: cache dir missing: $CACHE_DIR"
    exit 1
fi

$PYTHON -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name "prep_16ch_${RANGE}" \
    --data_start "$DATA_START" --pred_start 2022-05-01 --pred_end 2025-10-31 \
    --lead_end 45 \
    --channels "$CHANNELS" --in_days 21 \
    --decoder random \
    --batch_size 1024 --epochs 0 \
    --d_model 256 --nhead 8 --enc_layers 4 --dec_layers 4 --patch_size 16 \
    --dilate_radius 14 \
    --cache_dir "$CACHE_DIR" --chunk_patches 2000 --num_workers 8 \
    --skip_forecast \
    --prep_only \
    --label_fusion --nfdb_min_size_ha 1.0 \
    --fire_clim_dir data/fire_clim_annual_nbac

PY_EXIT=$?
echo "=== Done $(date) exit=$PY_EXIT ==="
echo "=== Cache contents after prep ==="
ls -lh "$CACHE_DIR"
exit $PY_EXIT
