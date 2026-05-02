#!/bin/bash
#SBATCH --job-name=wf-prep-13ch
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/jiaqi217/logs/prep_13ch_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/prep_13ch_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# 13ch NBAC+NFDB prep (mirror of prep_16ch_nbac_nfdb_narval.sh)
# Builds fire_dilated_nbac_nfdb_*.npy in existing cache dir.
# Usage: RANGE=22y sbatch slurm/prep_13ch_nbac_nfdb_narval.sh

set -uo pipefail
RANGE=${RANGE:?Must set RANGE (22y, 12y, 4y)}

case "$RANGE" in
    22y) DATA_START=2000-05-01; CACHE_DIR=/scratch/jiaqi217/meteo_cache/v3_13ch_2000     ;;
    12y) DATA_START=2014-05-01; CACHE_DIR=/scratch/jiaqi217/meteo_cache/v3_13ch_12y_2014 ;;
    4y)  DATA_START=2018-05-01; CACHE_DIR=/scratch/jiaqi217/meteo_cache/v3_13ch_4y_2018  ;;
    *) echo "ERROR: unknown RANGE"; exit 1 ;;
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

# Clean stale no-fusion-tag fire_patched.dat (won't be used post-fix, but takes disk)
buggy_pat="$CACHE_DIR/fire_patched_v3_r14_${DATA_START}_2025-12-20_*.dat"
ls $buggy_pat >/dev/null 2>&1 && rm -v $buggy_pat 2>/dev/null

CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,deep_soil,precip_def,NDVI,population,slope,burn_age,burn_count"

echo "============================================="
echo "  13ch NBAC+NFDB prep for $RANGE"
echo "  Cache: $CACHE_DIR"
echo "============================================="

if [ ! -d "$CACHE_DIR" ]; then echo "ERROR: cache dir missing: $CACHE_DIR"; exit 1; fi

$PYTHON -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name "prep_13ch_${RANGE}" \
    --data_start "$DATA_START" --pred_start 2022-05-01 --pred_end 2025-10-31 \
    --lead_end 45 \
    --channels "$CHANNELS" --in_days 21 \
    --decoder random --batch_size 1024 --epochs 0 \
    --d_model 256 --nhead 8 --enc_layers 4 --dec_layers 4 --patch_size 16 \
    --dilate_radius 14 \
    --cache_dir "$CACHE_DIR" --chunk_patches 2000 --num_workers 8 \
    --skip_forecast --prep_only \
    --label_fusion --nfdb_min_size_ha 1.0 \
    --fire_clim_dir data/fire_clim_annual_nbac

echo "=== Done $(date) exit=$? ==="
ls -lh "$CACHE_DIR"
