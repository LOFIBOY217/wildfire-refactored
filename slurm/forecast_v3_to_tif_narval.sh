#!/bin/bash
#SBATCH --job-name=wf-v3-forecast-tif
#SBATCH --gpus-per-node=1
#SBATCH --time=0-03:00:00
#SBATCH --mem=400G
#SBATCH --output=/scratch/jiaqi217/logs/v3_forecast_tif_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/v3_forecast_tif_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# Dump per-lead fire probability GeoTIFFs for V3 SOTA checkpoint
# (paper Fig 3 — Canada map figures).
#
# Issue dates (default): 2023-05-15, 2023-08-15, 2022-08-15
# Output: outputs/v3_9ch_enc21_12y_2014_fire_prob/{YYYYMMDD}/
#         fire_prob_lead{LL}d_{YYYYMMDD_target}.tif  (32 TIFs each)
#
# Usage:
#   sbatch slurm/forecast_v3_to_tif_narval.sh
#   # custom dates:
#   ISSUE_DATES="2023-06-01 2024-07-01" sbatch slurm/forecast_v3_to_tif_narval.sh
# ----------------------------------------------------------------

set -uo pipefail

ISSUE_DATES=${ISSUE_DATES:-"2023-05-15 2023-08-15 2022-08-15"}
CKPT=${CKPT:-"checkpoints/v3_9ch_enc21_12y_2014/best_model.pt"}
OUT_DIR=${OUT_DIR:-"outputs/v3_9ch_enc21_12y_2014_fire_prob"}

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 proj/9.4.1 eccodes/2.31.0
cd "$SCRATCH/wildfire-refactored"
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

source slurm/lib_copy_cache.sh
copy_venv $SCRATCH/venv-wildfire
cuda_probe || exit 1

# We only need the meteo cache for the issue dates (each issuance reads
# ~21 days of inputs). Full cache copy is overkill but matches existing
# eval jobs and avoids per-day Lustre reads.
LOCAL_CACHE=$SLURM_TMPDIR/cache
mkdir -p "$LOCAL_CACHE"
# Only the S2S decoder cache is needed locally — forecast_v3_to_tif.py
# reads raw FWI/ERA5 tifs straight from config paths (just ~21 days per
# issue date), so the pre-patched 12y meteo memmap is NOT required.
[ -z "${S2S_CACHE:-}" ] && copy_s2s_cache "$SCRATCH/meteo_cache" "$LOCAL_CACHE"

[ -f "$CKPT" ] || { echo "ERROR: checkpoint missing: $CKPT"; exit 1; }

echo "============================================="
echo "  V3 FORECAST → TIF"
echo "  ckpt        : $CKPT"
echo "  issue dates : $ISSUE_DATES"
echo "  out_dir     : $OUT_DIR"
echo "============================================="

$PYTHON -u -m src.forecasting.forecast_v3_to_tif \
    --config configs/paths_narval.yaml \
    --ckpt "$CKPT" \
    --issue_dates $ISSUE_DATES \
    --out_dir "$OUT_DIR" \
    --s2s_cache "${S2S_CACHE:-$LOCAL_CACHE/s2s_decoder_cache.dat}"

PY_EXIT=$?
echo "=== Done $(date) exit=$PY_EXIT ==="

echo "=== output summary ==="
for d in $ISSUE_DATES; do
    yyyymmdd=$(echo $d | tr -d '-')
    echo "  $d:"
    ls "$OUT_DIR/$yyyymmdd/" 2>/dev/null | head -3
    n=$(ls "$OUT_DIR/$yyyymmdd/" 2>/dev/null | wc -l)
    echo "    ... ($n total)"
done

exit $PY_EXIT
