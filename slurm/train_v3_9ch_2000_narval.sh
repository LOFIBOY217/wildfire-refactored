#!/bin/bash
#SBATCH --job-name=wf-v3-9ch-2000
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=500G
#SBATCH --output=/scratch/jiaqi217/logs/train_v3_9ch_2000_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/train_v3_9ch_2000_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca
#SBATCH --account=def-inghaw

# ----------------------------------------------------------------
#   V3 9ch x enc{ENC} x 2000-2025 EXTENDED TRAINING
# ----------------------------------------------------------------
# Channel set: 9ch baseline
# CHANNELS = FWI,2t,fire_clim,2d,tcw,sm20,population,slope,burn_age
#
# Fair-comparison baseline for 13ch/16ch sweeps. Existing 9ch enc21
# SOTA (6.47x Lift@5000) was trained on 2018-2022 (4y) — this sweep
# retrains on 2000-2022 (22y) so 9ch/13ch/16ch share identical data.
#
# Usage:
#   ENC=14 sbatch slurm/train_v3_9ch_2000_narval.sh
#   ENC=21 sbatch slurm/train_v3_9ch_2000_narval.sh
#   ENC=28 sbatch slurm/train_v3_9ch_2000_narval.sh
#   ENC=35 sbatch slurm/train_v3_9ch_2000_narval.sh
#
# PREREQUISITES:
#   - 9ch cache built at $SCRATCH/meteo_cache/v3_9ch_2000
#     run: CACHE_CHANNELS=9 sbatch slurm/rebuild_cache_2000_2025_narval.sh
#
# (Supersedes hardcoded train_v3_9ch_enc28_2000_narval.sh. Use ENC env
# var instead of one-script-per-enc.)
# ----------------------------------------------------------------

set -uo pipefail
ENC=${ENC:?Must set ENC (one of 14, 21, 28, 35)}
RANGE=${RANGE:-22y}   # 22y (2000-05-01) or 4y (2018-05-01)

# Map RANGE → date_start + cache dir + run-name tag
case "$RANGE" in
    22y)
        DATA_START=2000-05-01
        CACHE_TAG=2000
        RUN_TAG=2000
        ;;
    4y)
        DATA_START=2018-05-01
        CACHE_TAG=4y_2018
        RUN_TAG=4y_2018
        ;;
    *)
        echo "ERROR: unknown RANGE=$RANGE (expected '22y' or '4y')"
        exit 1
        ;;
esac

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 proj/9.4.1 eccodes/2.31.0
cd "$SCRATCH/wildfire-refactored"
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

source slurm/lib_copy_cache.sh
copy_venv $SCRATCH/venv-wildfire
$PYTHON -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())" || exit 1

LOCAL_CACHE=$SLURM_TMPDIR/cache
mkdir -p "$LOCAL_CACHE"
copy_s2s_cache "$SCRATCH/meteo_cache" "$LOCAL_CACHE"

CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,population,slope,burn_age"
CACHE_DIR_2000="$SCRATCH/meteo_cache/v3_9ch_${CACHE_TAG}"
RUN_NAME="v3_9ch_enc${ENC}_${RUN_TAG}"

if [ ! -d "$CACHE_DIR_2000" ] || [ -z "$(ls -A "$CACHE_DIR_2000" 2>/dev/null)" ]; then
    echo "ERROR: $CACHE_DIR_2000 is empty or missing."
    echo "Run cache build first:"
    echo "  CACHE_CHANNELS=9 DATA_START=$DATA_START sbatch slurm/rebuild_cache_2000_2025_narval.sh"
    exit 1
fi

# ----------------------------------------------------------------
# Copy meteo cache to local SSD (added 2026-04-22)
#
# Lustre random read on /scratch is ~5 MB/s for memmap chunked reads
# (observed via `cat /proc/<pid>/io` in run 59724826). Local SSD is
# ~500 MB/s. Without this copy, --load_train_to_ram alone takes 6+
# hours on Lustre vs ~5 min on SSD. After load, training is also
# faster because val + decoder_ctx still hit the meteo file.
#
# Falls back to Lustre if local SSD lacks space. Narval GPU nodes
# have 14T local SSD as of 2026-04-22, so copy almost always works.
# ----------------------------------------------------------------
LOCAL_METEO="$LOCAL_CACHE/meteo"
mkdir -p "$LOCAL_METEO"
SRC_BYTES=$(du -sb "$CACHE_DIR_2000" | awk '{print $1}')
SRC_GB=$((SRC_BYTES / 1024 / 1024 / 1024))
AVAIL_KB=$(df --output=avail "$SLURM_TMPDIR" | tail -1)
AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
NEEDED_GB=$((SRC_GB * 12 / 10))    # +20% margin
echo "=== meteo cache copy decision ==="
echo "  source       : $CACHE_DIR_2000 (${SRC_GB} GB)"
echo "  SSD avail    : ${AVAIL_GB} GB"
echo "  need (1.2x)  : ${NEEDED_GB} GB"

if [ "$AVAIL_GB" -gt "$NEEDED_GB" ]; then
    echo "  decision     : COPY to local SSD"
    t0=$SECONDS
    for f in "$CACHE_DIR_2000"/*; do
        [ -f "$f" ] || continue
        fname=$(basename "$f")
        sz=$(du -h "$f" | cut -f1)
        echo "  copy $fname ($sz)"
        cp "$f" "$LOCAL_METEO/" || {
            echo "  ERROR: copy $fname failed -- aborting copy, falling back to Lustre"
            rm -rf "$LOCAL_METEO"
            break
        }
    done
    if [ -d "$LOCAL_METEO" ]; then
        echo "  copy total   : $((SECONDS - t0))s"
        ls -lh "$LOCAL_METEO/"
        TRAIN_CACHE_DIR="$LOCAL_METEO"
    else
        TRAIN_CACHE_DIR="$CACHE_DIR_2000"
    fi
else
    echo "  decision     : insufficient SSD space, use Lustre directly"
    TRAIN_CACHE_DIR="$CACHE_DIR_2000"
fi
echo "  TRAIN_CACHE_DIR = $TRAIN_CACHE_DIR"

# Hard guard: Plan A requires NBAC-based fire_clim dir to exist before training.
# If the dir is missing, we'd silently fall back to CWFIS fire_clim — wasting 3 days.
if [ ! -d "$SCRATCH/wildfire-refactored/data/fire_clim_annual_nbac" ] || \
   [ -z "$(ls -A "$SCRATCH/wildfire-refactored/data/fire_clim_annual_nbac" 2>/dev/null)" ]; then
    echo "ERROR: data/fire_clim_annual_nbac is missing or empty."
    echo "Build NBAC fire_clim first: sbatch slurm/make_fire_clim_nbac_narval.sh"
    exit 1
fi

echo "============================================="
echo "  V3 9ch x enc${ENC} x RANGE=${RANGE}  (data_start=$DATA_START)"
echo "  Run name: $RUN_NAME"
echo "  Cache: $CACHE_DIR_2000"
echo "  ★★★ LABEL: NBAC + NFDB (Plan A, post 2026-04-21 fix) ★★★"
echo "  ★★★ FIRE_CLIM: data/fire_clim_annual_nbac (NBAC-derived) ★★★"
echo "============================================="

$PYTHON -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name "$RUN_NAME" \
    --data_start "$DATA_START" --pred_start 2022-05-01 --pred_end 2025-10-31 \
    --channels "$CHANNELS" --in_days "$ENC" \
    --decoder s2s_legacy --s2s_cache "$LOCAL_CACHE/s2s_decoder_cache.dat" --s2s_max_issue_lag 3 \
    --loss_fn focal --focal_alpha 0.25 --focal_gamma 2.0 \
    --hard_neg_fraction 0.5 --neg_ratio 20 --neg_buffer 2 \
    --batch_size 4096 --epochs 4 --lr 1e-4 --weight_decay 0.01 --dropout 0.2 \
    --d_model 256 --nhead 8 --enc_layers 4 --dec_layers 4 --patch_size 16 \
    --dilate_radius 14 --val_lift_k 5000 --val_lift_sample_wins 20 \
    --fire_season_only --cluster_eval --decoder_ctx --load_train_to_ram \
    --cache_dir "$TRAIN_CACHE_DIR" --chunk_patches 2000 --num_workers 4 \
    --log_interval 200 --skip_forecast \
    --label_fusion --nfdb_min_size_ha 1.0 \
    --fire_clim_dir data/fire_clim_annual_nbac \
    --save_per_window_json "$SCRATCH/wildfire-refactored/outputs/${RUN_NAME}_per_window.json"

# 2026-04-21 Plan A defaults:
#   --label_fusion           → use NBAC+NFDB labels (post LABEL_DECISION_2026_04_21.md)
#   --nfdb_min_size_ha 1.0   → drop <1ha NFDB micro-fires (noise reduction)
#   --fire_clim_dir nbac dir → use NBAC-derived fire_clim feature (no CWFIS drift)
#   --save_per_window_json   → enables filtering 2025 windows post-hoc (val period
#                              still includes 2025 due to cache constraint, but
#                              we'll exclude in analysis since NBAC only ships through 2024)

PY_EXIT=$?
echo "=== Done: $(date) exit=$PY_EXIT ==="
exit $PY_EXIT
