#!/bin/bash
#SBATCH --job-name=wf-v3-12ch-stat-12y
#SBATCH --gpus-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=480G
#SBATCH --output=/scratch/jiaqi217/logs/train_v3_12ch_static_12y_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/train_v3_12ch_static_12y_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# V3 12ch-static x enc{ENC} x 12y (2014-2022) — adds 2 free static channels
# ----------------------------------------------------------------
# Channel set: 9ch + elevation + aspect + lightning_climatology (3 free static channels)
# Both rasters already exist in data/terrain/{dem_cdem,aspect}.tif —
# adding them is essentially free (no new download, just a cache
# rebuild). Hypothesis: terrain context (elevation magnitude + slope
# direction) is complementary to the slope-magnitude channel already
# in 9ch and provides physiographic priors that climatology already
# encodes implicitly but our model doesn't see directly.
#
# Note: cache will be built on first run (~6-12h for 12y at 11ch); the
# build step is included in the same job. Subsequent runs at different
# enc reuse the cache.
#
# Usage:
#   ENC=21 sbatch slurm/train_v3_12ch_static_12y_narval.sh
# ----------------------------------------------------------------

set -uo pipefail
ENC=${ENC:?Must set ENC (e.g. 21)}

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 proj/9.4.1 eccodes/2.31.0
cd "$SCRATCH/wildfire-refactored"
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

export WANDB_MODE=offline
export WANDB_ENTITY=jiaaqii-huang-university-of-toronto
export WANDB_DIR=$SCRATCH/wandb_offline

source slurm/lib_copy_cache.sh
copy_venv $SCRATCH/venv-wildfire
cuda_probe || exit 1

LOCAL_CACHE=$SLURM_TMPDIR/cache
mkdir -p "$LOCAL_CACHE"
copy_s2s_cache "$SCRATCH/meteo_cache" "$LOCAL_CACHE"

CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,population,slope,burn_age,elevation,aspect,lightning_climatology"
CACHE_DIR_LUSTRE="$SCRATCH/meteo_cache/v3_12ch_static_12y_2014"
RUN_NAME="v3_12ch_static_enc${ENC}_12y_2014"

mkdir -p "$CACHE_DIR_LUSTRE"

# If cache exists, copy to SSD; otherwise let train_v3 build it
LOCAL_METEO="$LOCAL_CACHE/meteo"
mkdir -p "$LOCAL_METEO"
if [ -n "$(ls -A "$CACHE_DIR_LUSTRE" 2>/dev/null)" ]; then
    echo "=== copy 11ch cache to local SSD ==="
    for f in "$CACHE_DIR_LUSTRE"/*; do
        [ -f "$f" ] || continue
        cp "$f" "$LOCAL_METEO/" || { echo "FATAL"; exit 1; }
    done
    TRAIN_CACHE_DIR="$LOCAL_METEO"
else
    echo "=== cache missing — train_v3 will build it on Lustre ==="
    TRAIN_CACHE_DIR="$CACHE_DIR_LUSTRE"
fi

echo "============================================="
echo "  V3 11ch x enc${ENC} x 12y (2014-2022)"
echo "  Channels: $CHANNELS"
echo "  Run: $RUN_NAME"
echo "============================================="

$PYTHON -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name "$RUN_NAME" \
    --data_start 2014-05-01 --pred_start 2022-05-01 --pred_end 2025-10-31 \
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
    --wandb_project wildfire-s2s \
    --wandb_tags "12ch_static,enc${ENC},12y_2014,12y,terrain_lightning_static" \
    --save_per_window_json "$SCRATCH/wildfire-refactored/outputs/${RUN_NAME}_per_window.json"

PY_EXIT=$?
echo "=== Done $(date) exit=$PY_EXIT ==="
exit $PY_EXIT
