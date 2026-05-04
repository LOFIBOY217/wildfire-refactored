#!/bin/bash
#SBATCH --job-name=wf-12y-climblend
#SBATCH --gpus-per-node=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=400G
#SBATCH --output=/scratch/jiaqi217/logs/train_12y_climblend_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/train_12y_climblend_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# 12y enc{ENC} 9ch + CLIMATOLOGY BLENDING
# ----------------------------------------------------------------
# Adds α × log(fire_clim) as a fixed logit bias at every
# (patch, lead-day, sub-pixel). Model only needs to learn the
# RESIDUAL on top of the climatological prior.
#
# Hypothesis: directly attacks the Recall@budget gap to climatology
# baseline (model alone tied/lost on Recall@5%-10%; with blending
# we should match or beat climatology by definition + add dynamic
# signal on top).
#
# Required env: ENC, ALPHA
# Usage:
#   ENC=21 ALPHA=0.5 sbatch slurm/train_v3_12y_clim_blend_narval.sh
#   ENC=21 ALPHA=1.0 sbatch slurm/train_v3_12y_clim_blend_narval.sh
# ----------------------------------------------------------------

set -uo pipefail
ENC=${ENC:?Must set ENC (e.g. 21)}
ALPHA=${ALPHA:?Must set ALPHA (e.g. 0.5)}

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

CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,population,slope,burn_age"
CACHE_DIR_LUSTRE="$SCRATCH/meteo_cache/v3_9ch_12y_2014"
RUN_NAME="v3_9ch_enc${ENC}_12y_2014_climblend_a${ALPHA}"

LOCAL_METEO="$LOCAL_CACHE/meteo"
mkdir -p "$LOCAL_METEO"
echo "=== copy 12y meteo to local SSD ==="
t0=$SECONDS
for f in "$CACHE_DIR_LUSTRE"/*; do
    [ -f "$f" ] || continue
    cp "$f" "$LOCAL_METEO/" || { echo "FATAL"; exit 1; }
done
echo "  done in $((SECONDS - t0))s"
TRAIN_CACHE_DIR="$LOCAL_METEO"

echo "============================================="
echo "  V3 9ch enc${ENC} 12y CLIM-BLEND alpha=${ALPHA}"
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
    --clim_blend_alpha "$ALPHA" \
    --wandb_project wildfire-s2s \
    --wandb_tags "9ch,enc${ENC},12y_2014,12y,clim_blend_a${ALPHA}" \
    --save_per_window_json "$SCRATCH/wildfire-refactored/outputs/${RUN_NAME}_per_window.json"

PY_EXIT=$?
echo "=== Done $(date) exit=$PY_EXIT ==="
exit $PY_EXIT
