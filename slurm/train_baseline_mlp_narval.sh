#!/bin/bash
#SBATCH --job-name=wf-mlp-baseline
#SBATCH --gpus-per-node=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=480G
#SBATCH --output=/scratch/jiaqi217/logs/mlp_baseline_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/mlp_baseline_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# MLP BASELINE — 22y × 9ch
# ----------------------------------------------------------------
# Answers: "is the transformer overkill?" — flatten encoder + decoder
# input → 2 hidden 512 → 33 × 256 logits per patch.
#
# Range choice: 22y (data-rich, MLP can absorb).
# Channel choice: 9ch (only fully-built master cache for 22y).
# Recipe matches transformer for fair comparison.
# ----------------------------------------------------------------

set -uo pipefail

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
CACHE_DIR_LUSTRE="$SCRATCH/meteo_cache/v3_9ch_2000"
RUN_NAME="baseline_mlp_22y_2000_9ch"

LOCAL_METEO="$LOCAL_CACHE/meteo"
mkdir -p "$LOCAL_METEO"
echo "=== copy 22y meteo to local SSD (~960 GB) ==="
t0=$SECONDS
for f in "$CACHE_DIR_LUSTRE"/*; do
    [ -f "$f" ] || continue
    cp "$f" "$LOCAL_METEO/" || { echo "FATAL"; exit 1; }
done
echo "  done in $((SECONDS - t0))s"
TRAIN_CACHE_DIR="$LOCAL_METEO"

echo "============================================="
echo "  MLP BASELINE — 22y × 9ch"
echo "  Run: $RUN_NAME"
echo "============================================="

$PYTHON -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name "$RUN_NAME" \
    --model_type mlp \
    --data_start 2000-05-01 --pred_start 2022-05-01 --pred_end 2025-10-31 \
    --channels "$CHANNELS" --in_days 21 \
    --decoder s2s_legacy --s2s_cache "$LOCAL_CACHE/s2s_decoder_cache.dat" --s2s_max_issue_lag 3 \
    --loss_fn focal --focal_alpha 0.25 --focal_gamma 2.0 \
    --hard_neg_fraction 0.5 --neg_ratio 20 --neg_buffer 2 \
    --batch_size 4096 --epochs 4 --lr 1e-4 --weight_decay 0.01 --dropout 0.2 \
    --d_model 256 --patch_size 16 \
    --dilate_radius 14 --val_lift_k 5000 --val_lift_sample_wins 20 \
    --fire_season_only --cluster_eval --decoder_ctx --load_train_to_ram \
    --cache_dir "$TRAIN_CACHE_DIR" --chunk_patches 2000 --num_workers 4 \
    --log_interval 200 --skip_forecast \
    --label_fusion --nfdb_min_size_ha 1.0 \
    --fire_clim_dir data/fire_clim_annual_nbac \
    --wandb_project wildfire-s2s \
    --wandb_tags "baseline,mlp,9ch,22y_2000" \
    --save_per_window_json "$SCRATCH/wildfire-refactored/outputs/${RUN_NAME}_per_window.json"

PY_EXIT=$?
echo "=== Done $(date) exit=$PY_EXIT ==="
exit $PY_EXIT
