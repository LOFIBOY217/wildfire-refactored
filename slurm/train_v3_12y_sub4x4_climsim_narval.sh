#!/bin/bash
#SBATCH --job-name=wf-12y-sub4x4-cs
#SBATCH --gpus-per-node=1
#SBATCH --time=0-08:00:00
#SBATCH --mem=480G
#SBATCH --output=/scratch/jiaqi217/logs/train_12y_sub4x4_climsim_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/train_12y_sub4x4_climsim_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# 12y enc21 + ONI climsim + sub4x4 decoder (128-dim, spatially aware SEAS5)
# Replaces s2s_legacy (9-dim patch-mean) with s2s sub4x4 (128-dim per-sub-patch).
# Hypothesis: 14× richer decoder query → cross-attention can match SEAS5 forecast
# at sub-patch position → +5–15% Lift over current SOTA 8.07×.

set -uo pipefail
ENC=${ENC:-21}

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

# Sub4x4 decoder cache (309 GB)
S2S_SUB4_SCRATCH="$SCRATCH/meteo_cache/s2s_subpatch4x4_cache.dat"
S2S_SUB4_LOCAL="$LOCAL_CACHE/s2s_subpatch4x4_cache.dat"
SSD_AVAIL_GB=$(($(df --output=avail $SLURM_TMPDIR | tail -1) / 1048576))
echo "SSD avail: ${SSD_AVAIL_GB}G"
if [ "$SSD_AVAIL_GB" -ge 700 ]; then
    cp "$S2S_SUB4_SCRATCH" "$S2S_SUB4_LOCAL" && \
    cp "${S2S_SUB4_SCRATCH}.dates.npy" "${S2S_SUB4_LOCAL}.dates.npy" 2>/dev/null || \
    S2S_SUB4_LOCAL="$S2S_SUB4_SCRATCH"
else
    S2S_SUB4_LOCAL="$S2S_SUB4_SCRATCH"
fi

# Meteo cache (12y, 9ch)
CACHE_DIR_LUSTRE="$SCRATCH/meteo_cache/v3_9ch_12y_2014"
LOCAL_METEO="$LOCAL_CACHE/meteo"
mkdir -p "$LOCAL_METEO"
echo "=== copy 12y meteo to local SSD ==="
for f in "$CACHE_DIR_LUSTRE"/*; do
    [ -f "$f" ] || continue
    cp "$f" "$LOCAL_METEO/" || { echo "FATAL"; exit 1; }
done
TRAIN_CACHE_DIR="$LOCAL_METEO"

CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,population,slope,burn_age"
CSV_PATH="$SCRATCH/wildfire-refactored/data/climate_indices/oni_similarity_12y_to_val2022_2025.csv"
RUN_NAME="v3_9ch_enc${ENC}_12y_2014_sub4x4_climsim"

echo "============================================="
echo "  12y enc${ENC} + climsim + sub4x4 decoder (128-dim)"
echo "  Run: $RUN_NAME"
echo "============================================="

$PYTHON -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name "$RUN_NAME" \
    --data_start 2014-05-01 --pred_start 2022-05-01 --pred_end 2025-10-31 \
    --channels "$CHANNELS" --in_days "$ENC" \
    --decoder s2s --s2s_full_cache "$S2S_SUB4_LOCAL" --dec_dim 128 \
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
    --climate_similarity_csv "$CSV_PATH" \
    --wandb_project wildfire-s2s \
    --wandb_tags "9ch,enc${ENC},12y_2014,12y,sub4x4,climsim" \
    --save_per_window_json "$SCRATCH/wildfire-refactored/outputs/${RUN_NAME}_per_window.json"

echo "=== Done $(date) exit=$? ==="
