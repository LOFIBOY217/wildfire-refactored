#!/bin/bash
#SBATCH --job-name=wf-12y-dm-sweep
#SBATCH --gpus-per-node=1
#SBATCH --time=0-08:00:00
#SBATCH --mem=400G
#SBATCH --output=/scratch/jiaqi217/logs/train_12y_dm_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/train_12y_dm_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# d_model sweep on SOTA recipe (12y enc21 + climsim).
# Tests over- vs under-parameterization on 12y data range.
# Usage:
#   D_MODEL=128 sbatch ...   # smaller (test if 256 is overparameterized)
#   D_MODEL=384 sbatch ...   # larger (risk of overfit on 12y)
#   D_MODEL=512 sbatch ...   # large

set -uo pipefail
ENC=${ENC:-21}
D_MODEL=${D_MODEL:?Must set D_MODEL (e.g. 128, 384, 512)}

# Auto-scale enc/dec layers proportionally for 384/512 to keep dim/heads ratio
if [ "$D_MODEL" = "512" ]; then
    NHEAD=${NHEAD:-8}; ENC_L=${ENC_L:-6}; DEC_L=${DEC_L:-6}
elif [ "$D_MODEL" = "384" ]; then
    NHEAD=${NHEAD:-8}; ENC_L=${ENC_L:-6}; DEC_L=${DEC_L:-6}
else
    NHEAD=${NHEAD:-8}; ENC_L=${ENC_L:-4}; DEC_L=${DEC_L:-4}
fi

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
CSV_PATH="$SCRATCH/wildfire-refactored/data/climate_indices/oni_similarity_12y_to_val2022_2025.csv"
RUN_NAME="v3_9ch_enc${ENC}_12y_2014_climsim_dm${D_MODEL}"

LOCAL_METEO="$LOCAL_CACHE/meteo"
mkdir -p "$LOCAL_METEO"
echo "=== copy 12y meteo ==="
for f in "$CACHE_DIR_LUSTRE"/*; do
    [ -f "$f" ] || continue
    cp "$f" "$LOCAL_METEO/" || exit 1
done
TRAIN_CACHE_DIR="$LOCAL_METEO"

echo "  d_model=${D_MODEL} enc_layers=${ENC_L} dec_layers=${DEC_L}"

$PYTHON -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name "$RUN_NAME" \
    --data_start 2014-05-01 --pred_start 2022-05-01 --pred_end 2025-10-31 \
    --channels "$CHANNELS" --in_days "$ENC" \
    --decoder s2s_legacy --s2s_cache "$LOCAL_CACHE/s2s_decoder_cache.dat" --s2s_max_issue_lag 3 \
    --loss_fn focal --focal_alpha 0.25 --focal_gamma 2.0 \
    --hard_neg_fraction 0.5 --neg_ratio 20 --neg_buffer 2 \
    --batch_size 4096 --epochs 4 --lr 1e-4 --weight_decay 0.01 --dropout 0.2 \
    --d_model "$D_MODEL" --nhead "$NHEAD" --enc_layers "$ENC_L" --dec_layers "$DEC_L" --patch_size 16 \
    --dilate_radius 14 --val_lift_k 5000 --val_lift_sample_wins 20 \
    --fire_season_only --cluster_eval --decoder_ctx --load_train_to_ram \
    --cache_dir "$TRAIN_CACHE_DIR" --chunk_patches 2000 --num_workers 4 \
    --log_interval 200 --skip_forecast \
    --label_fusion --nfdb_min_size_ha 1.0 \
    --fire_clim_dir data/fire_clim_annual_nbac \
    --climate_similarity_csv "$CSV_PATH" \
    --wandb_project wildfire-s2s \
    --wandb_tags "9ch,enc${ENC},12y_2014,12y,climsim,dm${D_MODEL}" \
    --save_per_window_json "$SCRATCH/wildfire-refactored/outputs/${RUN_NAME}_per_window.json"

echo "=== Done $(date) exit=$? ==="
