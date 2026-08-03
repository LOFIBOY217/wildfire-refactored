#!/bin/bash
#SBATCH --job-name=wf-fcnhead
#SBATCH --account=def-inghaw
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=249G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/jiaqi217/logs/train_fcnhead_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/train_fcnhead_%j.err
# B: fully-convolutional (translation-equivariant) output head on the winning
# conv-stem arch (2t, 9ch). Targets the "West: right region, wrong pixel" flaw
# (sharper pixel-level output vs the patch->256-logit projection). Mirrors the
# PROVEN train_2tanom recipe (fresh cache build + 4 epochs fits in <24h); only
# delta: 2t (not 2t_anom) + --conv_output_head.
set -o pipefail
export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 proj/9.4.1 eccodes/2.31.0
cd "$SCRATCH/wildfire-refactored"
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
PYTHON=$SCRATCH/venv-wildfire/bin/python
source slurm/lib_copy_cache.sh
LOCAL_CACHE=$SLURM_TMPDIR/cache; mkdir -p "$LOCAL_CACHE"
copy_s2s_cache "$SCRATCH/meteo_cache" "$LOCAL_CACHE"
CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,population,slope,burn_age"
REAL="$SCRATCH/meteo_cache/v3_9ch_12y_2014"
LOCAL_METEO="$LOCAL_CACHE/meteo"; mkdir -p "$LOCAL_METEO"
RUN_NAME="${RUN_NAME:-v3_9ch_convstem_fcnhead}"
# Seed channel-independent fire caches; fresh 2t meteo build (T mismatch on all
# persistent caches forces a rebuild anyway — this is the proven-reliable path).
echo "=== seed fire caches only (fresh meteo build) ==="
for f in "$REAL"/fire_dilated_*.npy "$REAL"/fire_patched_*.dat; do
  [ -e "$f" ] && cp "$f" "$LOCAL_METEO/" && echo "  seeded $(basename $f)"
done
echo "=== TRAIN conv-stem + FCN output head  run=$RUN_NAME ==="
$PYTHON -u -m src.training.train_v3 \
  --config configs/paths_narval.yaml --run_name "$RUN_NAME" \
  --data_start 2014-05-01 --pred_start 2022-05-01 --pred_end 2025-10-31 \
  --channels "$CHANNELS" --in_days 21 \
  --decoder s2s_legacy --s2s_cache "$LOCAL_CACHE/s2s_decoder_cache.dat" --s2s_max_issue_lag 3 \
  --loss_fn focal --focal_alpha 0.25 --focal_gamma 2.0 \
  --hard_neg_fraction 0.5 --neg_ratio 20 --neg_buffer 2 \
  --batch_size 4096 --epochs 4 --lr 1e-4 --weight_decay 0.01 --dropout 0.2 \
  --d_model 256 --nhead 8 --enc_layers 4 --dec_layers 4 --patch_size 16 \
  --dilate_radius 14 --val_lift_k 5000 --val_lift_sample_wins 20 \
  --enc_conv_stem --conv_output_head \
  --fire_season_only --cluster_eval --decoder_ctx \
  --cache_dir "$LOCAL_METEO" --chunk_patches 2000 --num_workers 4 \
  --log_interval 200 --skip_forecast \
  --label_fusion --nfdb_min_size_ha 1.0 \
  --fire_clim_dir data/fire_clim_annual_nbac \
  --save_per_window_json "$SCRATCH/wildfire-refactored/outputs/${RUN_NAME}_per_window.json"
echo "=== done exit=$? ==="
