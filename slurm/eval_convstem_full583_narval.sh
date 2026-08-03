#!/bin/bash
#SBATCH --job-name=eval-cstem583
#SBATCH --account=def-inghaw
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=249G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/jiaqi217/logs/eval_cstem583_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/eval_cstem583_%j.err
# Clean head-to-head baseline: plain conv-stem (2t) on the SAME 583-window
# 2022-24 full val used by the 2t_anom eval. SLICES the persistent 1TB master
# cache v3_9ch_2000 (confirmed 2t: ch1 mean -1.64/std 12.68, matches 12y_2014)
# via --master_cache_dir → NO rebuild. Same data_start/pred range as 2t_anom eval.
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
RUN_NAME="v3_9ch_convstem_plain_full583"
CKPT="checkpoints/v3_9ch_enc21_12y_2014_convstem/best_model.pt"
MASTER="$SCRATCH/meteo_cache/v3_9ch_2000"   # 1TB 2000-2025 master (2t), slice-reuse
echo "=== EVAL plain conv-stem (2t) via MASTER slice  ckpt=$CKPT ==="
$PYTHON -u -m src.training.train_v3 \
  --config configs/paths_narval.yaml --run_name "$RUN_NAME" \
  --eval_checkpoint "$CKPT" --epochs 0 \
  --data_start 2014-05-01 --pred_start 2022-05-01 --pred_end 2025-10-31 \
  --channels "$CHANNELS" --in_days 21 \
  --decoder s2s_legacy --s2s_cache "$LOCAL_CACHE/s2s_decoder_cache.dat" --s2s_max_issue_lag 3 \
  --batch_size 1024 \
  --d_model 256 --nhead 8 --enc_layers 4 --dec_layers 4 --patch_size 16 \
  --dilate_radius 14 --val_lift_k 5000 --val_lift_sample_wins 9999 \
  --enc_conv_stem \
  --fire_season_only --cluster_eval --decoder_ctx \
  --cache_dir "$MASTER" --master_cache_dir "$MASTER" --master_data_start 2000-05-01 \
  --chunk_patches 2000 --num_workers 4 \
  --log_interval 200 --skip_forecast \
  --label_fusion --nfdb_min_size_ha 1.0 \
  --fire_clim_dir data/fire_clim_annual_nbac \
  --full_val \
  --save_per_window_json "$SCRATCH/wildfire-refactored/outputs/${RUN_NAME}_per_window.json"
echo "=== done exit=$? ==="
