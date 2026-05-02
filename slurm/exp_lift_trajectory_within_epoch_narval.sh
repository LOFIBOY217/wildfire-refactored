#!/bin/bash
#SBATCH --job-name=wf-lift-traj
#SBATCH --gpus-per-node=1
#SBATCH --time=1-12:00:00
#SBATCH --mem=750G
#SBATCH --output=/scratch/jiaqi217/logs/exp_lift_traj_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/exp_lift_traj_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ==================================================================
# EXPERIMENT: lift_trajectory_within_epoch
#
# HYPOTHESIS (2026-05-01, see docs/CALIBRATION_VS_RANK_HYPOTHESIS.md):
#
#   "Val Lift@K peaks at ~5,000-10,000 SGD updates regardless of data
#    range. After the peak, val Lift declines monotonically while
#    train_loss continues to decrease and ROC-AUC stays stable.
#    Mechanism: focal loss + cosine LR push model toward calibrated
#    smooth predictions; calibrated predictions hurt top-K Lift
#    because they spread probability mass across patches."
#
# CRITICAL TEST: Run 22y enc14 with mid-epoch eval every 500 batches.
# Each epoch has 10,887 batches → ~22 eval points per epoch.
#
# PREDICTION 1: Within ep1 (10,887 batches), val_lift will rise then
#               peak around batch 5,000-10,000, then decline before
#               epoch end. Currently we only see end-of-epoch (10,887
#               = 5.73x) — peak might be HIGHER.
#
# PREDICTION 2: ROC-AUC will be ~stable (0.86-0.91) throughout — this
#               confirms model learns globally, only top-K rank degrades.
#
# PREDICTION 3: Brier will improve monotonically — calibration gets
#               better as Lift gets worse.
#
# OUTCOME INTERPRETATION:
#   IF peak found in middle of ep1:
#     → Hypothesis CONFIRMED. We've been missing the real best ckpt.
#     → Action: implement step-level early stopping; report higher
#       val_lift in paper.
#   IF val_lift monotonically decreases through ep1:
#     → Hypothesis WRONG (peak is at step 0).
#     → Action: investigate alternative explanations (distribution
#       shift, hard negative pool changes, etc.)
#   IF val_lift monotonically increases:
#     → Need MORE updates; recipe is fine.
#
# CSV output: outputs/v3_9ch_enc14_2000_lift_traj_lift_trajectory.csv
# Plot via:  python scripts/plot_lift_trajectory.py
#
# Cost: 1 GPU job, ~30h (22h base + ~25% overhead from 88 mid-epoch evals)
# ==================================================================

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
$PYTHON -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())" || exit 1

LOCAL_CACHE=$SLURM_TMPDIR/cache
mkdir -p "$LOCAL_CACHE"
copy_s2s_cache "$SCRATCH/meteo_cache" "$LOCAL_CACHE"

CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,population,slope,burn_age"
TRAIN_CACHE_DIR="$SCRATCH/meteo_cache/v3_9ch_2000"
RUN_NAME="v3_9ch_enc14_2000_lift_traj"

echo "============================================="
echo "  EXP: lift_trajectory_within_epoch"
echo "  Run: $RUN_NAME"
echo "  mid_epoch_val_every=500 batches"
echo "  expected: ~22 eval points per epoch in 22y"
echo "============================================="

$PYTHON -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name "$RUN_NAME" \
    --data_start 2000-05-01 --pred_start 2022-05-01 --pred_end 2025-10-31 \
    --channels "$CHANNELS" --in_days 14 \
    --decoder s2s_legacy --s2s_cache "$LOCAL_CACHE/s2s_decoder_cache.dat" --s2s_max_issue_lag 3 \
    --loss_fn focal --focal_alpha 0.25 --focal_gamma 2.0 \
    --hard_neg_fraction 0.5 --neg_ratio 20 --neg_buffer 2 \
    --batch_size 4096 --epochs 4 --lr 1e-4 --weight_decay 0.01 --dropout 0.2 \
    --d_model 256 --nhead 8 --enc_layers 4 --dec_layers 4 --patch_size 16 \
    --dilate_radius 14 --val_lift_k 5000 --val_lift_sample_wins 20 \
    --fire_season_only --decoder_ctx --load_train_to_ram \
    --cache_dir "$TRAIN_CACHE_DIR" --chunk_patches 2000 --num_workers 4 \
    --log_interval 200 --skip_forecast \
    --label_fusion --nfdb_min_size_ha 1.0 \
    --fire_clim_dir data/fire_clim_annual_nbac \
    --mid_epoch_val_every 500 \
    --wandb_project wildfire-s2s \
    --wandb_tags "9ch,enc14,2000,22y,lift_traj_diagnostic" \
    --save_per_window_json "$SCRATCH/wildfire-refactored/outputs/${RUN_NAME}_per_window.json"

PY_EXIT=$?
echo "=== Done $(date) exit=$PY_EXIT ==="
echo "=== Lift trajectory CSV: ==="
ls -la outputs/${RUN_NAME}_lift_trajectory.csv 2>/dev/null
exit $PY_EXIT
