#!/bin/bash
#SBATCH --job-name=wf-22y-rec
#SBATCH --gpus-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=750G
#SBATCH --output=/scratch/jiaqi217/logs/train_22y_recency_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/train_22y_recency_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# DIAGNOSTIC: 22y enc14 with EXPONENTIAL RECENCY-WEIGHTED TRAINING
#
# Hypothesis: 4y > 22y observation reflects climate non-stationarity
# (Jain 2022, Buch 2023). Recency weighting w = exp(-(2024-y)/tau)
# down-weights older windows so model focuses on recent distribution.
#
# Expected (from Weyn 2020 JAMES on weather forecasting): +4-7% over
# unweighted. If 22y_recency > 4y enc14 (6.69x), confirms the
# diagnosis and gives us paper headline.
#
# Recipe difference from default:
#   --recency_tau ${TAU}   (default 0 = uniform; we test 6/10/15)
# All other hyperparams identical to 22y default.
#
# RUN_NAME: v3_9ch_enc14_2000_recency_tau${TAU}
#
# Usage:
#   TAU=6  sbatch slurm/train_v3_22y_recency_narval.sh
#   TAU=10 sbatch slurm/train_v3_22y_recency_narval.sh
#   TAU=15 sbatch slurm/train_v3_22y_recency_narval.sh
# ----------------------------------------------------------------

set -uo pipefail
TAU=${TAU:?Must set TAU (e.g. 6, 10, 15)}
ENC=${ENC:-14}

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 proj/9.4.1 eccodes/2.31.0
cd "$SCRATCH/wildfire-refactored"
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

# W&B
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
RUN_NAME="v3_9ch_enc${ENC}_2000_recency_tau${TAU}"

echo "============================================="
echo "  V3 9ch enc${ENC} 22y with RECENCY WEIGHTING (tau=${TAU})"
echo "  Run: $RUN_NAME"
echo "  weight = exp(-(2024 - window_year) / ${TAU})"
echo "============================================="

$PYTHON -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name "$RUN_NAME" \
    --data_start 2000-05-01 --pred_start 2022-05-01 --pred_end 2025-10-31 \
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
    --recency_tau "$TAU" --recency_base_year 2024 \
    --wandb_project wildfire-s2s \
    --wandb_tags "9ch,enc${ENC},2000,22y,recency_tau${TAU}" \
    --save_per_window_json "$SCRATCH/wildfire-refactored/outputs/${RUN_NAME}_per_window.json"

PY_EXIT=$?
echo "=== Done $(date) exit=$PY_EXIT ==="
exit $PY_EXIT
