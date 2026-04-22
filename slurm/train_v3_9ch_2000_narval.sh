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
CACHE_DIR_2000="$SCRATCH/meteo_cache/v3_9ch_2000"
RUN_NAME="v3_9ch_enc${ENC}_2000"

if [ ! -d "$CACHE_DIR_2000" ] || [ -z "$(ls -A "$CACHE_DIR_2000" 2>/dev/null)" ]; then
    echo "ERROR: $CACHE_DIR_2000 is empty or missing."
    echo "Run cache build first: CACHE_CHANNELS=9 sbatch slurm/rebuild_cache_2000_2025_narval.sh"
    exit 1
fi

echo "============================================="
echo "  V3 9ch x enc${ENC} x 2000-2025 (fair-comparison baseline)"
echo "  Run name: $RUN_NAME"
echo "  Cache: $CACHE_DIR_2000"
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
    --cache_dir "$CACHE_DIR_2000" --chunk_patches 2000 --num_workers 4 \
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
