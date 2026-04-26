#!/bin/bash
#SBATCH --job-name=wf-22y-enc28-cosine-lower
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=750G
#SBATCH --output=/scratch/jiaqi217/logs/train_22y_enc28_cosine_lower_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/train_22y_enc28_cosine_lower_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca
#SBATCH --account=def-inghaw

# ----------------------------------------------------------------
# 22y enc28 retrain with HALVED LR + lower LR_min.
#
# Diagnosis: original 22y enc28 (59870683) ep1 lift 5.59 -> ep2 3.81
# = 32% drop. CosineAnnealing IS on (lr_min=1e-6), but with --lr 1e-4
# and 4 epochs, the schedule keeps LR > 5e-5 through ep2, which is
# too aggressive for 22y data (5.5× more gradient updates per epoch
# than 4y at same LR). Halving initial LR + lowering min compresses
# the schedule into the safer regime.
#
# Cost: 1 extra GPU job (~10h). Benefit: validates the LR diagnosis,
# potentially gives true 22y > 4y SOTA.
# ----------------------------------------------------------------

set -uo pipefail
ENC=28

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

# Copy 22y meteo cache to local SSD (now includes the dilated label
# from earlier C-step; train_v3 will reuse it and skip dilation)
CACHE_DIR_LUSTRE="$SCRATCH/meteo_cache/v3_9ch_2000"
LOCAL_METEO="$LOCAL_CACHE/meteo"
mkdir -p "$LOCAL_METEO"
echo "=== copy 22y meteo to local SSD (incl. pre-built dilated) ==="
t0=$SECONDS
for f in "$CACHE_DIR_LUSTRE"/*; do
    [ -f "$f" ] || continue
    fname=$(basename "$f")
    sz=$(du -h "$f" | cut -f1)
    echo "  copy $fname ($sz)"
    cp "$f" "$LOCAL_METEO/" || { echo "FATAL"; exit 1; }
done
echo "  done in $((SECONDS - t0))s"

CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,population,slope,burn_age"
RUN_NAME="v3_9ch_enc${ENC}_2000_cosine_lower"

if [ ! -d "$SCRATCH/wildfire-refactored/data/fire_clim_annual_nbac" ]; then
    echo "ERROR: NBAC fire_clim missing"; exit 1
fi

echo "============================================="
echo "  V3 22y enc${ENC}  COSINE-LOWER  --lr 5e-5"
echo "  Run: $RUN_NAME"
echo "============================================="

$PYTHON -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name "$RUN_NAME" \
    --data_start 2000-05-01 --pred_start 2022-05-01 --pred_end 2025-10-31 \
    --channels "$CHANNELS" --in_days "$ENC" \
    --decoder s2s_legacy --s2s_cache "$LOCAL_CACHE/s2s_decoder_cache.dat" --s2s_max_issue_lag 3 \
    --loss_fn focal --focal_alpha 0.25 --focal_gamma 2.0 \
    --hard_neg_fraction 0.5 --neg_ratio 20 --neg_buffer 2 \
    --batch_size 4096 --epochs 4 --lr 5e-5 --lr_min 1e-7 --weight_decay 0.01 --dropout 0.2 \
    --d_model 256 --nhead 8 --enc_layers 4 --dec_layers 4 --patch_size 16 \
    --dilate_radius 14 --val_lift_k 5000 --val_lift_sample_wins 20 \
    --fire_season_only --cluster_eval --decoder_ctx --load_train_to_ram \
    --cache_dir "$LOCAL_METEO" --chunk_patches 2000 --num_workers 4 \
    --log_interval 200 --skip_forecast \
    --label_fusion --nfdb_min_size_ha 1.0 \
    --fire_clim_dir data/fire_clim_annual_nbac \
    --save_per_window_json "$SCRATCH/wildfire-refactored/outputs/${RUN_NAME}_per_window.json"

PY_EXIT=$?
echo "=== Done: $(date) exit=$PY_EXIT ==="
exit $PY_EXIT
