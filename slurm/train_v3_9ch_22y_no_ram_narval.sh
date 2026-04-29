#!/bin/bash
#SBATCH --job-name=wf-22y-no-ram
#SBATCH --gpus-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=400G
#SBATCH --output=/scratch/jiaqi217/logs/train_v3_22y_no_ram_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/train_v3_22y_no_ram_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca
#SBATCH --account=def-inghaw

# ----------------------------------------------------------------
# 22y rescue script: drops --load_train_to_ram so we only need 400G
# (fits standard 510G GPU nodes -> schedules fast). Trades training
# speed (~1.5-2x slower) for OOM safety. Used after 22y enc14
# (59815513) OOM'd at 750G during the np.array(memmap[...]) copy.
#
# Usage:
#   ENC=14 sbatch slurm/train_v3_9ch_22y_no_ram_narval.sh
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
CACHE_DIR_LUSTRE="$SCRATCH/meteo_cache/v3_9ch_2000"
LOCAL_METEO="$LOCAL_CACHE/meteo"
mkdir -p "$LOCAL_METEO"

# Copy 22y meteo cache to local SSD (~1.1TB, ~30-60min)
echo "=== copy 22y meteo to local SSD ==="
t0=$SECONDS
for f in "$CACHE_DIR_LUSTRE"/*; do
    [ -f "$f" ] || continue
    fname=$(basename "$f")
    sz=$(du -h "$f" | cut -f1)
    echo "  copy $fname ($sz)"
    cp "$f" "$LOCAL_METEO/" || { echo "FATAL: copy failed"; exit 1; }
done
echo "  done in $((SECONDS - t0))s"

RUN_NAME="v3_9ch_enc${ENC}_2000_no_ram"

if [ ! -d "$SCRATCH/wildfire-refactored/data/fire_clim_annual_nbac" ]; then
    echo "ERROR: data/fire_clim_annual_nbac missing"
    exit 1
fi

echo "============================================="
echo "  V3 9ch x enc${ENC} x 22y NO_LOAD_TO_RAM"
echo "  mem=400G, expects 1.5-2x slower training"
echo "============================================="

# NOTE: --load_train_to_ram OMITTED on purpose. Training reads from
# local SSD memmap each batch (slower than RAM but stable).
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
    --fire_season_only --cluster_eval --decoder_ctx \
    --cache_dir "$LOCAL_METEO" --chunk_patches 2000 --num_workers 4 \
    --log_interval 200 --skip_forecast \
    --label_fusion --nfdb_min_size_ha 1.0 \
    --fire_clim_dir data/fire_clim_annual_nbac \
    --save_per_window_json "$SCRATCH/wildfire-refactored/outputs/${RUN_NAME}_per_window.json"

PY_EXIT=$?
echo "=== Done: $(date) exit=$PY_EXIT ==="
exit $PY_EXIT
