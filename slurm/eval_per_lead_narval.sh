#!/bin/bash
#SBATCH --job-name=wf-per-lead
#SBATCH --gpus-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=400G
#SBATCH --output=/scratch/jiaqi217/logs/per_lead_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/per_lead_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# Per-lead-day Lift decay eval for a trained model.
#
# Produces outputs/per_lead/${RUN_NAME}.json with shape:
#   { k, coarsen_factor, n_windows,
#     per_window: [ { win_idx, date, ts, te,
#                     per_lead: [ { lead, n_fire, lift_k, lift_coarse },
#                                  ... 33 entries ... ] },
#                   ... ~583 entries ... ] }
#
# Combined with the per-lead-day baseline CSV (from
# baselines_all4_full_narval.sh), this gives the §6 lift-vs-lead-day
# figure (model curve + 4 baseline flat curves).
#
# Usage (default = SOTA 12y enc21):
#   sbatch slurm/eval_per_lead_narval.sh
#
# Override:
#   RUN_NAME=... CKPT=... sbatch slurm/eval_per_lead_narval.sh
# ----------------------------------------------------------------

set -uo pipefail
ENC=${ENC:-21}
RANGE_TAG=${RANGE_TAG:-12y_2014}
DATA_START=${DATA_START:-2014-05-01}
CACHE_LUSTRE=${CACHE_LUSTRE:-v3_9ch_12y_2014}
RUN_NAME=${RUN_NAME:-v3_9ch_enc${ENC}_${RANGE_TAG}}
CKPT=${CKPT:-$SCRATCH/wildfire-refactored/checkpoints/$RUN_NAME/best_model.pt}

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

CACHE_DIR_LUSTRE="$SCRATCH/meteo_cache/$CACHE_LUSTRE"
LOCAL_METEO="$LOCAL_CACHE/meteo"
mkdir -p "$LOCAL_METEO"
echo "=== copy meteo to local SSD ($CACHE_LUSTRE) ==="
t0=$SECONDS
for f in "$CACHE_DIR_LUSTRE"/*; do
    [ -f "$f" ] || continue
    cp "$f" "$LOCAL_METEO/" || { echo "FATAL"; exit 1; }
done
echo "  done in $((SECONDS - t0))s"

CHANNELS="FWI,2t,fire_clim,2d,tcw,sm20,population,slope,burn_age"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: ckpt missing: $CKPT"; exit 1
fi

OUT_DIR="$SCRATCH/wildfire-refactored/outputs/per_lead"
mkdir -p "$OUT_DIR"
OUT_JSON="$OUT_DIR/${RUN_NAME}.json"

echo "============================================="
echo "  PER-LEAD-DAY EVAL: $RUN_NAME"
echo "  ckpt: $CKPT"
echo "  out : $OUT_JSON"
echo "============================================="

$PYTHON -u -m src.training.train_v3 \
    --config configs/paths_narval.yaml \
    --run_name "${RUN_NAME}_per_lead" \
    --eval_checkpoint "$CKPT" \
    --per_lead_metrics_json "$OUT_JSON" \
    --full_val \
    --data_start "$DATA_START" --pred_start 2022-05-01 --pred_end 2025-09-30 \
    --channels "$CHANNELS" --in_days "$ENC" \
    --decoder s2s_legacy --s2s_cache "$LOCAL_CACHE/s2s_decoder_cache.dat" --s2s_max_issue_lag 3 \
    --loss_fn focal --focal_alpha 0.25 --focal_gamma 2.0 \
    --batch_size 4096 --epochs 0 \
    --d_model 256 --nhead 8 --enc_layers 4 --dec_layers 4 --patch_size 16 \
    --dilate_radius 14 --val_lift_k 5000 \
    --fire_season_only --decoder_ctx \
    --cache_dir "$LOCAL_METEO" --chunk_patches 2000 --num_workers 4 \
    --skip_forecast \
    --label_fusion --nfdb_min_size_ha 1.0 \
    --fire_clim_dir data/fire_clim_annual_nbac

PY_EXIT=$?
echo "=== done $(date) exit=$PY_EXIT ==="
if [ -f "$OUT_JSON" ]; then
    echo "per-lead-day JSON written: $(du -h $OUT_JSON | cut -f1)"
fi
exit $PY_EXIT
