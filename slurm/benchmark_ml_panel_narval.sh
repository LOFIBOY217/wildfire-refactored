#!/bin/bash
#SBATCH --job-name=wf-ml-panel
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=0-12:00:00
#SBATCH --output=/scratch/jiaqi217/logs/ml_panel_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/ml_panel_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# Run RF + XGBoost + MLP baselines back-to-back. CPU-only; reuses
# benchmark_ml.py with --model arg. Each model takes ~30-90 min;
# total ~3-4 h. Data loading is shared across runs by reusing the
# Python process.

set -uo pipefail
export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1
source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

LABEL_NPY="$SCRATCH/wildfire-refactored/data/fire_labels/fire_labels_nbac_nfdb_2000-05-01_2025-12-21_2281x2709_r14.npy"
CLIM_TIF="$SCRATCH/wildfire-refactored/data/fire_clim_annual_nbac/fire_clim_upto_2022.tif"
OUT_CSV="$SCRATCH/wildfire-refactored/outputs/benchmark_ml.csv"

echo "============================================="
echo "  ML BASELINE PANEL  (RF + XGBoost + MLP)"
echo "  label : $LABEL_NPY"
echo "  clim  : $CLIM_TIF"
echo "  out   : $OUT_CSV"
echo "============================================="

for MODEL in rf xgboost mlp; do
    echo
    echo ">>> $MODEL  $(date)"
    python3 -u -m scripts.benchmark_ml \
        --model "$MODEL" \
        --config configs/paths_narval.yaml \
        --pred_start 2022-05-01 --pred_end 2024-10-31 \
        --in_days 7 --lead_start 14 --lead_end 45 \
        --patch_size 16 --dilate_radius 14 \
        --fire_label_npy "$LABEL_NPY" \
        --climatology_tif "$CLIM_TIF" \
        --n_train_wins 80 \
        --n_sample_wins 20 \
        --output_csv "$OUT_CSV"
    echo "<<< $MODEL done  $(date)"
done

echo "=== ALL DONE  $(date) ==="
