#!/bin/bash
#SBATCH --job-name=wf-logreg-baseline
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
# 200G -> 400G after 59782192 OOM'd loading sm20 day 3000/5350: each
# daily channel stack (T*n_patches*P*P*2 bytes) ~ 65 GB; with 2t + sm20
# + FWI + fire_label ~ 227 GB > 200 GB cap. 400G covers all 5 channels
# plus logreg fit + scoring.
#SBATCH --time=0-06:00:00
#SBATCH --output=/scratch/jiaqi217/logs/logreg_baseline_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/logreg_baseline_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# Logistic regression baseline on NEW NBAC+NFDB labels.
# CPU-only (no GPU needed). Reuses benchmark_baselines.py data loaders.
#
# Inputs:
#   - same FWI, fire_label_npy, climatology, ERA5 channels as transformer
# Outputs:
#   - outputs/benchmark_logreg.csv (appendable to benchmark_baselines.csv)

set -uo pipefail
export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1
source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

# 22y NBAC+NFDB label stack (built by scripts/build_fire_labels.py)
LABEL_NPY="$SCRATCH/wildfire-refactored/data/fire_labels/fire_labels_nbac_nfdb_2000-05-01_2025-12-21_2281x2709_r14.npy"
CLIM_TIF="$SCRATCH/wildfire-refactored/data/fire_clim_annual_nbac/fire_clim_upto_2022.tif"

if [ ! -f "$LABEL_NPY" ]; then
    echo "ERROR: label npy missing: $LABEL_NPY"
    echo "Build it via slurm/build_fire_labels_narval.sh"
    exit 1
fi
if [ ! -f "$CLIM_TIF" ]; then
    echo "ERROR: NBAC climatology TIF missing: $CLIM_TIF"
    exit 1
fi

OUT_CSV="$SCRATCH/wildfire-refactored/outputs/benchmark_logreg.csv"

echo "============================================="
echo "  LogReg baseline (NBAC+NFDB labels, 22y train, 2022-2024 val)"
echo "  label  : $LABEL_NPY"
echo "  clim   : $CLIM_TIF"
echo "  output : $OUT_CSV"
echo "============================================="

python3 -u -m scripts.benchmark_logreg \
    --config configs/paths_narval.yaml \
    --pred_start 2022-05-01 --pred_end 2024-10-31 \
    --in_days 7 --lead_start 14 --lead_end 45 \
    --patch_size 16 --dilate_radius 14 \
    --fire_label_npy "$LABEL_NPY" \
    --climatology_tif "$CLIM_TIF" \
    --n_train_wins 80 \
    --n_sample_wins 20 \
    --output_csv "$OUT_CSV"

echo "=== done $(date) ==="
ls -lh "$OUT_CSV"
