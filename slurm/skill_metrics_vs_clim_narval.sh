#!/bin/bash
#SBATCH --job-name=wf-skill-metrics
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --output=/scratch/jiaqi217/logs/skill_metrics_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/skill_metrics_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# AUC / RSS / BSS / Anomaly Spearman ρ for SOTA + 3 baselines on the
# SAME windows + SAME labels.

set -uo pipefail
export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1 eccodes/2.31.0
cd "$SCRATCH/wildfire-refactored"
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

source slurm/lib_copy_cache.sh
copy_venv $SCRATCH/venv-wildfire

$PYTHON -u -m scripts.skill_metrics_vs_clim \
    --scores_dir "$SCRATCH/wildfire-refactored/outputs/window_scores_full/v3_9ch_enc21_12y_2014" \
    --label_npy "$SCRATCH/wildfire-refactored/data/fire_labels/fire_labels_nbac_nfdb_2000-05-01_2025-12-21_2281x2709_r14.npy" \
    --label_data_start 2000-05-01 \
    --fire_clim_dir "$SCRATCH/wildfire-refactored/data/fire_clim_annual_nbac" \
    --ecmwf_dir "$SCRATCH/wildfire-refactored/data/ecmwf_s2s_fire_epsg3978/fwinx" \
    --pred_start 2022-05-01 --pred_end 2025-10-31 \
    --lead_start 14 --lead_end 46 \
    --output "$SCRATCH/wildfire-refactored/outputs/skill_vs_clim_summary.json"

PY_EXIT=$?
echo "=== done $(date) exit=$PY_EXIT ==="
exit $PY_EXIT
