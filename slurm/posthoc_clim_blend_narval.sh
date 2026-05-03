#!/bin/bash
#SBATCH --job-name=wf-posthoc-blend
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --output=/scratch/jiaqi217/logs/posthoc_blend_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/posthoc_blend_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

set -uo pipefail
export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1
cd "$SCRATCH/wildfire-refactored"
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

source slurm/lib_copy_cache.sh
copy_venv $SCRATCH/venv-wildfire

$PYTHON -u -m scripts.posthoc_clim_blend_sweep \
    --scores_dir "$SCRATCH/wildfire-refactored/outputs/window_scores_full/v3_9ch_enc21_12y_2014" \
    --fire_clim_dir "$SCRATCH/wildfire-refactored/data/fire_clim_annual_nbac" \
    --pred_start 2022-05-01 --pred_end 2025-10-31 \
    --output "$SCRATCH/wildfire-refactored/outputs/posthoc_clim_blend_sweep.json"

PY_EXIT=$?
echo "=== done $(date) exit=$PY_EXIT ==="
exit $PY_EXIT
