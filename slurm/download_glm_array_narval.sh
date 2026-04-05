#!/bin/bash
#SBATCH --job-name=wf-dl-glm
#SBATCH --array=0-6
#SBATCH --time=48:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/jiaqi217/logs/dl_glm_%A_%a.log
#SBATCH --error=/scratch/jiaqi217/logs/dl_glm_%A_%a.err

# Array index 0-6 → years 2018-2024, fire season May-Oct each

set -euo pipefail
if [ -z "${SCRATCH:-}" ]; then export SCRATCH=/scratch/jiaqi217; fi
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1 2>/dev/null || true
source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored

YEARS=(2018 2019 2020 2021 2022 2023 2024)
YEAR=${YEARS[$SLURM_ARRAY_TASK_ID]}

echo "GLM download: ${YEAR}  task ${SLURM_ARRAY_TASK_ID}/6  node $(hostname)"

python3 -u -m src.data_ops.download.download_goes_glm \
    --start "${YEAR}0501" --end "${YEAR}1031" --workers 4 \
    --config configs/paths_narval.yaml
