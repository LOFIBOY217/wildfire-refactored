#!/bin/bash
#SBATCH --job-name=wf-proc-srtm
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/jiaqi217/logs/proc_srtm_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/proc_srtm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# Merge 2748 SRTM .hgt tiles → slope.tif + aspect.tif on FWI grid
# Needs large memory for tile merge

set -euo pipefail
if [ -z "${SCRATCH:-}" ]; then export SCRATCH=/scratch/jiaqi217; fi
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1 2>/dev/null || true
source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored

python3 -u -m src.data_ops.processing.process_srtm_slope \
    --config configs/paths_narval.yaml
