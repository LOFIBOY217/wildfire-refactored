#!/bin/bash
#SBATCH --job-name=wf-dl-cdem
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/jiaqi217/logs/dl_cdem_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/dl_cdem_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# Download CDEM tiles from NRCan (full Canada coverage incl. >60°N)
# ----------------------------------------------------------------

set -euo pipefail

if [ -z "${SCRATCH:-}" ]; then
    export SCRATCH=/scratch/jiaqi217
fi

module load StdEnv/2023 gcc/12.3 python/3.11.5 2>/dev/null || true
source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored

echo "============================================="
echo "  CDEM Download"
echo "  Node: $(hostname)  Time: $(date)"
echo "============================================="

python3 -u -m src.data_ops.download.download_cdem \
    --config configs/paths_narval.yaml \
    --workers 4

echo ""
echo "Done: $(date)"
