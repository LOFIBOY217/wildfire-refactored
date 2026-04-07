#!/bin/bash
#SBATCH --job-name=wf-hotspot-2017
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/jiaqi217/logs/hotspot_2017_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/hotspot_2017_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# Download CWFIS hotspot data for 2017 (May-Oct) via WFS
# and append to the main hotspot CSV.
#
# CWFIS WFS: https://cwfis.cfs.nrcan.gc.ca/geoserver/ows (public)
# No API key required.
#
# NOTE: If this fails with "Network is unreachable" on compute nodes,
# run directly on the login node:
#   cd $SCRATCH/wildfire-refactored
#   source $SCRATCH/venv-wildfire/bin/activate
#   python -m src.data_ops.download.download_hotspots \
#     --config configs/paths_narval.yaml \
#     --start_year 2017 --end_year 2017 \
#     --output data/hotspot/hotspot_2017.csv
#   # Then append (skip header of 2017 file):
#   tail -n +2 data/hotspot/hotspot_2017.csv >> data/hotspot/hotspot_2018_2025.csv
# ----------------------------------------------------------------

set -euo pipefail

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}

[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1

source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored

HOTSPOT_MAIN="data/hotspot/hotspot_2018_2025.csv"
HOTSPOT_2017="data/hotspot/hotspot_2017.csv"

echo "============================================="
echo "  CWFIS Hotspot 2017 Download"
echo "  Node: $(hostname)  Time: $(date)"
echo "============================================="

# Download 2017 May-Oct into separate file
python3 -u -m src.data_ops.download.download_hotspots \
    --config configs/paths_narval.yaml \
    --start_year 2017 \
    --end_year 2017 \
    --start_month 5 \
    --end_month 10 \
    --output "$HOTSPOT_2017"

N_2017=$(wc -l < "$HOTSPOT_2017")
echo "Downloaded $(( N_2017 - 1 )) records for 2017"

# Append 2017 records (skip header) to the main CSV
echo "Appending to $HOTSPOT_MAIN ..."
tail -n +2 "$HOTSPOT_2017" >> "$HOTSPOT_MAIN"

N_MAIN=$(wc -l < "$HOTSPOT_MAIN")
echo "Main CSV now has $(( N_MAIN - 1 )) total records"
echo "Done: $(date)"
