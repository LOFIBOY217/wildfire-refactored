#!/bin/bash
#SBATCH --job-name=wf-python
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/jiaqi217/logs/python_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/python_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ============================================================
#  Generic Python wrapper for Narval — ALWAYS use sbatch
#  NEVER run python directly on the login node (violates HPC policy)
#
#  Override SLURM params via #SBATCH directive on command line:
#    sbatch --time=12:00:00 --mem=64G --cpus-per-task=8 \
#           --job-name=my-task \
#           slurm/run_python_narval.sh \
#           "python3 -m src.path.to.module --args"
#
#  Or with env vars:
#    CMD="python3 -m src.xxx" sbatch slurm/run_python_narval.sh
#
#  For GPU jobs, also pass --gpus-per-node=1 (and use a training script).
# ============================================================

set -uo pipefail

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}

# Load modules (standard set — extend in derived scripts if needed)
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1

# Geospatial env
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

# Activate project venv
source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH

# Command to run:
#   priority 1: $1 (command-line arg, quoted)
#   priority 2: $CMD (env var)
#   else: error
CMD="${1:-${CMD:-}}"
if [ -z "$CMD" ]; then
    echo "ERROR: no command given. Pass as arg or set CMD env var."
    echo ""
    echo "Example:"
    echo "  sbatch --time=2:00:00 slurm/run_python_narval.sh 'python3 -m src.foo.bar --arg val'"
    echo "  CMD='python3 -m src.foo.bar' sbatch slurm/run_python_narval.sh"
    exit 2
fi

echo "============================================="
echo "  Python wrapper job"
echo "  Node: $(hostname)"
echo "  Time: $(date)"
echo "  Command: $CMD"
echo "============================================="

eval "$CMD"
EXIT=$?

echo ""
echo "============================================="
echo "  Done: $(date)  exit=$EXIT"
echo "============================================="
exit $EXIT
