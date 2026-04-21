#!/bin/bash
# ============================================================
#   Gate 3: verify training can reuse pre-built v3 cache
# ============================================================
# Submits ONE ENC=21 9ch training job and watches its startup log
# until it either:
#   (A) Prints "Loading cached memmap:" → cache reused correctly → PASS
#   (B) Prints "Streaming meteo_patched → float16 memmap" → cache MISS → FAIL
#   (C) Takes > 40 min to show either → timeout → FAIL (investigate)
#
# On PASS, the training job keeps running normally (don't cancel it).
# User can then submit the remaining 3 ENC values in parallel.
#
# Usage (on Narval login node):
#   cd /scratch/jiaqi217/wildfire-refactored
#   bash scripts/gate3_training_smoke.sh
#
# Exit codes:
#   0  → cache reused, safe to submit remaining sweep
#   1  → cache miss or timeout, DO NOT submit remaining
# ============================================================

set -uo pipefail

SCRIPT=slurm/train_v3_9ch_2000_narval.sh
ENC=21

if [ ! -f "$SCRIPT" ]; then
    echo "[FAIL] script not found: $SCRIPT"
    exit 1
fi

echo "[Gate 3] Submitting ENC=$ENC 9ch training as smoke probe..."
JID=$(ENC=$ENC sbatch --parsable "$SCRIPT")
if [ -z "$JID" ]; then
    echo "[FAIL] sbatch returned empty job id"
    exit 1
fi
LOG=/scratch/jiaqi217/logs/train_v3_9ch_2000_${JID}.log
echo "[Gate 3] Submitted job $JID"
echo "[Gate 3] log: $LOG"

# Wait for job to enter R state
echo "[Gate 3] Waiting for job to start (may be 0-60 min depending on queue)..."
wait_for_running() {
    local deadline=$(($(date +%s) + 3600 * 3))  # 3h ceiling
    while [ $(date +%s) -lt $deadline ]; do
        state=$(sacct -j $JID -n --format=State -X 2>/dev/null | tr -d ' ' | head -1)
        case "$state" in
            RUNNING) return 0 ;;
            COMPLETED|FAILED|CANCELLED*|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)
                echo "[FAIL] job ended in $state before running"
                return 1
                ;;
            *) sleep 60 ;;
        esac
    done
    echo "[FAIL] job did not start within 3h"
    return 1
}
wait_for_running || exit 1
echo "[Gate 3] job is RUNNING, watching log..."

# Watch log for cache signature (40 min budget from running start)
probe_deadline=$(($(date +%s) + 2400))  # 40 min
while [ $(date +%s) -lt $probe_deadline ]; do
    if [ -f "$LOG" ]; then
        if grep -q "Loading cached memmap:" "$LOG" 2>/dev/null; then
            echo "[PASS] cache reused — training is loading pre-built memmap"
            echo "[INFO] job $JID continues training; submit remaining sweep:"
            echo "  for ENC in 14 28 35; do ENC=\$ENC sbatch slurm/train_v3_9ch_2000_narval.sh; done"
            exit 0
        fi
        if grep -q "Streaming meteo_patched" "$LOG" 2>/dev/null; then
            echo "[FAIL] cache MISS — training is rebuilding memmap from scratch"
            echo "[INFO] cancel job and investigate cache key mismatch:"
            echo "  scancel $JID"
            echo "[INFO] check 'Aligned dates' line in $LOG vs cache filename"
            exit 1
        fi
    fi
    sleep 30
done

echo "[FAIL] Neither signature found within 40 min after job start"
echo "[INFO] inspect log manually: tail -100 $LOG"
exit 1
