#!/bin/bash
# Check readiness for S2S full-patch training (wf-s2s-fp)
# Run on Narval: bash scripts/check_fullpatch_ready.sh

SCRATCH=/scratch/jiaqi217
PROJECT=$SCRATCH/wildfire-refactored
METEO=$SCRATCH/meteo_cache

OK=0
FAIL=0

check() {
    local label="$1"
    local path="$2"
    local min_gb="${3:-0}"
    if [ -f "$path" ]; then
        size_gb=$(du -BG "$path" 2>/dev/null | awk '{print $1}' | tr -d 'G')
        if [ "$min_gb" -gt 0 ] && [ "${size_gb:-0}" -lt "$min_gb" ]; then
            echo "  ⚠  $label  →  ${size_gb}G (expected >=${min_gb}G, may be incomplete)"
            FAIL=$((FAIL+1))
        else
            echo "  ✅ $label  →  ${size_gb}G"
            OK=$((OK+1))
        fi
    else
        echo "  ❌ $label  →  NOT FOUND: $path"
        FAIL=$((FAIL+1))
    fi
}

check_dir() {
    local label="$1"
    local path="$2"
    if [ -d "$path" ]; then
        echo "  ✅ $label  →  $path"
        OK=$((OK+1))
    else
        echo "  ❌ $label  →  NOT FOUND: $path"
        FAIL=$((FAIL+1))
    fi
}

echo "=============================================="
echo " S2S Full-Patch Training Readiness Check"
echo "=============================================="

echo ""
echo "[1] S2S full-patch cache (the key blocker)"
check "s2s_full_patch_cache.dat" \
      "$PROJECT/data/s2s_full_patch_cache.dat" 5000
check "s2s_full_patch_cache.dat.dates.npy" \
      "$PROJECT/data/s2s_full_patch_cache.dat.dates.npy" 0

echo ""
echo "[2] Meteo encoder cache (STEP 6 skip)"
METEO_PF=$(ls $METEO/meteo_p16_C8_T*_pf.dat 2>/dev/null | tail -1)
METEO_STATS=$(ls $METEO/meteo_p16_C8_T*_stats.npy 2>/dev/null | tail -1)
[ -n "$METEO_PF" ]    && { echo "  ✅ meteo_pf.dat      →  $(du -BG $METEO_PF | awk '{print $1}')"; OK=$((OK+1)); } \
                       || { echo "  ❌ meteo_pf.dat      →  NOT FOUND"; FAIL=$((FAIL+1)); }
[ -n "$METEO_STATS" ] && { echo "  ✅ meteo_stats.npy   →  found"; OK=$((OK+1)); } \
                       || { echo "  ❌ meteo_stats.npy   →  NOT FOUND"; FAIL=$((FAIL+1)); }

echo ""
echo "[3] Fire labels cache (STEP 4 skip)"
FIRE_PF=$(ls $METEO/fire_patched_r14_*.dat 2>/dev/null | tail -1)
[ -n "$FIRE_PF" ] && { echo "  ✅ fire_patched.dat  →  $(du -BG $FIRE_PF | awk '{print $1}')"; OK=$((OK+1)); } \
                  || { echo "  ❌ fire_patched.dat  →  NOT FOUND"; FAIL=$((FAIL+1)); }

echo ""
echo "[4] Static inputs"
check "fire_climatology.tif" \
      "$PROJECT/data/fire_climatology.tif" 0

echo ""
echo "[5] Environment"
check_dir "venv-wildfire" "$SCRATCH/venv-wildfire"
check_dir "configs" "$PROJECT/configs"
[ -f "$PROJECT/slurm/train_v2_s2s_fullpatch_narval.sh" ] \
    && { echo "  ✅ SLURM script      →  found"; OK=$((OK+1)); } \
    || { echo "  ❌ SLURM script      →  NOT FOUND"; FAIL=$((FAIL+1)); }

echo ""
echo "=============================================="
echo " Result: $OK passed  |  $FAIL failed"
if [ "$FAIL" -eq 0 ]; then
    echo " ✅ READY — sbatch slurm/train_v2_s2s_fullpatch_narval.sh"
else
    echo " ❌ NOT READY — fix the items above first"
fi
echo "=============================================="
