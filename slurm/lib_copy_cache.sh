#!/bin/bash
# ============================================================
# Shared library: copy caches from Lustre to local SSD
# Source this file from SLURM scripts:
#   source slurm/lib_copy_cache.sh
# ============================================================

# Print timestamped message
ts() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Copy venv to local SSD and activate it
# Usage: copy_venv <scratch_venv_path>
copy_venv() {
    local src="$1"
    local dst="$SLURM_TMPDIR/venv"
    local timeout_sec="${2:-1800}"  # default 30 min

    ts "=== COPYING VENV TO LOCAL SSD ==="
    ts "  Source : $src"
    ts "  Dest   : $dst"
    ts "  Timeout: ${timeout_sec}s"
    local t0=$SECONDS

    timeout $timeout_sec cp -a "$src" "$dst"
    local rc=$?
    local elapsed=$((SECONDS - t0))

    if [ $rc -eq 0 ]; then
        local sz=$(du -sh "$dst" | cut -f1)
        ts "  DONE  venv copy: $sz in ${elapsed}s"
        source "$dst/bin/activate"
        export PYTHON="$dst/bin/python"
    elif [ $rc -eq 124 ]; then
        ts "  TIMEOUT: venv copy took >${timeout_sec}s, falling back to Lustre venv"
        rm -rf "$dst" 2>/dev/null
        export PYTHON="$src/bin/python"
    else
        ts "  WARNING: venv copy failed (rc=$rc), falling back to Lustre venv"
        export PYTHON="$src/bin/python"
    fi
    ts "  Python now: $PYTHON"
    ts "=== VENV COPY COMPLETE ==="
}

# Copy a single file with timeout and speed reporting
# Usage: copy_with_timeout <src> <dst_dir> <timeout_seconds>
copy_with_timeout() {
    local src="$1"
    local dst_dir="$2"
    local timeout_sec="${3:-3600}"  # default 1 hour
    local fname=$(basename "$src")
    local sz=$(du -h "$src" | cut -f1)
    local sz_bytes=$(stat --format="%s" "$src" 2>/dev/null || echo 0)

    ts "  START copy: $fname ($sz) timeout=${timeout_sec}s"
    local t0=$SECONDS

    timeout $timeout_sec cp "$src" "$dst_dir/"
    local rc=$?

    local elapsed=$((SECONDS - t0))
    if [ $rc -eq 0 ]; then
        local speed="N/A"
        if [ $elapsed -gt 0 ] && [ $sz_bytes -gt 0 ]; then
            speed=$(echo "$sz_bytes $elapsed" | awk '{printf "%.1f MB/s", $1/1048576/$2}')
        fi
        ts "  DONE  copy: $fname in ${elapsed}s ($speed)"
    elif [ $rc -eq 124 ]; then
        ts "  TIMEOUT copy: $fname after ${timeout_sec}s — Lustre likely stuck"
        return 1
    else
        ts "  FAILED copy: $fname (exit code $rc) after ${elapsed}s"
        return 1
    fi
    return 0
}

# Copy all meteo caches (meteo_pf, stats, fire) to LOCAL_CACHE
# Usage: copy_meteo_caches <scratch_cache_dir> <local_cache_dir> <timeout_per_file>
copy_meteo_caches() {
    local scratch="$1"
    local local_dir="$2"
    local timeout_sec="${3:-3600}"

    ts "=== COPYING CACHES TO LOCAL SSD ==="
    ts "  Source : $scratch"
    ts "  Dest   : $local_dir"
    echo "  SLURM_TMPDIR disk:"
    df -h $SLURM_TMPDIR
    mkdir -p "$local_dir"

    # Meteo cache (the big one)
    local copied=0
    local use_lustre=0
    for f in $scratch/meteo_p16_*_pf.dat; do
        [ -f "$f" ] || continue
        copy_with_timeout "$f" "$local_dir" "$timeout_sec" || {
            ts "WARNING: Lustre copy failed/timed out. Will use Lustre path directly."
            use_lustre=1
            # Symlink instead so training finds the file
            ln -sf "$f" "$local_dir/$(basename $f)"
        }
        copied=$((copied + 1))
    done

    if [ $copied -eq 0 ]; then
        ts "WARNING: No meteo_pf.dat cache found in $scratch"
        ts "  Training will build cache from scratch (slow but OK)"
    fi

    # Small files (stats, fire cache, norm_stats) — 5 min timeout each
    for pattern in "*_stats.npy" "fire_*.npy" "fire_*.dat" "norm_stats*"; do
        for f in $scratch/$pattern; do
            [ -f "$f" ] || continue
            copy_with_timeout "$f" "$local_dir" 300 || \
                ln -sf "$f" "$local_dir/$(basename $f)"
        done
    done

    ts "=== LOCAL CACHE CONTENTS ==="
    ls -lh "$local_dir/"
    echo "  SLURM_TMPDIR remaining:"
    df -h $SLURM_TMPDIR
    ts "=== COPY COMPLETE ==="
}

# Copy S2S decoder cache
# Usage: copy_s2s_cache <scratch_cache_dir> <local_cache_dir> <timeout>
copy_s2s_cache() {
    local scratch="$1"
    local local_dir="$2"
    local timeout_sec="${3:-1800}"

    if [ -f "$scratch/s2s_decoder_cache.dat" ]; then
        copy_with_timeout "$scratch/s2s_decoder_cache.dat" "$local_dir" "$timeout_sec" || {
            ts "FATAL: S2S cache copy failed/timed out."
            exit 1
        }
        [ -f "$scratch/s2s_decoder_cache.dat.dates.npy" ] && \
            copy_with_timeout "$scratch/s2s_decoder_cache.dat.dates.npy" "$local_dir" 60
    else
        ts "WARNING: s2s_decoder_cache.dat not found in $scratch"
    fi
}

# Copy results back from local SSD to scratch
# Usage: copy_back <local_dir> <scratch_dir>
copy_back() {
    local local_dir="$1"
    local scratch="$2"

    ts "=== COPYING RESULTS BACK TO SCRATCH ==="
    mkdir -p "$scratch"
    local t0=$SECONDS

    for f in "$local_dir"/*; do
        [ -f "$f" ] || continue
        local fname=$(basename "$f")
        local sz=$(du -h "$f" | cut -f1)
        ts "  $fname ($sz)..."
        timeout 7200 cp "$f" "$scratch/" || ts "  ERROR: failed to copy $fname back"
    done

    ts "  Total copy-back time: $((SECONDS - t0))s"
    ts "=== COPY BACK COMPLETE ==="
}
