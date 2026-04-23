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

# Copy venv to local SSD via tar (single-file sequential IO, avoids Lustre metadata overhead)
# Usage: copy_venv <scratch_venv_path>
#
# Strategy: tar archive on Lustre → single large file copy → extract on local SSD
# Why tar?  venv has ~50k small files. Lustre metadata ops are slow (can take hours).
#           tar packs them into one file → sequential read → fast even on slow nodes.
# Diagnostics: logs file count, tar size, copy speed, extract speed separately
#              so we can identify which phase is the bottleneck.
copy_venv() {
    local src="$1"
    local dst="$SLURM_TMPDIR/venv"
    local tar_path="${src}.tar"
    local timeout_sec="${2:-600}"  # default 10 min (tar should be fast)

    ts "=== COPYING VENV TO LOCAL SSD (tar method) ==="
    ts "  Source venv : $src"
    ts "  Tar archive : $tar_path"
    ts "  Dest        : $dst"

    # Step 1: Ensure tar exists on Lustre (one-time creation)
    # Validate existing tar:
    #   (a) must be > 1MB (corrupt/truncated check)
    #   (b) must be newer than the newest file inside the source venv
    #       (staleness check — added 2026-04-22 after run 59719579 failed:
    #        tar from Mar 28 was missing geopandas installed Apr 18, but
    #        cache logic only checked existence + size, silently reused
    #        the old tar and lost a 4y training run)
    if [ -f "$tar_path" ]; then
        local existing_sz=$(stat --format="%s" "$tar_path" 2>/dev/null || echo 0)
        if [ "$existing_sz" -lt 1048576 ]; then
            ts "  [tar] Existing archive too small (${existing_sz} bytes) — likely corrupt, removing"
            rm -f "$tar_path"
        else
            local tar_mtime=$(stat --format="%Y" "$tar_path" 2>/dev/null || echo 0)
            # newest mtime in the venv (any package install bumps a file mtime)
            local venv_newest=$(find "$src" -type f -printf '%T@\n' 2>/dev/null | sort -nr | head -1 | cut -d. -f1)
            venv_newest=${venv_newest:-0}
            if [ "$tar_mtime" -lt "$venv_newest" ]; then
                local tar_age=$(date -d @$tar_mtime '+%Y-%m-%d %H:%M' 2>/dev/null || echo "unknown")
                local venv_age=$(date -d @$venv_newest '+%Y-%m-%d %H:%M' 2>/dev/null || echo "unknown")
                ts "  [tar] STALE: tar from $tar_age, venv updated $venv_age — rebuilding"
                rm -f "$tar_path"
            fi
        fi
    fi

    if [ ! -f "$tar_path" ]; then
        ts "  [tar] Archive not found, creating from venv directory..."
        ts "  [tar] File count: $(find "$src" -type f | wc -l) files"
        local t_tar=$SECONDS
        tar cf "$tar_path" -C "$(dirname $src)" "$(basename $src)"
        local rc_tar=$?
        local elapsed_tar=$((SECONDS - t_tar))
        if [ $rc_tar -eq 0 ]; then
            local tar_sz=$(du -h "$tar_path" | cut -f1)
            ts "  [tar] Created $tar_path ($tar_sz) in ${elapsed_tar}s"
        else
            ts "  [tar] FAILED to create tar (rc=$rc_tar). Falling back to Lustre venv."
            export PYTHON="$src/bin/python"
            ts "  Python now: $PYTHON"
            ts "=== VENV COPY COMPLETE (fallback) ==="
            return
        fi
    else
        local tar_sz=$(du -h "$tar_path" | cut -f1)
        ts "  [tar] Found existing archive ($tar_sz)"
    fi

    # Step 2: Copy tar to local SSD (single file = fast sequential IO)
    local tar_bytes=$(stat --format="%s" "$tar_path" 2>/dev/null || echo 0)
    ts "  [copy] Copying tar to local SSD (timeout=${timeout_sec}s)..."
    local t_copy=$SECONDS
    timeout $timeout_sec cp "$tar_path" "$SLURM_TMPDIR/venv.tar"
    local rc_copy=$?
    local elapsed_copy=$((SECONDS - t_copy))

    if [ $rc_copy -eq 0 ]; then
        local speed="N/A"
        if [ $elapsed_copy -gt 0 ] && [ $tar_bytes -gt 0 ]; then
            speed=$(echo "$tar_bytes $elapsed_copy" | awk '{printf "%.1f MB/s", $1/1048576/$2}')
        fi
        ts "  [copy] DONE in ${elapsed_copy}s ($speed)"
    elif [ $rc_copy -eq 124 ]; then
        ts "  [copy] TIMEOUT after ${timeout_sec}s. Falling back to Lustre venv."
        rm -f "$SLURM_TMPDIR/venv.tar" 2>/dev/null
        export PYTHON="$src/bin/python"
        ts "  Python now: $PYTHON"
        ts "=== VENV COPY COMPLETE (fallback) ==="
        return
    else
        ts "  [copy] FAILED (rc=$rc_copy). Falling back to Lustre venv."
        export PYTHON="$src/bin/python"
        ts "  Python now: $PYTHON"
        ts "=== VENV COPY COMPLETE (fallback) ==="
        return
    fi

    # Step 3: Extract on local SSD (local IO, should be very fast)
    ts "  [extract] Extracting on local SSD..."
    local t_extract=$SECONDS
    mkdir -p "$SLURM_TMPDIR"
    tar xf "$SLURM_TMPDIR/venv.tar" -C "$SLURM_TMPDIR/"
    local rc_extract=$?
    local elapsed_extract=$((SECONDS - t_extract))

    # Rename extracted dir to expected name
    local extracted_name=$(basename "$src")
    if [ "$extracted_name" != "venv" ] && [ -d "$SLURM_TMPDIR/$extracted_name" ]; then
        mv "$SLURM_TMPDIR/$extracted_name" "$dst"
    fi

    rm -f "$SLURM_TMPDIR/venv.tar"

    if [ $rc_extract -eq 0 ]; then
        ts "  [extract] DONE in ${elapsed_extract}s"
        export PYTHON="$dst/bin/python"
    else
        ts "  [extract] FAILED (rc=$rc_extract). Falling back to Lustre venv."
        export PYTHON="$src/bin/python"
    fi

    ts "  Python now: $PYTHON"
    ts "  Total venv setup: $((SECONDS - t_copy))s (copy=${elapsed_copy}s + extract=${elapsed_extract}s)"
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

# Copy meteo caches to LOCAL_CACHE, filtered by data_start date
# Usage: copy_meteo_caches <scratch_cache_dir> <local_cache_dir> <timeout_per_file> [data_start]
# If data_start is given (e.g. "2018-05-01"), only copy files matching that date.
# Otherwise copies all files (backward compatible).
copy_meteo_caches() {
    local scratch="$1"
    local local_dir="$2"
    local timeout_sec="${3:-3600}"
    local data_start="${4:-}"  # optional filter

    ts "=== COPYING CACHES TO LOCAL SSD ==="
    ts "  Source : $scratch"
    ts "  Dest   : $local_dir"
    if [ -n "$data_start" ]; then
        ts "  Filter : data_start=$data_start"
    fi
    echo "  SLURM_TMPDIR disk:"
    df -h $SLURM_TMPDIR
    mkdir -p "$local_dir"

    # Meteo cache (the big one) — pick the best match
    local copied=0
    local best_pf=""

    if [ -n "$data_start" ]; then
        # Find the largest pf.dat matching data_start (largest T = most complete)
        for f in $scratch/meteo_p16_*_${data_start}_*_pf.dat; do
            [ -f "$f" ] || continue
            if [ -z "$best_pf" ] || [ "$(stat --format=%s "$f")" -gt "$(stat --format=%s "$best_pf")" ]; then
                best_pf="$f"
            fi
        done
    fi

    if [ -n "$best_pf" ]; then
        ts "  Selected pf.dat: $(basename $best_pf) ($(du -h "$best_pf" | cut -f1))"
        copy_with_timeout "$best_pf" "$local_dir" "$timeout_sec" || {
            ts "WARNING: copy failed/timed out. Symlinking to Lustre."
            ln -sf "$best_pf" "$local_dir/$(basename $best_pf)"
        }
        copied=1
    else
        # No filter or no match — copy all (backward compatible)
        for f in $scratch/meteo_p16_*_pf.dat; do
            [ -f "$f" ] || continue
            copy_with_timeout "$f" "$local_dir" "$timeout_sec" || {
                ts "WARNING: copy failed/timed out. Symlinking to Lustre."
                ln -sf "$f" "$local_dir/$(basename $f)"
            }
            copied=$((copied + 1))
        done
    fi

    if [ $copied -eq 0 ]; then
        ts "WARNING: No meteo_pf.dat cache found in $scratch"
        ts "  Training will build cache from scratch (slow but OK)"
    fi

    # Small files — filter by data_start if given
    for pattern in "*_stats.npy" "fire_*.npy" "fire_*.dat" "norm_stats*"; do
        for f in $scratch/$pattern; do
            [ -f "$f" ] || continue
            # Skip files that don't match data_start (if filter is set)
            if [ -n "$data_start" ]; then
                case "$(basename $f)" in
                    *${data_start}*|*_stats.npy|norm_stats*) ;;  # match or universal
                    *) ts "  SKIP (wrong date range): $(basename $f)"; continue ;;
                esac
            fi
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
