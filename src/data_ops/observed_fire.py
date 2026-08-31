#!/usr/bin/env python3
"""
Observed-fire raster builders (single source of truth).

This module owns the logic for turning raw fire records into a daily
(T, H, W) uint8 stack and, for verification, into a single collapsed
(H, W) "observed fire over a window" raster. Three sources are supported:

    - "nbac_nfdb" : NBAC burned-area polygons UNION NFDB agency points
                    (the training target, per LABEL_DECISION_2026_04_21.md).
    - "cwfis"     : legacy CWFIS hotspot points.
    - "ciffc"     : current-season CIFFC size-circle points (Route B area
                    expansion), the only source that covers 2025+ (NBAC/NFDB
                    are annual products that stop at 2024).

Historically these builders lived inside scripts/build_fire_labels.py, which
made them unimportable from src/. They are now here so both the label builder
and the inference entry point (src/forecasting/run_inference.py) share one
implementation. build_date_list / build_cwfis / build_nbac_nfdb / dilate_stack
are byte-identical moves from that script.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation

from src.data_ops.processing.rasterize_hotspots import (
    load_hotspot_data, rasterize_hotspots_batch, load_nfdb_as_hotspot_df,
)
from src.data_ops.processing.rasterize_burn_polygons import (
    load_nbac, rasterize_nbac_batch,
)
from src.data_ops.processing.rasterize_fires import (
    load_ciffc_data, rasterize_fires_batch,
)


def build_date_list(start: str, end: str):
    sd = date.fromisoformat(start)
    ed = date.fromisoformat(end)
    dates = []
    cur = sd
    while cur <= ed:
        dates.append(cur)
        cur += timedelta(days=1)
    return dates


def build_cwfis(hotspot_csv: str, dates, profile):
    df = load_hotspot_data(hotspot_csv)
    print(f"  CWFIS: {len(df):,} hotspot records loaded")
    return rasterize_hotspots_batch(df, dates, profile)


def build_nbac_nfdb(nbac_path: str, nfdb_path: str,
                    nbac_date_source: str,
                    nfdb_min_size_ha: float,
                    exclude_prescribed: bool,
                    dates, profile):
    H, W = int(profile["height"]), int(profile["width"])
    stack = np.zeros((len(dates), H, W), dtype=np.uint8)

    # --- NBAC polygons ---
    nbac = load_nbac(nbac_path)
    print(f"  NBAC: {len(nbac):,} polygons loaded")
    if exclude_prescribed and "PRESCRIBED" in nbac.columns:
        _before = len(nbac)
        # NBAC PRESCRIBED: 'true' = prescribed burn, NaN = wildfire (audit 2026-04-21).
        nbac = nbac[nbac["PRESCRIBED"].isna()].copy()
        print(f"  NBAC: dropped {_before - len(nbac)} prescribed polygons "
              f"({len(nbac):,} remain)")
    nbac_stack = rasterize_nbac_batch(nbac, dates, profile,
                                      date_source=nbac_date_source)
    np.maximum(stack, nbac_stack, out=stack)
    nbac_pos = int(stack.sum())
    print(f"  after NBAC: {nbac_pos:,} positive pixels")

    # --- NFDB points ---
    keep_causes = {"H", "N", "U"} if exclude_prescribed else None
    nfdb = load_nfdb_as_hotspot_df(
        nfdb_path,
        min_size_ha=nfdb_min_size_ha,
        causes=keep_causes,
    )
    print(f"  NFDB: {len(nfdb):,} fires loaded "
          f"(size >= {nfdb_min_size_ha} ha, excl prescribed={exclude_prescribed})")
    nfdb_stack = rasterize_hotspots_batch(nfdb, dates, profile)
    before = int(stack.sum())
    np.maximum(stack, nfdb_stack, out=stack)
    added = int(stack.sum()) - before
    print(f"  after NFDB: +{added:,} pixels")

    return stack, {"nbac_positive": nbac_pos, "nfdb_added": added}


def dilate_stack(stack: np.ndarray, r: int):
    if r <= 0:
        return stack
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disk = (xx ** 2 + yy ** 2 <= r ** 2)
    T = stack.shape[0]
    out = np.zeros_like(stack)
    print(f"  Dilating {T} frames with r={r} px disk...")
    t0 = time.time()
    for t in range(T):
        if stack[t].any():
            out[t] = binary_dilation(stack[t], structure=disk).astype(np.uint8)
        if (t + 1) % 1000 == 0:
            print(f"    {t+1}/{T}  ({time.time()-t0:.0f}s)")
    return out


# ---------------------------------------------------------------------------
# New: CIFFC current-season builder + high-level window collapse
# ---------------------------------------------------------------------------

def build_ciffc(ciffc_path: str, dates, profile):
    """Daily (T,H,W) uint8 stack from a CIFFC size/point report (Route B).

    Thin wrapper over rasterize_fires_batch so CIFFC lines up with the other
    build_* functions' (dates, profile) -> (T,H,W) contract.
    """
    df = load_ciffc_data(ciffc_path)
    print(f"  CIFFC: {len(df):,} fire records loaded")
    return rasterize_fires_batch(df, dates, profile)


def _dilate_2d(frame: np.ndarray, r: int) -> np.ndarray:
    """Dilate a single (H,W) frame with an r-pixel disk.

    Collapsing days first and dilating once is identical to dilating each day
    then taking the union, because binary dilation distributes over union:
    dilate(A | B) == dilate(A) | dilate(B).
    """
    if r <= 0:
        return frame.astype(np.uint8)
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disk = (xx ** 2 + yy ** 2 <= r ** 2)
    if not frame.any():
        return frame.astype(np.uint8)
    return binary_dilation(frame, structure=disk).astype(np.uint8)


def build_observed_window(
    target_start: str,
    target_end: str,
    source: str,
    profile: dict,
    *,
    dilate_radius: int = 14,
    nbac_path: Optional[str] = None,
    nfdb_path: Optional[str] = None,
    hotspot_csv: Optional[str] = None,
    ciffc_path: Optional[str] = None,
    nbac_date_source: str = "AG",
    nfdb_min_size_ha: float = 1.0,
    exclude_prescribed: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """Observed fire over [target_start, target_end], collapsed to (H,W).

    Builds the daily stack for the window, takes the pixel-wise union over
    days, then applies a single r-pixel dilation (matches the training label
    geometry; see the module docstring). Returns (obs_2d uint8, provenance).

    source selects the record set; the matching *_path argument must be given:
        "nbac_nfdb" -> nbac_path + nfdb_path
        "cwfis"     -> hotspot_csv
        "ciffc"     -> ciffc_path
    """
    dates = build_date_list(target_start, target_end)
    print(f"[observed] source={source} window={target_start}..{target_end} "
          f"({len(dates)} days) dilate_r={dilate_radius}")

    provenance: Dict = {}
    if source == "nbac_nfdb":
        if not nbac_path or not nfdb_path:
            raise ValueError("source='nbac_nfdb' requires nbac_path and nfdb_path")
        stack, provenance = build_nbac_nfdb(
            nbac_path, nfdb_path,
            nbac_date_source=nbac_date_source,
            nfdb_min_size_ha=nfdb_min_size_ha,
            exclude_prescribed=exclude_prescribed,
            dates=dates, profile=profile)
    elif source == "cwfis":
        if not hotspot_csv:
            raise ValueError("source='cwfis' requires hotspot_csv")
        stack = build_cwfis(hotspot_csv, dates, profile)
    elif source == "ciffc":
        if not ciffc_path:
            raise ValueError("source='ciffc' requires ciffc_path")
        stack = build_ciffc(ciffc_path, dates, profile)
    else:
        raise ValueError(
            f"unknown source '{source}' (expected nbac_nfdb|cwfis|ciffc)")

    obs_2d = (stack.max(axis=0) > 0).astype(np.uint8)
    raw_positive = int(obs_2d.sum())
    obs_2d = _dilate_2d(obs_2d, dilate_radius)
    dilated_positive = int(obs_2d.sum())
    print(f"[observed] raw={raw_positive:,} -> dilated={dilated_positive:,} px")

    provenance.update({
        "source": source,
        "window": [target_start, target_end],
        "dilate_radius": dilate_radius,
        "raw_positive": raw_positive,
        "dilated_positive": dilated_positive,
    })
    return obs_2d, provenance
