"""
V3 SOTA -> per-lead-day fire probability GeoTIFFs.
==================================================

Drop-in counterpart of V2's `_run_forecast_only` (in
src/training/train_s2s_hotspot_cwfis_v2.py around line 1044) for the V3
SOTA checkpoint `v3_9ch_enc21_12y_2014`. Matches V2's output layout
EXACTLY so downstream plotting (e.g. scripts/plot_fig3_canada_map.py)
keeps globbing the same file pattern:

    {out_dir}/{YYYYMMDD_issue}/fire_prob_lead{LL}d_{YYYYMMDD_target}.tif

V3-specific quirks reproduced here (DO NOT skip any of these — see
training-loop assertions in S2SHotspotTransformer.forward):

  * 9 channels in EXACTLY this order:
        FWI, 2t, fire_clim, 2d, tcw, sm20, population, slope, burn_age
    The order is fixed by the checkpoint's `args["channels"]` string
    (we re-read it from the ckpt and assert).
  * `--decoder s2s_legacy` with `--s2s_max_issue_lag 3`. The decoder
    input is a (dec_days, 9) S2S patch-mean tensor built by
    _make_dec_s2s — NOT the encoder ERA5 patches.
  * `--decoder_ctx` augmentation: at inference we MUST concatenate
        [s2s_legacy_9dim | static_ctx_per_patch | lead_time_enc_4dim]
    onto the decoder. `static_ctx_per_patch` is the per-patch mean of
    the static channels {fire_clim, population, slope, burn_age}, in
    that order (see DECODER_CTX_CHANNELS in train_v3.py:160). The
    lead-time encoding is sin/cos of (lead_day/60) and sin/cos of
    (doy/365), reproduced by `_build_lead_time_encoding`.
  * `--fire_clim_dir data/fire_clim_annual_nbac` — yearly TIFs of form
    `fire_clim_upto_{YEAR}.tif`. We pick the largest year <= current
    target year (strictly-prior-or-equal, leak-free per
    train_v3.py:2041 fix).
  * Encoder: in_days=21, lead_start=14, lead_end=45 (s2s_legacy clamps).
  * Population, slope, burn_age, fire_clim are normalized by the
    SAME meteo_means/meteo_stds saved in the checkpoint. We re-load
    these maps from disk and apply the saved per-channel z-score.
  * `burn_age` uses PREVIOUS year (year-1 offset) to prevent leakage:
    `years_since_burn_{cur_date.year - 1}.tif`.

Run example:

    python -m src.forecasting.forecast_v3_to_tif \
        --config configs/paths_narval.yaml \
        --ckpt $SCRATCH/wildfire-refactored/checkpoints/v3_9ch_enc21_12y_2014/best_model.pt \
        --s2s_cache $SCRATCH/wildfire-refactored/data/s2s_processed/s2s_decoder_cache.dat \
        --issue_dates 2023-05-15 2023-08-15 2022-08-15 \
        --out_dir outputs/v3_9ch_enc21_12y_2014_fire_prob
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import rasterio
import torch

# --- Project imports ------------------------------------------------------
# Add repo root for "python path/to/forecast_v3_to_tif.py" usage.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.config import load_config, get_path  # noqa: E402
from src.utils.date_utils import extract_date_from_filename  # noqa: E402
from src.utils.patch_utils import depatchify  # noqa: E402
from src.models.s2s_hotspot import S2SHotspotTransformer  # noqa: E402
from src.training.train_v3 import (  # noqa: E402
    DECODER_CTX_CHANNELS,
    V3_CHANNEL_DEFS,
    _build_decoder_ctx_static,
    _build_lead_time_encoding,
    _augment_decoder,
    _load_static_channel,
    _interpolate_ndvi,
    _build_ndvi_index,
)
from src.training.train_s2s_hotspot_cwfis_v2 import (  # noqa: E402
    _build_file_dict,
    _build_flat_file_dict,
    _read_tif_safe,
    _patchify_frame,
    _make_dec_s2s,
    _expand_s2s_date_mapping,
    S2S_DEC_DIM,
)


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
def _parse_date(s: str) -> date:
    """YYYY-MM-DD -> date."""
    y, m, d = s.split("-")
    return date(int(y), int(m), int(d))


def _resolve_paths(cfg: dict) -> dict:
    """Resolve every path used by the 9ch SOTA forecaster."""
    paths_cfg = cfg.get("paths", {})
    return {
        "fwi_dir":             get_path(cfg, "fwi_dir"),
        "obs_root":            get_path(cfg, "observation_dir")
                                   if "observation_dir" in paths_cfg
                                   else get_path(cfg, "ecmwf_dir"),
        "fire_clim_dir":       paths_cfg.get("fire_clim_dir"),
        "fire_clim_static":    paths_cfg.get("fire_climatology_tif"),
        "population_tif":      paths_cfg.get("population_tif",
                                             "data/population_density.tif"),
        "terrain_dir":         paths_cfg.get("terrain_dir", "data/terrain"),
        "burn_scars_dir":      paths_cfg.get("burn_scars_dir",
                                             "data/burn_scars"),
        "ndvi_dir":            paths_cfg.get("ndvi_dir", "data/ndvi_data"),
    }


def _build_fire_clim_index(fire_clim_dir: Optional[str],
                           fire_clim_static: Optional[str],
                           H: int, W: int) -> Dict[int, np.ndarray]:
    """Return {year: (H,W) float32 fire_clim map}. Falls back to a single
    static TIF treated as 'year 0' if the annual dir is missing.
    """
    out: Dict[int, np.ndarray] = {}
    if fire_clim_dir and os.path.isdir(fire_clim_dir):
        for p in sorted(glob.glob(os.path.join(fire_clim_dir,
                                               "fire_clim_upto_*.tif"))):
            bn = os.path.basename(p)
            try:
                yr = int(bn.replace("fire_clim_upto_", "").replace(".tif", ""))
            except ValueError:
                continue
            out[yr] = _load_static_channel(p, H, W, f"fire_clim_{yr}")
    if not out and fire_clim_static and os.path.exists(fire_clim_static):
        out[0] = _load_static_channel(fire_clim_static, H, W, "fire_clim")
    return out


def _pick_fire_clim(fire_clim_arrays: Dict[int, np.ndarray],
                    yr: int, H: int, W: int) -> np.ndarray:
    """Leak-free fire_clim selector — strictly prior-or-equal year
    (see train_v3.py:2041 bug-fix comment)."""
    if not fire_clim_arrays:
        return np.zeros((H, W), dtype=np.float32)
    if yr in fire_clim_arrays:
        return fire_clim_arrays[yr]
    prior = [y for y in fire_clim_arrays.keys() if y <= yr]
    if not prior:
        return np.zeros((H, W), dtype=np.float32)
    return fire_clim_arrays[max(prior)]


def _build_burn_age_index(burn_scars_dir: str,
                          H: int, W: int) -> Dict[int, np.ndarray]:
    """Map {year: log1p(years_since_burn) array}. Empty if dir missing."""
    out: Dict[int, np.ndarray] = {}
    if not burn_scars_dir or not os.path.isdir(burn_scars_dir):
        return out
    for p in sorted(glob.glob(os.path.join(burn_scars_dir,
                                           "years_since_burn_*.tif"))):
        bn = os.path.basename(p)
        try:
            yr = int(bn.replace("years_since_burn_", "").replace(".tif", ""))
        except ValueError:
            continue
        raw = _load_static_channel(p, H, W, f"burn_{yr}")
        raw = np.maximum(raw, 0)
        out[yr] = np.log1p(raw).astype(np.float32)
    return out


def _build_burn_count_index(burn_scars_dir: str,
                            H: int, W: int) -> Dict[int, np.ndarray]:
    """Map {year: log1p(burn_count) array}. Same encoding as burn_age
    (max(0) then log1p). Empty if dir missing."""
    out: Dict[int, np.ndarray] = {}
    if not burn_scars_dir or not os.path.isdir(burn_scars_dir):
        return out
    for p in sorted(glob.glob(os.path.join(burn_scars_dir,
                                           "burn_count_*.tif"))):
        bn = os.path.basename(p)
        try:
            yr = int(bn.replace("burn_count_", "").replace(".tif", ""))
        except ValueError:
            continue
        raw = _load_static_channel(p, H, W, f"burn_count_{yr}")
        raw = np.maximum(raw, 0)
        out[yr] = np.log1p(raw).astype(np.float32)
    return out


def _pick_burn_age(burn_index: Dict[int, np.ndarray],
                   cur_year: int, H: int, W: int) -> np.ndarray:
    """Year-1 offset to prevent leakage (train_v3.py:2089)."""
    if not burn_index:
        return np.zeros((H, W), dtype=np.float32)
    prev_year = cur_year - 1
    if prev_year in burn_index:
        return burn_index[prev_year]
    valid = [y for y in burn_index.keys() if y <= prev_year]
    if valid:
        return burn_index[max(valid)]
    return np.zeros((H, W), dtype=np.float32)


# ----------------------------------------------------------------
# Main forecast routine
# ----------------------------------------------------------------
def forecast(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 1. Load checkpoint ------------------------------------------------
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    print(f"[forecast_v3] Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    saved_args = ckpt["args"]
    P            = saved_args["patch_size"]
    in_days      = saved_args["in_days"]
    lead_start   = saved_args["lead_start"]
    lead_end     = saved_args["lead_end"]
    if saved_args.get("decoder") in ("s2s", "s2s_legacy") and lead_end > 45:
        lead_end = 45  # match train_v3.py:1260
    decoder_days = lead_end - lead_start + 1
    decoder_mode = saved_args.get("decoder", "s2s_legacy")
    s2s_max_lag  = saved_args.get("s2s_max_issue_lag", 3)

    meteo_means = ckpt["meteo_means"].astype(np.float32)
    meteo_stds  = ckpt["meteo_stds"].astype(np.float32)
    s2s_means   = ckpt.get("s2s_means", None)
    s2s_stds    = ckpt.get("s2s_stds", None)

    patch_dim_enc = ckpt["patch_dim_enc"]
    patch_dim_dec = ckpt["patch_dim_dec"]
    patch_dim_out = ckpt["patch_dim_out"]

    # Channel order is load-bearing — recover it from saved args.
    if "channels" in saved_args and saved_args["channels"]:
        CHANNEL_NAMES = [c.strip() for c in saved_args["channels"].split(",")]
    else:
        raise RuntimeError(
            "Checkpoint args missing 'channels' — cannot reconstruct channel order."
        )
    N_CHANNELS = len(CHANNEL_NAMES)
    enc_dim = P * P * N_CHANNELS
    if enc_dim != patch_dim_enc:
        raise RuntimeError(
            f"Channel mismatch: ckpt patch_dim_enc={patch_dim_enc} but "
            f"P²*N_CH={enc_dim} from saved channels={CHANNEL_NAMES}."
        )

    print(f"  N_CHANNELS={N_CHANNELS}  channels={CHANNEL_NAMES}")
    print(f"  in_days={in_days}  lead={lead_start}-{lead_end}  "
          f"decoder_days={decoder_days}  decoder={decoder_mode}")
    print(f"  patch_dim_enc={patch_dim_enc}  patch_dim_dec={patch_dim_dec}  "
          f"patch_dim_out={patch_dim_out}")

    # Decoder context: forecast_dim + ctx_dim must equal patch_dim_dec.
    # forecast_dim = S2S_DEC_DIM (9) for s2s_legacy.
    n_ctx_channels = sum(1 for n in CHANNEL_NAMES if n in DECODER_CTX_CHANNELS)
    use_decoder_ctx = bool(saved_args.get("decoder_ctx", False))
    ctx_extra_dim = (n_ctx_channels + 4) if use_decoder_ctx else 0
    if decoder_mode == "s2s_legacy":
        dec_dim_base = S2S_DEC_DIM
    else:
        dec_dim_base = patch_dim_dec - ctx_extra_dim
    expected_dec = dec_dim_base + ctx_extra_dim
    if expected_dec != patch_dim_dec:
        raise RuntimeError(
            f"Decoder dim mismatch: ckpt patch_dim_dec={patch_dim_dec} != "
            f"forecast({dec_dim_base}) + ctx({ctx_extra_dim}). "
            f"decoder_ctx={use_decoder_ctx}, n_ctx_channels={n_ctx_channels}."
        )

    # ---- 2. Build model ----------------------------------------------------
    _mt = saved_args.get("model_type", "transformer")
    if _mt in ("mlp", "convlstm"):
        from src.models.baselines import build_baseline as _bb
        _P = int(round(patch_dim_out ** 0.5)); _nch = patch_dim_enc // max(_P*_P, 1)
        print(f"  [BASELINE forecast] model_type={_mt} n_channels={_nch} P={_P}")
        model = _bb(_mt, patch_dim_enc=patch_dim_enc, patch_dim_dec=patch_dim_dec,
                    patch_dim_out=patch_dim_out, encoder_days=in_days,
                    decoder_days=decoder_days, n_channels=_nch, patch_size=_P,
                    d_model=saved_args["d_model"], dropout=saved_args.get("dropout", 0.2)).to(device)
    else:
        model = S2SHotspotTransformer(
            patch_dim_enc=patch_dim_enc,
            patch_dim_dec=patch_dim_dec,
            patch_dim_out=patch_dim_out,
            d_model=saved_args["d_model"],
            nhead=saved_args["nhead"],
            num_encoder_layers=saved_args["enc_layers"],
            num_decoder_layers=saved_args["dec_layers"],
            dim_feedforward=saved_args["d_model"] * 4,
            dropout=saved_args.get("dropout", 0.1),
            encoder_days=in_days,
            decoder_days=decoder_days,
            n_patches=0,            # SOTA was trained with use_patch_embed=False
            mlp_dec_embed=saved_args.get("mlp_dec_embed", False),
            dec_ctx_dim=ctx_extra_dim if use_decoder_ctx else 0,
            enc_conv_stem=saved_args.get("enc_conv_stem", False),
            patch_size=saved_args.get("patch_size", 16),
            conv_output_head=saved_args.get("conv_output_head", False),
            conv_head_encmean=saved_args.get("conv_head_encmean", False),
        ).to(device)

    # Support both naming conventions (V2 = model_state_dict, V3 = model_state)
    state = (ckpt.get("model_state")
             or ckpt.get("model_state_dict")
             or ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(unexpected) > len(state) / 2:
        raise RuntimeError(
            f"State-dict load failed: {len(unexpected)} unexpected keys "
            f"out of {len(state)}. First unexpected: {unexpected[:3]}")
    if missing:
        print(f"  [warn] {len(missing)} missing keys: {missing[:3]}")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys: {unexpected[:3]}")
    model.eval()
    print(f"  Loaded. device={device} params={sum(p.numel() for p in model.parameters()):,}")

    # ---- 3. Resolve data paths --------------------------------------------
    cfg = load_config(args.config)
    P_PATHS = _resolve_paths(cfg)

    # Build file indices for ONLY the channels we need.
    print(f"\n[forecast_v3] Indexing input TIFs for {N_CHANNELS} channels...")
    fwi_dict: Dict[date, str] = {}
    for p in sorted(glob.glob(os.path.join(P_PATHS["fwi_dir"], "*.tif"))):
        d = extract_date_from_filename(os.path.basename(p))
        if d:
            fwi_dict[d] = p
    if not fwi_dict:
        raise RuntimeError(f"No FWI tifs found in {P_PATHS['fwi_dir']}")

    daily_dicts: Dict[str, Dict[date, str]] = {"FWI": fwi_dict}
    # ERA5 observation channels (2t, 2d, tcw, sm20, st20)
    for ch in ("2t", "2d", "tcw", "sm20", "st20"):
        if ch in CHANNEL_NAMES:
            daily_dicts[ch] = _build_file_dict(P_PATHS["obs_root"], ch)
    # All other daily-dict-based optional channels not used by the
    # 9ch SOTA but supported here for forward-compat.
    flat_dict_specs = {
        "lightning": ("lightning_dir",  "lightning"),
        "deep_soil": ("deep_soil_dir",  "swvl2"),
        "precip_def":("precip_dir",     "tp"),
        "u10":       ("wind_u_dir",     "u10"),
        "v10":       ("wind_v_dir",     "v10"),
        "CAPE":      ("cape_dir",       "cape"),
        "FFMC":      ("ffmc_dir",       "ffmc"),
        "DMC":       ("dmc_dir",        "dmc"),
        "DC":        ("dc_dir",         "dc"),
        "BUI":       ("bui_dir",        "bui"),
        "ISI":       ("isi_dir",        "isi"),
    }
    for ch, (cfg_key, prefix) in flat_dict_specs.items():
        if ch in CHANNEL_NAMES:
            d_dir = cfg.get("paths", {}).get(cfg_key)
            if d_dir is None:
                raise RuntimeError(f"Path '{cfg_key}' missing in config for channel {ch}")
            daily_dicts[ch] = _build_flat_file_dict(d_dir, prefix)

    # Grid dims (from first FWI tif).
    first_fwi = next(iter(fwi_dict.values()))
    with rasterio.open(first_fwi) as src:
        ref_profile = src.profile.copy()
        H, W = src.height, src.width
    if (H, W) != (2281, 2709):
        print(f"  [warn] FWI grid {(H, W)} != expected (2281, 2709)")
    Hc, Wc = H - H % P, W - W % P
    nph, npw = Hc // P, Wc // P
    n_patches = nph * npw
    print(f"  grid: H={H} W={W}  patches={nph}x{npw}={n_patches}")

    # Output TIF profile — single band float32 deflate (caller asked for deflate).
    out_profile = ref_profile.copy()
    out_profile.update(dtype=rasterio.float32, count=1, compress="deflate",
                       predictor=2, zlevel=6)

    # ---- 4. Static channel arrays -----------------------------------------
    static_arrays: Dict[str, np.ndarray] = {}
    fire_clim_arrays = _build_fire_clim_index(
        P_PATHS["fire_clim_dir"], P_PATHS["fire_clim_static"], H, W)
    if "fire_clim" in CHANNEL_NAMES and not fire_clim_arrays:
        print("  [warn] fire_clim requested but no fire_clim_upto_*.tif found.")

    if "population" in CHANNEL_NAMES:
        static_arrays["population"] = _load_static_channel(
            P_PATHS["population_tif"], H, W, "population")
    if "slope" in CHANNEL_NAMES:
        static_arrays["slope"] = _load_static_channel(
            os.path.join(P_PATHS["terrain_dir"], "slope.tif"), H, W, "slope")
    if "elevation" in CHANNEL_NAMES:
        static_arrays["elevation"] = _load_static_channel(
            os.path.join(P_PATHS["terrain_dir"], "dem_cdem.tif"),
            H, W, "elevation")
    if "aspect" in CHANNEL_NAMES:
        static_arrays["aspect"] = _load_static_channel(
            os.path.join(P_PATHS["terrain_dir"], "aspect.tif"),
            H, W, "aspect")
    if "lightning_climatology" in CHANNEL_NAMES:
        # Static per-pixel mean annual lightning (log1p); matches
        # train_v3.py:1652. Without this the channel loaded as zeros.
        static_arrays["lightning_climatology"] = _load_static_channel(
            "data/lightning_climatology.tif", H, W, "lightning_climatology")

    burn_index = (_build_burn_age_index(P_PATHS["burn_scars_dir"], H, W)
                  if "burn_age" in CHANNEL_NAMES else {})
    burn_count_index = (_build_burn_count_index(P_PATHS["burn_scars_dir"], H, W)
                        if "burn_count" in CHANNEL_NAMES else {})

    ndvi_index = (_build_ndvi_index(P_PATHS["ndvi_dir"])
                  if "NDVI" in CHANNEL_NAMES else [])
    ndvi_cache: Dict = {}

    # ---- 5. S2S decoder cache (required for s2s_legacy) -------------------
    s2s_cache = None
    s2s_dates = None
    if decoder_mode == "s2s_legacy":
        if not args.s2s_cache:
            raise RuntimeError(
                "decoder=s2s_legacy requires --s2s_cache <path to "
                "s2s_decoder_cache.dat (with .dates.npy sibling)>")
        if not os.path.exists(args.s2s_cache):
            raise FileNotFoundError(f"S2S cache not found: {args.s2s_cache}")
        dates_file = args.s2s_cache + ".dates.npy"
        if not os.path.exists(dates_file):
            raise FileNotFoundError(f"S2S dates file not found: {dates_file}")
        s2s_dates = np.load(dates_file, allow_pickle=True)
        s2s_n_dates = len(s2s_dates)
        s2s_cache = np.memmap(args.s2s_cache, dtype="float16", mode="r",
                              shape=(s2s_n_dates, n_patches, 32,
                                     S2S_DEC_DIM - 3))
        print(f"  S2S cache loaded: {s2s_cache.shape} "
              f"({os.path.getsize(args.s2s_cache)/1e9:.2f} GB) "
              f"dates {s2s_dates[0]}..{s2s_dates[-1]}")

    # ---- 6. Issue-date loop ------------------------------------------------
    issue_dates = [_parse_date(s) for s in args.issue_dates]
    os.makedirs(args.out_dir, exist_ok=True)

    for issue_date in issue_dates:
        t0 = time.time()
        print(f"\n=== Forecasting issue_date={issue_date} ===")

        # The encoder needs days [issue - in_days, issue).
        enc_start = issue_date - timedelta(days=in_days)
        enc_dates = [enc_start + timedelta(days=i) for i in range(in_days)]

        # ------- Build encoder meteo stack (T=in_days, H, W, C) -------
        meteo_y = np.zeros((in_days, H, W, N_CHANNELS), dtype=np.float32)
        _fallback_fwi = None
        _fallback_t2m = None

        skip = False
        for t_idx, cur_date in enumerate(enc_dates):
            for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
                ch_def = V3_CHANNEL_DEFS.get(ch_name)
                if ch_def is None:
                    raise ValueError(f"Unknown channel in ckpt: {ch_name}")

                # FWI: required, fallback to previous day if missing.
                if ch_name == "FWI":
                    if cur_date in fwi_dict:
                        arr = _read_tif_safe(fwi_dict[cur_date], _fallback_fwi)
                        _fallback_fwi = arr
                    elif _fallback_fwi is not None:
                        arr = _fallback_fwi
                    else:
                        print(f"  [skip] no FWI for {cur_date} and no fallback")
                        skip = True
                        break
                    meteo_y[t_idx, ..., ch_idx] = np.nan_to_num(
                        arr, nan=float(meteo_means[ch_idx]))

                elif ch_name == "2t":
                    d = daily_dicts.get("2t", {})
                    if cur_date in d:
                        arr = _read_tif_safe(d[cur_date], _fallback_t2m)
                        _fallback_t2m = arr
                    elif _fallback_t2m is not None:
                        arr = _fallback_t2m
                    else:
                        print(f"  [skip] no 2t for {cur_date}")
                        skip = True
                        break
                    meteo_y[t_idx, ..., ch_idx] = np.nan_to_num(
                        arr, nan=float(meteo_means[ch_idx]))

                elif ch_def["type"] == "static":
                    meteo_y[t_idx, ..., ch_idx] = static_arrays.get(
                        ch_name, np.zeros((H, W), dtype=np.float32))

                elif ch_name == "fire_clim":
                    meteo_y[t_idx, ..., ch_idx] = _pick_fire_clim(
                        fire_clim_arrays, cur_date.year, H, W)

                elif ch_name == "burn_age":
                    meteo_y[t_idx, ..., ch_idx] = _pick_burn_age(
                        burn_index, cur_date.year, H, W)
                elif ch_name == "burn_count":
                    meteo_y[t_idx, ..., ch_idx] = _pick_burn_age(
                        burn_count_index, cur_date.year, H, W)

                elif ch_name == "NDVI":
                    meteo_y[t_idx, ..., ch_idx] = _interpolate_ndvi(
                        cur_date, ndvi_index, ndvi_cache, H, W)

                elif ch_name in daily_dicts:
                    d = daily_dicts[ch_name]
                    if cur_date in d:
                        arr = _read_tif_safe(d[cur_date], None)
                        if arr is not None:
                            meteo_y[t_idx, ..., ch_idx] = np.nan_to_num(
                                arr, nan=float(meteo_means[ch_idx]))
                else:
                    # Unsupported channel type for forecast inference.
                    raise NotImplementedError(
                        f"Channel '{ch_name}' (type={ch_def['type']}) is not "
                        f"supported by this forecaster yet."
                    )

            if skip:
                break

        if skip:
            print(f"  [skip] issue_date {issue_date}: missing inputs.")
            continue

        # Normalize using ckpt stats (matches train_v3.py:2112-2114).
        meteo_y -= meteo_means
        meteo_y /= meteo_stds
        np.clip(meteo_y, -10.0, 10.0, out=meteo_y)

        # Patchify entire (T, H, W, C) stack -> (n_patches, T, enc_dim).
        # train_v3 uses _patchify_frame per timestep then transposes; the
        # equivalent vectorized form is to call _patchify_frame on each
        # frame and stack along T.
        enc_patches = np.empty((n_patches, in_days, enc_dim), dtype=np.float16)
        for t in range(in_days):
            enc_patches[:, t, :] = _patchify_frame(meteo_y[t], P).astype(np.float16)

        # ------- Build decoder_ctx static + lead-time tensors -------
        decoder_ctx_fn = None
        if use_decoder_ctx:
            dec_ctx_np, ctx_indices = _build_decoder_ctx_static(
                enc_patches, CHANNEL_NAMES, P * P, patch_mean=True)
            ctx_names = [CHANNEL_NAMES[i] for i in ctx_indices]
            print(f"  decoder_ctx: static channels {ctx_names} "
                  f"-> shape {None if dec_ctx_np is None else dec_ctx_np.shape}")

            # Lead-time encoding (seasonal sin/cos uses base DOY = issue_date's DOY).
            base_doy = issue_date.timetuple().tm_yday
            lead_time_enc = _build_lead_time_encoding(
                decoder_days, lead_start, base_doy=base_doy, device=device)

            def decoder_ctx_fn(xb_dec, cs, ce):  # noqa: E306
                ctx_batch = torch.from_numpy(
                    dec_ctx_np[cs:ce].astype(np.float32)).to(xb_dec.device)
                return _augment_decoder(xb_dec, ctx_batch, lead_time_enc)

        # ------- S2S decoder mapping for THIS issue_date -------
        if decoder_mode == "s2s_legacy":
            issue_str = str(issue_date)
            date_to_idx, date_to_exact, date_to_lag = _expand_s2s_date_mapping(
                s2s_dates, [issue_date], max_lag_days=s2s_max_lag)
            if issue_str not in date_to_idx:
                print(f"  [skip] issue_date {issue_date} not in S2S cache "
                      f"(max_lag={s2s_max_lag}d).")
                continue
            lag = date_to_lag.get(issue_str, 0)
            exact = date_to_exact.get(issue_str, False)
            print(f"  S2S issue mapping: idx={date_to_idx[issue_str]} "
                  f"lag={lag}d exact={exact}")

        # ------- Forward pass in chunks -------
        chunk = args.pred_batch_size
        prob_list = []
        with torch.no_grad():
            for cs in range(0, n_patches, chunk):
                ce = min(cs + chunk, n_patches)
                xb_enc = torch.from_numpy(
                    enc_patches[cs:ce].astype(np.float32)).to(device)

                if decoder_mode == "s2s_legacy":
                    dec_list = [
                        _make_dec_s2s(
                            s2s_cache, date_to_idx,
                            str(issue_date), cs + pi,
                            decoder_days, S2S_DEC_DIM, P,
                            s2s_means=s2s_means, s2s_stds=s2s_stds,
                            date_to_s2s_lag=date_to_lag,
                            s2s_max_lag=s2s_max_lag,
                        ).astype(np.float32)
                        for pi in range(ce - cs)
                    ]
                    xb_dec = torch.from_numpy(
                        np.stack(dec_list, axis=0)).to(device)
                else:
                    # Other decoder modes (oracle/zeros/random/climatology/s2s)
                    # are not exercised by the 9ch SOTA; raise clear error.
                    raise NotImplementedError(
                        f"decoder='{decoder_mode}' is not implemented by this "
                        f"forecaster. SOTA is s2s_legacy."
                    )

                if decoder_ctx_fn is not None:
                    xb_dec = decoder_ctx_fn(xb_dec, cs, ce)

                logits = model(xb_enc, xb_dec)
                prob_list.append(torch.sigmoid(logits).cpu().numpy())

        probs = np.concatenate(prob_list, axis=0)
        # probs shape: (n_patches, decoder_days, P*P)

        # ------- Save 32 GeoTIFFs (lead 14..45 inclusive) -------
        base_str = issue_date.strftime("%Y%m%d")
        day_out = os.path.join(args.out_dir, base_str)
        os.makedirs(day_out, exist_ok=True)

        for li, lead in enumerate(range(lead_start, lead_end + 1)):
            target_date = issue_date + timedelta(days=lead)
            target_str = target_date.strftime("%Y%m%d")
            out_path = os.path.join(
                day_out, f"fire_prob_lead{lead:02d}d_{target_str}.tif")

            prob_patches_lead = probs[:, li, :]                    # (n_patches, P²)
            prob_vol = depatchify(
                prob_patches_lead[:, np.newaxis, :],
                (nph, npw), P, (Hc, Wc), num_channels=1)            # (1, Hc, Wc)
            prob_map = prob_vol[0] if prob_vol.ndim == 3 else prob_vol
            if prob_map.shape != (H, W):
                full = np.zeros((H, W), dtype=np.float32)
                full[:prob_map.shape[0], :prob_map.shape[1]] = prob_map
                prob_map = full

            with rasterio.open(out_path, "w", **out_profile) as dst:
                dst.write(prob_map.astype(np.float32), 1)

        elapsed = time.time() - t0
        print(f"  -> wrote {decoder_days} TIFs to {day_out} "
              f"in {elapsed:.1f}s")

    print(f"\n[forecast_v3] DONE. Output dir: {args.out_dir}")


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="V3 SOTA -> per-lead-day fire-probability GeoTIFFs.")
    ap.add_argument("--config", type=str, default="configs/paths_narval.yaml",
                    help="YAML config (default: paths_narval.yaml).")
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Path to V3 best_model.pt (e.g. "
                         "checkpoints/v3_9ch_enc21_12y_2014/best_model.pt).")
    ap.add_argument("--s2s_cache", type=str, default=None,
                    help="Path to s2s_decoder_cache.dat (required when "
                         "checkpoint decoder=s2s_legacy).")
    ap.add_argument("--issue_dates", type=str, nargs="+", required=True,
                    help="One or more YYYY-MM-DD forecast issue dates.")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output root dir; per-issue subdirs created.")
    ap.add_argument("--pred_batch_size", type=int, default=512,
                    help="Patch chunk for forward pass (default 512).")
    args = ap.parse_args()
    forecast(args)


if __name__ == "__main__":
    main()
