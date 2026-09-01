# Sub-seasonal Wildfire Ignition Prediction for Canada

Probabilistic forecasting of **where wildfires will ignite across Canada,
14–46 days in advance**, on a daily 2 km grid (~6M cells). The model conditions
sub-seasonal (S2S) meteorology and fire-weather indices on a learned map of
where fires tend to start, targeting the operational lead-time window where no
reliable ignition-location tools currently exist.

- **Input**: ERA5 / S2S meteorology, Fire Weather Index (FWI) components,
  vegetation, terrain, lightning, population, and burn history.
- **Output**: a daily ignition-probability GeoTIFF over all of Canada
  (EPSG:3978, 2709×2281, ~2 km).
- **Model**: patch-based S2S Transformer (encoder–decoder) trained with focal
  loss and hard-negative mining.

## Results

Evaluated on held-out fire seasons, the model substantially outperforms the
operational baselines (climatology, persistence, and an FWI oracle) in **lift** —
the fire density among its top-ranked pixels relative to the base rate.

The advantage is largest on **novel ignitions**: new fires, as opposed to the
continuation of already-burning areas. Persistence-style baselines score highly
on *total* fire (fires tend to keep burning) but collapse toward zero lift on
novel ignition, whereas the model retains strong lift there — which is the
operationally hard and important case for early warning. See `figures/` and
`docs/figure_interpretations.md` for the per-model, per-budget breakdowns and
forecast-vs-observed maps.

## Repository layout

```
src/
  config.py            # YAML config loader (env-var expansion, path resolution)
  data_ops/            # download + processing (FWI, ERA5, hotspots, labels, caches)
  datasets/            # torch datasets / patch samplers
  models/              # S2S transformer and baselines
  training/            # train_v3.py (main), baselines, losses
  forecasting/         # forecast_v3_to_tif.py (checkpoint -> probability GeoTIFF)
  evaluation/          # metrics (Lift@K, Lift@30km, BSS), baselines, comparisons
configs/               # default.yaml + per-machine path overrides
scripts/               # data builders, audits, analysis
tests/                 # unit tests (pytest)
docs/                  # data conventions, label methodology, metric definitions
```

## Installation

Requires Python 3.11. The conda environment installs everything, including a
CPU build of PyTorch.

```bash
conda env create -f environment.yml
conda activate wildfore-r
```

For a GPU build, reinstall PyTorch from the CUDA index that matches your driver:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Reading GRIB files (ERA5 downloads) needs a backend — apply one overlay:

```bash
conda env update -n wildfore-r -f environments/environment.local-pygrib.yml  # local (pygrib)
conda env update -n wildfore-r -f environments/environment.hpc-cfgrib.yml    # HPC (cfgrib)
```

Prefer pip? Use `requirements.txt` instead (same versions). Note that the
geospatial packages need a system GDAL, which conda otherwise provides for you:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration and credentials

All paths live in `configs/default.yaml` (relative to the repo root). To point
at a different data location, copy it and pass `--config your_paths.yaml`;
values are merged on top of the defaults.

Data-download scripts read credentials from environment variables (or a
`~/.cdsapirc` file) — never commit keys:

```bash
export CDS_API_KEY=your-copernicus-cds-key      # https://cds.climate.copernicus.eu
export ECDS_API_KEY=your-ecmwf-datastore-token  # https://ecds.ecmwf.int (S2S / TIGGE)
```

`CDS_API_KEY` covers the Copernicus CDS / CEMS datasets (ERA5, FWI). The S2S and
TIGGE downloaders use the ECMWF Data Store (ECDS); set `ECDS_API_KEY` to your
ECMWF personal access token (the same token works, so `CDS_API_KEY` is used as a
fallback), and accept each dataset's licence once on its ECDS Download tab.
NASA FIRMS/Earthdata scripts use their own keys (see each script's `--help`).

## Data

The raw datasets (hundreds of GB) are **not** included in the repository. They
are downloaded from public providers and processed onto the common 2 km grid.

The downloaders pull from live external services, which drift over time (layers
renamed, formats changed, whole APIs decommissioned). Every downloader carries a
**Maintenance** block in its docstring, and
[`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) is the central registry of each
dataset's authoritative source URL, credential, and *where to look when a
downloader breaks*.

| Data | Source | Downloader |
|------|--------|-----------|
| FWI / FFMC / DMC / DC / ISI / BUI | Copernicus CDS (cems-fire-historical-v1) | `python -m src.data_ops.download.download_fwi_historical` |
| ERA5 meteorology | Copernicus CDS (reanalysis-era5-single-levels) | `python -m src.data_ops.download.download_ecmwf_reanalysis_observations` |
| Fire hotspots | CWFIS | `python -m src.data_ops.download.download_hotspots` |

Then build the training targets and the meteorology cache:

```bash
# Fire labels (NBAC burned-area polygons + NFDB points, r14 dilation)
python scripts/build_fire_labels.py --scheme nbac_nfdb

# Per-pixel meteorology cache the training loop reads from
python scripts/build_meteo_cache.py --config configs/default.yaml
```

See `docs/DATA_CONVENTIONS.md` for the raster conventions (CRS, nodata, naming)
and `docs/LABEL_DECISION_2026_04_21.md` for how fire labels are defined.

## Quick start

The unit tests run without any external data and are the fastest way to verify
your environment:

```bash
pytest tests/
```

## Training

`src/training/train_v3.py` is the main entry point. A representative run
(9-channel, 21-day encoder history, focal loss, hard-negative mining):

```bash
python -m src.training.train_v3 \
    --config configs/default.yaml \
    --run_name v3_9ch_enc21 \
    --channels "FWI,2t,fire_clim,2d,tcw,sm20,population,slope,burn_age" \
    --in_days 21 \
    --loss_fn focal --focal_alpha 0.25 --focal_gamma 2.0 \
    --hard_neg_fraction 0.5 --neg_ratio 20 --neg_buffer 2 \
    --batch_size 4096 --epochs 4 --lr 1e-4 --weight_decay 0.01 --dropout 0.2
```

Run `python -m src.training.train_v3 --help` for the full flag list (channels,
architecture size, decoder mode, regularization, evaluation options). On a
SLURM cluster, wrap this command in a batch script (set your own account,
paths, and modules).

## Evaluation

Evaluate a trained checkpoint (add `--full_val` for the full 811-window eval):

```bash
python -m src.training.train_v3 \
    --run_name v3_9ch_enc21 \
    --channels "FWI,2t,fire_clim,2d,tcw,sm20,population,slope,burn_age" \
    --eval_checkpoint checkpoints/v3_9ch_enc21/best_model.pt \
    --full_val
```

Metrics live in `src/evaluation/metrics.py`: **Lift@K** (fire density in the
top-K pixels relative to the base rate), **Lift@30km** (cluster-level lift after
coarsening), and **BSS** (Brier Skill Score vs a base-rate reference). See
`docs/metrics/` for definitions.

## Forecasting

Produce ignition-probability GeoTIFFs for chosen issue dates:

```bash
python -m src.forecasting.forecast_v3_to_tif \
    --config configs/default.yaml \
    --ckpt checkpoints/v3_9ch_enc21/best_model.pt \
    --s2s_cache data/s2s_processed/s2s_decoder_cache.dat \
    --issue_dates 2023-05-15 2023-08-15 \
    --out_dir outputs/v3_9ch_enc21_fire_prob
```

### Unified inference

`src/forecasting/run_inference.py` wraps the forecast step into a single command
that also produces the downstream artifacts: a **top-K ranking CSV**
(pixel → lon/lat, ordered by ignition probability) and an optional **forecast
map** alongside the probability GeoTIFF.

```bash
python -m src.forecasting.run_inference \
    --config configs/default.yaml \
    --ckpt checkpoints/v3_9ch_enc21/best_model.pt \
    --s2s_cache data/s2s_processed/s2s_decoder_cache.dat \
    --issue_dates 2023-08-15 \
    --lead 30 --topk 5000 \
    --out_dir outputs/inference_20230815
```

Add `--eval_window N` to switch on **verification**: the observed fire raster is
built over the N-day window after each issue date and metrics (Lift@K, BSS,
ROC-AUC) are written to JSON. The truth source is explicit — `--truth_source`
defaults to `nbac_nfdb` (the training label; NBAC burned-area ∪ NFDB points),
with `ciffc` (current-season) and `cwfis` also supported:

```bash
python -m src.forecasting.run_inference \
    --config configs/default.yaml \
    --ckpt checkpoints/v3_9ch_enc21/best_model.pt \
    --s2s_cache data/s2s_processed/s2s_decoder_cache.dat \
    --issue_dates 2023-08-15 \
    --lead 30 --eval_window 30 --truth_source nbac_nfdb \
    --out_dir outputs/inference_20230815
```

Note that these verification metrics are computed over the full grid for a single
issue-date window, so they are **not** directly comparable to the multi-window,
valid-mask Lift figures reported in `docs/`. See
`python -m src.forecasting.run_inference --help` for map/ranking toggles and
truth-source path overrides.

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
