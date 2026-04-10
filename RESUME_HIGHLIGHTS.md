# Resume Highlights — Sub-seasonal Wildfire Prediction Project

*Choose 5-8 bullets depending on role. Technical ML Engineer / Research Scientist / Data Scientist all covered.*

---

## Full Version (All Highlights)

### Project Summary
**Sub-seasonal Wildfire Ignition Prediction for Canada** — probabilistic forecasting system for wildfire ignition locations across all of Canada at 2 km resolution (~6M grid cells), targeting the operationally critical 14–46 day lead time window where no reliable prediction tools currently exist.

---

### Model Architecture & ML Engineering

- Designed and implemented a **patch-based Sequence-to-Sequence Transformer** (encoder-decoder, 8.5M parameters) that ingests 7 days of historical weather (encoder) and 32 days of ECMWF S2S sub-seasonal weather forecasts (decoder) to predict per-pixel ignition probability on 16×16 pixel patches

- Introduced a **dual-path decoder embedding** that separately projects S2S forecast signals (9-dim) and spatial-temporal context (fire climatology, population, terrain, burn history, lead/season encoding), preventing high-dimensional context from drowning the small forecast signal

- Implemented multiple decoder input representations for ablation studies: patch-mean (9-dim), multi-statistic (24-dim: mean/std/max per channel), sub-patch 4×4 (128-dim), PCA-128 (128-dim via covariance eigendecomposition), and full-patch (2048-dim)

- **Focal BCE Loss** (α=0.25, γ=2.0) to handle extreme class imbalance (fire pixels < 0.01% of total), replacing standard BCE. Also implemented differentiable ApproxNDCG ranking loss and hybrid focal+ranking loss as alternatives

- **Hard negative mining** — 50% of negative samples drawn proportional to fire climatology (high-risk-but-no-fire patches), forcing the model to learn discriminative features near the decision boundary; plus spatial buffering (`neg_buffer=2`) to exclude 5×5 neighborhoods around fire patches

---

### Multi-Source Data Engineering (15+ TB pipeline)

- Built end-to-end ingestion pipelines for **8 heterogeneous data sources**: Copernicus ERA5 reanalysis, ECMWF S2S hindcasts, Canadian FWI System (6 components), MODIS NDVI, CWFIS fire hotspot records, Canadian Digital Elevation Model (CDEM), population density grids, and National Burned Area Composite (NBAC)

- Reprojected all spatial data to a unified **EPSG:3978 Canada Lambert Conformal Conic** grid at 2 km resolution (2709×2281 pixels), handling CRS conversions from WGS84 GRIB/NetCDF sources with bilinear resampling

- Engineered **derived features**: 30-day precipitation deficit, annual rolling fire climatology (leak-free temporal split), burn age/count from historical fire perimeters, vapor pressure deficit (VPD) from dewpoint, and CAPE as lightning proxy (after ruling out GOES GLM due to latitude limitations above 52°N)

- Designed a 5-stage **S2S forecast processing pipeline**: raw GRIB download → daily TIF extraction → EPSG:3978 reprojection with VPD computation → Van Wagner FWI forward integration → patchified memmap cache (supports multiple compression formats trading size for fidelity)

---

### HPC Infrastructure & Performance Optimization

- Managed training across **two HPC clusters** (Narval and Trillium) with SLURM job scheduling, GPU/CPU dependency chains, 20 TB scratch storage, and complex multi-step pipelines

- Designed a **reusable memory-mapped cache system**: pre-computed float16 patch-first memmaps (up to 549 GB for 16-channel encoder) built once on CPU nodes, then loaded in seconds for all subsequent experiments — eliminating redundant I/O-bound processing across dozens of experiment runs

- Implemented **SSD-accelerated training**: venv + S2S cache + encoder memmap copied to node-local NVMe at job start (3 GB/s vs 500 MB/s Lustre), with automatic capacity detection and fallback to network storage, and atomic copy-back to persistent storage on both success and failure

- Built **parallel data preparation pipelines** with SLURM dependency chains: resample → cache build → training, enabling continuous experiment throughput across multiple concurrent jobs

- Diagnosed and resolved I/O bottlenecks using iostat profiling: identified Lustre random-read penalties as the primary limitation for memmap loading, designed SSD copy-first strategy to overcome it

---

### Evaluation & Experimental Rigor

- Designed **rigorous per-window evaluation** over 811 fire-season validation windows (2022-05 to 2024-10) with mean±std metrics, replacing initial 20-window samples that had high variance

- Implemented **8 complementary metrics**: Lift@K, Precision@K, Recall@K, CSI (Critical Success Index), ETS (Equitable Threat Score, WMO-recommended), ROC-AUC, Brier Score, and PR-AUC — with analysis showing PR-AUC as the most honest metric under extreme class imbalance

- Built systematic **ablation studies**: oracle decoder (perfect future weather, upper bound), random decoder, zero decoder, climatology decoder, and null input baseline — establishing a complete performance hierarchy from trivial baselines to the theoretical ceiling

- Achieved **Lift@5000 = 7.35x** (top-5000 predictions are 7.35× more likely to contain fire than random baseline) using real S2S forecasts on 8-channel V2 model; oracle upper bound reaches 10.01x Lift, demonstrating clear headroom for V3 improvements

---

### Reliability Engineering & Debugging Methodology

- **Systematically categorized and fixed 9 distinct bug categories** in the training pipeline: silent sentinel value propagation (nodata=9999 treated as real data), temporal data leakage (current-year burn data leaking into encoder features), train/val interface mismatches (decoder_ctx missing from validation path), Python module caching (long-running jobs using stale imported code), cache loss on crash (`set -e` preventing cleanup), out-of-memory in PCA fitting, and more

- Implemented **9 defensive measures** to prevent regression of each bug category:
  - Runtime shape assertions in model forward pass (catches dim mismatches at batch 1)
  - Git SHA + dirty-status startup print (prevents stale-code confusion)
  - Data quality guards flagging channels with >99.9% identical values (catches sentinel bugs)
  - Normalization statistics sanity checks (warns on suspicious mean/std)
  - Validation probe with fake tensors before main loop (catches val path bugs in <1 second)
  - Smoke test SLURM job exercising full train/val/cluster-eval path in 1 hour
  - `EXIT` trap for atomic cache copy-back on any termination path
  - Source-inspection regression tests for temporal leakage and nodata handling (6 tests)
  - Memory-efficient PCA via covariance eigendecomposition (avoiding SVD OOM on large sample matrices)

- Developed a **bug classification framework** distinguishing "silent bugs" (code runs, loss decreases, model learns wrong things) from "loud bugs" (crashes) — silent bugs identified as the most dangerous class requiring proactive static analysis

- **Fixed critical temporal data leakage**: discovered that annual burn-scar files (e.g. `years_since_burn_2021.tif`) contained fires from the entire 2021 calendar year, causing models to see Sept-Dec 2021 fires when predicting for July 2021. Fixed by using year-minus-one data with consistent "upto" naming convention matching the fire climatology layer

- **Root-caused sentinel value propagation**: burn-scar TIFs use `nodata=9999` for "never burned", but `_load_static_channel` did not mask this, resulting in 99.8% of pixels having the same post-normalization value — effectively making the burn_age channel useless. Fixed by reading `src.nodata` from TIF metadata

---

### Disk & Resource Management

- Managed **20 TB scratch quota** across multiple simultaneous experiments: freed 463 GB through systematic cleanup of intermediate products (raw GRIBs, deprecated V2 memmaps), designed storage pyramid with raw data (~12 GB, immutable) → processed TIFs (~10 TB, regenerable) → training caches (~5 TB, regenerable from processed)

- Implemented **incremental cache rebuilding strategy**: documented full pipeline from raw GRIB to training-ready memmap, enabling any cache format to be regenerated from the 12 GB raw source in 12-24 hours rather than requiring re-download from external APIs

---

## Short Versions by Role

### ML Engineer (focus: infrastructure, reliability, optimization)

- Designed and trained an 8.5M-parameter S2S Transformer on 15+ TB geospatial pipeline across 2 HPC clusters, achieving 7.35x Lift@5000 for Canadian wildfire prediction 14-46 days ahead
- Built reusable float16 memmap caching system (549 GB for 16-channel encoder) with SSD copy-back optimization, achieving 5× speedup over Lustre baseline
- **Identified and fixed 9 silent ML bug categories** including sentinel value propagation, temporal data leakage, and train/val interface mismatches. Implemented 9 defensive measures (shape assertions, data quality guards, smoke tests, regression tests) to prevent regressions
- Managed SLURM dependency chains with automatic cache copy-back on crash, preventing 15+ hours of work loss during training failures
- Reduced PCA memory footprint from 30 GB (SVD) to <10 GB (covariance eigendecomposition) for 1M-sample dimensionality reduction

### Research Scientist (focus: modeling, evaluation, novelty)

- Designed a patch-based Sequence-to-Sequence Transformer for sub-seasonal wildfire prediction (14-46 day lead) combining ERA5 reanalysis (encoder) with ECMWF S2S forecasts (decoder) at 2 km resolution
- Introduced **dual-path decoder embedding** separating forecast signals from spatial context, preventing signal drowning under high-dimensional context concatenation
- Implemented **Focal BCE Loss + hard negative mining** (proportional to fire climatology) to handle <0.01% positive-class imbalance, plus differentiable ApproxNDCG ranking loss
- Achieved **Lift@5000 = 7.35x** with PR-AUC = 0.37 (vs. 4.20x climatology baseline, 10.01x oracle upper bound) on 811-window fire-season validation
- Designed systematic ablation framework with 5 decoder modes (oracle, S2S legacy, random, zeros, climatology) and 8 evaluation metrics (Lift@K, CSI, ETS, PR-AUC, etc.)
- **Identified and fixed a critical temporal data leakage**: annual burn-scar files contained future-year fires, causing implicit data leakage in the burn_age channel — corrected by implementing strict "upto-year" temporal boundary logic

### Data Scientist / Data Engineer (focus: pipelines, data quality)

- Built end-to-end ingestion pipelines for 8 heterogeneous geospatial data sources (ERA5, ECMWF S2S, FWI, MODIS NDVI, CWFIS hotspots, CDEM, WorldPop, NBAC) into a unified EPSG:3978 grid
- Designed 5-stage S2S forecast processing pipeline (GRIB → daily TIF → EPSG:3978 reproject → FWI forward integration → patchified cache) processing 1683 issue dates with 32 lead days each
- Engineered derived features including 30-day precipitation deficit, annual rolling fire climatology, and vapor pressure deficit from dewpoint
- **Identified and fixed silent data quality bugs**: burn-scar sentinel value (9999) propagating through normalization, temporal leakage in year-aligned channels. Added data quality guards flagging channels with >99.9% identical values
- Managed 15+ TB storage across multiple concurrent experiments with systematic cleanup (freed 463 GB) and designed rebuild strategy to regenerate any intermediate cache from 12 GB raw source

---

## One-Line Summary (for cover letter / summary section)

> Built a patch-based Sequence-to-Sequence Transformer for sub-seasonal (14-46 day) wildfire ignition prediction across Canada, achieving 7.35× lift over random baseline on 15+ TB of multi-source geospatial data; identified and systematically prevented 9 classes of silent ML bugs through defensive engineering (shape assertions, data quality guards, smoke tests, regression tests) to ensure training pipeline reliability.
