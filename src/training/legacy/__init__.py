"""Legacy training entry points from earlier project phases.

These predate the current train_v3.py pipeline:
  - train_s2s_hotspot_cwfis.py / _v2_4gpu.py : earlier S2S hotspot trainers
    (CWFIS labels, superseded by train_v3.py and the NBAC+NFDB label scheme)
  - train_transformer_7day_cwfis.py, ciffc/, fwi/ : the 7-day fire-probability
    and FWI-prediction projects (use src/models/legacy/ architectures)
  - train_logistic_cwfis.py : the CWFIS logistic-regression baseline

Kept for reproducibility of earlier experiments. Nothing in the current
pipeline (train_v3, forecasting) imports them. Note: train_s2s_hotspot_cwfis_v2
stays at src/training/ root because train_v3 and forecast_v3_to_tif reuse its
data-loading helpers.
"""
