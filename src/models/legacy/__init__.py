"""Legacy model architectures from earlier project phases.

These predate the current sub-seasonal (14-46 day) hotspot model in
src/models/s2s_hotspot.py: the transformer_7day* family (7-day fire
probability, CIFFC) and the FWI-prediction models (transformer_fwi,
transformer_7day_fwi, s2s_transformer). They are kept for reproducibility of
earlier experiments and are used only by the legacy training scripts under
src/training/ciffc/ and src/training/fwi/. The current pipeline (train_v3,
forecast_v3_to_tif) does not import them.
"""
