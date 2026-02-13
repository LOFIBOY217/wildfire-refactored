"""
Model Definitions
=================
Neural network architectures and baseline models for wildfire prediction.
Contains model classes only -- no training logic.

Modules:
    transformer_fwi   - PatchTemporalTransformer: patch-based encoder-decoder for 7-day FWI forecast
    s2s_transformer   - S2STransformer: subseasonal 14-46 day forecast using FWI + ECMWF inputs
    logistic_baseline - Logistic regression with 3 engineered features (FWI, dryness, recent fire)
"""
