"""
Training Scripts
================
Model training entry points. Run via:
    python -m src.training.train_transformer_fwi --config configs/default.yaml
    python -m src.training.train_s2s_transformer --config configs/default.yaml
    python -m src.training.train_logistic --config configs/default.yaml

Modules:
    train_transformer_fwi - Train PatchTemporalTransformer on FWI sequences
    train_s2s_transformer - Train S2STransformer on FWI + ECMWF for 14-46 day forecasts
    train_logistic        - Train logistic regression baseline with 7-day rolling features
"""
