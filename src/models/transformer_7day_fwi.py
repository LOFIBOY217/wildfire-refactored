"""
7-Day FWI Regression Transformer
=================================
Encoder-decoder Transformer that predicts future FWI (Fire Weather Index)
values for 7 future days using 7 days of historical meteorological observations.

Unlike the binary fire-probability variants, this model treats forecasting as a
regression task: the target is the continuous FWI value at each spatial location,
not binary CIFFC fire occurrence labels.

Motivation:
    CIFFC binary labels suffer from irreducible noise (sparse records, time lag,
    spatial mismatch). FWI is a physically-derived continuous index available at
    every grid cell every day — a much cleaner regression target that allows the
    Transformer to learn true time-series patterns without label noise.

Architecture:
    Encoder: (B, 7, patch_dim_in)   → TransformerEncoder → memory
    Decoder: lead_embed(0..6)        → TransformerDecoder(memory) → proj → FWI
    Output:  (B, 7, patch_dim_out)   raw FWI predictions (no sigmoid, use MSELoss)

patch_dim_in  = patch_size² × 3  (FWI + 2t + 2d flattened)
patch_dim_out = patch_size² × 1  (FWI only)
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (batch_first)."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class FWI7DayTransformer(nn.Module):
    """
    7-day FWI regression Transformer.

    Args:
        patch_dim_in:        Flattened input patch size = patch_size² × 3 (FWI+2t+2d)
        patch_dim_out:       Flattened output patch size = patch_size² (FWI only)
        d_model:             Transformer hidden dimension (default 128)
        nhead:               Number of attention heads (default 4)
        num_encoder_layers:  Number of encoder layers (default 2)
        num_decoder_layers:  Number of decoder layers (default 2)
        dim_feedforward:     FFN hidden dimension (default 512)
        dropout:             Dropout rate (default 0.1)
        forecast_days:       Number of future days to predict (default 7)
        encoder_days:        Number of historical days in encoder input (default 7)
    """

    def __init__(
        self,
        patch_dim_in,
        patch_dim_out,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        forecast_days=7,
        encoder_days=7,
    ):
        super().__init__()

        self.forecast_days = forecast_days
        self.encoder_days  = encoder_days

        # Project multi-variable input patches to model dimension
        self.enc_embed = nn.Linear(patch_dim_in, d_model)

        # Learnable lead-time queries: one vector per forecast day (0..forecast_days-1)
        self.lead_embed = nn.Embedding(forecast_days, d_model)

        # Positional encoding for encoder sequence
        self.pos_enc = PositionalEncoding(
            d_model, max_len=max(encoder_days, forecast_days) + 16
        )

        # Transformer stack
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Project decoder output to per-pixel FWI predictions
        # No activation — raw output trained with MSELoss
        self.proj = nn.Linear(d_model, patch_dim_out)

    def forward(self, enc_input):
        """
        Args:
            enc_input: (B, encoder_days, patch_dim_in)  normalised FWI+2t+2d patches

        Returns:
            fwi_pred: (B, forecast_days, patch_dim_out)
                      Raw predicted FWI values (not normalised, trained with MSELoss).
        """
        B = enc_input.size(0)

        # Encoder path
        enc    = self.enc_embed(enc_input)          # (B, encoder_days, d_model)
        enc    = self.pos_enc(enc)
        memory = self.transformer.encoder(enc)       # (B, encoder_days, d_model)

        # Decoder path: learnable lead-time queries
        lead_idx = torch.arange(self.forecast_days, device=enc_input.device)
        dec      = self.lead_embed(lead_idx).unsqueeze(0).expand(B, -1, -1)  # (B, 7, d_model)

        output   = self.transformer.decoder(dec, memory)  # (B, forecast_days, d_model)
        fwi_pred = self.proj(output)                       # (B, forecast_days, patch_dim_out)

        return fwi_pred
