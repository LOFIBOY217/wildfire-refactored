"""
7-Day Wildfire Probability Transformer — Future Meteo Decoder
=============================================================
Variant of FireProb7DayTransformer where the decoder receives the actual
future 7-day meteorological observations (FWI + 2m temperature + 2m dewpoint)
instead of learnable lead-time queries.

This is an oracle / upper-bound experiment: at both training and inference
the model is given the TRUE future weather as decoder input.  When S2S ECMWF
forecast data becomes available, the same interface accepts forecast values
with no code change.

Architecture:
    Encoder: (B, 7, patch_dim_in)  → TransformerEncoder → memory
    Decoder: dec_embed(future_meteo) → TransformerDecoder(memory) → proj → logits
    Output:  (B, 7, patch_dim_out)  logits (apply sigmoid for probabilities)

Key difference from transformer_7day.py:
    - Removes self.lead_embed (learnable Embedding)
    - Adds    self.dec_embed  (Linear projection of future meteo patches)
    - forward() requires a mandatory dec_input argument
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


class FireProb7DayTransformerWithFuture(nn.Module):
    """
    7-day fire probability forecasting Transformer with future meteo decoder.

    Args:
        patch_dim_in:        Flattened input patch size = patch_size² × 3 (FWI+2t+2d)
        patch_dim_out:       Flattened output patch size = patch_size²
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

        # Encoder: project historical meteo patches to model dimension
        self.enc_embed = nn.Linear(patch_dim_in, d_model)

        # Decoder: project future meteo patches to model dimension
        # (replaces learnable lead_embed from the original model)
        self.dec_embed = nn.Linear(patch_dim_in, d_model)

        # Shared positional encoding for both encoder and decoder sequences
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

        # Project decoder output to per-pixel logits
        self.proj = nn.Linear(d_model, patch_dim_out)

    def forward(self, enc_input, dec_input):
        """
        Args:
            enc_input: (B, encoder_days,  patch_dim_in)  normalised historical FWI+2t+2d
            dec_input: (B, forecast_days, patch_dim_in)  normalised future    FWI+2t+2d

        Returns:
            logits: (B, forecast_days, patch_dim_out)
                    Raw logits — apply sigmoid for fire probabilities.
        """
        # Encoder path
        enc = self.enc_embed(enc_input)   # (B, encoder_days, d_model)
        enc = self.pos_enc(enc)
        memory = self.transformer.encoder(enc)  # (B, encoder_days, d_model)

        # Decoder path: project future meteo (replaces learnable lead queries)
        dec = self.dec_embed(dec_input)   # (B, forecast_days, d_model)
        dec = self.pos_enc(dec)

        output = self.transformer.decoder(dec, memory)  # (B, forecast_days, d_model)
        logits = self.proj(output)                       # (B, forecast_days, patch_dim_out)

        return logits
