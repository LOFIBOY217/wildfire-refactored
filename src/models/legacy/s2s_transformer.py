"""
S2S (Subseasonal-to-Seasonal) Transformer for FWI Prediction
=============================================================
Uses historical FWI observations as encoder input and ECMWF S2S weather
forecasts as decoder input to predict FWI 14-46 days ahead.

Architecture:
    Encoder: FWI history patches -> Linear -> PositionalEncoding -> TransformerEncoder
    Decoder: ECMWF forecast patches -> Linear -> PositionalEncoding -> TransformerDecoder
    Output: Linear projection -> predicted FWI patches

Based on train_s2s_transformer.py.
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


class S2STransformer(nn.Module):
    """
    S2S FWI Prediction Transformer.

    Args:
        patch_dim_fwi: Flattened FWI patch size (patch_size^2)
        patch_dim_ecmwf: Flattened ECMWF patch size (patch_size^2 * num_vars)
        d_model: Transformer hidden dimension
        nhead: Number of attention heads
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
        encoder_days: Number of historical FWI days (encoder input)
        decoder_days: Number of forecast days (decoder input/output)
    """

    def __init__(self, patch_dim_fwi, patch_dim_ecmwf,
                 d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 encoder_days=7, decoder_days=33):
        super().__init__()

        self.encoder_days = encoder_days
        self.decoder_days = decoder_days

        # Separate embeddings for FWI and ECMWF inputs
        self.enc_embed = nn.Linear(patch_dim_fwi, d_model)
        self.dec_embed = nn.Linear(patch_dim_ecmwf, d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(
            d_model, max_len=max(encoder_days, decoder_days) + 64
        )

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Output projection back to FWI patch dimension
        self.proj = nn.Linear(d_model, patch_dim_fwi)

    def forward(self, encoder_input, decoder_input):
        """
        Args:
            encoder_input: (B, encoder_days, patch_dim_fwi)
            decoder_input: (B, decoder_days, patch_dim_ecmwf)

        Returns:
            output: (B, decoder_days, patch_dim_fwi)
        """
        enc = self.enc_embed(encoder_input)   # (B, encoder_days, d_model)
        dec = self.dec_embed(decoder_input)   # (B, decoder_days, d_model)

        enc = self.pos_enc(enc)
        dec = self.pos_enc(dec)

        memory = self.transformer.encoder(enc)
        output = self.transformer.decoder(dec, memory)

        output = self.proj(output)  # (B, decoder_days, patch_dim_fwi)
        return output
