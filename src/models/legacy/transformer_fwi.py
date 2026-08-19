"""
Patch Temporal Transformer for FWI Forecasting
===============================================
Encoder-decoder Transformer that processes spatial patches as temporal sequences.

Architecture:
    Input FWI Maps (T, H, W)
        -> Patchify -> (N_patches, T, patch_dim)
        -> Linear Embedding -> (N_patches, T, d_model)
        -> Positional Encoding
        -> Transformer Encoder-Decoder
        -> Linear Projection -> (N_patches, out_days, patch_dim)
        -> Depatchify -> (out_days, H, W)

Based on pytorch_transformer_fwi20260129.py (v2), which supersedes v1.
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
        # (1, max_len, d_model) for batch_first
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """x: (B, S, E) with batch_first=True"""
        S = x.size(1)
        return x + self.pe[:, :S, :]


class PatchTemporalTransformer(nn.Module):
    """
    Per-patch temporal Transformer encoder-decoder for FWI forecasting.

    Args:
        patch_dim: Flattened patch size (patch_size * patch_size)
        d_model: Transformer hidden dimension
        nhead: Number of attention heads
        num_enc: Number of encoder layers
        num_dec: Number of decoder layers
        in_days: Number of input time steps
        out_days: Number of output time steps
        dropout: Dropout rate
    """

    def __init__(self, patch_dim, d_model=128, nhead=4, num_enc=2, num_dec=2,
                 in_days=7, out_days=7, dropout=0.1):
        super().__init__()
        self.in_days = in_days
        self.out_days = out_days
        self.embed = nn.Linear(patch_dim, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max(in_days, out_days) + 32)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_enc,
            num_decoder_layers=num_dec,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        # Learnable query embedding: (1, out_days, d_model)
        self.query_embed = nn.Parameter(torch.randn(1, out_days, d_model))
        self.proj = nn.Linear(d_model, patch_dim)

    def forward(self, src_seq):
        """
        Args:
            src_seq: (B, in_days, patch_dim)

        Returns:
            (B, out_days, patch_dim)
        """
        B = src_seq.size(0)
        src = self.embed(src_seq)                    # (B, S, E)
        src = self.pos(src)                          # (B, S, E)
        tgt = self.query_embed.expand(B, -1, -1)    # (B, T, E)
        tgt = self.pos(tgt)                          # (B, T, E)
        mem = self.transformer.encoder(src)          # (B, S, E)
        out = self.transformer.decoder(tgt, mem)     # (B, T, E)
        rec = self.proj(out)                         # (B, T, patch_dim)
        return rec
