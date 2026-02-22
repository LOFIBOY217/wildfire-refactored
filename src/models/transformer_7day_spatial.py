"""
7-Day Wildfire Probability Transformer — Spatial Position Encoding + Future Meteo Decoder
==========================================================================================
Builds on FireProb7DayTransformerWithFuture with two improvements:

1. **patch_size=8 (default in training script)**
   Smaller patches (8×8=64 pixels vs 16×16=256) preserve more spatial detail
   within each patch, reducing false positives caused by coarse spatial averaging.

2. **Learnable spatial position embedding**
   Each patch has a unique learned position vector indexed by its linear patch id
   (row * grid_cols + col).  This lets the model learn geographic priors:
   e.g. western boreal forest patches are inherently higher-risk than eastern
   grasslands regardless of current weather.

Architecture:
    Encoder: (B, 7, patch_dim_in)  → enc_embed + spatial_embed + pos_enc
                                   → TransformerEncoder → memory
    Decoder: dec_embed(future_meteo) + spatial_embed + pos_enc
                                   → TransformerDecoder(memory) → proj → logits
    Output:  (B, 7, patch_dim_out) logits

Interface is forward-compatible with S2S ECMWF forecast data as decoder input.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal temporal positional encoding (batch_first)."""

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


class FireProb7DayTransformerSpatial(nn.Module):
    """
    7-day fire probability Transformer with future meteo decoder and spatial
    position encoding.

    Args:
        patch_dim_in:        Flattened input patch size = patch_size² × 3
        patch_dim_out:       Flattened output patch size = patch_size²
        n_patches:           Total number of spatial patches = grid_rows × grid_cols
                             Used to size the learnable spatial embedding table.
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
        n_patches,
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
        self.dec_embed = nn.Linear(patch_dim_in, d_model)

        # Learnable spatial position embedding: one vector per patch location
        # Indexed by linear patch id = row * grid_cols + col
        self.spatial_embed = nn.Embedding(n_patches, d_model)

        # Temporal positional encoding (sinusoidal, shared by enc and dec)
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

        # Initialise spatial embeddings with small values
        nn.init.normal_(self.spatial_embed.weight, mean=0.0, std=0.02)

    def forward(self, enc_input, dec_input, patch_idx):
        """
        Args:
            enc_input : (B, encoder_days,  patch_dim_in)  normalised historical meteo
            dec_input : (B, forecast_days, patch_dim_in)  normalised future meteo
            patch_idx : (B,)  LongTensor — linear spatial index of each patch in batch

        Returns:
            logits    : (B, forecast_days, patch_dim_out)
                        Raw logits — apply sigmoid for fire probabilities.
        """
        # Spatial position: one vector per patch, broadcast across all time steps
        sp = self.spatial_embed(patch_idx)   # (B, d_model)
        sp = sp.unsqueeze(1)                  # (B, 1, d_model)

        # Encoder path
        enc = self.enc_embed(enc_input)       # (B, encoder_days, d_model)
        enc = enc + sp                         # inject spatial position at every time step
        enc = self.pos_enc(enc)                # add temporal position encoding
        memory = self.transformer.encoder(enc)

        # Decoder path
        dec = self.dec_embed(dec_input)       # (B, forecast_days, d_model)
        dec = dec + sp                         # same spatial position for decoder
        dec = self.pos_enc(dec)
        output = self.transformer.decoder(dec, memory)  # (B, forecast_days, d_model)

        return self.proj(output)               # (B, forecast_days, patch_dim_out)
