"""
S2S Hotspot Transformer for Fire Probability Prediction
=======================================================
Predicts per-pixel fire probability 14–46 days ahead using a seq2seq
Transformer architecture:

    Encoder: 7-day historical weather (FWI + 2m temperature + 2m dewpoint)
             patches → Linear → PositionalEncoding → TransformerEncoder

    Decoder: future weather patches (ERA5 oracle OR ECMWF S2S hindcast)
             → Linear → PositionalEncoding → TransformerDecoder

    Output:  Linear projection → per-pixel logits → BCEWithLogitsLoss

Key differences from S2STransformer (FWI regression):
  - Encoder input:  patch_dim_enc = P²×3  (FWI + 2t + 2d, not FWI-only)
  - Decoder input:  patch_dim_dec = P²×3  (ERA5 oracle) or P²×5 (ECMWF)
  - Output:         patch_dim_out = P²    (one logit per pixel per day)
  - Loss:           BCEWithLogitsLoss (binary fire probability, not MSE FWI)
  - No sigmoid in forward() — caller applies sigmoid for probabilities.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (batch_first)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class S2SHotspotTransformer(nn.Module):
    """
    S2S Fire Probability Prediction Transformer.

    Treats each spatial patch as an independent sequence and predicts
    per-pixel fire probability logits for each decoder day.

    Args:
        patch_dim_enc:        Flattened encoder patch size.
                              P²×3 for FWI+2t+2d encoder history.
        patch_dim_dec:        Flattened decoder patch size.
                              P²×3 for ERA5 oracle; P²×5 for ECMWF S2S.
        patch_dim_out:        Output patch size = P² (one logit per pixel).
        d_model:              Transformer hidden dimension.
        nhead:                Number of attention heads.
        num_encoder_layers:   Depth of TransformerEncoder stack.
        num_decoder_layers:   Depth of TransformerDecoder stack.
        dim_feedforward:      FFN hidden dimension (typically d_model × 4).
        dropout:              Dropout rate.
        encoder_days:         Length of encoder input sequence (default 7).
        decoder_days:         Length of decoder input/output sequence (default 33
                              for lead 14–46).
        n_patches:            Number of spatial patches (for learnable patch
                              embedding). 0 = disabled (default).
        mlp_dec_embed:        If True, use a 2-layer MLP for the decoder
                              embedding instead of a single Linear layer.
                              Provides better compression of 2048-dim oracle
                              features into d_model. Default: False.
    """

    def __init__(
        self,
        patch_dim_enc: int,
        patch_dim_dec: int,
        patch_dim_out: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        encoder_days: int = 7,
        decoder_days: int = 33,
        n_patches: int = 0,
        mlp_dec_embed: bool = False,
        dec_ctx_dim: int = 0,
    ):
        super().__init__()

        self.encoder_days  = encoder_days
        self.decoder_days  = decoder_days
        self.patch_dim_out = patch_dim_out
        self.dec_ctx_dim   = dec_ctx_dim

        # Separate linear projections for encoder (history) and decoder (forecast)
        self.enc_embed = nn.Linear(patch_dim_enc, d_model)

        if dec_ctx_dim > 0 and dec_ctx_dim < patch_dim_dec:
            # Dual-path decoder embedding: separate projections for forecast signal
            # and spatial/temporal context, then add. Prevents the small forecast
            # signal (e.g. 9-dim S2S) from being drowned by large context (1284-dim).
            dec_forecast_dim = patch_dim_dec - dec_ctx_dim
            self._dec_forecast_dim = dec_forecast_dim  # remembered for shape assertions
            self.dec_forecast_embed = nn.Sequential(
                nn.Linear(dec_forecast_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            self.dec_ctx_embed = nn.Sequential(
                nn.Linear(dec_ctx_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            self.dec_embed = None  # unused, forward uses dual path
        elif mlp_dec_embed and patch_dim_dec > d_model:
            # 2-layer MLP for decoder: better preserves structure when compressing
            # large oracle/S2S inputs (e.g. 2048-dim full patch → d_model).
            hidden_dec = max(d_model * 2, patch_dim_dec // 2)
            self.dec_embed = nn.Sequential(
                nn.Linear(patch_dim_dec, hidden_dec),
                nn.GELU(),
                nn.Linear(hidden_dec, d_model),
            )
            self._dec_embed_in_dim = patch_dim_dec
        else:
            self.dec_embed = nn.Linear(patch_dim_dec, d_model)
            self._dec_embed_in_dim = patch_dim_dec
        self.embed_drop = nn.Dropout(dropout)

        # Learnable spatial patch embedding (geographic location encoding).
        # Maps patch index → d_model vector, added to both enc and dec after
        # the temporal projection. Allows the model to learn location-specific
        # fire-behavior patterns beyond what fire_clim (a single scalar) captures.
        if n_patches > 0:
            self.patch_embed = nn.Embedding(n_patches, d_model)
            nn.init.normal_(self.patch_embed.weight, std=0.01)
        else:
            self.patch_embed = None

        # Shared sinusoidal positional encoding
        self.pos_enc = PositionalEncoding(
            d_model, max_len=max(encoder_days, decoder_days) + 64
        )

        # Transformer backbone (batch_first so dim 0 = batch)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Output head: project d_model → P² binary logits (no sigmoid here)
        self.proj = nn.Linear(d_model, patch_dim_out)

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
        patch_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            encoder_input: (B, encoder_days, patch_dim_enc)
                           Historical weather patches (FWI + 2t + 2d).
            decoder_input: (B, decoder_days, patch_dim_dec)
                           Future weather patches (ERA5 oracle or ECMWF S2S).
            patch_ids:     (B,) integer patch indices, used to look up spatial
                           patch embeddings. Optional; if None or patch_embed is
                           disabled, no spatial embedding is added.

        Returns:
            logits: (B, decoder_days, patch_dim_out)
                    Raw fire-probability logits (apply sigmoid for probabilities).
        """
        # Shape assertions — catches train/val interface mismatches at batch 1
        assert encoder_input.dim() == 3, \
            f"encoder_input must be 3D (B,T,D), got shape {tuple(encoder_input.shape)}"
        assert encoder_input.shape[-1] == self.enc_embed.in_features, \
            f"encoder_input dim {encoder_input.shape[-1]} != expected {self.enc_embed.in_features}"
        assert decoder_input.dim() == 3, \
            f"decoder_input must be 3D (B,T,D), got shape {tuple(decoder_input.shape)}"

        enc = self.pos_enc(self.embed_drop(self.enc_embed(encoder_input)))  # (B, enc_days, d_model)

        if self.dec_ctx_dim > 0 and self.dec_embed is None:
            # Dual-path: split decoder input into forecast + context, project separately, add
            expected_dec_dim = self._dec_forecast_dim + self.dec_ctx_dim
            assert decoder_input.shape[-1] == expected_dec_dim, (
                f"decoder_input dim {decoder_input.shape[-1]} != expected "
                f"{expected_dec_dim} (forecast {self._dec_forecast_dim} "
                f"+ ctx {self.dec_ctx_dim}). Did training augment with decoder_ctx "
                f"but validation forget to?"
            )
            split_at = decoder_input.shape[-1] - self.dec_ctx_dim
            dec_forecast = decoder_input[..., :split_at]   # (B, dec_days, forecast_dim)
            dec_context  = decoder_input[..., split_at:]   # (B, dec_days, ctx_dim)
            dec = self.dec_forecast_embed(dec_forecast) + self.dec_ctx_embed(dec_context)
            dec = self.pos_enc(self.embed_drop(dec))
        else:
            assert decoder_input.shape[-1] == self._dec_embed_in_dim, \
                f"decoder_input dim {decoder_input.shape[-1]} != expected {self._dec_embed_in_dim}"
            dec = self.pos_enc(self.embed_drop(self.dec_embed(decoder_input)))  # (B, dec_days, d_model)

        # Add learnable spatial embedding (same for all time steps in enc & dec)
        if self.patch_embed is not None and patch_ids is not None:
            spatial = self.patch_embed(patch_ids).unsqueeze(1)  # (B, 1, d_model)
            enc = enc + spatial
            dec = dec + spatial

        memory = self.transformer.encoder(enc)              # (B, enc_days, d_model)
        output = self.transformer.decoder(dec, memory)      # (B, dec_days, d_model)

        return self.proj(output)                            # (B, dec_days, patch_dim_out)
