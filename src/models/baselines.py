"""Drop-in baselines for the S2SHotspotTransformer.

Both classes share the exact forward signature
(encoder_input, decoder_input, patch_ids) → (B, decoder_days, patch_dim_out)
so they can replace the transformer in train_v3.py without touching the
training loop, eval code, loss, or data pipeline.

Models:
  - MLPBaseline      — per-patch MLP, flattens encoder + decoder input,
                       2 hidden layers, predicts 33 × 256 logits per patch.
                       Answers: "is the transformer overkill?"
  - ConvLSTMBaseline — 2-layer ConvLSTM2D over the 21-day encoder history,
                       1×1 conv head to 33 lead-day logits per sub-pixel.
                       Answers: "is attention better than recurrent?"
                       This is the standard sacred baseline in geo DL
                       (FireCastNet, ClimateBench, DeepCube all compare).
"""
from __future__ import annotations
import torch
import torch.nn as nn


# ----------------------------------------------------------------
#  MLP
# ----------------------------------------------------------------
class MLPBaseline(nn.Module):
    """
    Per-patch MLP. Flattens (encoder + decoder) input → 2 hidden → output.

    Args mirror S2SHotspotTransformer for drop-in replacement.
    """

    def __init__(
        self,
        patch_dim_enc: int,
        patch_dim_dec: int,
        patch_dim_out: int,
        encoder_days: int,
        decoder_days: int,
        d_model: int = 512,
        dropout: float = 0.2,
        **_unused,
    ):
        super().__init__()
        in_dim = encoder_days * patch_dim_enc + decoder_days * patch_dim_dec
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, decoder_days * patch_dim_out),
        )
        self.decoder_days = decoder_days
        self.patch_dim_out = patch_dim_out
        self.encoder_days = encoder_days

    def forward(self, encoder_input, decoder_input, patch_ids=None):
        B = encoder_input.shape[0]
        x = torch.cat(
            [encoder_input.reshape(B, -1), decoder_input.reshape(B, -1)],
            dim=-1,
        )
        out = self.net(x)
        return out.reshape(B, self.decoder_days, self.patch_dim_out)


# ----------------------------------------------------------------
#  ConvLSTM
# ----------------------------------------------------------------
class ConvLSTMCell(nn.Module):
    """Standard ConvLSTM2D cell (Shi et al. 2015)."""

    def __init__(self, in_ch: int, hidden_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(
            in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=pad
        )
        self.hidden_ch = hidden_ch

    def forward(self, x, state):
        h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

    def init_state(self, B, H, W, device, dtype):
        return (
            torch.zeros(B, self.hidden_ch, H, W, device=device, dtype=dtype),
            torch.zeros(B, self.hidden_ch, H, W, device=device, dtype=dtype),
        )


class ConvLSTMBaseline(nn.Module):
    """
    2-layer ConvLSTM over the 21-day encoder history; 1×1 Conv head to
    decoder_days × (patch_size²) sub-pixel logits per patch.

    Decoder-side input is reduced to a per-patch mean and concatenated
    with the final hidden state via 1×1 Conv. This keeps the model
    minimal but still consumes the same inputs as the transformer.
    """

    def __init__(
        self,
        patch_dim_enc: int,
        patch_dim_dec: int,
        patch_dim_out: int,
        encoder_days: int,
        decoder_days: int,
        n_channels: int,
        patch_size: int = 16,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        **_unused,
    ):
        super().__init__()
        assert patch_dim_enc == n_channels * patch_size * patch_size, (
            f"patch_dim_enc={patch_dim_enc} != n_channels*P*P="
            f"{n_channels * patch_size * patch_size}; check --channels and "
            f"--patch_size."
        )
        assert patch_dim_out == patch_size * patch_size, (
            f"patch_dim_out={patch_dim_out} != P*P={patch_size * patch_size}"
        )
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.encoder_days = encoder_days
        self.decoder_days = decoder_days
        self.patch_dim_dec = patch_dim_dec
        self.hidden_dim = hidden_dim

        self.cell1 = ConvLSTMCell(n_channels, hidden_dim, kernel_size=3)
        self.cell2 = ConvLSTMCell(hidden_dim, hidden_dim, kernel_size=3)
        self.dropout = nn.Dropout2d(dropout)

        # Project per-patch decoder mean to a hidden_dim × P × P feature
        # map, broadcast spatially, then add to ConvLSTM final hidden.
        self.dec_proj = nn.Linear(patch_dim_dec, hidden_dim)

        # Head: hidden_dim → decoder_days channels (1 logit per sub-pixel
        # per lead day).
        self.head = nn.Conv2d(hidden_dim, decoder_days, kernel_size=1)

    def forward(self, encoder_input, decoder_input, patch_ids=None):
        B, T, _ = encoder_input.shape
        C, P = self.n_channels, self.patch_size
        x_seq = encoder_input.reshape(B, T, C, P, P).float()

        h1, c1 = self.cell1.init_state(B, P, P, encoder_input.device, x_seq.dtype)
        h2, c2 = self.cell2.init_state(B, P, P, encoder_input.device, x_seq.dtype)

        for t in range(T):
            h1, c1 = self.cell1(x_seq[:, t], (h1, c1))
            h2, c2 = self.cell2(h1, (h2, c2))

        # Decoder: per-patch mean over decoder days → linear → broadcast
        dec_mean = decoder_input.mean(dim=1).float()              # (B, dec_dim)
        dec_feat = self.dec_proj(dec_mean)                         # (B, hidden_dim)
        dec_feat = dec_feat.view(B, self.hidden_dim, 1, 1)
        h2 = h2 + dec_feat                                         # broadcast add
        h2 = self.dropout(h2)

        out = self.head(h2)                                        # (B, T_dec, P, P)
        return out.reshape(B, self.decoder_days, P * P)


# ----------------------------------------------------------------
#  Factory
# ----------------------------------------------------------------
def build_baseline(
    model_type: str,
    *,
    patch_dim_enc: int,
    patch_dim_dec: int,
    patch_dim_out: int,
    encoder_days: int,
    decoder_days: int,
    n_channels: int,
    patch_size: int,
    d_model: int = 256,
    dropout: float = 0.2,
):
    """Construct an MLP or ConvLSTM baseline. Raises if model_type unknown."""
    if model_type == "mlp":
        return MLPBaseline(
            patch_dim_enc=patch_dim_enc,
            patch_dim_dec=patch_dim_dec,
            patch_dim_out=patch_dim_out,
            encoder_days=encoder_days,
            decoder_days=decoder_days,
            d_model=d_model * 2,
            dropout=dropout,
        )
    if model_type == "convlstm":
        return ConvLSTMBaseline(
            patch_dim_enc=patch_dim_enc,
            patch_dim_dec=patch_dim_dec,
            patch_dim_out=patch_dim_out,
            encoder_days=encoder_days,
            decoder_days=decoder_days,
            n_channels=n_channels,
            patch_size=patch_size,
            hidden_dim=64,
            dropout=dropout,
        )
    raise ValueError(f"Unknown model_type={model_type!r}")
