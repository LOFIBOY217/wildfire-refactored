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
#  U-Net
# ----------------------------------------------------------------
class _DoubleConv(nn.Module):
    """(Conv3x3 → BN → ReLU) × 2 — the canonical U-Net block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetBaseline(nn.Module):
    """
    Fully-convolutional U-Net over one 16×16 patch. The 21-day encoder history
    is folded into the channel axis (T×C input channels), so the model is the
    canonical CNN / segmentation family (the burned-area architecture reviewers
    expect), complementing the recurrent ConvLSTM and the attention Transformer.

    Two downsampling stages (16→8→4) with skip connections, then symmetric
    upsampling back to 16×16. Decoder-side input is reduced to a per-patch mean,
    projected, and broadcast-added to the final feature map (same conditioning
    scheme as ConvLSTMBaseline) so the model consumes identical inputs.

    Output: 1 logit per sub-pixel per lead day → (B, decoder_days, P*P).
    Same forward signature as the transformer / ConvLSTM → drop-in.
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
        base_ch: int = 64,
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

        in_ch = encoder_days * n_channels           # time folded into channels
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4

        # Encoder (contracting) path
        self.enc1 = _DoubleConv(in_ch, c1)          # 16×16
        self.enc2 = _DoubleConv(c1, c2)             # 8×8
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _DoubleConv(c2, c3)       # 4×4

        # Decoder (expanding) path with skip connections
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)   # 4→8
        self.dec2 = _DoubleConv(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)   # 8→16
        self.dec1 = _DoubleConv(c1 + c1, c1)

        # Decoder-context conditioning (per-patch mean → broadcast add)
        self.dec_proj = nn.Linear(patch_dim_dec, c1)
        self.drop = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(c1, decoder_days, kernel_size=1)

    def forward(self, encoder_input, decoder_input, patch_ids=None):
        B = encoder_input.shape[0]
        C, P = self.n_channels, self.patch_size
        # (B, T, C*P*P) → (B, T*C, P, P): fold the 21-day history into channels
        x = encoder_input.reshape(B, self.encoder_days * C, P, P).float()

        e1 = self.enc1(x)                    # (B, c1, 16, 16)
        e2 = self.enc2(self.pool(e1))        # (B, c2, 8, 8)
        b = self.bottleneck(self.pool(e2))   # (B, c3, 4, 4)

        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))   # (B, c2, 8, 8)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (B, c1, 16, 16)

        # Broadcast-add decoder context (per-patch mean over lead days)
        dec_mean = decoder_input.mean(dim=1).float()          # (B, dec_dim)
        dec_feat = self.dec_proj(dec_mean).view(B, -1, 1, 1)  # (B, c1, 1, 1)
        d1 = self.drop(d1 + dec_feat)

        out = self.head(d1)                                   # (B, T_dec, P, P)
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
    if model_type == "unet":
        return UNetBaseline(
            patch_dim_enc=patch_dim_enc,
            patch_dim_dec=patch_dim_dec,
            patch_dim_out=patch_dim_out,
            encoder_days=encoder_days,
            decoder_days=decoder_days,
            n_channels=n_channels,
            patch_size=patch_size,
            base_ch=64,
            dropout=dropout,
        )
    raise ValueError(f"Unknown model_type={model_type!r}")
