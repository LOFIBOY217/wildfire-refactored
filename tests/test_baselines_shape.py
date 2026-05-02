"""Verify baseline models produce output shape compatible with the
transformer's training / eval pipeline:
  forward(encoder_input, decoder_input) -> (B, decoder_days, patch_dim_out)
"""
import torch
from src.models.baselines import build_baseline


def test_mlp_shape():
    B, T_enc, T_dec, C, P = 4, 21, 33, 9, 16
    enc_dim = C * P * P            # 2304
    dec_dim = 12                    # arbitrary (s2s_legacy + ctx)
    out_dim = P * P                 # 256
    model = build_baseline(
        "mlp",
        patch_dim_enc=enc_dim, patch_dim_dec=dec_dim, patch_dim_out=out_dim,
        encoder_days=T_enc, decoder_days=T_dec,
        n_channels=C, patch_size=P,
    )
    enc = torch.randn(B, T_enc, enc_dim)
    dec = torch.randn(B, T_dec, dec_dim)
    out = model(enc, dec)
    assert out.shape == (B, T_dec, out_dim), (
        f"MLP output shape {out.shape} != expected {(B, T_dec, out_dim)}"
    )


def test_convlstm_shape():
    B, T_enc, T_dec, C, P = 4, 21, 33, 9, 16
    enc_dim = C * P * P
    dec_dim = 12
    out_dim = P * P
    model = build_baseline(
        "convlstm",
        patch_dim_enc=enc_dim, patch_dim_dec=dec_dim, patch_dim_out=out_dim,
        encoder_days=T_enc, decoder_days=T_dec,
        n_channels=C, patch_size=P,
    )
    enc = torch.randn(B, T_enc, enc_dim)
    dec = torch.randn(B, T_dec, dec_dim)
    out = model(enc, dec)
    assert out.shape == (B, T_dec, out_dim), (
        f"ConvLSTM output shape {out.shape} != expected {(B, T_dec, out_dim)}"
    )


def test_mlp_param_count_reasonable():
    B, T_enc, T_dec, C, P = 4, 21, 33, 9, 16
    enc_dim = C * P * P
    dec_dim = 12
    out_dim = P * P
    model = build_baseline(
        "mlp",
        patch_dim_enc=enc_dim, patch_dim_dec=dec_dim, patch_dim_out=out_dim,
        encoder_days=T_enc, decoder_days=T_dec,
        n_channels=C, patch_size=P,
    )
    n = sum(p.numel() for p in model.parameters())
    # ~50M params is fine for a flatten-MLP at this scale; warn if absurd
    assert 1e6 < n < 200e6, f"MLP params={n:,} unexpected"


def test_convlstm_param_count_reasonable():
    B, T_enc, T_dec, C, P = 4, 21, 33, 9, 16
    enc_dim = C * P * P
    dec_dim = 12
    out_dim = P * P
    model = build_baseline(
        "convlstm",
        patch_dim_enc=enc_dim, patch_dim_dec=dec_dim, patch_dim_out=out_dim,
        encoder_days=T_enc, decoder_days=T_dec,
        n_channels=C, patch_size=P,
    )
    n = sum(p.numel() for p in model.parameters())
    # ConvLSTM should be ~100K-500K
    assert 50_000 < n < 5_000_000, f"ConvLSTM params={n:,} unexpected"


if __name__ == "__main__":
    test_mlp_shape()
    test_convlstm_shape()
    test_mlp_param_count_reasonable()
    test_convlstm_param_count_reasonable()
    print("OK — baselines shape + param count verified")
