"""
Forward-shape and differentiability tests for the main model
S2SHotspotTransformer (src/models/s2s_hotspot.py).

Baselines are covered by test_baselines_shape.py; this file covers the
Transformer itself, including the conv-stem patch embed and the conv output
head variants that are the paper's SOTA configurations.

Run: python -m pytest tests/test_model_forward.py -v
"""
import torch

from src.models.s2s_hotspot import S2SHotspotTransformer

# Small but representative shapes (9-channel, 21-day encoder, 33-day decoder).
B, T_ENC, T_DEC, C, P = 2, 21, 33, 9, 16
ENC_DIM = C * P * P   # 2304
DEC_DIM = 12          # s2s_legacy-sized decoder input
OUT_DIM = P * P       # 256 (one logit per pixel)


def _make(**kw):
    """Tiny model so the tests run fast on CPU."""
    return S2SHotspotTransformer(
        patch_dim_enc=ENC_DIM, patch_dim_dec=DEC_DIM, patch_dim_out=OUT_DIM,
        d_model=64, nhead=4, num_encoder_layers=1, num_decoder_layers=1,
        dim_feedforward=128, dropout=0.0,
        encoder_days=T_ENC, decoder_days=T_DEC, patch_size=P, **kw,
    )


def _forward(model):
    enc = torch.randn(B, T_ENC, ENC_DIM)
    dec = torch.randn(B, T_DEC, DEC_DIM)
    return model(enc, dec)


def test_forward_flatten_shape():
    out = _forward(_make().eval())
    assert out.shape == (B, T_DEC, OUT_DIM)


def test_forward_conv_stem_shape():
    out = _forward(_make(enc_conv_stem=True).eval())
    assert out.shape == (B, T_DEC, OUT_DIM)


def test_forward_conv_output_head_shape():
    out = _forward(_make(conv_output_head=True).eval())
    assert out.shape == (B, T_DEC, OUT_DIM)


def test_forward_sota_config_shape():
    """conv-stem + conv output head together (the SOTA config)."""
    out = _forward(_make(enc_conv_stem=True, conv_output_head=True).eval())
    assert out.shape == (B, T_DEC, OUT_DIM)


def test_forward_with_patch_ids():
    n_patches = 50
    model = _make(n_patches=n_patches).eval()
    enc = torch.randn(B, T_ENC, ENC_DIM)
    dec = torch.randn(B, T_DEC, DEC_DIM)
    patch_ids = torch.randint(0, n_patches, (B,))
    out = model(enc, dec, patch_ids)
    assert out.shape == (B, T_DEC, OUT_DIM)


def test_forward_output_finite_and_non_degenerate():
    out = _forward(_make().eval())
    assert torch.isfinite(out).all()
    # Raw logits (no sigmoid) should carry some spread, not a constant map.
    assert out.std().item() > 0.0


def test_forward_differentiable():
    model = _make()
    enc = torch.randn(B, T_ENC, ENC_DIM)
    dec = torch.randn(B, T_DEC, DEC_DIM)
    model(enc, dec).sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(
        g is not None and torch.isfinite(g).all() for g in grads
    )


def test_forward_patch_batch_independence():
    """Each patch is an independent sequence: row i of the output must depend
    only on row i of the input (eval mode, dropout=0)."""
    model = _make().eval()
    enc = torch.randn(B, T_ENC, ENC_DIM)
    dec = torch.randn(B, T_DEC, DEC_DIM)
    out_full = model(enc, dec)
    out_first = model(enc[:1], dec[:1])
    assert torch.allclose(out_full[:1], out_first, atol=1e-4)
