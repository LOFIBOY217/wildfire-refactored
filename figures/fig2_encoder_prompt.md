# Fig 2 — Encoder Detail Prompt (FINAL)

Companion to Fig 1. After 14 cumulative corrections including spatial-
embed removal.

## Architecture truth (verified against `src/models/s2s_hotspot.py`)

- Standard PyTorch `nn.Transformer` (line 151), batch_first=True
- Encoder sequence dimension = 21 history days
- Each patch processed independently; "B" in the transformer is B·P
- Time positional encoding is sinusoidal (`PositionalEncoding`, lines 28-40,
  `register_buffer` — non-learnable)
- ★ `--use_patch_embed` defaults to False; SOTA baseline does NOT use
  learnable spatial embedding. Only auto-enabled for clim_blend / gating
  variants
- No factorized attention, no cross-patch attention
- Encoder output `memory` shape: `(B·P, 21, 256)`

## Visual identity

- White canvas, white panel interior
- Green `#10b981` used ONLY for: panel border, numbered step circles
  (with white digit), and the two 3D tensor cubes (Steps 1 and 5)
- Amber `#f59e0b` left-edge stripe marks the embedding-input card (Step 3)
- Grey attention-matrix icon, no green inside
- Dark orange `#ea580c` dashed arrow exits Step 5 toward "→ Decoder"

---

## Prompt (paste verbatim into image2)

```
Generate a detailed Encoder diagram. White background, restrained colour
(green used only as encoder identity, not as panel wash). This is a
paper figure.

================================================================
GLOBAL VISUAL
================================================================
- Canvas background: white (#ffffff).
- Panel interior fill: WHITE.  Only the outer border is green.
- Outer panel border: 2.5 px rounded rectangle, GREEN (#10b981).
- All body text: black (#000000), clean sans-serif (Inter / Helvetica).
- Title (bold black, top-centred):
    "Encoder — 21-day history → per-patch memory"
- NO "Fig 2" badge anywhere. NO sidebar.
- Aspect ratio: 4:3 landscape.

GREEN IS USED ONLY FOR "ENCODER IDENTITY" ELEMENTS:
- Outer panel border (#10b981)
- Numbered step circles 1–5 (solid #10b981 + white digit), ~28 px
- The two 3D tensor cubes (Steps 1 and 5):
    Front #a7f3d0, Top #6ee7b7, Right side #34d399
    Black 1.5 px edges, thin grid lines on front face.
Everything else black-on-white.

OTHER STYLE:
- Standard step cards: 1.5 px thin BLACK border, white fill.
- Positional signal card (Step 3): white fill + 3 px AMBER (#f59e0b)
  left-edge stripe.
- Attention matrix icon (Step 4a): 21×21 grid, very faintly grey-tinted
  (#f3f4f6), 0.5 px black grid lines. No green.
- Arrows: black solid 1.5 px for data flow.
- Cross-figure arrow exiting Step 5: DARK ORANGE (#ea580c) dashed 2 px.

================================================================
STEPS  (top-to-bottom)
================================================================

1 INPUT TENSOR
  Green 3D cube + "Encoder input (B, 21 days, C, 16, 16)" +
  "C = 9/13/16 channels · 16×16 patch ≈ 32 km tile / 21 days of
  meteorology ending at issue date t".
  Reshape annotation: "(B·P, 21, C·16·16)".

2 CONV2D STEM
  "Conv2D stem → 256-dim token per (patch, day)" / "3×3 conv + GELU +
  flatten".  Output "(B·P, 21, 256)".

3 ADD TIME POSITIONAL ENCODING   ★ no spatial embed
  Amber-striped card: "Time pos enc — sinusoidal (21, 256)" /
  "non-learnable; sin/cos buffer (Vaswani et al.)".
  Output "(B·P, 21, 256)".

4 TRANSFORMER ENCODER STACK × N = 4
  Outer block "Standard Transformer encoder × N = 4".
  (a) TIME SELF-ATTENTION: "MHA, 8 heads, full attention over 21
      history tokens (one independent sequence per patch)".
      21×21 grey-tinted full attention matrix icon.
  (b) MLP: "FFN: 256 → 1024 → 256, GELU, dropout 0.2".
  "residual + LayerNorm after each sub-card" footer.
  ★ No space-attention sub-card.

5 OUTPUT MEMORY
  Smaller green 3D cube + "Encoder memory (B·P, 21, 256)" /
  "21 history tokens per patch, ready for decoder cross-attention".
  Dark orange dashed arrow → "→ Decoder (next figure)".

PANEL-WIDTH FOOTER (italic grey-black):
  "Per-patch model: each patch is anonymous to the transformer — no
   spatial embedding, no cross-patch attention. Spatial coupling enters
   only at evaluation (Lift@30 km pooling)."
```
