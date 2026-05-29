# Fig 3 — Decoder Detail Prompt (FINAL, simplified v2)

Companion to Fig 1 + Fig 2. Closes the three-figure architecture set.
Density matched to Fig 2 (~14 text blocks).

## Architecture truth (verified against `src/models/s2s_hotspot.py`)

- Standard PyTorch `nn.Transformer` decoder layers
- Decoder sequence dimension = 33 lead days
- Each patch processed independently
- Cross-attention: `Q = decoder tokens (B·P, 33, 256)`,
                   `K, V = encoder memory of THE SAME patch (B·P, 21, 256)`
- 33 × 21 cross-attention matrix per patch (NOT 33 × P)
- Lead-day positional encoding is sinusoidal (same `PositionalEncoding`
  module as encoder, non-learnable)
- Decoder context = `fire climatology + burn age + day-of-year`
  (NO burn count — that lives only in 12ch_static variant, not in SOTA)
- Sub-pixel head: 256-dim token → 16×16 sub-pixel logits per patch
- Output: `(B·P, 33, 16, 16)` → sigmoid → 2 km probability map,
  33 lead days

## v1 → v2 simplification log

v1 was visually cluttered (~22 text blocks). v2 cuts:
- Step 3 yellow→red gradient bar (kept only the amber-striped card)
- Step 4(a) italic "Each lead is a standalone..." (3 lines → 1 line shape)
- Step 4(b) italic "Each lead-day decides..." (5 lines → 2 lines shape)
- Step 4(a) 33×33 mask icon (deleted — "causal mask" in text suffices)
- Step 4(b) 33×21 heatmap icon (deleted — shape annotation suffices)
- Step 5 three-line block consolidated into one tight three-row card
- Step 4 overall block shrunk from ~50% to ~30% of figure height

Result: ~14 text blocks, matching Fig 2.

## Visual identity

- White canvas, white panel interior
- Rose `#f43f5e` for: panel border, numbered step circles, decoder
  3D cubes (Steps 1 and 6)
- Green `#10b981` used ONCE: small encoder-memory cube anchor next
  to Step 4 — the visual handshake with Fig 2
- Amber `#f59e0b` left-edge stripe marks the embedding-input card (Step 3)
- Dark orange `#ea580c` dashed arrow for cross-attention signal

---

## Prompt (paste verbatim into image2)

```
Generate a detailed Decoder diagram. White background, restrained colour.
LESS IS MORE — cut explanatory italics and icons from the previous
version. Companion to the Encoder figure. This is a paper figure.

================================================================
GLOBAL VISUAL  (same as Fig 2 family)
================================================================
- Canvas background: white (#ffffff).
- Panel interior fill: WHITE. Only the outer border is rose.
- Outer panel border: 2.5 px rounded rectangle, ROSE (#f43f5e).
- Title (bold black, top-centred):
    "Decoder — 33-day forecast window → per-sub-pixel logits"
- NO "Fig 3" badge. NO sidebar.
- Aspect ratio: 4:3 landscape.

ROSE = DECODER IDENTITY:
- Outer panel border (#f43f5e)
- Numbered step circles 1–6 (solid #f43f5e + white digit), ~28 px
- The two decoder 3D tensor cubes (Steps 1 and 6):
    Front #fecdd3, Top #fda4af, Right side #fb7185

GREEN HANDSHAKE — single non-rose cube in the body:
- Small encoder-memory 3D cube next to Step 4 (front #a7f3d0, top
  #6ee7b7, right #34d399). Matches Fig 2 cube colour.

OTHER STYLE:
- Standard step cards: 1.5 px thin BLACK border, white fill.
- Positional card (Step 3): white fill + 3 px AMBER (#f59e0b) left stripe.
- Cross-attention arrow from green cube into Step 4(b): DARK ORANGE
  (#ea580c) dashed 2 px.
- Shape annotations: monospace black.

★★★ SIMPLIFICATION RULES (apply throughout) ★★★
- NO italic "why" sentences inside sub-cards. Keep operation + shape only.
- NO gradient bars / colour scales / decorative time hints.
- NO mask icons, NO heatmap icons inside Step 4 sub-cards. Architecture
  detail is conveyed by text shape annotations, not pictograms.
- All sub-card text is ≤ 2 lines per card.

================================================================
STEPS
================================================================

1 INPUT TENSORS
  Two rose 3D cubes side by side, joined by a small black "concat" glyph.
    (a) "Decoder input  (B, 33 leads, C, 16, 16)"
        small caption: "S2S ECMWF forecast, lead t+14 … t+46"
    (b) "Decoder context  (B, 33, ctx_dim)"
        small caption: "fire clim + burn age + day-of-year"
  Down arrow + annotation: "reshape per patch → (B·P, 33, C·16·16 + ctx_dim)"

2 PROJECTION
  Single card: "Linear → 256-dim token per (lead, patch)"
  Output: "(B·P, 33, 256)"

3 ADD LEAD-DAY POSITIONAL ENCODING
  Single amber-striped white card:
    "Lead-day pos enc — sinusoidal  (33, 256)"
    small italic underneath: "non-learnable; sin/cos buffer"
  Output: "(B·P, 33, 256)"
  ★ NO gradient bar to the right. Just the card.

4 DECODER LAYER STACK   × L = 4
  LEFT ANCHOR (vertically centred next to the L× block):
    Small green 3D encoder-memory cube + label "encoder memory (B·P, 21, 256)"
    Dark-orange dashed arrow exits the cube horizontally and enters
    sub-card (b) only.

  RIGHT MAIN BLOCK titled "Standard Transformer decoder layer × L = 4":
    Three SHORT sub-cards stacked vertically, each ≤ 2 lines:

      (a) Masked self-attn over the 33 leads
          shape: (B·P, 33, 256) → (B·P, 33, 256), causal mask

      (b) Cross-attn — Q from decoder, K/V from encoder memory of SAME patch
          Q: (B·P, 33, 256)   K, V: (B·P, 21, 256)

      (c) MLP — FFN 256 → 1024 → 256, GELU, dropout 0.2

  Right of the block: "× L = 4" thin black bracket.
  Single line under the block (small italic, panel-style not sub-card):
    "residual + LayerNorm after each sub-card"

  ★ No mask icon. No heatmap icon. The architectural fact "Q has 33,
    K/V has 21" is conveyed by the explicit shape annotations in 4(b).

5 SUB-PIXEL HEAD
  Single tight card:
    "MLP head — 256-dim token → 16×16 sub-pixels per (lead, patch)"
    "(B·P, 33, 256)  reshape  (B·P, 33, 16, 16)"
    "33 × (16×16) = 8,448 logits per patch"

6 OUTPUT
  Small rose 3D cube + "Per-patch logits  (B·P, 33, 16, 16)"
  Down arrow → small line-art Canada outline tinted orange-red.
  Caption: "→ sigmoid → 2 km daily fire-probability map, 33 lead days"

================================================================
PANEL-WIDTH FOOTER  (italic grey-black, ≤ 2 lines)
================================================================
  "Cross-attention happens only within a patch's own 21 history tokens.
   No cross-patch coupling — spatial structure recovered at evaluation."

================================================================
STYLE
================================================================
- Density target: ≤ 14 text blocks total in the panel (matches Fig 2).
- The figure should read top-down in under 10 seconds.
- The ONLY pictorial elements are the rose decoder cubes (Steps 1, 6),
  the green encoder-memory cube (Step 4 anchor), and the tiny Canada
  outline (Step 6). No mask grids, no heatmaps, no decorative bars.
```
