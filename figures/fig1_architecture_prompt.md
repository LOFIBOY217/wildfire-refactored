# Fig 1 — Architecture Overview Prompt (FINAL v4)

Used to generate the main paper architecture figure.
After 13 cumulative corrections across v1→v4.

## Cumulative correction log

1. Channel stack merged (no separate "static priors")
2. burn_count removed everywhere
3. Cross-attn arrow + "encoder memory (same patch)" label
4. Focal `α=0.25, γ=2`
5. Canada map "Illustrative single-day forecast, 14-day lead" caption
6. MLP head "33 × (16×16) = 8,448" explicit
7. "time pos enc (sinusoidal, 21 d)" — not learnable
8. "lead-day pos enc (sinusoidal, 33 steps)"
9. Solid filled number circles, white digits
10. Cross-attn arrow dark-orange dashed
11. "Per-patch model: no cross-patch attention" footer
12. Output shape `(B·P, 21, 256)`
13. Patchify 3D tensor cubes filled (blue / red)

## Architecture truth (verified against `src/models/s2s_hotspot.py`)

- Standard PyTorch `nn.Transformer` (line 151)
- Encoder sequence = 21 days, decoder sequence = 33 leads
- Each patch processed independently (B·P, T, 256)
- Spatial info via learnable `nn.Embedding(n_patches, d_model)` only
- Time / lead positional encoding is sinusoidal (`PositionalEncoding`, lines 28-40)
- No factorized attention, no cross-patch attention
- Encoder memory shape `(B·P, 21, 256)`
- Cross-attention queries the 21 history tokens of THE SAME patch

---

## Prompt (paste verbatim into image2)

```
Generate a clean architecture diagram in a minimalist white-and-line-art
style. This is Figure 1 of a paper.

================================================================
GLOBAL VISUAL
================================================================
- Background: white (#ffffff). NO pastel fills on panels.
- All body text: black (#000000), clean sans-serif (Inter / Helvetica).
- Panel borders: 2.5 px rounded rectangles, no fill, coloured outlines:
    • blue  (#2563eb) for panels 1, 2, 3, 6
    • red   (#dc2626) for panels 4, 5
- Numbered circles at each panel header:
    SOLID FILLED circles in the panel's border colour
    (blue for 1/2/3/6, red for 4/5), with the digit in WHITE inside.
    Diameter ~36 px. Place just left of the panel header text.
- Inner sub-blocks: 1.5 px thin BLACK borders, white fill.
- Icons (attention heatmaps, mask triangles, weather glyph, flame glyph,
  Canada outline): line-art, thin black outlines, white interior.
- Arrows:
    • black solid 1.5 px arrow                →  data flow
    • DARK ORANGE (#ea580c) DASHED 2 px arrow →  cross-attention signal
    • red dotted 1.5 px arrow                 →  loss / supervision
- Title: bold black, two lines, top-centred.
- Aspect ratio: 16:9 landscape.

PANEL 2 tensor cubes (★ KEY VISUAL ANCHOR):
- Encoder tensor cube: blue 3D filled (#bfdbfe front, #93c5fd top,
  #60a5fa right side)
- Decoder tensor cube: red 3D filled (#fecaca front, #fca5a5 top,
  #f87171 right side)

ARCHITECTURE TRUTH IN TEXT:
- Encoder: "time pos enc (sinusoidal, 21 d)" + "spatial pos embed
  (learnable, per patch)" + "N × [time self-attn over 21 history days
  + MLP]" + "Output: encoder memory (B·P, 21, 256)"
- Encoder footer (italic): "Per-patch model: no cross-patch attention.
  Spatial info enters via learnable patch embedding."
- Decoder: "+ lead-day pos enc (sinusoidal, 33 steps)" + "L × [masked
  self-attn over leads → cross-attn to encoder memory of same patch
  → MLP]" + "MLP head: 33 leads × (16×16 sub-pixels) = 8,448 logits
  per patch"
- Cross-attn arrow from Panel 3 → Panel 4 labelled "encoder memory
  (same patch)" in dark orange (#ea580c) dashed.

PANEL 1 content:
- (a) Channel stack (9/13/16): meteorology + climatological priors +
  terrain. Example list: FWI, 2t, 2d, tcw, sm20, fire_clim, burn_age,
  population, slope.
- (b) Labels: NBAC + NFDB fire polygons, 33-day window starting at t+14.

PANEL 2 content:
- 16×16 patches (~32 km tile)
- Two filled 3D cubes (see above)
- Decoder context: fire climatology + burn age + day-of-year
  (NO burn count)

PANEL 5 content:
- Canada heatmap, log scale 10⁻⁴ → 1
- Italic: "Illustrative single-day forecast, 14-day lead"
- Bullets: Lift@30 km event-level; Recall@budget top 1/5/10%;
  Leak-free baselines pre-window climatology; Application 2–6 week
  risk surface for fire agencies

PANEL 6 content (single line):
"Focal BCE (α=0.25, γ=2) · 1:20 pos:neg · 50% hard-neg · AdamW ·
cosine LR · batch 4096 · 4 epochs · dropout 0.2"
```
