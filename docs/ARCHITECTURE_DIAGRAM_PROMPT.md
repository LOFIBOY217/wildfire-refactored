# Architecture Diagram Prompt (image2 / GPT-image)

> **Future self: this is THE file for the model architecture-diagram prompt. When asked to
> (re)draw the architecture, come here first. Update this file when the SOTA architecture
> changes.** Single-model conv-stem SOTA. Do NOT add an ensemble stage (ensemble is a
> product/eval detail, not architecture).

## How to generate
Use the full §A prompt (text-only) at `quality="high"`, wide 16:9. Expect 2-3 iterations to clean
up dense labels. (User does NOT want image-edit mode.) §A is CODE-ALIGNED as of 2026-07-18 —
see the "Code-aligned corrections" note below for what was fixed vs the earlier figure.

## What it depicts (current SOTA, 2026-07-18)
conv-stem Patch S2S Transformer, encoder–decoder, ~7.8M params. 5-panel horizontal figure:
Inputs → Patchification → Encoder (Conv2D stem) → Decoder (cross-attn) → Output/eval.
Verified against `src/models/s2s_hotspot.py` + `src/training/train_v3.py` on Narval:
- Encoder self-attention is TEMPORAL-only (over 21 days); each patch is an independent sequence,
  NO cross-patch spatial attention. Spatial structure comes from the Conv2D stem.
- Positional encoding is SINUSOIDAL (not learned). Learned spatial patch-embedding is OFF in the
  conv-stem SOTA (`use_patch_embed=False`).
- Decoder self-attention over lead days is UNMASKED (no causal/tgt_mask); 33 leads predicted in
  parallel. Decoder cross-attends to encoder memory.
- Output head `nn.Linear(d_model, P²=256)` → per lead day 256 logits → 33×256 = 8,448 per patch.
- Inputs: FWI is the ONLY fire-weather index (FFMC/DMC/DC/BUI dropped as redundant). 9ch =
  FWI,2t,fire_clim,2d,tcw,sm20,population,slope,burn_age; 16ch adds wind/precip/deep_soil/etc.

### Code-aligned corrections (fixed 2026-07-18 vs the earlier flatten-era figure)
① Encoder "factorized time-then-space attention" → temporal-only, no cross-patch attention.
② Decoder "masked/causal self-attention" + triangular matrix → unmasked, parallel (full grid).
③ "learned time/lead-day/spatial position embeddings" → sinusoidal PE; spatial embed disabled.
④ Panel-1 "FWI, FFMC, DMC, DC, ISI, BUI" → "FWI" only.
Also: title-case "Patchification"; dropped "batch 4096, 4 epochs" from the training caption.
NOTE: the earlier reference PNG still contains errors ①–④ — do NOT reproduce it; use §A.

---

## §A — FULL TEXT PROMPT (paste into image2, quality="high", wide 16:9)

A clean, flat-vector scientific infographic for an ML paper, NOT a photo. White background, five
color-coded rounded-rectangle panels in a horizontal row connected left-to-right by thick black
arrows, each panel with a small numbered circular badge and a bold colored title. Restrained
palette: panel 1 blue, panel 2 amber/yellow, panel 3 green, panel 4 salmon/red, panel 5 purple.
Crisp thin strokes, small flat icons, clean uppercase-and-sentence-case sans-serif labels, subtle
shadows, generous white space. Rendered at high resolution with sharp legible small text.

TOP TITLE (bold, centered, black): "Patch Transformer for Subseasonal Wildfire Forecasting — 14 to 46 day lead, 2 km grid, Canada"

PANEL 1 — badge "1", title "Inputs" (blue): a stack of about five offset semi-transparent
Canada-shaped raster map layers tinted blue-to-red (heatmaps). Below, four labeled input groups
each with a small icon:
  - flame icon "fire-weather index: FWI"
  - cloud icon "ERA5 weather: temperature, humidity, total column water, soil moisture"
  - mountain icon "static priors: leak-free fire climatology, burn age, population, slope"
  - database icon "labels: NBAC polygons + NFDB points, 33-day window at +14 days"
  Footer in blue: "9 to 16 channels, all on a 2 km Canada grid (EPSG:3978)".

PANEL 2 — badge "2", title "Patchification" (amber): heading "Split 2 km Canada grid into 16x16
patches (≈ 32 km each)". A Canada map overlaid with a coarse grid, one cell zoomed out with a dashed
line to a small 4x4 grid square. Below, two small 3D tensor cubes side by side: a blue cube labeled
"encoder tensor: 21-day history" with a vertical "time" arrow and base label "21", and a yellow
cube labeled "decoder tensor: 33 lead days" with a "time" arrow and base label "33". An amber
sub-box at the bottom: "decoder context: climatology + burn age/count + day-of-year".

PANEL 3 — badge "3", title "Encoder" (green): "Input patch (16x16 × C)" with a small multicolored
16x16 pixel patch thumbnail, arrow down to a green box "Conv2D stem: 16x16 patch to 256-d token",
arrow down to a short row of small token squares labeled "256-d", a green dashed box "+ sinusoidal
temporal positional encoding", a green box "temporal self-attention over 21 days + MLP (× N); no
cross-patch attention", and a green box "encoder memory (patches, 256), used as keys and values".

PANEL 4 — badge "4", title "Decoder" (salmon): a box "project lead patch + context to 256-d", a
dashed box "+ sinusoidal lead-day positional encoding (33 steps)", "self-attention over lead days
(unmasked, parallel)" above a red FULL (fully filled, non-triangular) square attention grid with
vertical axis "lead day", horizontal axis "lead day", and ticks "1 ... 33". Below: box
"cross-attention to encoder memory (L times)", box "MLP head", then bold text "33 leads x 256
sub-pixels = 8,448 logits per patch".

PANEL 5 — badge "5", title "Output and evaluation" (purple): heading "daily probability map, 14 to
46 day lead". A Canada-shaped map with an orange-to-dark-red wildfire probability heatmap and a
horizontal logarithmic colorbar labeled from "10^-6" to "1" captioned "wildfire ignition
probability". Below, three small callout rows with icons: "Lift@30 km — event-scale skill",
"Recall@budget — fires caught in top 1 to 10%", "leak-free baselines — trained strictly
pre-validation". Purple footer: "2 to 6 weeks earlier than 1 to 10 day fire-weather guidance".

BOTTOM: a long red dashed arrow curving from panel 5 back under to panel 1, with centered red
caption: "training: focal loss (gamma = 2), 1:20 positive:negative sampling, 50% hard-negative
mining, AdamW".

Constraints: reproduce ALL the quoted text accurately and legibly, keep the five-panel color
coding, no extra panels, no watermark, no logos, no photographic elements. Everything flat vector,
one horizontal band of five panels with the title on top and the training arrow underneath.

---

## §B — REFERENCE-IMAGE EDIT PROMPT (preferred; attach the reference PNG as Image 1)

Image 1 is a reference architecture figure. Recreate it as a clean, high-resolution flat-vector
infographic. Keep the exact same five color-coded panels (Inputs, Patchify, Encoder, Decoder,
Output and evaluation), the same layout, the same numbered badges, the same icons, the same maps
and tensor cubes and the causal attention triangle, and reproduce ALL text labels verbatim and
legibly. Do not add, remove, or reword any panel or label. Only improve rendering sharpness, text
legibility, alignment, and spacing. Keep the top title and the bottom red training-caption arrow
identical.

---

## Iteration notes (single-change follow-ups)
- Dense text garbled: "regenerate at higher resolution; keep layout identical; make all small labels sharper and correctly spelled."
- Panel too cramped: "give panel 3 more vertical room and enlarge its box text, keep everything else the same."
- Colors off: "use panel colors blue/amber/green/salmon/purple in that left-to-right order, keep all text."

> NOTE: SINGLE-MODEL ONLY (conv-stem). The reference figure's encoder already shows the Conv2D
> stem, so it is conv-stem-correct. Do NOT add an ensemble stage.
