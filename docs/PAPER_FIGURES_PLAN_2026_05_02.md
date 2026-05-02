# Paper Figures — AI Frame + Python Overlay Pipeline (2026-05-02)

This file tracks every paper figure being produced via the **hybrid
"AI-frame + Python-overlay" workflow**:

1. **AI (image2)** generates the figure frame: layout, colour scheme,
   typography, legends, captions, decorative elements, base maps. AI is
   GOOD at: aesthetics, NeurIPS-style layout, consistent colours, icons.
   AI is BAD at: real geographic shapes, exact numbers, real data
   distributions. Final image2 prompts are stored verbatim per figure
   below.

2. **Python (matplotlib + cartopy + geopandas + rasterio)** renders the
   real data overlays as transparent PNGs and composites them back into
   the AI-generated frame. Python is GOOD at: EPSG:3978 projection,
   real NBAC polygons, real hotspot positions, exact statistics. Each
   overlay is saved as a separate transparent PNG so the frame can be
   regenerated independently.

3. **Compositor**: a small Python script (Pillow / cairosvg) overlays
   the Python PNGs onto the image2 PNG at known anchor coordinates.

This document is the single source of truth for: (a) which figures need
which overlays, (b) what real data each overlay needs, (c) which
overlays are done / pending. **Append new figures at the bottom — do
not edit existing entries once "FROZEN" is marked.**

---

## Status table

| # | Figure name | AI frame status | Python overlay status | Final composite status |
|---|-------------|-----------------|------------------------|------------------------|
| 1 | Three-tier evaluation protocol | ✅ APPROVED v3 (2026-05-02) | ⏳ TODO | ⏳ |
| 2 | S2S timeline (14–46 day lead) | ✅ APPROVED v1 | n/a (no real-data overlay needed) | ✅ |
| 3 | Patchification | ✅ APPROVED v2 (2026-05-02) — correct 4 colormaps + 6×6 grid + token vector | n/a (illustrative crops, no real-data overlay needed) | ✅ |
| 4 | Fire label construction | ✅ APPROVED v2 (2026-05-02) — frame has clean placeholder, no fake geography | ⏳ TODO (render real NBAC+NFDB raster crop into placeholder) | ⏳ |
| 5 | Standard vs Novel-30 d evaluation | ✅ APPROVED v2 (2026-05-02) — clean USGS-style base, A/B base map pixel-identical, polygons consistent | ⏳ TODO (replace illustrative polygons with real central-BC NBAC + ignitions) | ⏳ |
| 6 | Study area (Canada) | ⏳ AI v1 had bad geography | ⏳ TODO (entire Python rebuild planned) | ⏳ |
| 7 | Architecture diagram | ✅ APPROVED v1 | n/a | ✅ |

---

## Figure 1 — Novel three-tier evaluation protocol

**Status**: AI frame v3 APPROVED 2026-05-02 (9.5/10). Three Python data
overlays pending.

### What's good in the AI frame
- Three-column layout, colour-coded headers (blue/green/red), top borders
- Canada outline pixel-identical across columns; provinces labelled
- 5 orange flame icons in identical positions across all three columns
  (BC interior / AB foothills / S Saskatchewan / S NWT / central QC)
- Panel 1 red dot density balanced with Panel 2/3 visual weight
- Panel 2 inset (30 km pooling example) has solid border + scale bar
  inside
- Panel 3 callouts colour-matched to contour intensity (dark/medium/light
  red) with leader lines
- Footer "To our knowledge..." is a quiet grey footnote
- Province codes (YT/NT/NU/BC/AB/SK/MB/ON/QC/NL) all visible

### Python overlay TODOs (replace AI-drawn overlays with real data)

**Overlay 1.A — Panel 1 red dots (real top-K = 5 000 predicted-risk pixels)**
- Input data: `outputs/window_scores_full/v3_9ch_enc21_12y_2014/*.npz`
  (current SOTA, 583 windows). Pick representative window e.g.
  `window_NNNN_2023-07-15.npz` (peak BC fire season).
- Logic: load score map, take top-5000 pixels by predicted prob, plot as
  red dots on EPSG:3978 Canada base.
- Output: `figures/overlays/three_tier_panel1_topK_pixels.png`
  (transparent background, 1600 × 1000 px, EPSG:3978 alignment-matched
  to AI frame map area).

**Overlay 1.B — Panel 2 pooled cells (30 km coarsened top-K)**
- Same source data as 1.A.
- Logic: 15 × 15 max-pool the score map (15 px × 2 km/px = 30 km),
  then take top-K cells by pooled prob. Render as soft-edged red squares.
- Output: `figures/overlays/three_tier_panel2_pooled_cells.png`.

**Overlay 1.C — Panel 3 risk-budget contours (real model quantiles)**
- Same source. Compute quantile thresholds at 99 % / 95 % / 90 % of all
  Canada land pixels (top 1 % / 5 % / 10 % budget).
- Plot 3 nested contour lines using `matplotlib.pyplot.contour`.
- Output: `figures/overlays/three_tier_panel3_budget_contours.png`.

**Overlay 1.D — Real flame icon positions (ground-truth fire events)**
- Input: NBAC polygons + NFDB ignitions for the SAME window date.
- Logic: pick the 5–6 largest active fires that day, plot flame icons at
  their centroids. Use same 5 positions across all three panels (so the
  reader can compare overlays directly).
- Output: `figures/overlays/three_tier_panelABC_flame_icons.png`.

### Composite step
- Use Pillow. AI frame PNG → measure pixel coordinates of each Canada
  base map → resize each overlay PNG to those coordinates → alpha-blend.
- Anchor coordinates TBD once AI frame export resolution is fixed
  (target: 1600 × 700 final).

---

## Figure 2 — S2S timeline (14–46 day lead)

**Status**: AI frame v1 APPROVED. **No Python overlay needed** — pure
conceptual figure with no real data.

Composited and ready as-is.

---

## Figure 3 — Patchification (raster → patch tokens)

**Status**: AI frame v2 APPROVED 2026-05-02 (9.5/10). 4 scientific
colormaps render as expected (YlOrRd / RdBu_r / YlGnBu / Purples),
6 × 6 patch grid is exact, "patch i" red border consistent between
Panel 2 and Panel 3, all numbers preserved.

### What's good in the AI frame
- Panel 1: isometric 4-layer stack with correct colormaps per channel;
  "4 channels (C)" curly brace below
- Panel 2: clean 6 × 6 dashed grid (NOT 5 × 5), exact 16 px cells,
  red-bordered "patch i" highlight, tick labels 0/16/32/48/64/80/96 on
  both axes
- Panel 3: same red-bordered patch on the left → Linear Conv2D arrow →
  256-dim soft-green token column on the right; "× ~24 000 patches per
  timestep" subtitle correct
- Panel 4: 2 visible rows with vertical "..." between, 21 days curly
  brace on the right, "~24 000 patch tokens per day" arrow below
- Numbered badges 1/2/3/4 colour-coded (blue/orange/green/grey)

### Minor issue
- Panel 2 X-axis tick "16" is positioned slightly right of the first
  dashed line. Cosmetic; reviewer unlikely to notice.

### Python overlay
- **No overlay needed**. The 4 rasters in Panel 1 are illustrative
  crops representing concepts (FWI / ERA5 / soil moisture / fire
  climatology) — reviewer will not check whether they correspond to a
  real southern-BC location on a specific date. AI frame is final.
- (Skipped: optional Python real-raster crop. Not worth the effort
  given current frame quality.)

---

## Figure 4 — Fire label construction

**Status**: AI frame v2 APPROVED 2026-05-02 (9.5/10). Stage 5 output
panel correctly contains a dashed placeholder with text
"[Insert real NBAC + NFDB binary raster — Python-rendered]".
No fake Canada / US / lake geography anywhere in the figure.

### What's good in the AI frame
- 4 stages (1: blue, 2: orange, 3: green, 4: purple) + 5: output panel
  with thicker dark border, all rounded rectangles, numbered circle
  badges
- Stage 1 sub-panels: clean polygon outline (NBAC) + filled red dot
  (NFDB)
- Right-side callout "Replaces CWFIS hotspot labels..." with bold blue
  border + dashed blue connector from Stage 1
- "Adopted 2026-04-21" subtitle, "350×" preserved, "AG_SDATE..AG_EDATE"
  preserved exactly
- Stage 5 placeholder is clearly marked dashed rectangle, italic text,
  ready to be replaced by Python overlay
- Two-row "1 (fire) / 0 (no fire)" legend already in place

### Python overlay TODOs

**Overlay 4.A — Real binary label raster crop**
- Input: `data/fire_labels/fire_labels_nbac_nfdb_2000-05-01_2025-12-21_2281x2709_r14.npy`
  (the canonical NBAC + NFDB binary stack, dilated by 14 px).
- Logic: pick a peak-fire-day in 2023 (e.g. 2023-07-15 — record QC fire
  season), crop a representative region (e.g. western Canada full extent
  or central BC zoom). Render binary raster (red = 1, beige = 0) on
  EPSG:3978 Canada base with thin grey province lines.
- Output: `figures/overlays/fire_label_stage5_real_raster.png`
  (sized to fit the AI frame's dashed placeholder area).

### Composite step
- Drop the PNG into the AI-frame's placeholder rectangle, alpha = 1.0
  (full opaque, replaces placeholder text).

---

## Figure 5 — Standard vs Novel-30 d evaluation

**Status**: AI frame v2 APPROVED 2026-05-02 (9.5/10). Panel A/B base
maps are pixel-identical; clean USGS-style cartography (no green
vegetation tint, no satellite look); 3 polygon shapes consistent
between A (solid red + orange halo) and B (diagonal hatched);
"17.1×" / "0.0×" preserved exactly in the bottom bar chart.

### What's good in the AI frame
- USGS-style base: light beige land, thin blue rivers, faint grey
  contour lines, NO satellite imagery, NO green vegetation
- A and B share IDENTICAL base map (same hill texture, same rivers,
  same lake shapes) — only the overlay differs
- 3 polygon shapes are pixel-replicated between A and B (solid in A,
  hatched in B)
- Centre divider with red callout arrow + "Same model, same data — only
  the evaluation definition changes" works as visual anchor
- Bottom bar chart: Y-axis labelled, gridlines at 0×/10×/20×, grey bar
  reaches 17×, red ✗ on the right side
- Scale bar "0–25–50 km" identical in both panels
- "First time this distinction has been reported..." footnote in quiet
  grey italic

### Minor issue
- The red ✗ in the right mini-chart sits at ~Y = 3× instead of exactly
  Y = 0×. Visual nit; reviewer unlikely to notice. If touching up, ask
  image2 to "move the red ✗ down so its baseline sits exactly on the
  Y = 0× gridline".

### Python overlay TODOs

**Overlay 5.A — Real central-BC fire polygons + dilated halos**
- Input: NBAC polygons in central BC for a peak fire day in 2023.
  Pick 3 spatially-distinct fires (e.g. ~50–100 km apart, each 5–20 km
  diameter).
- Logic: render 3 polygons in two styles for the two panels.
  - Panel A: deep red filled (#c32525 80 % alpha) + soft orange halo
    around (representing 14–46 d future fire window, dilated by ~30 px).
  - Panel B: same 3 polygon shapes, rendered as diagonal black hatching
    on white fill (representing the "excluded — past 30 d" mask).
- Add 5 small orange dots scattered in the white space (representing
  novel ignitions at locations OUTSIDE the hatched polygons).
- Both panels share the same base crop with EPSG:3978 alignment, same
  rivers, same contour lines.
- Output:
  - `figures/overlays/novel30d_panelA_fire_polygons.png`
  - `figures/overlays/novel30d_panelB_hatched_polygons.png`

### Composite step
- Drop 5.A into Panel A map slot.
- Drop 5.B into Panel B map slot.
- AI frame keeps the bottom bar chart ("17.1× vs 0.0×") and the centre
  red callout arrow as-is.

---

## Figure 6 — Study area (Canada)

**Status**: AI v1 had multiple geographic errors (wrong projection,
wrong ecozone shapes, fake hotspot positions). **Plan: full Python
rebuild — no AI frame.**

### Python rebuild plan

Single matplotlib + cartopy figure:
- Main panel (left, ~70 % width): full Canada at EPSG:3978
- Right inset stack (3 panels):
  - (a) Validation 2022–2025 NBAC polygons (semi-transparent red fill)
  - (b) NBAC + NFDB ignitions 2000–2024 (orange dots)
  - (c) Ecozones reference (categorical colourmap from Natural Resources
    Canada ecozone shapefile)
- Province boundaries (Statistics Canada cartographic boundary file)
- Scale bar (0–500–1000 km), north arrow, EPSG:3978 caption
- Title: "Study area: Canada, 2 km grid (EPSG:3978, 2 709 × 2 281 px)"

### Required data files (all in repo or freely downloadable)
- `data/hotspot/hotspot_2000_2025.csv` — 10.83M rows
- NBAC shapefile (already used by `src/data_ops/processing/rasterize_burn_polygons.py`)
- Statistics Canada provincial boundary shapefile
- Natural Resources Canada ecozone shapefile

### Output
- `figures/figure6_study_area.pdf` (vector)
- `figures/figure6_study_area.png` (raster, 300 dpi)

No AI involvement in this one — geography is too important to risk.

---

## Figure 7 — Architecture diagram

**Status**: AI frame v1 APPROVED. No Python overlay needed.

Composited and ready as-is.

---

## Open tasks (when resuming this work)

1. **Run image2 v2 prompts for Figures 3, 4, 5** with the full prompts
   already sent (stored in chat 2026-05-02). Iterate until each frame
   matches the spec.
2. **Write `scripts/render_paper_figure_overlays.py`** — single CLI
   script that produces every transparent PNG overlay listed above.
   Args: `--figure {1,4,5} --window_date YYYY-MM-DD --out_dir figures/overlays/`.
   Reuses existing data loaders from `src/training/train_v3.py` and
   `src/evaluation/benchmark_baselines.py`.
3. **Write `scripts/composite_paper_figures.py`** — Pillow script that
   takes the AI frame PNG + the overlay PNGs, alpha-blends them at
   pre-measured anchor coordinates, exports final figure as PDF + PNG.
4. **Build Figure 6 standalone** in pure Python — see plan above.

## Conventions

- All figure PNGs land in `figures/`.
- Overlays land in `figures/overlays/`.
- Final composites land in `figures/final/` as both PDF (vector) + PNG
  (300 dpi raster).
- Use `figures/CITATIONS.md` (not yet created) to record any external
  data source / shapefile / colormap that needs attribution in paper
  acknowledgements.
- Naming: `figure{N}_{slug}.{pdf,png}`, e.g. `figure1_three_tier_eval.pdf`.
