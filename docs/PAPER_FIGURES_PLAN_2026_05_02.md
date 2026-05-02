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
| 3 | Patchification | ⏳ Refining (full prompt sent) | n/a | ⏳ |
| 4 | Fire label construction | ⏳ Refining (full prompt sent, frame has placeholder) | ⏳ TODO (replace placeholder with real raster crop) | ⏳ |
| 5 | Standard vs Novel-30 d evaluation | ⏳ Refining (full prompt sent) | ⏳ TODO (real polygon shapes) | ⏳ |
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

**Status**: AI frame in refinement (full prompt sent v2 with strict
6 × 6 grid + scientific colormaps + isometric stack rules).

### Python overlay TODOs
- **None** if AI frame v2+ uses real-looking colormaps. AI is allowed to
  draw the rasters here because they're illustrative crops, not paper-grade
  data. Reviewer will not check whether the FWI raster is "really
  southern BC".
- **Optional**: if reviewer cares, generate one real 96 × 96 px crop
  from FWI 2018-07-15 (peak fire season) at southern BC, plot 4 channels
  with the named colormaps, save as a single composite PNG, drop into
  Panel 1 of the AI frame.
- Source: `data/fwi_data/fwi_20180715.tif`,
  `data/era5_on_fwi_grid/...` etc.

---

## Figure 4 — Fire label construction

**Status**: AI frame in refinement (full prompt sent — explicitly asks
for placeholder dashed rectangle in Stage 5 output panel, no map of
Canada anywhere in the flow).

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

**Status**: AI frame in refinement (full prompt sent — strict
identical-base-map between Panel A and B, 3 polygons in same positions,
clean USGS-style topographic look not satellite).

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
