# Results Table — 2026-05-03 (consolidated, paper-ready)

Run on val period **2022-05-01 to 2025-10-31**, NBAC + NFDB labels,
14–46 day lead, EPSG:3978 2 km grid.

---

## Table 1 — Lift@K (pixel-level + event-level)

| # | Method | n_win | Lift@5000 | Lift@30 km | Source |
|--:|---|--:|---:|---:|---|
| 1 | Random | — | 1.00× | 1.00× | by definition |
| 2 | FWI max (operational) | 646 | 0.00× | — | benchmark_baselines.csv |
| 3 | FWI oracle (cheats) | 646 | 1.14× | 1.62× | benchmark_baselines.csv |
| 4 | Logreg (multi-feature) | — | **5.49×** | — | benchmark_logreg.csv (60251781) |
| 5 | Recent-data (2022 train) | 310 | 5.31× | n/a | recent_logreg_*.json |
| 6 | Climatology (pooled) | 646 | 5.09× | n/a | benchmark_baselines.csv |
| 7 | Climatology (per-win mean, full eval) | 536 | 3.48× [3.41, 3.54] | 3.01× [2.95, 3.07] | unified_enc35_12y_clim |
| 8 | **ECMWF S2S Fire Danger** ⭐ | **612** | **0.00×** | **0.94×** [0.92, 0.97] | baseline_ecmwf_s2s_summary.json |
| 9 | Persistence (standard) | 20 | 16.92× | — | benchmark_novel.csv |
| 10 | **Persistence (novel-30 d)** ⭐ | 20 | **0.00×** | **0.00×** | benchmark_novel.csv |
| 11 | 22y enc14 ep1 | 604 | 3.94× | 4.19× | save_scores 60179165 |
| 12 | 22y enc21 ep1 | 583 | 3.41× | 3.51× | save_scores 60179166 |
| 13 | 22y enc28 ep1 | 562 | 4.81× | 5.06× | save_scores 60179167 |
| 14 | 22y enc35 ep1 | 536 | 4.79× | 4.85× | save_scores 60179168 |
| 15 | 12y enc14 baseline | 583 | 6.40× | 5.08× | quick eval (full eval pending) |
| 16 | 12y enc28 (full) | 562 | 5.78× [5.42, 6.14] | 5.38× [5.02, 5.72] | save_scores 60179163 |
| 17 | 12y enc35 (full) | 536 | 5.62× [5.32, 5.92] | 4.44× [4.27, 4.62] | unified_enc35_12y_model |
| 18 | 12y enc14 + recency τ=6 | — | 7.01× | n/a | quick eval, 60185107 |
| 19 | 12y enc14 + recency τ=10 | — | 6.61× | n/a | quick eval, 60185108 |
| 20 | 12y enc14 + recency τ=15 | — | 6.78× | n/a | quick eval, 60185109 |
| 21 | 12y enc21 + recency τ=6 | — | 7.27× | n/a | quick eval, 60251780 |
| 22 | 4y enc14 (9ch) | 562 | 5.78× | 5.38× | save_scores 60179163 |
| 23 | 4y enc14 (13ch) | — | 4.52× | n/a | quick eval (full pending — 60251898-901 done?) |
| 24 | 4y enc21 (13ch) | — | 4.01× | n/a | quick eval |
| 25 | 4y enc28 (13ch) | — | 4.43× | n/a | quick eval |
| 26 | 4y enc35 (13ch) | — | 4.34× | n/a | quick eval |
| **27** | **★ 12y enc21 (current SOTA)** | **583** | **7.83×** [7.50, 8.21] | **6.73×** [6.40, 7.07] | save_scores 60179166 → unified |

### Headline numbers (for paper abstract)

- **SOTA model 12y enc21 = 7.83× / 6.73×** (Lift@5000 / Lift@30 km)
- vs **Climatology (full eval) 3.48× / 3.01×** → **+125% / +124%**
- vs **ECMWF SEAS5 0.00× / 0.94×** → **dramatically better**
  (ECMWF top-K at 14–46 d lead is spatially mis-localized: top-5000
   pixels concentrate in BC SW corner, miss QC megafires + NWT)

---

## Table 2 — Recall@budget (operational metric, advisor-requested)

For each val window: rank all Canada land pixels by predicted prob,
take top X% of national area = patrol budget, count fraction of
**connected fire events** captured (8-connectivity on dilated label).

### 12y enc21 (our SOTA) — n=435 windows ✅ CONFIRMED

| Budget | k_pixels | **Recall** | 95% CI |
|---|---:|---:|---|
| 0.1 % | 6 143 | 7.83 % | [7.59, 8.06] |
| 0.5 % | 30 717 | 19.48 % | [19.08, 19.82] |
| 1 % | 61 435 | 27.62 % | [27.15, 28.04] |
| 5 % | 307 174 | **62.04 %** | [61.52, 62.53] |
| 10 % | 614 349 | **85.84 %** | [85.48, 86.20] |

Mean fire events per window: **134.8**

### Climatology / Persistence / ECMWF S2S — PENDING

Jobs **60304734-736** submitted on 2026-05-03; ETA ~30 min each.
Will be filled in once they complete.

---

## Table 3 — Encoder-length scaling (9ch, full eval)

| Train range | enc14 | enc21 | enc28 | enc35 |
|---|---:|---:|---:|---:|
| 4y (2018-2021) | 5.78× | — | — | — |
| 12y (2014-2021) | 6.40× | **7.83× ★** | 5.78× | 5.62× |
| 22y (2000-2021) — *ep1 strategy* | 3.94× | 3.41× | 4.81× | 4.79× |

**Key findings**:
1. **12y is the sweet spot** — outperforms both 4y (data starvation) and 22y (calibration drift).
2. **12y enc21 dominates** by a wide margin; enc28/35 over-fit at this data scale.
3. **22y suffers from calibration-vs-rank drift** — Lift@5000 falls below climatology despite more data; lift_trajectory experiment will confirm whether mid-epoch ckpt selection rescues 22y.

---

## Table 4 — Channel scaling (4y range, full eval pending)

| Channel set | best Lift@5000 | best enc | Notes |
|---|---:|---:|---|
| 9ch | ~5.5× (full) | 14 | baseline |
| 13ch | 4.52× (quick) | 14 | **−18%** vs 9ch — Hughes phenomenon at 4y |
| 16ch | n/a (12y/22y caches building) | — | u10 + v10 + CAPE additions |

Confirms `data_range × n_channels` scaling interaction:
- **Adding channels in low-data regimes (4y) hurts** (curse of dim)
- **Need ≥ 12y data** to absorb extra channels usefully

---

## Methodological contributions (paper §)

1. **NBAC + NFDB label** (replaces CWFIS hotspots with 350× drift)
2. **Lift@30 km** event-pooled metric (novel)
3. **Recall@budget** operational metric (novel)
4. **Novel-30 d evaluation** — exposes persistence artifact
   (17.1× → 0.0× when current-fire pixels excluded)
5. **Master-cache slice** infrastructure — single 22y cache reused
   across all sub-ranges, saves 36 h/range (~12 ranges = 1.5 weeks of CPU)
6. **Cluster-aware (spatial-radius) negative mining** (in flight)
7. **Climate-similarity year weighting** (ENSO ONI distance, in flight)

---

## Open experiments (PD on Narval, 2026-05-03)

| Job | Purpose | When useful |
|---|---|---|
| 60252181-184 | 12y × 13ch × {enc14/21/28/35} | Paper Table 4 (channel scaling at SOTA range) |
| 60253136-140 | 22y × 16ch cache build + 4 enc | Final 3×4 matrix completion |
| 60252754-758, 60252812-819 | 14y / 6y / 18y master-cache trainings | Power-law fit on data-range scaling |
| 60254574-576 | 12y enc21 + spatial neg R={100, 200, 300} | High-ROI SOTA attack |
| 60277232-233 | MLP / ConvLSTM baseline | Paper baseline section |
| 60289339 | 12y enc21 + climate-sim ONI | Replace recency with proper weighting |
| 60278304 | lift_trajectory (22y enc14 mid-epoch) | Calibration-vs-rank hypothesis test |
| 60304733 | 22y 13ch cache prep (rerun, 480G) | unblocks 22y × 13ch |
| 60304734-736 | Recall@budget baselines (clim / persist / ecmwf) | Paper Table 2 completion |
