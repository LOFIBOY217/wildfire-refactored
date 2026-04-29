# Competitive Positioning — Where We Sit in the Wildfire DL Landscape

*Drafted 2026-04-29. Synthesis of 4 rounds of literature research conducted
this session: (1) S2S wildfire DL gap, (2) 4 direct competitor metrics,
(3) main-track wildfire papers verification, (4) cross-venue precise
search. Captures all findings + venue strategy recommendation.*

---

## TL;DR

- **S2S wildfire DL is a sparse but populated field.** 3 direct competitors
  (Kondylatos, TeleViT, FireCastNet) plus 1 close peer (BCWildfire). NOT
  blank.
- **Our genuine novelty axes**: (a) 2 km daily ignition probability across
  ALL of Canada, (b) 14-46 d S2S lead at this resolution, (c)
  polygon-aware novel-ignition metric methodology.
- **Realistic publishing target**: AAAI main / NeurIPS D&B / Scientific
  Reports / NHESS / JAMES — all four are achievable.
- **Out of reach as currently framed**: NeurIPS / ICML / ICLR main track
  unless we add architectural primitive, foundation-model framing, or
  theoretical contribution.
- **Strongest near-term move**: AAAI main 2027 with S2S-lead-time
  differentiator over BCWildfire's 1-day baseline.

---

## 1. The 8 papers we MUST cite

Verified by direct URL access (OpenReview, AAAI ojs, Nature, Copernicus).

| # | Paper | Venue | Year | Why we cite |
|---|---|---|---|---|
| 1 | **WildfireDB** | NeurIPS 2021 D&B | 2021 | Earliest fire D&B benchmark |
| 2 | **WildfireSpreadTS** | NeurIPS 2023 D&B | 2023 | US fire spread time-series |
| 3 | **Mesogeos** | NeurIPS 2023 D&B **Oral** | 2023 | Highest-tier fire DL paper, methodological precedent |
| 4 | **Sim2Real-Fire** | NeurIPS 2024 D&B | 2024 | Multi-modal sim dataset |
| 5 | **Kondylatos 2022** (`arXiv:2211.00534`) | NeurIPS CCAI workshop | 2022 | First S2S DL fire paper |
| 6 | **TeleViT** (`arXiv:2306.10940`) | ICCV 2023 Workshop | 2023 | **Closest architecture to ours**: ViT for S2S fire forecast |
| 7 | **FireCastNet** (`arXiv:2502.01550`) | Scientific Reports | 2025 | Strongest S2S DL fire paper, 6mo lead |
| 8 | **BCWildfire** (`arXiv:2511.17597`) | AAAI 2026 | 2026 | **Our most direct competitor**: Canada, daily, DL |

Plus 2 honorable mentions:
- **Son et al. JAMES 2022** (`10.1029/2022MS002995`) + **Son 2024** (`10.1029/2023MS003710`) — DL fire weather forecasting precedent at JAMES
- **McNorton GRL 2024** (`10.1029/2023GL107929`) — Global Probability-of-Fire forecast, ECMWF

---

## 2. Detailed comparison vs BCWildfire (most direct competitor)

### What they did

| Aspect | BCWildfire (AAAI 2026) |
|---|---|
| Labels | MODIS Active Fire (MOD/MYD14A1), high-confidence |
| Spatial resolution | 1 km |
| Temporal | Daily |
| Region | British Columbia + adjacent (~240 Mha) |
| Channels | **38**: fuel (LAI/FPAR/NDVI/EVI/MODIS bands), met (T/u10/v10/precip/LH/SM 4 layers/LST), terrain, anthro (land use, OSM dist), fire history |
| Train years | 2000-2020 (21y) |
| Val/test | 2021-2022 / 2023-2024 |
| Total samples | 1,015,275 (338K positive, 676K negative; **2:1 negative ratio, 1:1 in test**) |
| Architecture | **6 off-the-shelf 1D time-series models** per-pixel (no spatial structure): SCINet, TSMixer, CrossLinear, **Crossformer**, FEDformer, S_Mamba |
| Params | Not reported |
| Lead time | **1 day only** |
| Input window | Past 10 days |
| Metrics | Precision, Recall, F1, PR-AUC (no ROC-AUC, no Lift, no IoU, no Brier) |
| Best result | Crossformer + PE: **F1=88.70**, **PR-AUC=96.34** |
| Baselines | **None** (no climatology, no persistence, no FWI) |
| Bootstrap CIs | **No** |

### Where we beat BCWildfire

| Axis | Our advantage |
|---|---|
| **Lead time** | **14-46 days** vs their 1 day — totally different problem class |
| **Region scope** | All Canada (~1,250 Mha) vs BC alone (~240 Mha) — 5× area |
| **Label quality** | NBAC perimeters + NFDB ignitions vs MODIS active fire (cloud-limited, misses small/under-cloud fires — **they explicitly admit this limitation**) |
| **Architecture** | Custom S2S patch transformer with encoder/decoder, decoder context, focal loss, hard-neg mining vs off-the-shelf 1D time-series per-pixel |
| **Spatial modeling** | We model spatial structure with patch attention vs their per-pixel independence |
| **Evaluation rigor** | Lift@K + Lift@30km + Cluster Lift + ROC-AUC + PR-AUC + Brier + bootstrap CI vs P/R/F1/PR-AUC only |
| **Realistic deployment** | Lift on natural rare-event base rate (~10⁻⁴) vs their balanced 1:1 test set (60km/3day buffer around positives REMOVED → artificially easy) |
| **Novel-fire skill** | Novel-30d Lift = 6.55× explicitly reported vs they admit "performance limitations remain in predicting **new ignitions and small-scale fires**" |

### Where BCWildfire beats us

| Axis | Their advantage |
|---|---|
| **Spatial resolution** | 1 km vs our 2 km |
| **Channel count** | 38 vs our 9 |
| **Headline F1** | F1=88.70 looks higher than our PR-AUC=0.305 — **but not comparable** because their test is balanced 1:1 with positive buffer removed |
| **Acceptance** | Already at AAAI 2026 main track |

### The honest framing

> Direct comparison of headline numbers is **structurally invalid**:
> different lead time (1d vs 14-46d), different label (MODIS vs NBAC),
> different region (BC vs Canada), different test construction (balanced
> vs natural rare-event). They are a **next-day satellite-detection
> benchmark**; we are a **sub-seasonal NBAC-truth forecaster**. We
> operate in the harder regime and report harder metrics.

---

## 3. AAAI as target — viability assessment

### Why AAAI is realistic
- **BCWildfire just landed at AAAI 2026** → door is proven open for
  Canadian wildfire DL papers
- We have a clear differentiator: **S2S lead (14-46d) vs their 1-day**
- AAAI accepts applied AI papers if framed with deployed-system narrative
- Our evaluation rigor (CIs, multiple baselines, novel-fire metric)
  exceeds their published bar

### Risk of AAAI
- They may not appreciate Lift-style metrics — AAAI reviewers are trained
  on F1/AUC, not domain-specific operational metrics
- "Same country, similar covariates" → reviewers may say "scooped" if we
  don't sharply differentiate
- Single-region papers historically rare at AAAI

### USP framing for AAAI submission
```
Title: "Sub-Seasonal Wildfire Forecasting Across Canada at 2 km Daily
       Resolution: A Patch Transformer Approach with Polygon-Aware
       Evaluation"

Core claims:
1. First DL S2S (14-46d) wildfire model at 2 km daily over all Canada
2. Novel polygon-aware evaluation methodology (novel-ignition Lift)
   exposes label artifact common to all NBAC-style benchmarks
3. Empirical: +49% Lift@30km / +74% Lift@5000 over climatology with
   non-overlapping bootstrap CIs across 604 val windows
4. Robustness to climate-driven extremes: 2023-2024 surge years
   (4.3×/3.2× normal burn) handled OOD with precision INCREASING
   (41%→52%)
```

### AAAI strategy
- Submit by **AAAI 2027 deadline** (typically August 2026) — gives us
  time to finish 12-model scaling table + 13ch/16ch + 22y cluster_lift
  audit + write polished paper
- Compare against BCWildfire on overlapping setup (re-evaluate our
  predictions at 1-day lead on BC region for one ablation)
- Frame as **complementary** to BCWildfire, not competing — they own
  next-day, we own S2S; together cover the operational spectrum

---

## 4. What gets into CVPR / ICML / ICLR main track from Earth-science angle

### 6 Earth-science papers that DID land at top ML main tracks

| Paper | Venue | Year | What got it accepted |
|---|---|---|---|
| ClimaX (2301.10343) | ICML | 2023 | Foundation-model framing + 4 downstream tasks |
| SFNO / Spherical FourCastNet (2306.03838) | ICML | 2023 | New architectural primitive: neural operator on the sphere |
| DYffusion (2306.01984) | NeurIPS | 2023 | New diffusion conditioning scheme for spatiotemporal forecasting |
| ClimODE (2404.10024) | ICLR | 2024 (Oral) | Continuity-equation PDE constraint baked into Neural ODE |
| SatMAE (2207.08051) | NeurIPS | 2022 | New temporal/multi-spectral pretraining trick |
| Stormer (2312.03876) | NeurIPS | 2024 | Randomized lead-time training scheme |

### What does NOT get accepted at main track
- "Apply existing transformer to weather/fire/climate data" — this is the
  modal Earth-science workshop paper
- Regional studies (single country/region) without architectural novelty
- Strong empirical results vs climatology baseline alone

### 6 acceptance patterns (distilled)

1. **New architectural primitive** demoed on weather/Earth (SFNO,
   ClimODE, ConvLSTM)
2. **Foundation model / generalist** with multi-task evaluation (ClimaX,
   SatMAE, Aurora, Stormer)
3. **Benchmark + dataset** → goes to D&B track, not main (ClimateLearn,
   WeatherBench, our work fits here)
4. **Genuine NWP-beating results** → goes to Nature/Science, not ICLR
   (GraphCast, Pangu, NeuralGCM, GenCast)
5. **Physics-informed inductive bias** (ClimODE, Spherical CNNs)
6. **Scale + global + open weights** when paired with one of (1)-(5)

### Where we sit

Our current contribution as framed:
- (1) Architectural primitive: ❌ patch ViT + enc-dec is standard
- (2) Foundation model: ❌ single region, single task
- (3) Benchmark: ⚠️ partial (novel-ignition metric is reusable, but one
  metric isn't a full benchmark)
- (4) Beat operational baseline: ⚠️ need to beat ECMWF S2S, not just
  climatology
- (5) Physics: ❌
- (6) Scale + global: ❌ Canada only, 8.5M params

**Honest verdict: as currently framed, NeurIPS/ICML/ICLR main track is
out of reach.**

### 5 "if we did X, we'd unlock Y" scenarios

| Scenario | Effort | Realism | Unlocks |
|---|---|---|---|
| **A. Foundation-model pivot** (pretrain SeasFire global, finetune Canada+Australia+Med) | 6+ months, 5-10× compute | 4/10 | NeurIPS main (if multi-region transfer works) |
| **B. Polygon-artifact method paper** (generalize novel-ignition to deforestation, floods, oil spills + theoretical bias characterization) | 3-4 months, low compute | **6/10** | NeurIPS main as method paper — **most aligned with our work** |
| **C. Physics-informed angle** (Rothermel PDE in decoder) | 6+ months | 3/10 | ICLR main (mismatch: ignition is point-process, not PDE) |
| **D. Beat ECMWF S2S fire forecast rigorously** | 3 months + ECMWF data access | 5/10 | Nature Comms / PNAS — better than NeurIPS main |
| **E. NeurIPS D&B with novel-ignition benchmark** (full pipeline + multi-baseline + metric as public benchmark) | 2 months, low compute | **8/10** | NeurIPS D&B — high-quality tier, proven path |

### Recommended sequencing

**Year 1 (now → 2026-12)**: Combine B + E.
- Submit polygon-artifact method paper to NeurIPS main 2026 deadline
- Submit benchmark version (data + baselines + metric) to NeurIPS D&B
  same cycle
- AAAI 2027 main track as backup (deadline ~Aug 2026)

**Year 2 (2027 if Year 1 lands)**: Scenario A or D.
- Foundation model pivot OR
- ECMWF S2S head-to-head for Nature Communications

---

## 5. Final venue recommendation

### Primary target (highest realism × highest tier)
**🥇 AAAI 2027 main track** — proven door (BCWildfire 2026), our S2S
differentiator, 6 months to polish. Aim deadline Aug 2026.

### Secondary target (parallel submission)
**🥈 NeurIPS Datasets & Benchmarks 2026** — frame our 22y NBAC+NFDB
pipeline + novel-ignition metric as benchmark. 4 fire papers landed
here including 1 Oral. Best ML-tier shot.

### Tertiary (safety net + journal version)
**🥉 Scientific Reports** (Nature subsidiary) — FireCastNet 2025
precedent. Domain-friendly, transparent reviews.

### Workshop preview (low cost, high visibility)
**CCAI @ NeurIPS 2026** workshop — 4-page abstract, fast feedback,
builds visibility while main paper in review.

### Avoid
- ICML / ICLR / NeurIPS main Conference track (zero wildfire precedent;
  out of reach as currently framed unless we pursue Scenario B)
- CVPR / ECCV main (zero EarthVision wildfire papers)
- IEEE TGRS (couldn't verify any S2S DL fire paper there; long review)
- Fire (MDPI) / Forests (MDPI) — high acceptance but won't move the
  needle for academic placement

---

## 6. Activation criteria — when to upgrade target

### Go for AAAI/NeurIPS D&B if (now):
- ✅ 12-model scaling table complete (D+1)
- ✅ 22y cluster_lift audit done (D+2)
- ✅ At least one channel-extension result (13ch or 16ch) by D+10

### Go for NeurIPS main (Scenario B) if:
- novel-ignition metric works on at least 2 non-fire datasets (deforestation, flood)
- Theoretical characterization of polygon-induced bias written up
- Comparable Lift improvement holds in non-fire setting

### Go for Nature Comms (Scenario D) if:
- Beat ECMWF S2S on Canadian fire (need their forecast access)
- 22y cluster lift > BCWildfire bootstrap CI low-bound
- 13ch/16ch sweep shows incremental gains stack predictably

### Stay at AAAI / D&B / Sci Reports tier if:
- 22y vs 4y scaling shows plateau or regression
- 13ch / 16ch don't add predictive power
- novel-ignition metric doesn't generalize beyond fire labels

---

## Appendix A — Full venue verified count (2020-2025)

| Venue | Track | Verified count | Source |
|---|---|---|---|
| NeurIPS main | D&B | **4** | OpenReview |
| NeurIPS CCAI workshop | Workshop | **14** | climatechange.ai |
| ICML main | Main | **0** | OpenReview |
| ICML CCAI | Workshop | **1** | climatechange.ai |
| ICLR main | Main | **0** | OpenReview |
| ICLR CCAI | Workshop | **5** | climatechange.ai |
| **AAAI main** | Main | **2** (Where there's Smoke 2021, BCWildfire 2026) | ojs.aaai.org |
| CVPR EarthVision | Workshop | **0** | openaccess.thecvf.com |
| ICCV workshops | Workshop | **3** (FireFly, TeleViT 2023; PyroFocus 2025) | openaccess.thecvf.com |
| ECCV main | Main | **0** | search |
| IEEE TGRS | Journal | ⚠️ couldn't verify (anti-scrape) | IEEE Xplore blocked |
| RSE | Journal | ⚠️ couldn't verify (login required) | ScienceDirect blocked |
| AGU JAMES | Journal | **2** (Son 2022, 2024) | agupubs |
| AGU GRL/JGR/EF | Journal | **6** | agupubs |
| NHESS | Journal | **5** | nhess.copernicus.org |
| IJWF (CSIRO) | Journal | **6+** | publish.csiro.au |
| Nature family (Sci Rep, Nat Comms, Sci Data) | Journal | **8+** | nature.com |
| Fire (MDPI) | Journal | **5+** in 2024 alone | mdpi.com |

---

## Appendix B — What we should add to the paper to clinch AAAI

1. **Per-lead breakdown**: Lift@5000 + Lift@30km curve for each lead
   day 14, 21, 28, 35, 42, 46. Compare to TeleViT Fig 2 layout.
2. **AUPRC at coarsened (25 km) resolution**: makes our number
   directly comparable to FireCastNet 0.633 / TeleViT visual.
3. **Per-year breakdown** showing surge-year robustness (2023-2024
   already done for 4y enc14; replicate for 22y).
4. **Cluster lift audit**: p50 (current), p75, p90 + stratified small/
   medium/large bins. Honest reporting of bias.
5. **vs BCWildfire**: subset our 22y model predictions to BC region
   only, evaluate at 1-day lead, compare F1/PR-AUC head-to-head on
   their setup.
6. **Ablations**: 9ch vs 13ch vs 16ch. Encoder length 14/21/28/35.
   Training data 4y/12y/22y. With/without decoder_ctx.
7. **Limitations section**: be explicit about polygon labels, lack of
   ECMWF S2S head-to-head, single-region validation.
