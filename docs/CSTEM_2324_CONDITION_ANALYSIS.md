# conv-stem 2023/2024 Condition Analysis — where it wins vs loses

**Date:** 2026-07-18
**Model:** conv-stem Patch S2S Transformer (9ch, enc21), single-model SOTA.
**Test set:** 2023-05-01 → 2024-10-31 held-out, 287 rolling windows (lead 14–46d).
**Method:** `analyze_cstem_2324.py` — joins per-window eval JSON
(`outputs/eval_convstem_2324_per_window.json`, which already carries lift/base_rate/n_fire per
window) with SPATIAL target features computed from the daily NBAC+NFDB r14 fire-label stack
(`data/fire_labels/fire_labels_nbac_nfdb_2014-...-r14.npy`, 4252×2281×2709).
Raw table: `outputs/cstem_2324_condition_analysis.csv`.

## Headline
**conv-stem is a NEW-FIRE / EARLY-SEASON / SPREAD-OUT-FIRE specialist.**
It wins when fires are novel ignitions scattered across new places (early season), and loses when
fires are dense, persistent, already-burning megafires (late-season 2023 West).

## Overall numbers (Lift@5000)
- 2023: **6.38**  |  2024: **7.91**  |  combined **7.17**.
- 2023 had higher base_rate (0.062 vs 0.050) and LOWER novel-fire fraction (0.342 vs 0.436).

## Strongest driver: novelty
Correlation of window Lift@5000 with each condition (n=287):
- **novel_frac  +0.354**  ← strongest positive. conv-stem shines on fire NOT burning in prior 30 days.
- lon_spread   +0.279   → better when fire is geographically dispersed.
- c_lon        +0.229   → better when fire centroid is further east (less negative lon).
- base_rate / n_fire / fire_px  ≈ **−0.28**  → worse when fire is dense/abundant.
- bc_frac      −0.217, west_frac −0.176 → worse (at pixel scale) when fire is in the West.
- month        −0.208   → worse later in the season.

## ★ Key nuance: pixel-scale vs cluster-scale disagree on the West
Lift@5000 (pixel) and Lift@30km (cluster) have OPPOSITE signs on West / late-season:

| condition   | corr with Lift@5000 | corr with Lift@30km |
|-------------|--------------------:|--------------------:|
| west_frac   |              −0.176 |             **+0.229** |
| bc_frac     |              −0.217 |             +0.222 |
| month       |              −0.208 |             **+0.375** |
| novel_frac  |              +0.354 |             −0.206 |

Interpretation: **in the West, conv-stem gets the REGION/cluster right but misses the exact pixel.**
So "misses the West" is really "right macro-region, imprecise pixel." West-fraction tercile:
- low-West:  Lift@5000 **7.84**, Lift@30km 6.10
- high-West: Lift@5000 **6.49**, Lift@30km **6.79**  (pixel drops, cluster rises)

## By month
| month | Lift@5000 | Lift@30km | base_rate | west_frac |
|------:|----------:|----------:|----------:|----------:|
| 5 | 7.88 | 5.86 | 0.037 | 0.44 |
| 6 | **8.21** | 6.33 | 0.065 | 0.58 |
| 7 | 6.63 | 6.18 | 0.072 | 0.64 |
| 8 | **6.14** | 6.15 | 0.057 | 0.70 |
| 9 | 7.24 | **8.42** | 0.041 | 0.72 |

June best at pixel scale; August worst; September best at cluster scale.

## Best 12 windows (highest Lift@5000)
Almost all **2024 May–June**, novel_frac ≈ **0.78**, high lon_spread (~24–25).
- 2024-05-19 → **17.48**, 2024-05-20 → 16.87, 2024-05-21 → 16.34 ... (all novel_frac ~0.78–0.80).
- One 2023 entry: 2023-05-02 (11.37, novel_frac 0.76) — early-season, novel.

## Worst 12 windows (lowest Lift@5000)
**All 2023, almost all late Aug–Sept**, west_frac ≈ **0.77**, novel_frac ≈ **0.03–0.09**.
- 2023-09-14 → **3.75**, 2023-09-13 → 3.90, 2023-08-31 → 3.96 ...
These are the 2023 record-year persistent western megafires: fire already burning for weeks
(near-zero novelty), dense, in BC/West.

## Why (mechanism)
conv-stem's convolutional patch-embed does NOT memorize position → it predicts new ignitions from
the meteo signal (its real skill) but cannot simply "copy where it's already burning." For
persistent old fires (novel~0.03) the answer is essentially "where it burned last month," which a
position-memorizing model (flatten) copies well → that's why flatten beats conv-stem in 2023, and
conv-stem beats flatten in 2024/2026.

## Implications
1. conv-stem's weakness (persistent old fire) is partly **benign**: you don't need a 14–46 day
   forecast for a fire already burning — current fire state suffices. Its strength (novel
   early-season ignitions) is the hard, operationally valuable case.
2. This **motivates the rank-mean ensemble** ([[ensemble_rank_mean_2026]]): flatten covers the
   persistent-fire regime, conv-stem covers the novel-fire regime → ensemble beats both.
3. Candidate fixes to lift the weak regime: add a fresher "recent/active fire" channel (burn_age
   exists but may be too coarse), or up-weight novel-fire in the loss, so conv-stem degrades less
   on dense persistent-fire windows.

## Pending
- Full-epoch conv-stem 287-window test (job 65884186, GPU queued behind 2t_anom) → confirm it
  matches the 7.17 combined number.
