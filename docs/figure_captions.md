# Figure captions and prose drafts

English, AAAI-style, NO em-dashes. One section per figure. Numbers are the
real full-window leak-free NBAC+NFDB results (see results/eval/).

---

## Figure 5: Forward-chaining LOYO robustness
Script: `scripts/plot_fig5_loyo_robustness.py`. Data: `results/eval/loyo/`.

**Caption.**
Figure 5. Forward-chaining leave-one-year-out (LOYO) robustness. Each lower
row is one held-out fire season (val 2020 to 2024); the dot is that fold's
full-window Lift after training on 2014 to the prior year. The top row
(MACRO) shows the per-fold mean with its standard deviation (diamond and
thick bar) and the pooled-window 95 percent bootstrap interval (thin bar).
The dashed vertical line marks the single chronological-split value reported
elsewhere in the paper. Left: pixel scale (Lift@5000). Right: cluster scale
(Lift@30km). On both scales the macro mean coincides with the single-split
line, showing that the single split is not optimistically biased. The low
2023 fold reflects the anomalous record Quebec fire season.

---

## Section 5.3 Evaluation: robustness rewrite

A single chronological train and validation split (train 2014 to 2021,
validate 2022 to 2024) risks an optimistic estimate if the chosen validation
period happens to be easy. To test this we run forward-chaining
leave-one-year-out (LOYO) cross-validation. For each target year Y from 2020
to 2024 we train on every season from 2014 to year Y minus one and evaluate
only year Y's fire season at full window. Training therefore stays strictly
in the past of every evaluation, so no future information leaks backward, and
the procedure yields five independent estimates of skill on five different
fire seasons.

We report two aggregates. The macro mean weights each year equally and
summarizes year-to-year stability. The micro estimate pools all per-window
scores and attaches a 1000-sample bootstrap confidence interval. The two
agree closely: the macro Lift@5000 is 8.05 plus or minus 2.10 and the macro
Lift@30km is 7.21 plus or minus 1.84, while the micro estimates are 8.05
[7.04, 9.05] and 7.21 [6.32, 8.19] respectively. The aggregation was computed
two ways independently and matches to four decimal places.

The central finding is that the single-split value sits inside the LOYO
spread rather than above it. The single-split full evaluation gives
Lift@5000 = 7.83 and Lift@30km = 6.73, both slightly below the corresponding
LOYO macro means and well within the bootstrap intervals. The single split
therefore does not overstate skill, and if anything it is marginally
conservative.

Per-year results also locate the model's weakest case. Skill is high and
stable in 2020 to 2022 (Lift@5000 of 8.54, 10.38 and 9.07) and in 2024
(7.47), but drops in 2023 (4.80). 2023 was Canada's record fire season,
dominated by very large early-season Quebec fires whose ignition locations
were poorly anticipated by the climatological prior and the meteorological
precursors the model relies on. This single hard year, not a systematic
bias, accounts for most of the macro spread, and the same 2023 dip appears
in every figure in which that year is broken out.
