# Figure interpretations

Paper body text for each results figure. Plain prose for pasting into the manuscript.
No hyphens and no colons by house style. Figure numbering starts at Figure 4 (the
in distribution total lift) and increases by one for every figure that follows.

Sequence
- Figure 4  fig_lift_multik_287_total_macro   287 window in distribution, total fire lift across budgets
- Figure 5  fig_lift_multik_287_novel_micro   287 window in distribution, novel ignition lift across budgets
- Figure 6  fig_metrics_287                    287 window validation across F2, MCC, Brier skill score, and PR AUC
- Figure 7  fig_lift30km_287                    287 window cluster scale lift at 30 km, total and novel
- Figure 8  fig_lift_multik_2026_stacked        2026 out of sample lift, total on top and novel on the bottom

---

## Figure 4

Image: figures/fig_lift_multik_287_total_macro.png

Interpretation

Figure 4 ranks the nine systems by lift on the in distribution test at three operational budgets, the highest ranked one thousand, five thousand, and ten thousand cells. The FWI oracle is the most instructive entry. It is handed the true fire weather index over the target window, so it represents the ceiling of what a fire danger index can offer, and still its lift stays near zero at every budget. Fire weather marks when conditions are dangerous across a broad region, but it does not resolve which cell within that region actually burns. The index that operational agencies rely on most therefore carries almost no information for the ranking task, even when its future values are known exactly.

The remaining baselines do better by reading the landscape rather than the weather alone. Climatology reaches a lift close to six by concentrating the budget on places that have burned before, a real signal because fire recurs in the same boreal terrain from one year to the next. The multilayer perceptron and the ConvLSTM sit in the fives, comfortably above chance yet short of the leaders. Our patch transformer models hold the top, with the fully convolutional head and the novel ignition loss near eight and a half and the plain conv stem near seven. The margin between these models and climatology is the skill drawn from the current atmosphere rather than from the static record of where fire usually occurs, and that margin is the operationally meaningful quantity, since a climatological map is free while the fires that matter are the ones that break from the historical pattern.

Persistence is the exception and demands separate comment. Its lift reaches roughly eighteen and is clipped so the other systems stay legible, yet it does no more than repeat the fire already burning at issue time. This dominance says two things. Canadian fires are severe and long lived, so a fire active at issue time is usually still active four to six weeks later, and merely repeating the present is rewarded on the total target. At the same time, a baseline that only copies the present should never top a forecasting comparison, which shows that the total target credits continuation rather than genuine prediction. This is why Figure 5 turns to novel ignition, where persistence has no active fire to copy and falls to zero. Across all three budgets the ordering barely changes, so no single cutoff drives the result.

---

## Figure 5

Image: figures/fig_lift_multik_287_novel_micro.png

Interpretation

Figure 5 repeats the comparison on the novel ignition target, which keeps only fires that were not already burning in the thirty days before issue time. This is the operationally demanding task, since it asks each system to find new fires rather than to track continuations. The effect on the winner of the total target is immediate. Persistence, which reached roughly eighteen in Figure 4, falls to zero, because a system that copies the present has nothing to copy once the ongoing fires are removed. The FWI oracle stays at zero as well. The two entries that looked strongest or most authoritative on the total target thus contribute nothing to the prediction of new ignitions.

Our patch transformer models lead on this harder target. The variant trained with the novel ignition loss is the strongest at every budget, approaching eight at the tightest cutoff, and the fully convolutional head follows close behind. Climatology remains a serious baseline near five and a half, because many new fires still start where fire is historically likely, yet our models sit clearly above it. That gap is again the dynamic skill that a static map cannot supply, and it is the skill that matters most here, since a new ignition by definition departs from what was already burning.

The lift in Figure 5 is aggregated as a fire weighted mean, so that each ignition counts once rather than each forecast window counting once. Under this aggregation the multilayer perceptron sits in the middle of the field and the remaining learned baselines fall below it, which places our models ahead on the task that carries operational value. As in Figure 4 the three budgets tell the same story, so the ranking does not rest on a single choice of cutoff.

---

## Figure 6

Image: figures/fig_metrics_287.png

Interpretation

Figure 6 evaluates all nine systems on the four metrics that the machine learning and meteorology communities use most often, which are F2, the Matthews correlation coefficient, the Brier skill score, and the area under the precision recall curve. We report them so that the evaluation is legible to readers who expect these numbers, and the pattern they produce is itself worth reading.

On the three ranking metrics, persistence sits at or near the top and the learned systems, ours included, fall into a narrow band beneath or beside it. This repeats the degeneracy seen on the total target. A system that copies the fire already burning scores well whenever the target is dominated by continuation, and none of these three metrics isolate the harder question of where new fire appears. The conventional ranking metrics therefore do not separate the systems cleanly on this problem, which is exactly why the earlier figures lead with lift and with the novel ignition target.

The Brier skill score behaves differently because it rewards calibrated probability rather than ordering. Only the fully convolutional head and the multilayer perceptron exceed zero, at about 0.05 and 0.06, while the novel ignition variant and the other systems are overconfident and fall below zero, and climatology is the least calibrated of all. The FWI oracle has no score here since it is not a probability. Read together, Figure 6 shows that our systems are competitive on the familiar metrics and that one of them is genuinely well calibrated, while it also shows that these metrics blend continuation with prediction, which is the reason the rest of the results rely on the operational ranking view.

---

## Figure 7

Image: figures/fig_lift30km_287.png

Interpretation

The pixel scale lift in Figures 4 and 5 asks each system to place the single two kilometre cell that burns inside a tight budget, but that demand is stricter than the labels themselves support. The fire targets are formed by dilating each ignition to a radius close to thirty kilometres, so the exact cell is already uncertain at that scale, and a system that identifies the correct neighbourhood while missing the exact cell is penalised as though it had failed. Figure 7 removes that unfair penalty by scoring the same forecasts at a coarser resolution.

To build the cluster scale we pool the two kilometre map into thirty kilometre cells, taking the highest predicted probability inside each cell and marking a cell as fire if any pixel within it burned, then computing lift on the coarse map exactly as before. The choice of thirty kilometres is not arbitrary, since it matches the spatial uncertainty that the dilation already writes into the target, and it also matches the scale at which an agency actually responds, because resources are pre positioned by region rather than by individual cell. A forecast that points to the right district while missing the exact cell now receives the credit it deserves.

Figure 7 shows that the ranking is stable across scale. Our models lead on both the total and the novel target at thirty kilometres, just as they do at the pixel scale, so the advantage is not an artefact of one resolution. Persistence again tops the total target and collapses on the novel target, which means the cluster view inherits the same diagnostic. Climatology closes part of the gap here because getting the region right is easier than getting the cell right, yet it remains below our models on both targets. Reading the pixel and cluster figures together gives a more robust picture than either alone, since a system must rank well at the exact cell and at the operational neighbourhood in order to lead across both.

---

## Figure 8

Image: figures/fig_lift_multik_2026_stacked.png

Interpretation

Figure 8 tests every system on the 2026 fire season, which sits well beyond the training data, with the total target on top and the novel target on the bottom. The models are fitted on years through 2022, and the annual burned area products that define our targets end in 2024, so 2026 can only be scored against the current season records that agencies release in near real time. The forecast has to extrapolate across several years and into a season the model has never seen. This is partly a limitation of the data, since no completed record closes the gap, but it turns that limitation into the sharpest available measure of generalisation, because a system is now judged purely on how well what it learned transfers to conditions it was never shown.

The season also moves the problem in space, since early spring fire concentrates in the west rather than in the central boreal region that dominates the training fire, so a system must generalise in time and in place at the same time. Generalisation is exactly what separates the systems here. The conv stem model leads on the total target at roughly eight and a half and on the novel target at roughly seven and a half, almost the level it reached in distribution, so its generalisation gap is small. The fully convolutional head, which was the strongest system in distribution, falls by nearly half, and the multilayer perceptron drops further still, so their apparent strength was largely fitted to the training season and does not transfer. A model that memorises the training distribution, or that pins probability to the exact historical cell, has little to offer once the distribution moves, while a model that reads the current atmosphere and reasons about space generalises.

Read against Figures 4 and 5, Figure 8 carries the central message of the paper. Skill in distribution is not the same as the ability to generalise, and the ordering that looks settled on the familiar season is rearranged the moment the season moves far from the training years. What an operational forecast actually needs is generalisation rather than a high score on the season it was trained near, and our model is the one that keeps its lead on both targets and across every budget when the test demands it. The fires of next year will not resemble the fires the model learned from, and Figure 8 shows that our model is built to handle exactly that.
