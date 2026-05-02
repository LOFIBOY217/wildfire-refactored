# Calibration-vs-Rank Tradeoff Hypothesis (research log)

*Drafted 2026-05-01. Records the hypothesis, evidence, literature
review, and the experiment designed to test it.*

---

## 1. Observation that triggered this

After the fire_patched cache bug was fixed, we re-trained 4 model
configurations and observed:

| Range × Enc14 | Ep1 | Ep2 | Ep3 | Ep4 | Pattern |
|---|---|---|---|---|---|
| 4y default     | **6.69** | 4.94 | 4.96 | 4.91 | declines after ep1 |
| 4y strongreg   | 6.51 | **2.53** | (timeout) | -     | declines harder with more reg |
| 12y default    | 4.93 | 6.51 | 6.29 | **6.53** | climbs through epochs |
| 22y default    | **5.73** | 5.16 | 5.10 | 4.31 | declines after ep1 |

Across all configurations:
- **train_loss decreases monotonically** (model is learning)
- **ROC-AUC is stable at 0.83-0.91** (global ranking unaffected)
- **Brier score improves** (calibration is improving)
- **Lift@5000 declines** in 4y/22y (top-K rank degrades)

This rules out:
- Classic overfit (would show train_loss ↓ + val_loss ↑)
- Insufficient regularization (stronger reg made it worse)
- LR schedule mismatch (cosine_lower experiment already disconfirmed)
- Cache bug (label-consistency audit passes)

---

## 2. The hypothesis

**Lift@K rewards prediction sharpness; calibration training pushes
predictions toward smoothness; tradeoff makes top-K rank degrade
even as global rank stays stable.**

### Concrete mechanism
- focal loss + cosine LR + many SGD updates → model learns to predict
  probabilities closer to the true marginal rate (~0.5%).
- Calibrated predictions are smoother — high-risk patches get prob
  ~0.03-0.05 instead of 0.95.
- Smoother predictions hurt Lift@5000: the top-5000 contains more
  marginal cases, so precision drops.
- Smoother predictions don't hurt ROC-AUC: global rank order is
  preserved, only the magnitude of differences shrinks.

### Why 12y is exempt
The peak val_lift happens around 5,000-10,000 SGD updates:
- 4y ep1 = 1,602 updates → already at peak by end of ep1
- 12y ep1 = 5,800 updates → peak around ep2-3 (matches data)
- 22y ep1 = 10,887 updates → peak somewhere DURING ep1 (we don't see)

Our checkpoints are at end of each epoch, so for 22y we miss the
in-epoch peak.

---

## 3. Literature support

### Direct support (calibration vs rank tradeoff)
- **Mukhoti et al. 2020 NeurIPS** "Calibrating Deep Neural Networks
  Using Focal Loss" (arXiv:2002.09437) — focal loss improves
  calibration over BCE *but* sharpness decreases.
- **Wang et al. 2021** "Rethinking Calibration of Deep Neural Networks"
  (arXiv:2105.05031) — explicitly states "over-confidence is not
  always bad — it can improve top-K accuracy."
- **Müller et al. 2019 NeurIPS** "When Does Label Smoothing Help?"
  (arXiv:1906.02629) — analogous: smoothing hurts top-K rank in
  many tasks.
- **Guo et al. 2017 ICML** "On Calibration of Modern Neural Networks"
  (arXiv:1706.04599) — modern NNs are systematically over-confident;
  calibration techniques exist precisely because uncalibrated models
  rank better than they predict.

### Indirect support (training dynamics in long-tail tasks)
- **Lin et al. 2017 ICCV** "Focal Loss for Dense Object Detection"
  (arXiv:1708.02002) — focal loss has alpha+gamma tradeoff between
  precision (calibration) and recall (top-K).
- **Cui et al. 2019 CVPR** "Class-Balanced Loss" (arXiv:1901.05555) —
  effective sample number framework; relevant because rare-event tasks
  have strong calibration-vs-discrimination tension.

### What's NOT in the literature
- Direct study of "Lift@K vs SGD updates trajectory" in spatiotemporal
  forecasting.
- Specific "5,000-10,000 update sweet spot" claim — this is our
  hypothesis from limited data.

---

## 4. The experiment to test it

**Name**: `lift_trajectory_within_epoch`

**Script**: `slurm/exp_lift_trajectory_within_epoch_narval.sh`

**Setup**:
- Single 22y enc14 training (10,887 batches/epoch × 4 epochs = 43,548 updates)
- Mid-epoch validation every 500 batches (≈22 eval points per epoch, 88 total)
- Each eval = quick val_lift on 20 random fire windows (~3-5 min)
- Total overhead: ~25% slower than baseline 22y

**CSV output**: `outputs/v3_9ch_enc14_2000_lift_traj_lift_trajectory.csv`
Columns: `epoch, batch, global_step, train_loss, val_lift_k, val_roc_auc, val_brier, eval_sec`

**Cost**: 1 GPU job, ~30h wall time

---

## 5. Predictions (made BEFORE running)

If hypothesis is correct:

1. **val_lift_k will rise then peak around step 5,000-10,000**, then decline before end-of-ep1.
2. **ROC-AUC will be stable (~0.86-0.91) throughout**, with no obvious correlation to val_lift_k changes.
3. **Brier will decrease monotonically** (calibration improving) even as val_lift_k drops.
4. **Best in-epoch val_lift will be HIGHER than the end-of-epoch 5.73x** that we currently report.
5. The sweet-spot peak step will be in the **3,000-15,000 range**.

If hypothesis is WRONG, alternatives:

- val_lift_k flat throughout → randomly selected windows differ; not informative
- val_lift_k monotone decrease from step 0 → "more training is always worse"; need smaller LR
- val_lift_k monotone increase → just need MORE updates; recipe is fine
- ROC-AUC also drops → not just calibration; some other degradation

---

## 6. Action based on outcome

### If confirmed (peak in middle of ep1):
- Implement step-level early stopping (save best ckpt across ALL training steps, not just epoch ends)
- Re-evaluate all 22y models — true best ckpt may be at e.g. step 6,000 not step 10,887
- Update boss report: **22y might actually be SOTA** if we find the true peak
- Publish as paper finding: "spatiotemporal forecasting with focal loss exhibits a non-monotonic Lift@K trajectory; standard end-of-epoch checkpoint selection misses the true optimum at 5-10K updates"

### If disconfirmed:
- Drop the calibration-vs-rank framing
- Investigate alternative hypotheses:
  - Hard-negative mining pool changes between epochs (pool re-sampled each epoch with different seed?)
  - Distribution shift in val period (2023-2024 surge years)
  - Spurious features the model latches onto in later epochs

---

## 7. What this is NOT

**Not** a proven theory. Only 4 trajectory observations (4y default,
4y strongreg, 12y default, 22y default), all under similar recipe.
Could be confounded by:
- Random seed variability
- Hard negative pool re-sampling
- Specific to focal loss (BCE might behave differently)

The experiment above will give us mid-epoch granularity and either
strongly support or refute the hypothesis.

---

## 8. Practical metric recommendation (added 2026-05-01)

Independent of this hypothesis, **the most operationally important
metric for fire prediction is**:

> **Recall@30km within budget** = "of the top X% of Canada we flag at
> 14-46 d lead, what fraction of subsequent fire events fall there?"

For wildfire agencies, resources are pre-positioned by region. The top
1% of Canada (~12,500 km²) is a realistic deployment budget. A model
that catches X% of fire events in that 1% IS the model that matters.

For paper headline:
- **Lift@30km event-level** — operationally interpretable + standard +
  novel-fire artifact-resistant. Currently 4.50× for 4y default, 4.26×
  for 22y enc35. After bug fix and recency, target is >5×.

For internal model selection:
- **Best-in-trajectory val_lift** (depends on this experiment).
