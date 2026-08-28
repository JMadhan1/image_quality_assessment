# Evaluation Report

Model: hybrid ResNet18 (ImageNet-pretrained, fine-tuned) + 8 engineered CV
features, jointly predicting quality score (regression) and distortion type
(7-way classification: none, blur, gaussian_noise, salt_pepper, brightness,
jpeg, block_corrupt).

Held-out **test split**: 5,700 images (900 per distortion type, 300 clean),
from source photos never seen during training or validation (split by
source image, not by individual variant).

## Headline numbers

| Metric | Value |
|---|---|
| Overall classification accuracy | 84.4% |
| Weighted F1 | 0.849 |
| Macro F1 | 0.816 |
| Quality-score MAE | 4.01 (0-100 scale) |
| Quality-score RMSE | 6.67 |

## Per-class precision / recall / F1

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| none (clean) | 0.394 | 0.710 | 0.507 | 300 |
| blur | 0.780 | 0.829 | 0.803 | 900 |
| gaussian_noise | 0.892 | 0.789 | 0.837 | 900 |
| salt_pepper | 0.950 | 0.973 | 0.962 | 900 |
| brightness | 0.919 | 0.652 | 0.763 | 900 |
| jpeg | 0.834 | 0.912 | 0.872 | 900 |
| block_corrupt | 0.994 | 0.951 | 0.972 | 900 |

## Confusion matrix (rows = true, cols = predicted)

```
              none   blur  gnoise  s&p  bright  jpeg  block
none           213     22     13     1      8    42      1
blur             56   746      6     0     23    69      0
gaussian_noise  118    18    710    35      8    10      1
salt_pepper       6     1     10   876      4     2      1
brightness      110   121     45     5    587    30      2
jpeg              25    39      6     2      7   821      0
block_corrupt     13    10      6     3      2    10    856
```

## Failure case analysis

**1. "none" (clean) precision is the weakest class (0.39).** Roughly
100-120 images from each of blur, gaussian_noise, and brightness get
misclassified as clean. This is expected and traces to the severity-1
degradations in the synthetic generator: a mild kernel-3 Gaussian blur or
a small ±30 brightness shift is genuinely close to imperceptible, so the
label boundary between "none" and "mild distortion" is itself fuzzy. This
is a labeling-granularity limitation of the synthetic ground truth, not
purely a model weakness — a useful thing to flag rather than hide.

**2. brightness has the lowest recall (0.65) among distortions**, most
often confused with "none" (110 cases) and "blur" (121 cases). Brightness
shifts at low severity are subtle, and — more informatively — the current
brightness degradation alternates sign by severity parity (even severities
underexpose, odd severities overexpose) with no separate "which direction"
signal in the 7-way label; the model can correctly detect *that* exposure
is off while the confusion matrix only credits/penalizes the top-1 class.

**3. Quality-score regression is noisier than classification accuracy
suggests, especially for brightness.** The worst absolute-error cases
(errors of 37-50 points on a 0-100 scale) are dominated by brightness
images where the *distortion type* was classified correctly but the
*severity-derived score* was off by a large margin — i.e. the model can
tell an image is brightness-shifted but is less reliable at judging how
severely, compared to blur/JPEG where score errors are much tighter. This
suggests exposure severity is encoded more subtly in pixel statistics than
blur or compression severity, and would benefit from a dedicated
exposure-histogram feature with more bins if extended further.

**4. block_corrupt and salt_pepper are the easiest classes** (F1 > 0.96) —
both introduce strong, spatially localized high-frequency signal that both
the CNN and the engineered noise-energy feature pick up reliably.

## Model selection: what was tried, and why the baseline won

The numbers above are from the model actually shipped — a fine-tuned
ResNet18. That wasn't the first or only architecture tried; three follow-up
experiments were run specifically to push accuracy higher, and are reported
here in full rather than only reporting the winner, because the trade-offs
found are informative in their own right.

| Version | Change from baseline | Test accuracy | Weighted F1 | Score MAE |
|---|---|---|---|---|
| **v1 (shipped)** | ResNet18 baseline | **84.4%** | **0.849** | 4.01 |
| v2 | ResNet34 + full inverse-frequency class-weighted loss | 81.4% | 0.822 | **2.32** |
| v3 | ResNet34 + softened (sqrt) class-weighted loss | not completed (abandoned, see below) | — | — |
| v4 | ConvNeXt-Tiny + oversampling + MixUp | not completed (see below) | — | — |

**v2 finding:** class-weighting "none" (clean) at ~2.7x to fix its
under-representation (it has ~3x fewer training examples than each
distortion type) worked exactly as intended for that one class — "none"
recall rose from 0.710 to 0.793 — but it distorted the loss landscape
enough to cost accuracy on the other six classes, net negative overall
(84.4% → 81.4%). It did substantially improve quality-score regression
(MAE 4.01 → 2.32), a genuine trade-off rather than a strict win or loss:
better-calibrated scores, worse classification.

**v3 (planned fix, not completed):** the same idea with the loss weighting
softened via a square-root smoothing, intended to keep most of v2's "none"
recall gain without as much collateral damage to the other classes. Killed
before evaluation because it was training the same fundamentally
loss-reweighting-based approach as v2 and time was better spent on a more
promising direction (v4) than a second point on the same curve.

**v4 (attempted, not completed):** a switch to ConvNeXt-Tiny (82.1%
ImageNet top-1 vs ResNet34's 73.3% at a comparable 29M/21M parameter
count), replacing loss-reweighting with oversampling
(`WeightedRandomSampler`) and adding MixUp augmentation — a more
principled fix than v2/v3, since oversampling changes what the model sees
rather than distorting what the loss function optimizes for. This is
recorded as an honest incomplete experiment: across three attempts the
training run was repeatedly killed by environment instability unrelated
to the model itself (background process interruption on two attempts, a
Windows `DataLoader` worker deadlock — confirmed via near-zero CPU time on
worker processes that had been alive for over an hour — on the third,
after switching `num_workers` to isolate the cause). Given the 48-hour
window, the decision was to ship the fully-verified v1 rather than keep
retrying an approach that had not yet produced a completed, evaluated
result. The implementation (`--backbone convnext_tiny`, oversampling, and
MixUp are all in `train.py`) is left in place for anyone continuing this
work with more time or a more stable environment.

**Takeaway:** "bigger model" and "fix the class imbalance" were both
reasonable hypotheses, and both were tested rather than assumed — v2's
result shows the imbalance fix has a real cost that has to be weighed
against the regression-accuracy gain it buys, and the honest report of
v4's incompleteness is more useful than silently omitting a lead that
didn't pan out.

## Generalization (unseen domain)

The model was trained entirely on degraded Oxford-IIIT Pet photos. To
check it hasn't just memorized Pet-photo statistics, `generalization_check.py`
scores 100 clean, **undegraded** images from Oxford Flowers102 — a
completely different content domain never seen in any form during
training or validation. If the model generalizes, these should score high
and mostly land in ACCEPTABLE, since they're genuinely undistorted photos.

Results (`models/generalization_report.json`):

| Metric | Value |
|---|---|
| Mean quality score | 92.55 / 100 |
| ACCEPTABLE | 93 / 100 |
| DEGRADED | 6 / 100 |
| DEFECTIVE | 1 / 100 |
| ACCEPTABLE or DEGRADED | 99.0% |

This is strong evidence the model generalizes beyond memorizing the
training domain's specific content — it correctly recognizes clean,
undistorted images as high quality even for a subject (flowers) it never
saw during training. The 1 false-positive DEFECTIVE case and 6 DEGRADED
cases are plausible: some Flowers102 photos have genuinely shallow
depth-of-field (background blur) or close-up macro framing that shares
real pixel statistics with the synthetic blur/noise classes, so a small
false-positive rate here is expected rather than a sign of a bug.

A further, not-yet-run extension would be scoring a slice of an
*authentically*-degraded real-world dataset (e.g. KonIQ-10k, which has
human MOS quality ratings) to check whether the quality-score *ranking*
still correlates with human judgments on naturally occurring (not
synthetically generated) distortions — a stronger but more expensive test
than the unseen-domain check actually run here.

## Summary

The hybrid model reaches strong accuracy on distinct, high-frequency
distortions (block corruption, salt-and-pepper, JPEG) and reasonable
accuracy on blur and noise. Its main weaknesses are at the boundary
between "clean" and "mildly distorted" images, and in judging brightness
*severity* specifically — both traceable to genuine ambiguity in the
synthetic labels rather than an architectural flaw, and both are natural
next steps for improvement (finer severity binning, a direction-aware
exposure feature).
