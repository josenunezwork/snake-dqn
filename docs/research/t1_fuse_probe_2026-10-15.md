# T1 Part C: B8 fuse representation probe (2026-10-15)

**Development diagnostic, not gate evidence.** No policy training or promotion decision.

R2/R2' verdict: **value learning is the limit**, using the pooled-five-B8 nested LR H100 point AUC
**0.837 [0.773, 0.897]**. The rule uses the point estimate
(>=0.75 value learning, <0.65 perception, otherwise calibrated death head).
This is the panel's diagnostic routing rule; predictive association does not establish
a causal learning bottleneck or demonstrate that a learner intervention will work.

## Method

Captured 9 own-checkpoint arms on the same 32 scripted development worlds
as M7: round 52, promotion-v2-watch-rect, GridBatchSim, H5000, rowwise ego2s
forward, advisory mask, normal-only fallback when no action is safe, no veto,
numba vector61 backend and fast scripted anchors. Recorded L>=800 at frame%3==0.
The hook reads fuse[2]'s **post-ReLU output**, 256 dimensions, stored as float16;
Q vectors remain float32. Each NPZ includes arm, numeric world seed, frame, length,
chosen action, all labels and eligibility masks. All inference stays on CPU.

A positive label means self/enemy-body death with death_frame-frame < H; the fatal
decision counts. Wall/head-on deaths within H are excluded. A surviving row is
excluded when frame > 5000-H. These are exactly M7's label and censoring formulas.

LR uses train-only standardization, L2 regularization and C in {0.01,0.1,1}.
For every outer training set, C is selected by three inner folds grouped by world,
maximizing concatenated inner out-of-fold AUC. Five outer world folds produce every
reported prediction. Numeric worlds stay together across all five B8 arms in the
pooled model. HistGradientBoostingClassifier is a fixed-parameter comparison.
Settings, selected C values, warnings and timings are in metrics.json.

CIs are percentile 95% intervals from 2,000 bootstrap resamples of numeric worlds,
with replacement, carrying all rows and arms for each sampled world together. They
condition on the fitted out-of-fold predictions; models are not refit in each bootstrap.
Rows are decision weighted, and world clustering accounts for temporal dependence.
Of 2,000 attempted draws, CTRL_s12 has 1,999 valid draws at each horizon: one
single-class draw has undefined AUC and is omitted. All other intervals use 2,000
valid draws. Valid and discarded counts are recorded for every metric.

## Capture compatibility and integrity

All nine M7 raw NPZs exist. They have 60 scalars and 10 plane summaries, but lack
full local/global rasters and fuse activations. Summaries cannot reconstruct the
representation. Instead, recaptured each included arm with its own checkpoint.
For every full capture, compared selected row coordinates, lengths, actions, all six Q
values, world seeds, death frames and death causes bitwise against M7, with matching
checkpoint sha256. Observer checks also verify the actions actually executed.

- B8_s0: 22,080 rows; M7 identity `identical`; observer checks `{'observer_action_mismatch': 0}`.
- B8_s1: 14,527 rows; M7 identity `identical`; observer checks `{'observer_action_mismatch': 0}`.
- B8_s2: 21,654 rows; M7 identity `identical`; observer checks `{'observer_action_mismatch': 0}`.
- B8_s3: 18,340 rows; M7 identity `identical`; observer checks `{'observer_action_mismatch': 0}`.
- B8_s4: 16,180 rows; M7 identity `identical`; observer checks `{'observer_action_mismatch': 0}`.
- CTRL_s10: 20,734 rows; M7 identity `identical`; observer checks `{'observer_action_mismatch': 0}`.
- CTRL_s11: 18,822 rows; M7 identity `identical`; observer checks `{'observer_action_mismatch': 0}`.
- CTRL_s12: 16,678 rows; M7 identity `identical`; observer checks `{'observer_action_mismatch': 0}`.
- CTRL_s13: 19,652 rows; M7 identity `identical`; observer checks `{'observer_action_mismatch': 0}`.

## Probe AUCs

Every entry is AUC [95% CI].

| Scope | H | Kept rows | Death-positive rows | Nested LR | HGB |
|---|---:|---:|---:|---|---|
| B8_s0 | 50 | 21,744 | 132 | 0.766 [0.594, 0.929] | 0.792 [0.619, 0.954] |
| B8_s0 | 100 | 21,387 | 265 | 0.682 [0.478, 0.884] | 0.707 [0.488, 0.917] |
| B8_s0 | 300 | 19,980 | 800 | 0.604 [0.387, 0.805] | 0.585 [0.371, 0.790] |
| B8_s1 | 50 | 14,335 | 166 | 0.871 [0.749, 0.979] | 0.884 [0.724, 0.999] |
| B8_s1 | 100 | 14,131 | 331 | 0.810 [0.607, 0.980] | 0.828 [0.633, 0.995] |
| B8_s1 | 300 | 13,327 | 910 | 0.834 [0.689, 0.951] | 0.824 [0.673, 0.948] |
| B8_s2 | 50 | 21,334 | 170 | 0.883 [0.750, 0.991] | 0.928 [0.818, 0.999] |
| B8_s2 | 100 | 20,994 | 321 | 0.808 [0.639, 0.955] | 0.843 [0.674, 0.980] |
| B8_s2 | 300 | 19,654 | 920 | 0.797 [0.572, 0.981] | 0.790 [0.559, 0.980] |
| B8_s3 | 50 | 18,068 | 169 | 0.787 [0.566, 0.964] | 0.820 [0.613, 0.982] |
| B8_s3 | 100 | 17,779 | 337 | 0.734 [0.517, 0.923] | 0.788 [0.605, 0.948] |
| B8_s3 | 300 | 16,640 | 968 | 0.717 [0.555, 0.872] | 0.710 [0.549, 0.867] |
| B8_s4 | 50 | 15,940 | 185 | 0.842 [0.723, 0.938] | 0.868 [0.737, 0.962] |
| B8_s4 | 100 | 15,685 | 351 | 0.856 [0.770, 0.942] | 0.849 [0.749, 0.945] |
| B8_s4 | 300 | 14,680 | 963 | 0.757 [0.622, 0.894] | 0.779 [0.668, 0.898] |
| CTRL_s10 | 50 | 20,446 | 185 | 0.711 [0.561, 0.846] | 0.728 [0.580, 0.864] |
| CTRL_s10 | 100 | 20,140 | 367 | 0.652 [0.480, 0.814] | 0.645 [0.484, 0.806] |
| CTRL_s10 | 300 | 18,934 | 1,100 | 0.546 [0.386, 0.715] | 0.571 [0.406, 0.745] |
| CTRL_s11 | 50 | 18,550 | 150 | 0.753 [0.568, 0.901] | 0.830 [0.751, 0.912] |
| CTRL_s11 | 100 | 18,261 | 300 | 0.554 [0.402, 0.712] | 0.624 [0.480, 0.769] |
| CTRL_s11 | 300 | 17,122 | 900 | 0.401 [0.270, 0.527] | 0.436 [0.281, 0.594] |
| CTRL_s12 | 50 | 16,406 | 119 | 0.604 [0.433, 0.784] | 0.610 [0.430, 0.794] |
| CTRL_s12 | 100 | 16,117 | 234 | 0.518 [0.405, 0.616] | 0.572 [0.415, 0.747] |
| CTRL_s12 | 300 | 14,978 | 652 | 0.446 [0.317, 0.578] | 0.533 [0.424, 0.623] |
| CTRL_s13 | 50 | 19,364 | 149 | 0.894 [0.781, 0.998] | 0.903 [0.792, 0.998] |
| CTRL_s13 | 100 | 19,058 | 299 | 0.841 [0.666, 0.995] | 0.846 [0.674, 0.996] |
| CTRL_s13 | 300 | 17,852 | 900 | 0.666 [0.435, 0.885] | 0.672 [0.441, 0.885] |
| pooled_B8 | 50 | 91,421 | 822 | 0.880 [0.826, 0.932] | 0.880 [0.824, 0.930] |
| pooled_B8 | 100 | 89,976 | 1,605 | 0.837 [0.773, 0.897] | 0.848 [0.787, 0.907] |
| pooled_B8 | 300 | 84,281 | 4,561 | 0.795 [0.731, 0.855] | 0.800 [0.738, 0.855] |

## M7 existing-input baselines

These historical fits pooled all nine arms, whereas this report's pooled
probe uses five B8 arms. They use the same world bank and label definitions,
but are contextual baselines, not a paired improvement estimate. M7 LR used
fixed C=0.1, rather than nested selection. Its existing-input set includes
scalars, plane summaries and derived aggregates, not full raster tensors.

| H | M7 LR | M7 GBM |
|---:|---|---|
| 50 | 0.888 [0.856, 0.916] | 0.910 [0.879, 0.935] |
| 100 | 0.825 [0.777, 0.863] | 0.811 [0.761, 0.853] |
| 300 | 0.726 [0.673, 0.773] | 0.695 [0.641, 0.746] |

## Hero Q signals

At pooled B8 H100, lower max Q predicts death with AUC
**0.851 [0.784, 0.908]**.
The top-two gap in its raw orientation gives
**0.736 [0.687, 0.780]**.
The hero's Q therefore already contains a useful death-ranking signal; this
does not establish that it identifies a surviving alternative or an early commitment.

Raw-score orientation: larger max Q or top-two gap predicts death.
The negative-score AUC is 1 minus this AUC (CI endpoints reversed); both
orientations and their CIs are preserved in metrics.json. No sign was chosen
by maximizing held-out performance, and no probe was fitted for these scores.

| Scope | H | max Q raw | Top-two gap raw |
|---|---:|---|---|
| B8_s0 | 50 | 0.111 [0.007, 0.221] | 0.686 [0.546, 0.820] |
| B8_s0 | 100 | 0.149 [0.027, 0.277] | 0.672 [0.531, 0.800] |
| B8_s0 | 300 | 0.271 [0.112, 0.427] | 0.640 [0.546, 0.749] |
| B8_s1 | 50 | 0.098 [0.001, 0.281] | 0.749 [0.651, 0.843] |
| B8_s1 | 100 | 0.102 [0.002, 0.280] | 0.726 [0.626, 0.815] |
| B8_s1 | 300 | 0.091 [0.015, 0.216] | 0.704 [0.644, 0.758] |
| B8_s2 | 50 | 0.019 [0.001, 0.050] | 0.820 [0.766, 0.875] |
| B8_s2 | 100 | 0.064 [0.002, 0.142] | 0.814 [0.765, 0.865] |
| B8_s2 | 300 | 0.100 [0.003, 0.239] | 0.834 [0.738, 0.913] |
| B8_s3 | 50 | 0.152 [0.026, 0.325] | 0.761 [0.646, 0.867] |
| B8_s3 | 100 | 0.200 [0.045, 0.386] | 0.731 [0.599, 0.859] |
| B8_s3 | 300 | 0.211 [0.088, 0.338] | 0.659 [0.583, 0.735] |
| B8_s4 | 50 | 0.204 [0.036, 0.379] | 0.752 [0.661, 0.840] |
| B8_s4 | 100 | 0.208 [0.053, 0.393] | 0.723 [0.599, 0.827] |
| B8_s4 | 300 | 0.283 [0.105, 0.488] | 0.694 [0.561, 0.817] |
| CTRL_s10 | 50 | 0.252 [0.108, 0.428] | 0.726 [0.625, 0.819] |
| CTRL_s10 | 100 | 0.318 [0.161, 0.502] | 0.716 [0.615, 0.808] |
| CTRL_s10 | 300 | 0.332 [0.184, 0.478] | 0.651 [0.551, 0.755] |
| CTRL_s11 | 50 | 0.227 [0.088, 0.389] | 0.645 [0.547, 0.754] |
| CTRL_s11 | 100 | 0.265 [0.092, 0.468] | 0.607 [0.487, 0.737] |
| CTRL_s11 | 300 | 0.280 [0.116, 0.471] | 0.552 [0.463, 0.637] |
| CTRL_s12 | 50 | 0.299 [0.152, 0.442] | 0.646 [0.551, 0.755] |
| CTRL_s12 | 100 | 0.365 [0.223, 0.497] | 0.628 [0.520, 0.746] |
| CTRL_s12 | 300 | 0.478 [0.295, 0.589] | 0.571 [0.478, 0.666] |
| CTRL_s13 | 50 | 0.088 [0.002, 0.191] | 0.815 [0.730, 0.884] |
| CTRL_s13 | 100 | 0.122 [0.001, 0.275] | 0.784 [0.663, 0.884] |
| CTRL_s13 | 300 | 0.217 [0.033, 0.439] | 0.732 [0.618, 0.829] |
| pooled_B8 | 50 | 0.121 [0.065, 0.188] | 0.757 [0.718, 0.798] |
| pooled_B8 | 100 | 0.149 [0.092, 0.216] | 0.736 [0.687, 0.780] |
| pooled_B8 | 300 | 0.195 [0.135, 0.257] | 0.709 [0.665, 0.748] |

## H100 LR reliability

Fixed-width probability bins, using outer held-world predictions; no
additional calibrator was fit. Empty bins show —. This describes calibration
under the captured late-game decision distribution.

| Scope | Probability bin | Rows | Mean predicted | Observed death rate |
|---|---|---:|---:|---:|
| B8_s0 | 0.0–0.1 | 20,960 | 0.0064 | 0.0074 |
| B8_s0 | 0.1–0.2 | 149 | 0.1392 | 0.0067 |
| B8_s0 | 0.2–0.3 | 48 | 0.2436 | 0.0833 |
| B8_s0 | 0.3–0.4 | 38 | 0.3523 | 0.1053 |
| B8_s0 | 0.4–0.5 | 27 | 0.4429 | 0.1481 |
| B8_s0 | 0.5–0.6 | 22 | 0.5576 | 0.5000 |
| B8_s0 | 0.6–0.7 | 17 | 0.6375 | 0.3529 |
| B8_s0 | 0.7–0.8 | 30 | 0.7501 | 0.5333 |
| B8_s0 | 0.8–0.9 | 26 | 0.8610 | 0.5385 |
| B8_s0 | 0.9–1.0 | 70 | 0.9620 | 0.7143 |
| B8_s1 | 0.0–0.1 | 13,685 | 0.0062 | 0.0060 |
| B8_s1 | 0.1–0.2 | 116 | 0.1402 | 0.1207 |
| B8_s1 | 0.2–0.3 | 40 | 0.2496 | 0.2000 |
| B8_s1 | 0.3–0.4 | 37 | 0.3538 | 0.2162 |
| B8_s1 | 0.4–0.5 | 18 | 0.4504 | 0.3889 |
| B8_s1 | 0.5–0.6 | 23 | 0.5508 | 0.4348 |
| B8_s1 | 0.6–0.7 | 6 | 0.6506 | 0.5000 |
| B8_s1 | 0.7–0.8 | 7 | 0.7311 | 0.2857 |
| B8_s1 | 0.8–0.9 | 9 | 0.8453 | 0.8889 |
| B8_s1 | 0.9–1.0 | 190 | 0.9868 | 0.9947 |
| B8_s2 | 0.0–0.1 | 20,512 | 0.0032 | 0.0042 |
| B8_s2 | 0.1–0.2 | 123 | 0.1395 | 0.0569 |
| B8_s2 | 0.2–0.3 | 40 | 0.2535 | 0.1000 |
| B8_s2 | 0.3–0.4 | 26 | 0.3521 | 0.0769 |
| B8_s2 | 0.4–0.5 | 26 | 0.4425 | 0.1923 |
| B8_s2 | 0.5–0.6 | 21 | 0.5532 | 0.2381 |
| B8_s2 | 0.6–0.7 | 22 | 0.6451 | 0.3636 |
| B8_s2 | 0.7–0.8 | 16 | 0.7420 | 0.4375 |
| B8_s2 | 0.8–0.9 | 17 | 0.8563 | 0.7647 |
| B8_s2 | 0.9–1.0 | 191 | 0.9826 | 0.9634 |
| B8_s3 | 0.0–0.1 | 17,367 | 0.0083 | 0.0094 |
| B8_s3 | 0.1–0.2 | 142 | 0.1337 | 0.0634 |
| B8_s3 | 0.2–0.3 | 50 | 0.2484 | 0.0600 |
| B8_s3 | 0.3–0.4 | 27 | 0.3369 | 0.2593 |
| B8_s3 | 0.4–0.5 | 17 | 0.4466 | 0.4706 |
| B8_s3 | 0.5–0.6 | 18 | 0.5585 | 0.4444 |
| B8_s3 | 0.6–0.7 | 13 | 0.6561 | 0.3077 |
| B8_s3 | 0.7–0.8 | 6 | 0.7447 | 0.6667 |
| B8_s3 | 0.8–0.9 | 16 | 0.8585 | 0.9375 |
| B8_s3 | 0.9–1.0 | 123 | 0.9776 | 0.9431 |
| B8_s4 | 0.0–0.1 | 15,226 | 0.0080 | 0.0114 |
| B8_s4 | 0.1–0.2 | 176 | 0.1408 | 0.0966 |
| B8_s4 | 0.2–0.3 | 49 | 0.2424 | 0.1429 |
| B8_s4 | 0.3–0.4 | 20 | 0.3524 | 0.1500 |
| B8_s4 | 0.4–0.5 | 20 | 0.4541 | 0.0000 |
| B8_s4 | 0.5–0.6 | 18 | 0.5571 | 0.2778 |
| B8_s4 | 0.6–0.7 | 12 | 0.6401 | 0.3333 |
| B8_s4 | 0.7–0.8 | 12 | 0.7509 | 0.5833 |
| B8_s4 | 0.8–0.9 | 17 | 0.8506 | 0.7647 |
| B8_s4 | 0.9–1.0 | 135 | 0.9839 | 0.8963 |
| CTRL_s10 | 0.0–0.1 | 19,856 | 0.0100 | 0.0113 |
| CTRL_s10 | 0.1–0.2 | 106 | 0.1355 | 0.0283 |
| CTRL_s10 | 0.2–0.3 | 19 | 0.2581 | 0.4211 |
| CTRL_s10 | 0.3–0.4 | 12 | 0.3461 | 0.3333 |
| CTRL_s10 | 0.4–0.5 | 16 | 0.4473 | 0.5625 |
| CTRL_s10 | 0.5–0.6 | 8 | 0.5261 | 0.7500 |
| CTRL_s10 | 0.6–0.7 | 4 | 0.6659 | 1.0000 |
| CTRL_s10 | 0.7–0.8 | 9 | 0.7538 | 0.8889 |
| CTRL_s10 | 0.8–0.9 | 18 | 0.8673 | 0.7778 |
| CTRL_s10 | 0.9–1.0 | 92 | 0.9736 | 0.9457 |
| CTRL_s11 | 0.0–0.1 | 17,982 | 0.0099 | 0.0128 |
| CTRL_s11 | 0.1–0.2 | 128 | 0.1361 | 0.0469 |
| CTRL_s11 | 0.2–0.3 | 37 | 0.2424 | 0.0811 |
| CTRL_s11 | 0.3–0.4 | 22 | 0.3494 | 0.2273 |
| CTRL_s11 | 0.4–0.5 | 19 | 0.4458 | 0.2105 |
| CTRL_s11 | 0.5–0.6 | 7 | 0.5479 | 0.5714 |
| CTRL_s11 | 0.6–0.7 | 3 | 0.6439 | 0.6667 |
| CTRL_s11 | 0.7–0.8 | 7 | 0.7419 | 0.5714 |
| CTRL_s11 | 0.8–0.9 | 8 | 0.8609 | 0.3750 |
| CTRL_s11 | 0.9–1.0 | 48 | 0.9798 | 0.8125 |
| CTRL_s12 | 0.0–0.1 | 15,861 | 0.0085 | 0.0125 |
| CTRL_s12 | 0.1–0.2 | 138 | 0.1353 | 0.0217 |
| CTRL_s12 | 0.2–0.3 | 25 | 0.2491 | 0.0400 |
| CTRL_s12 | 0.3–0.4 | 19 | 0.3512 | 0.0000 |
| CTRL_s12 | 0.4–0.5 | 5 | 0.4445 | 0.0000 |
| CTRL_s12 | 0.5–0.6 | 9 | 0.5294 | 0.1111 |
| CTRL_s12 | 0.6–0.7 | 9 | 0.6480 | 0.0000 |
| CTRL_s12 | 0.7–0.8 | 6 | 0.7437 | 0.1667 |
| CTRL_s12 | 0.8–0.9 | 6 | 0.8381 | 0.1667 |
| CTRL_s12 | 0.9–1.0 | 39 | 0.9869 | 0.7436 |
| CTRL_s13 | 0.0–0.1 | 18,701 | 0.0044 | 0.0056 |
| CTRL_s13 | 0.1–0.2 | 84 | 0.1415 | 0.0714 |
| CTRL_s13 | 0.2–0.3 | 22 | 0.2453 | 0.0909 |
| CTRL_s13 | 0.3–0.4 | 18 | 0.3537 | 0.3333 |
| CTRL_s13 | 0.4–0.5 | 19 | 0.4555 | 0.3158 |
| CTRL_s13 | 0.5–0.6 | 14 | 0.5590 | 0.4286 |
| CTRL_s13 | 0.6–0.7 | 15 | 0.6401 | 0.6000 |
| CTRL_s13 | 0.7–0.8 | 20 | 0.7521 | 0.7000 |
| CTRL_s13 | 0.8–0.9 | 28 | 0.8535 | 0.8929 |
| CTRL_s13 | 0.9–1.0 | 137 | 0.9659 | 0.8832 |
| pooled_B8 | 0.0–0.1 | 88,281 | 0.0065 | 0.0074 |
| pooled_B8 | 0.1–0.2 | 448 | 0.1390 | 0.0915 |
| pooled_B8 | 0.2–0.3 | 141 | 0.2421 | 0.1348 |
| pooled_B8 | 0.3–0.4 | 84 | 0.3479 | 0.1429 |
| pooled_B8 | 0.4–0.5 | 42 | 0.4497 | 0.3571 |
| pooled_B8 | 0.5–0.6 | 35 | 0.5582 | 0.2571 |
| pooled_B8 | 0.6–0.7 | 47 | 0.6569 | 0.5532 |
| pooled_B8 | 0.7–0.8 | 47 | 0.7543 | 0.8723 |
| pooled_B8 | 0.8–0.9 | 105 | 0.8571 | 0.8095 |
| pooled_B8 | 0.9–1.0 | 746 | 0.9767 | 0.9450 |

## Timing and limits

| Arm | Evaluation seconds | Total capture seconds |
|---|---:|---:|
| B8_s0 | 172.7 | 173.0 |
| B8_s1 | 134.5 | 134.6 |
| B8_s2 | 159.1 | 159.1 |
| B8_s3 | 192.3 | 192.4 |
| B8_s4 | 185.1 | 185.2 |
| CTRL_s10 | 169.8 | 169.9 |
| CTRL_s11 | 170.7 | 170.8 |
| CTRL_s12 | 164.1 | 164.1 |
| CTRL_s13 | 158.4 | 158.5 |

Serial capture total: 1507.5s. Analysis wall time:
1656.7s. Torch and native NumPy/BLAS/OpenMP pools were
limited to two threads. Captures, full tests and fits ran as serial numerical
steps in detached tmux session `t1c-run`, with logs and exit receipts.

Only 32 shared development worlds support this diagnostic. Death-positive
episodes are few, rows are correlated, CIs exclude fit/model-selection and
training-seed uncertainty, and the bank was already used for diagnosis.
This is neither a fresh-world gate nor a calibrated deployment risk model.
Each arm has its own learned latent coordinates: pooled feature coordinates
are not guaranteed to align across checkpoints. Read the per-arm results
alongside the pooled point estimate; the 5-arm pool is not a seed bootstrap.
AUC measures ranking rather than intervention benefit or calibration quality.
CTRL_s12 OOF probabilities are float64 after repair; other archives are float32. All archived-prediction AUCs reproduce the float64 metrics within 0.000001. Rounded table values are unaffected.

## Reproduction and artifacts

Artifact root: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t1-fuse-probe-20261015`. Input SHA256s: `inputs.sha256`; result SHA256s:
`outputs.sha256`. See `README.md` for commands, dependencies and verification.

## Reviewer check (Claude, 2026-10-15): the max-Q signal is not a length proxy

On the pooled B8 H100 rows (89,976 rows, 1,605 positives), length alone has AUC 0.513 and frame
0.608, while −max Q has 0.851. Within length bands −max Q stays strong: L 800–950 0.716,
950–1100 0.858, 1100–1300 0.949, 1300+ 0.845. After regressing max Q on L, L², frame and frame²,
the residual still scores 0.841. B8's value head does know a state is dangerous ~100 decisions
ahead; whether it ranks the surviving action above the fatal one is T1 Part A's question.
