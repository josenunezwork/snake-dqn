# T3 label-arm diagnostics — 2026-10-16

Development diagnostics, not gate evidence. These measurements do not change the T3 decision or authorize promotion. All three label arms are bare final networks at evaluation; KEEPV8 describes training anchors, not a serving veto.

## Scope and evidence

All five M6 CTRL finals and all 15 completed T3 finals (training seeds 10–14) are included. Every T3 run had `exit=0` in `status.txt` before inclusion, including seed 14. Trajectory diagnostics use only round 62, scripted mix, 96 identical world seeds per checkpoint, H=5000. Round 63 is excluded. Pairing is by training seed and world seed; the 480 seed/world pairs reuse the same 96 worlds, so they are not 480 independent worlds.

All numerical work ran on CPU with one thread per process and at most four numerical workers. No MPS, tmux session control, or interaction with other owners' processes was used.

The companion `t3_diagnostics_2026-10-16_evidence/` directory preserves bank results, replay summaries, sampled indices, input hashes, and replay receipts. Full per-frame NPZ series remain in the local artifact paths recorded in those receipts.

## 1. Separation on the bank

The effective margin is **0.0631999254 × 1.0 = 0.0631999254 Q units**. This is well below the T1 median fatal-versus-survivor gap of 0.45 Q. The advisory note's proposed oversized-margin explanation is ruled out; the reported separation quartiles are also sub-unit rather than several Q units. This does not establish whether a different margin would improve gameplay.

Fixed sample seed `20261016` selects 2,000 rows without replacement: 1,333 decisive and 667 counter rows, preserving the bank's 10,885/5,442 proportion to integer rounding. The same rows are used for every final. Separation is `max Q(stored surviving set) − Q(stored B8 action)`. Agreement uses raw six-action argmax, matching training `bank_agree`. The surviving sets are B8-continuation labels, not freshly established survival outcomes under these students.

Pooled quartiles below are computed over all 5 × 2,000 model/row evaluations per arm, not averages of seed quartiles. They are descriptive; bank fit is not held-out performance.

| Arm | Rows | Model/row evaluations | Separation Q25 / Q50 / Q75 | Argmax in surviving set |
| --- | --- | --- | --- | --- |
| CTRL | overall | 10000 | -0.134 / -0.067 / 0.000 | 39.83% |
| CTRL | decisive | 6665 | -0.168 / -0.112 / -0.067 | 12.35% |
| CTRL | counter | 3335 | 0.000 / 0.000 / 0.000 | 94.75% |
| LABELS | overall | 10000 | 0.146 / 0.236 / 0.333 | 99.66% |
| LABELS | decisive | 6665 | 0.162 / 0.246 / 0.341 | 99.64% |
| LABELS | counter | 3335 | 0.094 / 0.210 / 0.313 | 99.70% |
| LABELS_KEEPV8 | overall | 10000 | 0.113 / 0.207 / 0.308 | 96.90% |
| LABELS_KEEPV8 | decisive | 6665 | 0.134 / 0.220 / 0.316 | 96.68% |
| LABELS_KEEPV8 | counter | 3335 | 0.051 / 0.176 / 0.286 | 97.33% |
| LABELS_KEEPV8_LO | overall | 10000 | 0.116 / 0.207 / 0.308 | 94.66% |
| LABELS_KEEPV8_LO | decisive | 6665 | 0.135 / 0.220 / 0.315 | 94.76% |
| LABELS_KEEPV8_LO | counter | 3335 | 0.054 / 0.175 / 0.291 | 94.45% |

Per-final results on the same 2,000 rows:

| Final | Separation Q25 / Q50 / Q75 | Raw agreement |
| --- | --- | --- |
| CTRL_s10 | -0.142 / -0.077 / 0.000 | 40.20% |
| CTRL_s11 | -0.121 / -0.060 / 0.000 | 39.80% |
| CTRL_s12 | -0.131 / -0.068 / 0.000 | 39.80% |
| CTRL_s13 | -0.153 / -0.084 / 0.000 | 39.40% |
| CTRL_s14 | -0.121 / -0.061 / 0.000 | 39.95% |
| LABELS_KEEPV8_LO_s10 | 0.115 / 0.208 / 0.312 | 94.00% |
| LABELS_KEEPV8_LO_s11 | 0.122 / 0.210 / 0.310 | 96.05% |
| LABELS_KEEPV8_LO_s12 | 0.114 / 0.209 / 0.311 | 94.30% |
| LABELS_KEEPV8_LO_s13 | 0.126 / 0.220 / 0.329 | 94.70% |
| LABELS_KEEPV8_LO_s14 | 0.105 / 0.189 / 0.279 | 94.25% |
| LABELS_KEEPV8_s10 | 0.104 / 0.197 / 0.312 | 96.65% |
| LABELS_KEEPV8_s11 | 0.108 / 0.195 / 0.292 | 97.30% |
| LABELS_KEEPV8_s12 | 0.118 / 0.209 / 0.300 | 97.20% |
| LABELS_KEEPV8_s13 | 0.134 / 0.225 / 0.328 | 96.95% |
| LABELS_KEEPV8_s14 | 0.103 / 0.204 / 0.306 | 96.40% |
| LABELS_s10 | 0.156 / 0.254 / 0.352 | 99.50% |
| LABELS_s11 | 0.138 / 0.227 / 0.315 | 99.80% |
| LABELS_s12 | 0.135 / 0.232 / 0.336 | 99.55% |
| LABELS_s13 | 0.151 / 0.236 / 0.333 | 99.70% |
| LABELS_s14 | 0.150 / 0.231 / 0.330 | 99.75% |

## 2. Scripted mass decomposition

For each paired world and post-transition frame, dead mass is zero. The exact arm-minus-CTRL mass delta is the sum of (a) mass differences on frames when both are alive and (b) arm-only surviving mass minus CTRL-only surviving mass, divided by 5000. The second term captures death timing/incidence; its sign can favor an arm even when that arm loses overall mass. This is an accounting decomposition, not a causal estimate.

| Arm vs CTRL | Total Δ mass | Both-alive size term | Death timing/incidence term | Arm-only mass | CTRL-only mass |
| --- | --- | --- | --- | --- | --- |
| LABELS | -75.145 | -55.327 | -19.818 | 106.894 | 126.711 |
| LABELS_KEEPV8 | -90.209 | -53.354 | -36.855 | 101.674 | 138.529 |
| LABELS_KEEPV8_LO | -83.246 | -48.722 | -34.524 | 104.902 | 139.426 |

Seed-paired decomposition (each row averages its 96 paired worlds):

| Arm | Seed | Total Δ | Both-alive size | Death timing/incidence |
| --- | --- | --- | --- | --- |
| LABELS | 10 | 6.781 | -60.255 | 67.036 |
| LABELS | 11 | -64.091 | -49.124 | -14.966 |
| LABELS | 12 | -131.650 | -65.366 | -66.283 |
| LABELS | 13 | -100.587 | -46.813 | -53.773 |
| LABELS | 14 | -86.181 | -55.078 | -31.102 |
| LABELS_KEEPV8 | 10 | -46.811 | -47.900 | 1.089 |
| LABELS_KEEPV8 | 11 | -131.654 | -52.554 | -79.100 |
| LABELS_KEEPV8 | 12 | -115.328 | -63.047 | -52.281 |
| LABELS_KEEPV8 | 13 | -90.307 | -53.581 | -36.726 |
| LABELS_KEEPV8 | 14 | -66.943 | -49.688 | -17.256 |
| LABELS_KEEPV8_LO | 10 | -71.170 | -43.881 | -27.289 |
| LABELS_KEEPV8_LO | 11 | -117.485 | -39.900 | -77.584 |
| LABELS_KEEPV8_LO | 12 | -116.062 | -54.969 | -61.093 |
| LABELS_KEEPV8_LO | 13 | -46.254 | -54.886 | 8.633 |
| LABELS_KEEPV8_LO | 14 | -65.259 | -49.972 | -15.287 |

Death-frame quartiles describe deaths only, with survivors censored at H5000. The fatal frame is one-based and contributes zero mass/alive frames.

| Arm | Arm deaths / 480 | CTRL deaths / 480 | Arm death Q25/Q50/Q75 | CTRL death Q25/Q50/Q75 |
| --- | --- | --- | --- | --- |
| LABELS | 199 | 217 | 1879.5 / 3151.0 / 4075.5 | 1982.0 / 3379.0 / 4269.0 |
| LABELS_KEEPV8 | 200 | 217 | 1599.5 / 2958.5 / 3843.2 | 1982.0 / 3379.0 / 4269.0 |
| LABELS_KEEPV8_LO | 212 | 217 | 2061.8 / 3122.0 / 3982.5 | 1982.0 / 3379.0 / 4269.0 |


| Arm | Seed | Arm deaths / 96 | CTRL deaths / 96 | Arm death quartiles | CTRL death quartiles |
| --- | --- | --- | --- | --- | --- |
| LABELS | 10 | 32 | 52 | 1950.0 / 3144.0 / 3925.2 | 1419.8 / 2920.5 / 4021.0 |
| LABELS | 11 | 34 | 33 | 2783.2 / 3849.5 / 4406.2 | 2101.0 / 3440.0 / 4422.0 |
| LABELS | 12 | 43 | 46 | 1671.0 / 2892.0 / 3747.0 | 2748.8 / 3438.0 / 4329.5 |
| LABELS | 13 | 47 | 41 | 1711.0 / 2817.0 / 4104.5 | 1980.0 / 2825.0 / 4132.0 |
| LABELS | 14 | 43 | 45 | 1926.0 / 3181.0 / 3930.0 | 2011.0 / 3658.0 / 4247.0 |
| LABELS_KEEPV8 | 10 | 39 | 52 | 1240.0 / 2298.0 / 3344.0 | 1419.8 / 2920.5 / 4021.0 |
| LABELS_KEEPV8 | 11 | 42 | 33 | 1823.2 / 3407.5 / 4046.8 | 2101.0 / 3440.0 / 4422.0 |
| LABELS_KEEPV8 | 12 | 44 | 46 | 2242.0 / 3136.0 / 3936.8 | 2748.8 / 3438.0 / 4329.5 |
| LABELS_KEEPV8 | 13 | 39 | 41 | 1573.0 / 2691.0 / 3707.0 | 1980.0 / 2825.0 / 4132.0 |
| LABELS_KEEPV8 | 14 | 36 | 45 | 1814.8 / 2977.5 / 3833.0 | 2011.0 / 3658.0 / 4247.0 |
| LABELS_KEEPV8_LO | 10 | 51 | 52 | 1607.0 / 3114.0 / 3716.0 | 1419.8 / 2920.5 / 4021.0 |
| LABELS_KEEPV8_LO | 11 | 41 | 33 | 2094.0 / 3114.0 / 3832.0 | 2101.0 / 3440.0 / 4422.0 |
| LABELS_KEEPV8_LO | 12 | 46 | 46 | 2223.5 / 3164.5 / 3927.2 | 2748.8 / 3438.0 / 4329.5 |
| LABELS_KEEPV8_LO | 13 | 36 | 41 | 2450.2 / 3282.0 / 4285.0 | 1980.0 / 2825.0 / 4132.0 |
| LABELS_KEEPV8_LO | 14 | 38 | 45 | 1642.5 / 3203.5 / 4110.0 | 2011.0 / 3658.0 / 4247.0 |

The length table uses a **fixed paired cohort where both policies survive to frame 5000** for all four timepoints. Means are post-transition hero logical length. The cohort can differ across label arms; its count is reported. These descriptive means pool matched survivor world pairs across seeds; seeds with more shared survivors contribute more pairs. Per-seed cohort means are retained in the evidence.

| Arm | Frame | Paired H5000 survivors | Arm length | CTRL length | Δ length |
| --- | --- | --- | --- | --- | --- |
| LABELS | 2000 | 153 | 665.2 | 667.4 | -2.2 |
| LABELS | 3000 | 153 | 839.5 | 964.6 | -125.1 |
| LABELS | 4000 | 153 | 931.4 | 1221.8 | -290.5 |
| LABELS | 5000 | 153 | 984.4 | 1444.8 | -460.4 |
| LABELS_KEEPV8 | 2000 | 151 | 648.3 | 667.7 | -19.4 |
| LABELS_KEEPV8 | 3000 | 151 | 845.0 | 958.3 | -113.2 |
| LABELS_KEEPV8 | 4000 | 151 | 936.4 | 1213.3 | -276.9 |
| LABELS_KEEPV8 | 5000 | 151 | 985.6 | 1431.8 | -446.2 |
| LABELS_KEEPV8_LO | 2000 | 144 | 662.7 | 669.5 | -6.8 |
| LABELS_KEEPV8_LO | 3000 | 144 | 854.8 | 960.5 | -105.7 |
| LABELS_KEEPV8_LO | 4000 | 144 | 941.2 | 1213.4 | -272.2 |
| LABELS_KEEPV8_LO | 5000 | 144 | 988.3 | 1430.8 | -442.5 |

## 3. Where the final arms disagree

Every living-hero decision on each CTRL final's round-62 scripted trajectory is evaluated by the matching-seed LABELS and LABELS_KEEPV8 finals on the same observation. CPU rowwise inference and advisory action masks match the original probes. These are action disagreements, not measured counterfactual outcomes. Boost disagreement means the actions differ in boost mode; remaining disagreements change direction within the same speed mode.

| Arm | CTRL length | Decisions | Disagreements | Rate | Cross-boost disagreements | Same-speed direction disagreements |
| --- | --- | --- | --- | --- | --- | --- |
| LABELS | all | 1985709 | 1335800 | 67.27% | 1005921 | 329879 |
| LABELS | <400 | 574277 | 222424 | 38.73% | 76645 | 145779 |
| LABELS | 400–<700 | 392986 | 161155 | 41.01% | 70244 | 90911 |
| LABELS | 700–<1000 | 425378 | 382279 | 89.87% | 341969 | 40310 |
| LABELS | ≥1000 | 593068 | 569942 | 96.10% | 517063 | 52879 |
| LABELS_KEEPV8 | all | 1985709 | 1011327 | 50.93% | 854968 | 156359 |
| LABELS_KEEPV8 | <400 | 574277 | 47855 | 8.33% | 4230 | 43625 |
| LABELS_KEEPV8 | 400–<700 | 392986 | 69122 | 17.59% | 37362 | 31760 |
| LABELS_KEEPV8 | 700–<1000 | 425378 | 330033 | 77.59% | 302834 | 27199 |
| LABELS_KEEPV8 | ≥1000 | 593068 | 564317 | 95.15% | 510542 | 53775 |

Q deciles use `−max` over all six raw CTRL Q outputs. Cutpoints pool all five CTRL trajectories once (not once per shadow model); D1 is lowest score and D10 highest. Ties go to the upper bin, so counts need not be exactly equal. Cross-seed Q scale differences can affect the pooled bins; per-seed counts/rates are retained in the evidence.

Cutpoints: -13.90814, -12.89241, -12.21915, -11.69228, -11.18663, -10.65534, -10.03173, -9.33360, -8.34059.

| Decile | Arm | Decisions | Disagreements | Rate (%) | Cross-boost disagreements |
| --- | --- | --- | --- | --- | --- |
| 1 | LABELS | 198571 | 80950 | 40.77 | 39263 |
| 1 | LABELS_KEEPV8 | 198571 | 24591 | 12.38 | 12690 |
| 2 | LABELS | 198571 | 92882 | 46.78 | 53683 |
| 2 | LABELS_KEEPV8 | 198571 | 42235 | 21.27 | 30106 |
| 3 | LABELS | 198571 | 97807 | 49.26 | 57946 |
| 3 | LABELS_KEEPV8 | 198571 | 50000 | 25.18 | 37324 |
| 4 | LABELS | 198571 | 102648 | 51.69 | 61383 |
| 4 | LABELS_KEEPV8 | 198571 | 56572 | 28.49 | 43145 |
| 5 | LABELS | 198570 | 112110 | 56.46 | 71860 |
| 5 | LABELS_KEEPV8 | 198570 | 70164 | 35.33 | 56729 |
| 6 | LABELS | 198570 | 131831 | 66.39 | 96425 |
| 6 | LABELS_KEEPV8 | 198570 | 96478 | 48.59 | 82532 |
| 7 | LABELS | 198572 | 167225 | 84.21 | 142090 |
| 7 | LABELS_KEEPV8 | 198572 | 141164 | 71.09 | 128286 |
| 8 | LABELS | 198571 | 187804 | 94.58 | 169175 |
| 8 | LABELS_KEEPV8 | 198571 | 175615 | 88.44 | 161167 |
| 9 | LABELS | 198571 | 193728 | 97.56 | 172522 |
| 9 | LABELS_KEEPV8 | 198571 | 190310 | 95.84 | 169643 |
| 10 | LABELS | 198571 | 168815 | 85.01 | 141574 |
| 10 | LABELS_KEEPV8 | 198571 | 164198 | 82.69 | 133346 |

## Reading

The bank correction is learned: median separation moves from −0.067 Q in CTRL to +0.207–0.236 Q, with raw agreement 94.66–99.66% versus 39.83%. The effective 0.06320-Q margin is small; the proposed large-margin overshoot explanation is ruled out. Strong agreement on training-bank rows does not show that their B8-continuation labels improve a student's own trajectories.

All three arms lose mass. Lower size on common-alive frames accounts for 55.33 of the 75.15 LABELS loss, 53.35 of the 90.21 KEEPV8 loss, and 48.72 of the 83.25 LO loss. The remaining 19.82/36.85/34.52 losses arise on frames when only one member of a pair remains alive. Total deaths are slightly lower (199/200/212 versus 217), but the pooled death-frame medians are earlier (3151/2958.5/3122 versus 3379). Thus “more deaths” is not the explanation; which worlds lose their hero, when, and at what mass still matters. The unequal-survival-window term includes the surviving policy's size and is not a pure causal death effect. Within fixed shared-survivor cohorts, all three arms finish approximately 443–460 length units smaller, supporting a substantial late-growth deficit.

Action changes are broad but concentrated at large lengths. LABELS and KEEPV8 disagree on 67.27% and 50.93% of all CTRL decisions. Below length 700, the rates are still 39.66% and 12.09%; however, those states account for only 28.72% and 11.57% of all disagreements, so most changes do not occur in ordinary short-length states. Disagreements also occur in the lowest raw-Q risk decile (40.77%/12.38%) and are much more frequent in upper deciles. Boost-mode changes account for 75.30%/84.54% of disagreements. This supports behavioral spillover beyond the bank's late-state region, while the on-CTRL comparison alone cannot identify which changes cause lower growth or deaths. No new T3 decision, promotion, or training recommendation follows from these descriptive diagnostics.

## Reproduction and validation

Specification: [T3 diagnostics spec](specs/t3_diagnostics_spec_2026-10-16.md), with the requested diagnostic questions from [advisory section 3](advice/2026-10-15_t3_interim_next.md). The margin scale was also checked directly in all 15 training summaries; their hashes are retained in the evidence.

The [evidence README](t3_diagnostics_2026-10-16_evidence/README.md) gives the CPU-only reproduction commands. Tooling was committed and pushed in `5e87c7c` (integration revision `d2117a1`); subsequent renderer adjustments only clarify table labels, ordering, and wording. The [bank JSON](t3_diagnostics_2026-10-16_evidence/bank-separation.json), [replay summary](t3_diagnostics_2026-10-16_evidence/replay-summary.json), and [validation receipt](t3_diagnostics_2026-10-16_evidence/validation.json) retain the numerical evidence and input identities.

- All 20 checkpoint replays passed exact per-world comparison with the round-62 source probes for mass, deaths, alive frames, death frames, and pre-death length.
- Independent artifact checks covered all 1,920 checkpoint/world pairs: source and array hashes, zero mass after death, source mass/alive equality, shared CTRL observations for both shadow arms, decision counts, death counts, fixed H5000 cohorts, and all 15 seed/arm integer mass-decomposition identities. Length, decile, and per-seed counts partition the full disagreement population exactly.
- Focused diagnostic and existing label-bank trainer tests: **21 passed**. Black, isort, flake8, and `git diff --check` passed for the changed code and tests.
- All task-owned numerical workers exited and were reaped. No training was launched; all diagnostic inference used CPU, never MPS.

The full NPZ trajectories remain local and hash-bound, rather than committed as large binary files. Raw bank inference uses batches of 256; all replay and shadow decisions use CPU rowwise inference. These remain development measurements and do not alter the registered T3 decision.
