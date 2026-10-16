# T4a post-hoc spillover and growth diagnostics — 2026-10-16

Diagnostic only. These results do not change GO/KILL/EXTEND or authorize promotion.

## Scope and measurement

All five TWO_SIDED finals and corresponding CTRL, LABELS and LABELS_KEEPV8 finals (training seeds 10–14) are included. The shared diagnostic worlds are the 96 round-62 scripted H5000 worlds used in the T3 report. These are separate from the registered round-65 gameplay evaluation. The same worlds recur across training seeds; the 480 seed/world pairs are not 480 independent worlds.

Action changes use raw six-action first argmax for all three sets. Decile scores are −max of raw CTRL Q; common cutpoints pool all five CTRL trajectories once. D1 is the lowest score and D10 the highest; ties enter the upper bin. Bank sides use original rollout survival of the stored B8 action. The bank is the complete v2 bank, with preferred surviving labels for decisive agreement and exact stored B8 action for counter agreement. Set (c) excludes exact canonical bank-observation fingerprints. These are action comparisons, not counterfactual survival outcomes.

The tables standardize (a), (b) and (c) to identical pooled-bank weights within common length-bucket × CTRL-Q-decile strata. Length buckets are <400, 400–<700, 700–<1000 and ≥1000. Within each decile the common length weights are renormalized. Unsupported strata are excluded without extrapolation; raw counts, exclusions, stratum rates and weights are preserved in the JSON evidence. Cross-boost share is the standardized cross-speed change rate divided by the standardized total action-change rate, not a mean of stratum percentages.

Observed common support is the 700–<1000 and ≥1000 buckets in all ten deciles. It retains all
54,425 decisive and 27,210 counter model/row evaluations (the same 10,885/5,442 bank rows
evaluated by five finals), and 1,018,446 of 1,985,709 CTRL decisions. No CTRL observation
fingerprint occurs in the bank; the 967,263 excluded CTRL decisions are outside the matched
length support, not exact bank matches. No claim is made about matched spillover below
length 700. Without matching/standardization, the all-CTRL raw action-change rates are
36.66% for TWO_SIDED, 67.36% for LABELS, and 51.01% for KEEPV8.

Shared raw-Q decile cutpoints are −13.908135, −12.892414, −12.219148, −11.692276,
−11.186625, −10.655344, −10.031733, −9.333603, and −8.340591.

## Matched spillover

| Arm | State set | Retained decisions | Argmax change | Cross-boost share of changes |
| --- | --- | ---: | ---: | ---: |
| TWO_SIDED | (a) decisive bank | 54425 | 88.42% | 13.02% |
| TWO_SIDED | (b) counter bank | 27210 | 16.06% | 9.55% |
| TWO_SIDED | (c) outside-bank CTRL | 1018446 | 53.22% | 10.98% |
| LABELS | (a) decisive bank | 54425 | 97.76% | 90.67% |
| LABELS | (b) counter bank | 27210 | 87.98% | 88.79% |
| LABELS | (c) outside-bank CTRL | 1018446 | 92.74% | 89.55% |
| LABELS_KEEPV8 | (a) decisive bank | 54425 | 96.14% | 91.37% |
| LABELS_KEEPV8 | (b) counter bank | 27210 | 83.09% | 90.13% |
| LABELS_KEEPV8 | (c) outside-bank CTRL | 1018446 | 86.68% | 90.05% |

Each decile cell is **argmax-change rate (cross-boost share of changes)**.

| Decile | Arm | (a) Decisive bank | (b) Counter bank | (c) Outside-bank CTRL |
| --- | --- | ---: | ---: | ---: |
| 1 | TWO_SIDED | 90.04% (3.89%) | 8.07% (0.00%) | 44.42% (3.50%) |
| 1 | LABELS | 98.75% (92.84%) | 86.54% (88.67%) | 82.92% (87.11%) |
| 1 | LABELS_KEEPV8 | 93.97% (94.69%) | 77.12% (96.01%) | 66.25% (93.03%) |
| 2 | TWO_SIDED | 87.62% (2.48%) | 3.49% (0.00%) | 49.63% (3.51%) |
| 2 | LABELS | 98.52% (93.65%) | 87.63% (82.28%) | 89.58% (90.95%) |
| 2 | LABELS_KEEPV8 | 94.58% (95.53%) | 82.99% (90.19%) | 78.41% (95.35%) |
| 3 | TWO_SIDED | 88.40% (3.20%) | 3.78% (10.85%) | 50.15% (3.37%) |
| 3 | LABELS | 99.09% (94.28%) | 87.28% (94.68%) | 91.30% (91.95%) |
| 3 | LABELS_KEEPV8 | 96.32% (96.28%) | 78.77% (97.34%) | 81.36% (95.82%) |
| 4 | TWO_SIDED | 89.58% (6.48%) | 8.03% (13.04%) | 47.62% (3.69%) |
| 4 | LABELS | 99.06% (92.49%) | 84.02% (90.93%) | 91.96% (92.21%) |
| 4 | LABELS_KEEPV8 | 94.85% (94.77%) | 72.89% (95.95%) | 80.92% (95.72%) |
| 5 | TWO_SIDED | 88.56% (7.12%) | 8.61% (3.93%) | 48.00% (3.89%) |
| 5 | LABELS | 98.66% (92.33%) | 87.87% (93.20%) | 92.24% (92.57%) |
| 5 | LABELS_KEEPV8 | 95.00% (95.13%) | 80.30% (96.83%) | 81.31% (95.53%) |
| 6 | TWO_SIDED | 89.68% (6.08%) | 8.93% (5.16%) | 47.86% (3.91%) |
| 6 | LABELS | 98.92% (93.82%) | 89.75% (92.92%) | 93.58% (92.50%) |
| 6 | LABELS_KEEPV8 | 96.15% (95.57%) | 81.34% (94.91%) | 82.83% (94.76%) |
| 7 | TWO_SIDED | 90.51% (5.77%) | 11.95% (3.86%) | 48.60% (4.53%) |
| 7 | LABELS | 99.21% (93.90%) | 92.62% (91.98%) | 94.95% (92.39%) |
| 7 | LABELS_KEEPV8 | 97.61% (94.88%) | 84.47% (93.43%) | 85.57% (94.36%) |
| 8 | TWO_SIDED | 91.57% (7.97%) | 12.39% (5.00%) | 51.97% (5.83%) |
| 8 | LABELS | 99.30% (92.50%) | 92.31% (90.87%) | 96.28% (91.16%) |
| 8 | LABELS_KEEPV8 | 98.16% (93.48%) | 86.70% (92.89%) | 90.62% (92.33%) |
| 9 | TWO_SIDED | 90.74% (11.38%) | 13.22% (7.14%) | 58.22% (9.63%) |
| 9 | LABELS | 98.85% (90.09%) | 92.09% (88.03%) | 97.43% (89.09%) |
| 9 | LABELS_KEEPV8 | 98.37% (90.77%) | 90.51% (88.97%) | 95.04% (89.11%) |
| 10 | TWO_SIDED | 81.66% (32.45%) | 32.40% (14.10%) | 57.75% (25.48%) |
| 10 | LABELS | 93.47% (84.58%) | 77.65% (81.69%) | 84.74% (83.70%) |
| 10 | LABELS_KEEPV8 | 92.05% (83.57%) | 75.83% (80.99%) | 81.93% (81.02%) |

## Growth on shared survivors

The primary curve uses worlds alive under both policies at each named frame. The cohort can change with time and differs by arm. Means pool matched survivor pairs across training seeds. The fixed-H5000 curve below uses the same cohort at every timepoint and is directly comparable to the T3 report.

### Frame-specific shared survivors

| Arm | Frame | Shared survivors | Arm mean length | CTRL mean length | Δ length |
| --- | ---: | ---: | ---: | ---: | ---: |
| TWO_SIDED | 1000 | 440 | 326.49 | 324.30 | 2.19 |
| TWO_SIDED | 2000 | 373 | 662.38 | 665.49 | -3.10 |
| TWO_SIDED | 3000 | 307 | 917.79 | 959.52 | -41.73 |
| TWO_SIDED | 4000 | 221 | 1126.43 | 1212.36 | -85.93 |
| TWO_SIDED | 5000 | 139 | 1310.24 | 1444.75 | -134.51 |
| LABELS | 1000 | 442 | 330.92 | 323.85 | 7.07 |
| LABELS | 2000 | 376 | 669.62 | 664.09 | 5.53 |
| LABELS | 3000 | 317 | 841.91 | 959.86 | -117.95 |
| LABELS | 4000 | 234 | 931.48 | 1209.67 | -278.18 |
| LABELS | 5000 | 153 | 984.37 | 1444.75 | -460.38 |
| LABELS_KEEPV8 | 1000 | 441 | 325.12 | 323.60 | 1.52 |
| LABELS_KEEPV8 | 2000 | 373 | 657.46 | 664.10 | -6.65 |
| LABELS_KEEPV8 | 3000 | 309 | 849.49 | 955.96 | -106.48 |
| LABELS_KEEPV8 | 4000 | 225 | 936.73 | 1203.51 | -266.78 |
| LABELS_KEEPV8 | 5000 | 151 | 985.65 | 1431.81 | -446.16 |

### Fixed H5000 shared survivors

| Arm | Frame | Shared survivors | Arm mean length | CTRL mean length | Δ length |
| --- | ---: | ---: | ---: | ---: | ---: |
| TWO_SIDED | 1000 | 139 | 333.17 | 328.43 | 4.74 |
| TWO_SIDED | 2000 | 139 | 673.60 | 675.96 | -2.35 |
| TWO_SIDED | 3000 | 139 | 923.96 | 967.94 | -43.98 |
| TWO_SIDED | 4000 | 139 | 1129.20 | 1225.65 | -96.45 |
| TWO_SIDED | 5000 | 139 | 1310.24 | 1444.75 | -134.51 |
| LABELS | 1000 | 153 | 333.16 | 325.28 | 7.88 |
| LABELS | 2000 | 153 | 665.17 | 667.41 | -2.24 |
| LABELS | 3000 | 153 | 839.49 | 964.57 | -125.08 |
| LABELS | 4000 | 153 | 931.35 | 1221.82 | -290.47 |
| LABELS | 5000 | 153 | 984.37 | 1444.75 | -460.38 |
| LABELS_KEEPV8 | 1000 | 151 | 318.28 | 325.50 | -7.21 |
| LABELS_KEEPV8 | 2000 | 151 | 648.27 | 667.67 | -19.40 |
| LABELS_KEEPV8 | 3000 | 151 | 845.04 | 958.28 | -113.25 |
| LABELS_KEEPV8 | 4000 | 151 | 936.43 | 1213.28 | -276.85 |
| LABELS_KEEPV8 | 5000 | 151 | 985.65 | 1431.81 | -446.16 |

## Bank target agreement

These full-bank final-checkpoint agreements are distinct from agreement with CTRL's argmax. Decisive targets use the v2 preferred surviving set; counter targets require the stored B8 action. All arms use the same v2 rows for contrast.

The low T3 agreement numbers here do not contradict the earlier T3 report's high agreement
with its original surviving sets: v2 prefers nonboost survivors, and its counter target is
the single stored B8 action. This is a different target definition, applied consistently
across the three finals for this contrast.

| Arm | Seed | Decisive target agreement | Counter target agreement |
| --- | --- | ---: | ---: |
| TWO_SIDED | 10 | 92.72% | 99.93% |
| TWO_SIDED | 11 | 95.08% | 99.98% |
| TWO_SIDED | 12 | 92.58% | 99.89% |
| TWO_SIDED | 13 | 92.84% | 99.93% |
| TWO_SIDED | 14 | 93.28% | 99.89% |
| TWO_SIDED | five-seed mean | 93.30% | 99.92% |
| LABELS | 10 | 26.90% | 11.96% |
| LABELS | 11 | 25.59% | 13.74% |
| LABELS | 12 | 25.40% | 12.05% |
| LABELS | 13 | 24.45% | 11.89% |
| LABELS | 14 | 26.19% | 11.14% |
| LABELS | five-seed mean | 25.71% | 12.16% |
| LABELS_KEEPV8 | 10 | 24.11% | 17.48% |
| LABELS_KEEPV8 | 11 | 24.70% | 18.71% |
| LABELS_KEEPV8 | 12 | 23.26% | 16.72% |
| LABELS_KEEPV8 | 13 | 21.66% | 13.45% |
| LABELS_KEEPV8 | 14 | 26.43% | 17.84% |
| LABELS_KEEPV8 | five-seed mean | 24.03% | 16.84% |

Registered TWO_SIDED final-100k training-window readout:

| Seed | Transitions | Decisive agreement | Counter agreement | Greedy boost |
| --- | ---: | ---: | ---: | ---: |
| 10 | 8000000 | 92.80% | 99.91% | 0.37% |
| 11 | 8000000 | 94.87% | 99.98% | 0.20% |
| 12 | 8000128 | 92.46% | 99.87% | 0.21% |
| 13 | 8000000 | 92.72% | 99.93% | 0.36% |
| 14 | 8000128 | 93.17% | 99.94% | 0.22% |
| five-seed mean | — | 93.20% | 99.93% | 0.27% |

## Declared interpretation (verbatim)

Interpretation, written before the data: counter agree > 0.9 AND (c) low AND decisive agree > 0.9
-> inputs carry the distinction, expert iteration next; decisive high but (c) high -> the loss
still generalises, tighten margin/counter matching before return regression; decisive agree stuck
below ~0.8 while counter agree holds -> states ambiguous at the input, move to inputs.

## Application to these results

**The observed reading is the second declared branch: decisive agreement is high, but
outside-bank spillover remains high. The loss still generalises; tighten margin/counter
matching before return regression.** This is a diagnostic direction, not authorization
to launch another experiment.

TWO_SIDED's five-seed final training-window agreement is 93.20% decisive and 99.93% counter;
full-bank final-checkpoint agreement independently gives 93.30% and 99.92%. Every seed
exceeds 0.9 on both sides. Yet matched outside-bank action change is 53.22%, ranging from
44.42% to 58.22% across deciles. The pre-registration does not define a numerical cutoff
for “low” or “high”; changes on roughly half the matched states in every decile support
the high-spillover reading without creating a new decision threshold. This is not the
low-(c) condition for the expert-iteration branch, nor the decisive-below-0.8 input-limit
pattern. Strong bank discrimination alone does not establish targeted behavioural change.

Compared with LABELS/KEEPV8, TWO_SIDED substantially reduces counter-bank action change
(16.06% versus 87.98%/83.09%) and outside-bank change (53.22% versus 92.74%/86.68%). Its
remaining outside-bank changes are mainly within the same speed mode: cross-boost share
is 10.98%, versus 89.55%/90.05%. The 16.06% counter change is relative to CTRL, while the
99.92% counter target agreement is relative to B8's stored action; these are different
comparisons and are not inconsistent.

Growth improves descriptively relative to the T3 label arms but still trails CTRL. On
frame-specific shared survivors, TWO_SIDED's length delta evolves from +2.19 at frame 1000
to −3.10, −41.73, −85.93 and −134.51 at frames 2000–5000. At 5000, its mean length is
1310.24 against paired CTRL's 1444.75, compared with T3 deficits of −460.38/−446.16. The
fixed-H5000 cohort also shows a late deficit (−43.98 at 3000, −96.45 at 4000, −134.51 at
5000). Different arms have different survivor cohorts, so this is not a causal estimate
of how much two-sided training restores growth. Neither this curve nor the spillover
readout alters the separately registered gameplay decision.

## Reproduction and evidence

The [tooling README](../../research/t4a_posthoc_20261016/README.md) documents all commands.
The [readiness receipt](t4a_posthoc_2026-10-16_evidence/part2-ready.json) records all five
successful terminal finals before Part 2 execution. The
[complete summary](t4a_posthoc_2026-10-16_evidence/posthoc-summary.json) preserves producer
receipts, input/output SHA-256 bindings, full raw and matched stratum counts, weights,
per-seed growth and agreement, and final training-window values. Full NPZ captures remain
at `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t4a-posthoc-20261016/`.

All five CTRL replays passed exact historical round-62 record parity. TWO_SIDED growth
series passed independent consistency checks against their emitted evaluator records for
mass, alive frames, deaths and terminal death frames; no historical TWO_SIDED round-62
source row exists. The T3 growth series were reused after their original receipt and input
hashes were verified. All 15 bank captures used full-bank rowwise CPU inference.

The final focused suite passed **30 tests**; flake8 and `git diff --check` passed. Numerical
work used CPU only, at most three single-thread workers, never MPS. All capture workers
exited successfully and were reaped. No protected tmux session, protected worktree, or
other owner's process was modified or controlled.

The [independent raw-artifact audit](t4a_posthoc_2026-10-16_evidence/part2-independent-audit.json)
passed **85/85 checks** without invoking the summary pipeline. It verified bank receipts
and hashes, exact shared CTRL observations and historical T3 identities, all 480
TWO_SIDED world-record closures, and independently recomputed matched per-decile
spillover and both pooled growth cohorts. A separate read-only reporting review found
no actionable issues. These are diagnostic checks, not gameplay-gate evidence.
