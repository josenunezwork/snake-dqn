# T4c tightened two-sided screen result

**Decision: KILL**. This is development-screen evidence only.

## Registered decision record

Round 66: 96 scripted / 32 frozen / 32 mixed worlds per checkpoint. The single round-67 extension adds 96 scripted worlds to candidate and CTRL; when complete, scripted comparisons use 192 worlds per seed. Incumbent comparisons and shared-survivor lengths retain round 66 only.

Initial decision: EXTEND; extension started: True; extension complete: True.

| mix | survival vs CTRL | survival SE | mass vs CTRL | mass SE | mass vs incumbent | incumbent SE |
|---|---:|---:|---:|---:|---:|---:|
| scripted | -0.013542 | 0.010233 | -9.899558 | 6.811934 | 171.194558 | 13.004060 |
| frozen | 0.012500 | 0.028980 | -12.586586 | 9.283476 | 90.858341 | 11.486366 |
| mixed | 0.037500 | 0.031869 | 9.048476 | 17.660262 | 154.055384 | 11.281359 |

GO guardrails: scripted survival >= +0.06; mass >= -30 on every mix; frozen and mixed survival >= -0.05. KILL if scripted survival <= 0 or a training exit is nonzero.

## Required diagnostics

### Final raw bank agreement

| family | five-seed mean | n |
|---|---:|---:|
| retained_decisive_all | 0.478254 | 5 |
| retained_decisive_gap_lt100 | 0.735028 | 5 |
| retained_decisive_gap_100_300 | 0.394592 | 5 |
| in_window_counter | 0.998199 | 5 |
| outside_counter | 0.898852 | 5 |

| seed | decisive all | <100 gap | 100-300 gap | in-window | outside |
|---:|---:|---:|---:|---:|---:|
| TWO_SIDED_T_s10 | 10885 / 0.454019 | 2675 / 0.719252 | 8210 / 0.367600 | 5442 / 0.997611 | 10885 / 0.892513 |
| TWO_SIDED_T_s11 | 10885 / 0.556086 | 2675 / 0.815701 | 8210 / 0.471498 | 5442 / 0.999265 | 10885 / 0.913275 |
| TWO_SIDED_T_s12 | 10885 / 0.445935 | 2675 / 0.699813 | 8210 / 0.363216 | 5442 / 0.998346 | 10885 / 0.891502 |
| TWO_SIDED_T_s13 | 10885 / 0.491318 | 2675 / 0.740561 | 8210 / 0.410110 | 5442 / 0.997795 | 10885 / 0.881488 |
| TWO_SIDED_T_s14 | 10885 / 0.443914 | 2675 / 0.699813 | 8210 / 0.360536 | 5442 / 0.997979 | 10885 / 0.915480 |

### Final 100k sampled telemetry

| family | five-seed mean | n |
|---|---:|---:|
| bank_agree_decisive | 0.462018 | 5 |
| bank_agree_counter | 0.998276 | 5 |
| bank_agree_outside | 0.904194 | 5 |

| seed | decisive | counter | outside |
|---:|---:|---:|---:|
| TWO_SIDED_T_s10 | 0.431050 | 0.997960 | 0.897888 |
| TWO_SIDED_T_s11 | 0.536762 | 0.999120 | 0.917556 |
| TWO_SIDED_T_s12 | 0.427483 | 0.998180 | 0.902467 |
| TWO_SIDED_T_s13 | 0.462300 | 0.997679 | 0.901771 |
| TWO_SIDED_T_s14 | 0.452496 | 0.998440 | 0.901286 |

### Same-init drift readout (r62)

| seed | candidate raw | candidate consequential | B8 raw | B8 consequential |
|---:|---:|---:|---:|---:|
| 10 | 0.203666 | 0.041826 | 0.088149 | 0.008654 |
| 11 | 0.232440 | 0.029162 | 0.101685 | 0.012134 |
| 12 | 0.209468 | 0.042987 | 0.084699 | 0.008792 |
| 13 | 0.241830 | 0.042323 | 0.095694 | 0.012806 |
| 14 | 0.201657 | 0.034159 | 0.081643 | 0.007130 |

Candidate five-seed raw / consequential: 0.217812 / 0.038091.
B8 five-seed raw / consequential: 0.090374 / 0.009903. Within 10 percentage points: True.

### Shared-survivor H5000 length

Five-seed mean: -7.275440 across 5 seed means; pooled: -7.376712 across 146 shared survivors; threshold >= -50.000000.

| seed | shared survivors | candidate minus CTRL |
|---:|---:|---:|
| 10 | 31 | -46.000000 |
| 11 | 30 | 9.400000 |
| 12 | 24 | -17.875000 |
| 13 | 35 | 2.828571 |
| 14 | 26 | 15.269231 |

## Context: T4a TWO_SIDED and T3

- t4a TWO_SIDED: KILL; scripted survival -0.027083, mass -24.126891.
- t3 LABELS: EXTEND; scripted survival 0.008333, mass -87.970490.
- t3 LABELS_KEEPV8: EXTEND; scripted survival 0.016667, mass -87.907298.
- t3 LABELS_KEEPV8_LO: KILL; scripted survival -0.010417, mass -81.155722.

## Declared fork

(c) falls but survival does not move: return regression next, with outside-bank branches rolled out too, which is the principled form of item 2 and supplies "greedy is fine here" targets with magnitudes.

## Interpretation and execution evidence

The initial round-66 primary was +0.016666667 (seed SE 0.015659684), requiring EXTEND.
After the sole extension, candidate survived 537/960 scripted episodes versus CTRL 550/960.
The primary is -0.013541666667 (seed SE 0.010232751409; five training seeds), so KILL applies.
All three mass guards and both secondary survival guards pass. No second extension was run.

Full-bank decisive agreement misses 0.85, including both distance strata (weights 1.0 and 0.3).
In-window agreement passes 0.95; outside-counter agreement misses 0.95. Table cells show
row count / raw six-action agreement, with separate family denominators. Sampled telemetry
uses each run's final post-7.9M records through ordinary packet overshoot; it is separate
from full-bank inference. Drift uses CTRL's round-62 outside-bank states at length >=700,
the original pooled length-bucket × CTRL -max-Q-decile standardization, and a consequential
flip threshold strictly greater than 0.3 CTRL Q. The consequential mean exceeds same-init
B8 drift by 0.028188 (2.819 percentage points), within the registered 10-point band.
The fork above is a research direction, not authorization to launch it or promotion evidence.
T3 EXTEND labels above are the saved contextual screen decisions, not completed extensions.

All five final training summaries have no halt and 8,000,000–8,000,128 transitions; all five
terminal training statuses are zero. All 33 round-66 rows and 10 round-67 rows completed.
CTRL and incumbent round-66 probes finished before candidate gameplay. Probes used CPU
rowwise inference, advisory masks, numba vector61, fast anchors, no candidate veto, and
at most two single-thread workers. The final watcher invocation exited zero. The retained
session-exit log also contains the earlier owned watcher's orderly restart status 143.

The initial freshness audit passed. The launch-time round-67 audit has `extension_pass: true`;
its aggregate `status: FAIL` records expected round-66 consumption by this screen, while
round 67 has no collision. Both receipts are retained unchanged. Independent review
reconstructed all registered world IDs, checked disjoint rounds and paired worlds, and
verified checkpoint hashes and decision arithmetic. Historical drift receipts bind all
five captures and their final checkpoint identities. Large captures remain in the artifact
root; their hashes are preserved in the completion receipt.

Portable evidence: [analysis](../../research/t3_screen_20261015/t4c_results/analysis.json),
[initial analysis](../../research/t3_screen_20261015/t4c_results/initial_analysis.json),
[world-level probes](../../research/t3_screen_20261015/t4c_results/probes.jsonl),
[diagnostics](../../research/t3_screen_20261015/t4c_results/diagnostics.json),
[training receipts](../../research/t3_screen_20261015/t4c_results/training_receipts.json),
and [SHA-256 manifest](../../research/t3_screen_20261015/t4c_results/manifest.json).

Validation: 28 focused T4c tests passed (probe parity, analysis, diagnostics, drift, watcher, regressions, seed audits); flake8 and `git diff --check` passed. Independent final review confirmed all family means, weighted drift rates, capture/checkpoint hashes, and the exact second fork branch.
