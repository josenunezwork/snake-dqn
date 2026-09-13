# Exploration-floor diagnostic — September 13, 2026

Status: **NOT_ADVANCED.** Raising the epsilon floor from 0.02 to 0.20 passed eight
of twelve predeclared fixed-world conditions, but did not replicate across two fresh
matched training seeds and both opponent mixes. T150 exceeded C150 in three panels.
In the remaining panel, seed 2026091502 against random-safe, T150 lost to C150 on
six of eight worlds. Both random-safe treatment finals were below their shared
initial mean. The greedy-food gains do not establish general improvement. No
checkpoint advanced, no incumbent comparison ran, and no model was promoted.

## Question and frozen method

This screen asked whether epsilon ending at 0.20 rather than 0.02 improves retained
full-horizon mass under canonical Watch opponents. Each arm used frozen source
`c777b04345e5c18aeb9721083622cb8a0850958c`, corrected-v3/raster31v3,
`watch_pre_move_v1`, beta zero, the same native Adam and exact one-epoch sampler,
and a fixed random-safe policy source. Within each seed, arms started from identical
initial tensors; `eps_end` was the only declared pair difference.

Both arms annealed from epsilon 1.0 during the first 30k useful hero transitions.
The floor therefore changes the full 0-to-30k exploration curve, rather than only a
final exploration setting. Each of four MPS arms targeted 150k useful hero
transitions and saved initial, 50k, and 150k checkpoints. Actual training totaled
600,665 useful transitions.

Ten CPU evaluation batches scored the shared initial and four arm-phase roles per
seed on eight fixed fresh worlds (2026091511–2026091518), both random-safe and
greedy-food mixes, and 5,000 frames. The 20 calls produced 160 finite, uniquely
keyed rows. Twelve native snapshots passed preflight before any world was allocated.

The rule required, in **both** seeds and **both** mixes: mean T150 > C150, mean
T150 >= T50, and mean T150 > shared initial. Any failed condition gives
NOT_ADVANCED. Food and survival are descriptive only and cannot rescue a failed mass
condition. Eight worlds are paired diagnostic blocks, not a population sample or a
confidence interval.

## Full-horizon mass

Values are means across the eight fixed worlds. I is the shared initial checkpoint;
C and T are control and treatment. The last column counts the passed conditions in
that panel.

| Training seed | Opponent mix | I | C50 | C150 | T50 | T150 | Conditions |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026091501 | Random-safe | 2.8442 | 0.9954 | 2.1353 | 2.4307 | 2.5610 | 2/3 |
| 2026091501 | Greedy-food | 1.9048 | 0.6230 | 1.4533 | 1.7201 | 1.9760 | 3/3 |
| 2026091502 | Random-safe | 1.6412 | 1.5705 | 1.6912 | 1.1999 | 1.1302 | 0/3 |
| 2026091502 | Greedy-food | 1.3063 | 0.9494 | 0.7975 | 1.1347 | 2.3651 | 3/3 |

The paired summaries retain world-level direction and sensitivity. `LOO` is the
range of the paired mean after omitting one of the eight worlds; it is descriptive,
not a confidence interval.

| Seed | Mix | Contrast | Mean | Median | World signs | LOO mean range |
| ---: | --- | --- | ---: | ---: | --- | --- |
| 2026091501 | Random-safe | T150 − C150 | +0.4258 | +0.7347 | +5 / −3 | +0.0832 to +0.7611 |
| 2026091501 | Random-safe | T150 − T50 | +0.1303 | +0.2168 | +4 / −4 | −0.3321 to +0.6964 |
| 2026091501 | Random-safe | T150 − I | −0.2832 | −0.0889 | +3 / −5 | −0.7161 to +0.1958 |
| 2026091501 | Greedy-food | T150 − C150 | +0.5227 | +0.5158 | +5 / −3 | +0.0315 to +0.9172 |
| 2026091501 | Greedy-food | T150 − T50 | +0.2559 | +0.4197 | +5 / −3 | −0.1552 to +0.6053 |
| 2026091501 | Greedy-food | T150 − I | +0.0712 | −0.1583 | +4 / −4 | −0.4469 to +0.3827 |
| 2026091502 | Random-safe | T150 − C150 | −0.5610 | −0.6824 | +2 / −6 | −0.7893 to −0.3780 |
| 2026091502 | Random-safe | T150 − T50 | −0.0697 | −0.2773 | +2 / −6 | −0.2967 to +0.1815 |
| 2026091502 | Random-safe | T150 − I | −0.5110 | −0.5757 | +3 / −5 | −0.7829 to −0.2681 |
| 2026091502 | Greedy-food | T150 − C150 | +1.5676 | +0.3345 | +5 / −3 | +0.4819 to +1.8246 |
| 2026091502 | Greedy-food | T150 − T50 | +1.2304 | +0.5365 | +4 / −4 | +0.1951 to +1.7344 |
| 2026091502 | Greedy-food | T150 − I | +1.0589 | +0.6348 | +6 / −2 | +0.1524 to +1.5113 |

![Fixed-world mass trajectories](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/exploration-floor/plot-v3/exploration-floor-mass-trajectories.png>)

The readable V3 plot was visually inspected and passed. It supersedes V1, whose
initial visual claim was premature because its footer overlapped the x-axis labels,
and V2, whose title and footer overlapped. The inputs and failed layout attempts
remain preserved in the [V1 visual inspection](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/exploration-floor/plot-v1/visual-qa.json>)
and [V2 visual inspection](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/exploration-floor/plot-v2/visual-qa.json>).

## Secondary food and survival diagnostics

These are evaluation outcomes, not decision criteria. Each entry is mean food eaten
or survival fraction in I / C50 / C150 / T50 / T150 order. This campaign preceded
the later `633cf6f` training-side valid-hero food-contact telemetry, so it contains
no such training metric.

| Training seed | Opponent mix | Food eaten by phase | Survival by phase |
| ---: | --- | --- | --- |
| 2026091501 | Random-safe | 81.750 / 0.375 / 20.875 / 63.375 / 23.375 | 0.7113 / 0.6987 / 0.1974 / 0.5971 / 0.7452 |
| 2026091501 | Greedy-food | 8.250 / 0.875 / 17.750 / 43.625 / 38.250 | 0.6249 / 0.3641 / 0.1231 / 0.3999 / 0.5299 |
| 2026091502 | Random-safe | 1.250 / 16.375 / 11.875 / 29.375 / 10.375 | 0.8728 / 0.2730 / 0.2963 / 0.1876 / 0.2557 |
| 2026091502 | Greedy-food | 3.875 / 13.875 / 11.250 / 29.750 / 18.250 | 0.5654 / 0.1284 / 0.1321 / 0.1847 / 0.2349 |

No direct counter measured actual exploratory actions or spatial-state coverage.
These behavior outcomes therefore cannot identify whether the epsilon schedule
changed either mechanism.

## Qualification, execution, and resources

The final independent audit passed 48 checks with no failures or pending items. It
verified frozen inputs, matched initial tensors and the allowlisted pair difference,
12 native checkpoint preflights, 146 input-role hashes, the complete 20-call/160-row
matrix, finite rows, serial closure, and the final resource closure. The final
analysis SHA-256 is `cf4d005d4a9bd3419dcfdfacc5f1fb91b03584e48fc58977ff0d2c297831dc65`.

All numerical work used the exclusive serialized slot and CPU thread limit two.
Training ran on MPS under frozen 4 GiB child-RSS, 8 GiB MPS-driver, and 12 GiB
available-memory safety limits. Every training arm exited naturally within its
600-second cap.

| Arm | Useful steps | Elapsed | Peak RSS from receipt |
| --- | ---: | ---: | ---: |
| Seed 2026091501 control | 150,142 | 293.213 s | 1,595,310,080 B |
| Seed 2026091501 treatment | 150,205 | 286.100 s | 1,476,395,008 B |
| Seed 2026091502 treatment | 150,094 | 281.231 s | 1,480,900,608 B |
| Seed 2026091502 control | 150,224 | 292.448 s | 1,588,215,808 B |

The closed wave-local log has 1,064 samples and no sampling errors. Minimum host
available memory was 27,589,197,824 B (25.69 GiB); aggregate CPU reached 94.9% and
aggregate GPU 83%. These are host readings, not learner attribution, and GPU load
may include other applications. The raw thermal strings record no thermal or
performance warning. The log covers all primary work through V1; V2 and V3 each
retain their own plot-supervisor receipts.

## Evidence, context, and next decision

- [Frozen protocol](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/exploration-floor/protocol.json>), [execution freeze](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/exploration-floor/execution-freeze.json>), [final analysis](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/exploration-floor/analysis.json>), and [audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/exploration-floor/audit.md>)
- Full scored rows: [batch 0](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/exploration-floor/evaluation/batch-0/calls.jsonl>), [1](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/exploration-floor/evaluation/batch-1/calls.jsonl>), [2](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/exploration-floor/evaluation/batch-2/calls.jsonl>), [3](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/exploration-floor/evaluation/batch-3/calls.jsonl>), [4](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/exploration-floor/evaluation/batch-4/calls.jsonl>), [5](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/exploration-floor/evaluation/batch-5/calls.jsonl>), [6](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/exploration-floor/evaluation/batch-6/calls.jsonl>), [7](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/exploration-floor/evaluation/batch-7/calls.jsonl>), [8](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/exploration-floor/evaluation/batch-8/calls.jsonl>), and [9](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/exploration-floor/evaluation/batch-9/calls.jsonl>)
- Training receipts: [1501 control](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/exploration-floor/supervisor-runs/seed1501-control/receipt.json>), [1501 treatment](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/exploration-floor/supervisor-runs/seed1501-treatment/receipt.json>), [1502 treatment](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/exploration-floor/supervisor-runs/seed1502-treatment/receipt.json>), and [1502 control](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/exploration-floor/supervisor-runs/seed1502-control/receipt.json>)
- [Resource closure](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/exploration-floor/resources-closure.json>), [resource log](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/exploration-floor/resources.jsonl>), [V3 PNG](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/exploration-floor/plot-v3/exploration-floor-mass-trajectories.png>), [V3 SVG](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/exploration-floor/plot-v3/exploration-floor-mass-trajectories.svg>), [visual QA](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/exploration-floor/plot-v3/visual-qa.json>), and [V3 output closure](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/exploration-floor/supervisor-runs/plot-v3/output-closure.json>)

The [PQN paper](https://arxiv.org/abs/2407.04811) and its [pinned CartPole configuration](https://raw.githubusercontent.com/mttga/purejaxql/47af6d7b35c89ddfe633aaf7341bdb8964cb7cce/purejaxql/config/alg/pqn_cartpole.yaml) provide scale context for a later exposure question only. They do not promise that more steps or a different epsilon floor improves this Snake recipe.

The next possible action is the conditional two-seed, 50k solo food-and-survival
diagnostic, and only after the isolated solo lifecycle and profile infrastructure
receives combined qualification. It has no frozen protocol, numerical launch, or
promotion authority. This result does not select an exploration schedule or a new
incumbent.
