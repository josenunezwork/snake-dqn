# Task-aligned CZ continuation — 24 September 2026

**Saved scientific decision: `INCONCLUSIVE_SCREEN`. Independent saved-output audit: `PASS`. No promotion.** Two of the three fixed lineages passed the practical living-mass gain checks in both opponent profiles. Only lineage 2026096102 also passed every solo-retention check. These results show useful improvements in several fixed comparisons, but do not satisfy the prospective package advancement rule.

This closeout transcribes completed saved outputs. The all-cell tables and ten figures beside this document are unchanged copies of the admitted presentation artifacts; no games, model queries, training-log analysis, or scientific reduction were rerun to write it. The controller's saved `completed.json` records its handoff state, `AUDITED_PENDING_ROOT_CLOSEOUT`. Root subsequently verified the audit's own receipt crosslinks, all 13 stages' natural exits with code 0, and controller termination; that verification completes the operational closeout.

## What was compared

All three CZ `enemy_visible` update-4608 parents—2026096101, 2026096102, and 2026096103—were compared with their prescribed final continuation endpoints. The cohort choice was outcome-informed. There was no favorable-seed or intermediate-checkpoint selection. The package retained the inherited learner and restricted input contract while changing the native experience distribution, collection horizon, and dose; it does not isolate the causal contribution of any one change.

The common fixed dose was 31,005 continuation rollouts per lineage. Saved audit totals record **23,776,196 scientific training hero transitions and 93,015 scientific optimizer updates** across the three lineages. These exclude qualification and evaluation work; the broader controller totals must not be relabelled as scientific training dose. Training telemetry is presented separately from greedy behavior.

Final evaluation used 32 fixed procedural native worlds per profile, H3000, the same serving contract for parent and candidate, terminal heroes, and respawning scripted opponents:

| Profile | Opponents |
|---|---|
| `solo` | None |
| `food_pressure` | Four GreedyFood and one RandomSafe |
| `mixed` | Two GreedyFood and three RandomSafe |

There were **768 evaluation episodes**: 576 learned-policy episodes and 192 scripted-anchor episodes. GreedyFood and RandomSafe hero anchors ran once per profile and were shared across lineages. All controllers began from paired initial descriptors within each world/profile; later policy-dependent random trajectories may diverge. Declared held IDs were excluded from generated training IDs. This is not an all-history geometry-novelty claim.

The primary metric is physical living mass summed over observed frames and divided by the full 3,000-frame horizon, with death and subsequent frames contributing zero. Ambient food is a physical event count. Alive fraction, endpoint survival, boost fraction, and encounter exposure remain visible in the [complete table](all-cells.md) and [full-precision CSV](all-cells.csv).

## Saved results and unchanged decision rules

The following values are rounded displays of saved paired mass results. Intervals are candidate-minus-parent, two-sided marginal 95% Student-t intervals over 32 paired worlds, with 31 degrees of freedom.

| Lineage | Profile | Parent mass/frame | Candidate mass/frame | Paired mass delta [95% CI] | Saved check |
|---|---|---:|---:|---|---|
| 2026096101 | solo | 62.494 | 60.401 | −2.093 [−25.720, 21.534] | Mass retention not established; food/alive pass |
| 2026096101 | food_pressure | 23.908 | 48.325 | 24.417 [10.893, 37.941] | Gain pass |
| 2026096101 | mixed | 35.725 | 50.256 | 14.531 [0.716, 28.345] | Gain pass |
| 2026096102 | solo | 36.516 | 74.462 | 37.946 [25.150, 50.742] | All retention checks pass |
| 2026096102 | food_pressure | 15.327 | 55.203 | 39.876 [25.825, 53.927] | Gain pass |
| 2026096102 | mixed | 17.136 | 55.239 | 38.104 [25.394, 50.813] | Gain pass |
| 2026096103 | solo | 72.291 | 55.092 | −17.199 [−33.093, −1.304] | Mass, food, and alive retention not established |
| 2026096103 | food_pressure | 19.950 | 33.599 | 13.649 [4.002, 23.295] | Gain pass |
| 2026096103 | mixed | 36.031 | 40.374 | 4.343 [−9.863, 18.548] | Gain not established |

Advancement required **every** lineage to achieve an opponent-profile mean mass gain of at least `max(0.10 × parent mean mass, 1.0)` in **both** opponent profiles, with a strictly positive paired lower confidence bound. Each lineage also needed solo lower bounds at least −5% of its parent mean for mass and ambient food, and at least −0.02 for alive fraction. These are prospective engineering tolerances, not power-derived margins or revised historical criteria.

The ordered rule first classifies clear harm if a solo upper confidence bound is below its allowed loss or an opponent mass upper bound is below zero. It then checks baseline encounter opportunity: every parent and the GreedyFood anchor must encounter an opponent after frame 16 in at least eight worlds in each opponent profile. Advancement requires all gain and retention checks. Practical-gain futility requires all six opponent mass upper bounds below their respective practical margins. Other outcomes are inconclusive. The [frozen evaluation protocol](../../../research/task_aligned_20260924/protocol-evaluation.md) gives the complete rule and measurement definitions.

Baseline exposure passed: each parent had exposure in all 32 worlds of each opponent profile; the GreedyFood anchor had 30 in each. Candidate exposure was descriptive and did not filter results. Lineage 6103's solo mass interval lies below zero, but its upper bound (−1.304) remains above the allowed loss boundary (−3.615), so it did not trigger the predefined clear-harm classification. A failed noninferiority check is not automatically evidence of harm beyond the allowed margin. The saved outcome therefore remains **`INCONCLUSIVE_SCREEN`**, including the retained negative and uncertain results.

## Saved figures

Each paired figure presents the fixed parent/candidate mass, food, and alive comparison. All nine lineage/profile figures are retained:

| Lineage | Solo | Food pressure | Mixed |
|---|---|---|---|
| 2026096101 | [Figure](paired-2026096101-solo.png) | [Figure](paired-2026096101-food_pressure.png) | [Figure](paired-2026096101-mixed.png) |
| 2026096102 | [Figure](paired-2026096102-solo.png) | [Figure](paired-2026096102-food_pressure.png) | [Figure](paired-2026096102-mixed.png) |
| 2026096103 | [Figure](paired-2026096103-solo.png) | [Figure](paired-2026096103-food_pressure.png) | [Figure](paired-2026096103-mixed.png) |

The separate [training telemetry figure](training-telemetry.png) preserves the saved loss and actual-valid-row curves. Optimizer loss and training throughput do not establish greedy gameplay improvement.

## Evidence and limits

The [copied presentation manifest](presentation.json) preserves source decision, cell-report, training-log, and artifact hashes. Its training-log paths are provenance only; those logs were not reopened for this documentation. The saved `decision.json` file-byte SHA-256 is `abb1adb87f8d38bf325d2799a8326c34b8fff3f0685c7ef20d7858db87eb27f2`; the saved audit report SHA-256 recorded by the controller is `33d0e475c0a383fadbbadf41307aae8d6657ffdf62f6f1f497f9bcddf92f1442`. The admitted intent is `9ea0f3b4e7fd449bde4cd95ba91827a6b1b3897770a86ba2db58c72939955099`.

The verified [root closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/task-aligned-cz-integration-20260924/admission-v2/closeout.json) has SHA-256 `57ddf3c520274287839fe4ea3bdcfa86313f15cb2b64bd187e4d9c70412afd2b`. It records 13 natural exits with code 0 and 36,853.561972498894 guarded seconds (about 10 hours 14 minutes). Sampled maximum RSS was 4,991,975,424 bytes and minimum available host RAM was 29,935,943,680 bytes. These are saved operational measurements, separate from behavioral evidence.

The local evidence root is `/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/task-aligned-cz-integration-20260924/admission-v2/run`. Authoritative records are `analysis/output/decision.json`, `analysis/output/presentation/`, `audit/output/report.json`, and `completed.json`. The audit records independent saved-frame-column arithmetic verification; its scientific decision check uses the shared frozen reducer. Audit `PASS` denotes evidence integrity, not scientific advancement.

The 32-world intervals describe each **fixed trained lineage**, not a population of trained models. Three trained lineages remain three replications; no familywise inference, causal factor attribution, or broad recipe-reliability claim follows. The native scripted-opponent results do not demonstrate full Slither transfer or stock-serving compatibility. This study neither promotes a candidate nor changes earlier study outcomes or the separate tournament promotion authority. The first-lineage exploratory preview is separate and is not substituted for this complete official matrix.
