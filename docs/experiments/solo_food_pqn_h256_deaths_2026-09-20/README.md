# CJ: exact saved-action diagnosis of H256 deaths

**Complete and independently audited.** All 607 deaths from CI's six policies replayed exactly as self-collisions under native action-mask fallback. There were zero advisory-safe fatal decisions, no raw-frame discrepancy, and no change to any policy's endpoint count.

At each fatal decision, no legal action remained advisory-safe. This identifies the terminal mechanism. It does not prove that the earlier trajectory was inevitable, that a specific earlier action would rescue the snake, or that absent body inputs caused the learned choices. The result prioritizes a matched body-observation learning experiment.

## Every policy

CJ selected every CI policy that failed a survival readout, preserving all three seeds and both training-cap arms at mark2048. Each trace contains 384 lanes and 256 frames, replayed in the original four 96-lane chunks. No policy, checkpoint, teacher, or new action selection was executed.

| Seed | Training cap | Self-collision fallback | Advisory-safe fatal | Endpoint survivors | Earliest death: frame / lane / length / action |
|---|---|---|---|---|---|
| 2026095401 | 64 | 98 | 0 | 286/384 | 54 / 171 / 12 / 1 |
| 2026095401 | 128 | 105 | 0 | 279/384 | 77 / 93 / 15 / 2 |
| 2026095402 | 64 | 74 | 0 | 310/384 | 54 / 118 / 13 / 2 |
| 2026095402 | 128 | 114 | 0 | 270/384 | 42 / 304 / 10 / 0 |
| 2026095403 | 64 | 90 | 0 | 294/384 | 45 / 223 / 10 / 1 |
| 2026095403 | 128 | 126 | 0 | 258/384 | 52 / 169 / 13 / 2 |

Actions0/1/2 mean normal-speed left/straight/right. Every earliest-death record has all six actions domain-legal, all six advisory-unsafe, and a resolved mask equal to the legal mask, as the native fallback contract specifies. The full archive retains each terminal body, head, direction, length, food, action and three masks, plus 16 preceding states. There are 607 windows and 9,712 preceding states; no terminal case was sampled away.

## Exactness and scope

The replay compared all fourteen saved fields on every frame: actions, resolved masks, ambient/corpse/total food events, validity, alive, mass, length, boosted, done, death cause, head coordinates, and directions. It also checked each chunk's initial food and preserved global lane identity. Exact terminal tuples, causes, first-death frames, counts and survivors reconcile to the audited CI death census. Native mask classification uses the qualified existing helper, independently recomputed from saved masks by the auditor.

This is reconstruction evidence, not another policy evaluation or learning curve. The [CI behavior and fixed gameplay figures](../solo_food_pqn_h256_diagnostic_2026-09-20/README.md) remain the behavioral comparison, and the [CG learning curves](../solo_food_pqn_training_horizon_2026-09-20/README.md) remain the training record. No inference call, model construction, checkpoint deserialization, new policy decision, or optimizer update occurred in CJ.

## Execution and next experiment

One serialized CPU replay finished naturally in 63.215904 of the declared 120 seconds. Thirteen synthetic contract tests passed in 1.423 of 60 seconds with zero qualification replay frames or inference. The science job peaked at 962,838,528 RSS bytes and retained at least 37,363,073,024 available bytes; qualification peaked at 513,916,928 RSS bytes. Both CPU locks and the existing two-thread, memory and watchdog guards remained intact, with no source drift or tripwire.

Review strengthened the adapter before freezing: it now enforces exact raw dtypes, reconciles terminal events directly to the audited CI census, and rejects conflicting inherited input hashes. A synthetic fixture's initially dead default lanes were corrected before any test. No scientific run was repeated. The independent auditor required an operational correction: a stale process was still writing its obsolete failure output. That process was stopped, and the final audit ran to a unique output path with a verified natural exit and no remaining writer. The authoritative final audit passes all closure and census checks.

The next study compares matched native PQN continuations with and without the existing own-body tactical channel. Both will start with identical Q behavior and receive the same H256 training exposure, with three fresh training seeds and fresh evaluation worlds. That test can measure whether learning with body information improves survival while retaining food. It cannot be assumed successful from these replay results. Apex remains incumbent pending the shared tournament gate.

- [Frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-h256-deaths/intent.json)
- [Every terminal and predecessor state](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-h256-deaths/evidence/report.json)
- [Independent audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-h256-deaths/independent-audit.json)
- [Final closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-h256-deaths/closeout.json)
