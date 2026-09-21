# CL: matched PQN credit-trace continuation

**Status: R4 is running. The scripted anchors and all three starting policies
are complete; the final matched learning comparison and audit remain pending.**

CL asks a narrow question: after the same completed CK body-aware parent state,
does a longer native PQN trace (`lambda095`, λ=.95) behave more reliably than the
current trace (`lambda065`, λ=.65)? It is a matched continuation, not a new
architecture, objective, reward, action, or teacher-label experiment. Apex
remains the operational incumbent until the shared tournament gate supports a
replacement. Its larger historical training budget is not evidence of inherent
architectural superiority.

## Frozen comparison

The three common parents are completed CK `body_food` checkpoints at mark 2560.
CL maps them one-to-one to fresh seeds 2026095601–5603. Each pair restores the
complete network, Adam moments and ages, counters, input transform, masks,
rewards, optimizer settings, and native PQN configuration. Both arms receive a
fresh runtime. Within a pair they share the same fresh environment and SGD
streams until their policies diverge; λ is the only paired-arm setting that
changes.

Each arm performs 512 new native updates from mark 2560 through 3072. Mark 2816
is a learning-curve checkpoint only. Mark 3072 is the only decision checkpoint.
The fresh gameplay bank contains 32 worlds (2026108200–8231), four headings, and
three balanced reachable-food placements per world: 384 lanes per role. One H256
archive supplies exact H64 and H128 prefixes. Training never receives evaluation
arrays, and the inherited CG teacher-label fit is historical report-only context,
not a CL measurement.

The long-trace arm must pass all 16 inherited absolute checks in every seed:
food, all 12 pose cells, survival time, endpoints, and paired food versus
random-safe at H64/H128/H256, plus late H256 food. The H256 endpoint threshold is
348/384. It must also pass three final paired-world checks in every seed, using
32 world means and a Student-t interval with 31 degrees of freedom:

1. `lambda095 − lambda065` endpoint CI lower bound is strictly above zero.
2. `lambda095 − lambda065` food CI lower bound is above −5% of the `lambda065`
   world mean.
3. `lambda095 − shared parent` food CI lower bound is above −5% of the parent
   world mean.

No pooling across seeds or lanes is allowed. The `lambda065` absolute result is
reported separately but is not an overall prerequisite. A midpoint, a selected
lane, or an old teacher-label fit cannot rescue a missed final criterion.

## Qualification and operational history

The original numerical qualification remains valid: it used two discarded MPS
updates, zero CPU optimizer updates, 10,764 discarded gameplay lane-frames, and
two separate 16-step × 16-environment training rollouts. It consumed 23.999031
seconds across its preserved attempts. The current R4 metadata repair adds a
new exact-fit-reference check only: **17 tests passed in 2.889 seconds**, with no
new forward pass, optimizer update, or simulation. Cumulative qualification is
**26.888031 / 180 seconds**.

Four earlier scientific attempts remain charged and preserved, totaling
**57.881651 seconds**:

| Attempt | Result | Charged seconds |
| --- | --- | ---: |
| R2 teacher evaluation | Wall guard stopped after archived gameplay, during a broad final input check | 30.101149 |
| R3 teacher recovery | Completed naturally by reusing the archived raw evidence | 12.543018 |
| R3 random-safe evaluation | Completed naturally | 10.435133 |
| R3 initial-parent evaluation | Failed on metadata before yielding a learned comparison | 4.802350 |

R4 adopts the completed R3 teacher and random-safe reports instead of replaying
them. The new metadata-only jobs completed naturally in 4.169698 seconds and
2.938298 seconds. Their adoption records bind the original commands, input
freezes, receipts, raw report pointers, identical bank bytes, and identical
world specifications. They created **zero** gameplay lane-frames, model
forwards, scripted selections, optimizer updates, metric reductions, or NPZ
loads.

The R4 remaining job caps total 5,840 seconds. Together with the preserved
57.881651 seconds, the admitted maximum is 5,897.881651 seconds, within the
unchanged 5,940-second science cap. Heavy jobs remain serialized under the
two-CPU-slot, CPU/BLAS-two-thread, interop-one-thread, 4 GiB RSS, 12 GiB
available-memory, 8 GiB MPS-driver, and 20-second watchdog guards.

## Current evidence

The R4 teacher and random-safe reports are adopted provenance records, not new
calibration runs. All three starting policies have completed their fresh-world
evaluation. Matched training and post-training evaluations are in progress.

| Starting-policy seed | H256 food | H256 survival time | H256 endpoints |
| --- | ---: | ---: | ---: |
| 2026095601 | 46.578125 | 0.944163005 | 316/384 |
| 2026095602 | 38.143229 | 0.967987061 | 351/384 |
| 2026095603 | 37.804688 | 0.907389323 | 284/384 |

The teacher reference has H256 food 43.5390625 and 325 survivors; random-safe has
food 4.6770833 and 376 survivors. These are descriptive anchors. The learner's
unchanged endpoint requirement is 348/384, alongside its food and time criteria.

A result will separately report training telemetry, historical fit references,
actual greedy H64/H128/H256 gameplay, all per-seed absolute gates, the three
paired trace-effect checks, confidence intervals, representative raw evidence,
resources, and independent audits. A reliable three-seed result would still not
establish that terminal credit assignment alone caused any effect, because λ
changes the return calculation for all valid rows.

## Primary records

- [R4 frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-credit-trace-r4/intent.json)
- [R4 design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-credit-trace-r4/design.md)
- [R4 qualification accounting](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-credit-trace-r4/qualification-accounting.json)
- [R4 metadata qualification](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-credit-trace-r4/qualification-complete.json)
- [R4 teacher adoption](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-credit-trace-r4/evaluation/teacher-seed0-mark0/adoption.json)
- [R4 random-safe adoption](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-credit-trace-r4/evaluation/random_safe-seed0-mark0/adoption.json)
- [Completed CK body-access comparison](../solo_food_pqn_body_access_2026-09-20/README.md)
