# PQN valid-hero food-contact telemetry — September 13, 2026

**Status: integrated locally on `main` at `633cf6f`; no remote push.** This change
adds an engineering metric. It does not establish policy quality, training progress,
or eligibility for promotion.

## Delivered behavior

The integration records an exact boolean food-contact event for valid HERO
transitions in both PQN rollout collectors. The JSON training history now persists
the resulting `hero_food_contact_events` count. It deliberately does not change
action selection, rewards, learning targets, optimizer state, checkpoint semantics,
or random-number use.

The source is available in [the trainer](../../../src/training/pqn_trainer.py) and
[the training-history writer](../../../src/scripts/train_pqn.py). The candidate was
qualified as `633cf6f`, then fast-forwarded into the local `main` checkout at
`633cf6fc18b57ff60dc6f7a014851b81afcb8783` after the V4 closure review. The
qualified worktree remains clean; this record does not imply a remote publication.

## Final V4 verification

V4 closed with 161 passing tests across the three passing suites. The two parent
regressions below are expected failures from the unfixed parent and are separate
from that 161-test total.

| Check | Result | Duration |
| --- | ---: | ---: |
| Focused telemetry and history tests | 63 passed | 2.47 s |
| Trainer tests | 61 passed | 9.57 s |
| Lifecycle tests | 37 passed | 1.55 s |
| Unfixed-parent regression proof | 2 expected call failures | — |
| Mutation screen | 5/5 killed; no survivors | — |

The parent proof runs each collection phase against `04544aa` and reaches the
expected missing `food_ate` failure. The mutation screen separately removes the
Watch collector, standard collector, HERO filtering, valid-transition filtering,
and JSON-history persistence; every mutant was detected.

Every actual-import witness used CPU with two PyTorch intra-op threads and one
interop thread. V4 samples the recursive pytest process tree and explicitly excludes
the outer supervisor from the reported RSS. Across ten checks it recorded 202 RSS
samples, a peak verifier-plus-pytest-tree RSS of 837,468,160 B (about 0.780 GiB),
and a lowest host-available-memory value of 28,852,764,672 B (about 26.87 GiB). Each
qualification phase enforced a 4 GiB recursive process-tree RSS guard, a 12 GiB
host-available-memory guard, and a 90-second cap; all resource guards passed.

The final V4 evidence is retained with its output closures:

- [focused result](../../../../snake-dqn-artifacts/ongoing-research-20260913/food-telemetry-verification/focused-v4/result.json) and [closure](../../../../snake-dqn-artifacts/ongoing-research-20260913/food-telemetry-verification/focused-v4/output-closure.json)
- [trainer result](../../../../snake-dqn-artifacts/ongoing-research-20260913/food-telemetry-verification/trainer-v4/result.json) and [closure](../../../../snake-dqn-artifacts/ongoing-research-20260913/food-telemetry-verification/trainer-v4/output-closure.json)
- [lifecycle result](../../../../snake-dqn-artifacts/ongoing-research-20260913/food-telemetry-verification/lifecycle-v4/result.json) and [closure](../../../../snake-dqn-artifacts/ongoing-research-20260913/food-telemetry-verification/lifecycle-v4/output-closure.json)
- [parent-regression result](../../../../snake-dqn-artifacts/ongoing-research-20260913/food-telemetry-verification/parent-regressions-v4/result.json) and [closure](../../../../snake-dqn-artifacts/ongoing-research-20260913/food-telemetry-verification/parent-regressions-v4/output-closure.json)
- [mutation result](../../../../snake-dqn-artifacts/ongoing-research-20260913/food-telemetry-verification/mutations-v4/result.json) and [closure](../../../../snake-dqn-artifacts/ongoing-research-20260913/food-telemetry-verification/mutations-v4/output-closure.json)
- [root execution record](../../../../snake-dqn-artifacts/ongoing-research-20260913/food-telemetry-verification/root-execution-v4.json) and [root launch identity](../../../../snake-dqn-artifacts/ongoing-research-20260913/food-telemetry-verification/root-launch-identity-v4.json)

## Retained qualification history

V2 failed before test collection because its venv symlink resolved to global Python;
it has no valid import witness. See the preserved [V2 result](../../../../snake-dqn-artifacts/ongoing-research-20260913/food-telemetry-verification/focused-v2/result.json).

V3 established functional behavior using the exact interpreter and repository
import witness, but its RSS observation covered only the direct driver and excluded
pytest descendants. It is useful functional evidence, but it is not the final
resource qualification. V4 supplies that process-tree coverage. The retained V3
[result](../../../../snake-dqn-artifacts/ongoing-research-20260913/food-telemetry-verification/focused-v3/result.json) and [closure](../../../../snake-dqn-artifacts/ongoing-research-20260913/food-telemetry-verification/focused-v3/output-closure.json) preserve the distinction.

## Study boundary

The active exploration run used the earlier `c777b04` source and therefore did not
produce this telemetry. No historical run is being relabeled, and the metric alone
does not demonstrate food-seeking, survival, generalization, or a reason to change
the incumbent.
