# Solo-learning status: evidence map

**Current milestone:** CG's warm-start native PQN policies pass the declared H64/H128
food-and-survival retention benchmark in all three lineages after the additional training
dose. Both training-cap arms pass, but the longer cap's strict relative survival benefit
passes 0/3 seeds. The first CH H256 anchor screen then stopped at calibration: GreedyFood
missed the frozen H256 survival-time and endpoint floors, so no learned CH policy ran.
The next learner-facing question requires a separately frozen diagnostic, not a post-hoc
change to CH's criteria.

This map is a recovery guide, not a pooled benchmark. Each linked study has its own fixed parents,
data, worlds, horizon, criteria, and statistical unit. “Passes 3/3” means every predeclared per-seed
requirement for that study passed; it does not transfer a result to another policy family or task.

## Recovered exploration-floor baseline

The earlier [solo exploration-floor experiment](solo_exploration_floor_2026-09-13/README.md)
is complete and independently audited. Its unchanged all-seed rule required all 24
learning conditions and all 12 C2 baseline conditions. It passed 16/24 and 6/12,
respectively: `STAGE_A_SCREEN_NOT_ESTABLISHED`, with no promotion eligibility.

| Training seed | Final floor-.10 food / mass integral / survival | Learning checks | C2 checks |
|---|---|---|---|
| 2026091901 | 1.375 / 1.761475 / 1.0 | 5/8 | 4/4 |
| 2026091902 | 5.000 / 3.334900 / 1.0 | 5/8 | 1/4 |
| 2026091903 | 2.375 / 3.341175 / 1.0 | 6/8 | 1/4 |

The three training receipts, final v5 evaluator receipts, analysis, and independent
v2 audit establish completion. The immutable early intent still says implementation
pending and there is no separate historical closeout file; neither means the completed
runs are missing. Earlier failed evaluator attempts remain part of the repair history.
The completed evidence is reused and the original thresholds are preserved.

## What has been established

| Line | Evidence | What it establishes | Boundary that remains |
|---|---|---|---|
| Food representation | [AN point-versus-raster](solo_food_point_model_2026-09-14/README.md) | A wide food-only raster can learn short H64 food seeking; the compact point model did not fit or play reliably. | H64 games all survived, so this was not a survival stress test. It also changed representation and capacity together. |
| Supervised correction | [BM fresh shared-head](solo_food_shared_head_fresh_init_confirmation_2026-09-19/README.md) and [BN H256 evaluation](solo_food_longer_solo_confirmation_2026-09-19/README.md) | A separately trained, shared safety head repeatedly corrected the defined safety labels while preserving a frozen food parent; fixed policies then passed fresh H256 worlds. | This is imitation learning with a body/safety-head objective, frozen food parents, and selected correction data. It is not native PQN learning, a whole-learner result, or a promotion. |
| Native PQN recovery | [CC training dose](solo_food_pqn_training_dose_2026-09-20/README.md) | Continuing the unchanged native PQN recipe to 1,024 updates recovered food relative to its mark-64 parent and met its recovery and survival-retention checks in all three seeds. | Final food exceeded the original released reference with a positive paired interval in only 1/3 seeds, so the all-seed primary result failed. Training and held label fit are descriptive references, separate from greedy fresh-world play. |
| Unanchored continuation | [CA no-CE versus restored-CE](solo_food_pqn_unanchored_continuation_2026-09-20/README.md) | Restoring teacher cross-entropy retained the short-game behavioral package in 3/3 seeds; the unanchored arm retained it in 1/3. | The restored anchor did not produce a reliable relative or beyond-initial improvement. This isolates neither a general PQN cure nor a causal reason for the result. |
| H128 retention | [CF 32-world replication](solo_food_pqn_h128_replication_2026-09-20/README.md) | CC's fixed mark-1024 policies retained H128 food and time across all three seeds. In CF, every H64 condition passed and H128 food/time conditions passed in all three. | Absolute H128 endpoint survival passed only 2/3 seeds: CF seed 5302 ended 342/384, below the unchanged 348 floor. The all-three requirement therefore fails; no averaging or pooling rescues it. |
| Death diagnosis | [CE saved-action H128 deaths](solo_food_pqn_h128_deaths_2026-09-20/README.md) | The 54 recorded H128 deaths replayed exactly as self-collisions under fallback; none was classified as wall, enemy, or advisory-safe fatal. | Replay classification does not prove earlier inevitability, a counterfactual rescue, or the cause of the learned policy's choice. |
| Native H128 milestone | [CG training-horizon comparison](solo_food_pqn_training_horizon_2026-09-20/README.md) | Both fixed final arms pass all 17 food/survival-retention checks in all three seeds on 32 fresh worlds. The six learners added 6,144 updates and 1,568,614 valid hero transitions. | Longer-cap survival superiority passes 0/3; CG overall success remains false. This is a warm-start, bounded H128 result, not from-scratch mastery, long-game reliability, or promotion. |

## How to interpret fit and gameplay

- **Training fit** asks whether a network matches labels in rows it was optimized on. **Held fit** asks
  the same question on a separated label set. Neither metric is greedy gameplay.
- **Greedy fresh-world gameplay** measures food, survival time, and endpoint survival in a bank not
  used for that study's optimization. The native-PQN reports keep its fresh gameplay separate from
  the archived teacher-label reference sets.
- The safety-head results and native-PQN results answer different questions. The former learns
  corrections over frozen food-Q behavior; the latter updates native TD objectives from warm-start
  parents and then tests unanchored or re-anchored continuation packages. They must not be merged
  into a single “model success” score.
- A two-seed survival result is still a failure when the frozen rule requires all three. CD and CF
  preserve that distinction even where food or relative-retention statistics look favorable.

## Operational decision

Apex remains the operational incumbent. It was trained for a much larger historical budget, which
is a confound rather than evidence of inherent architectural superiority. None of the studies above
ran the shared tournament promotion gate or earned a replacement decision.
Historical Apex logs include runs with hundreds of thousands of learner updates, but
those partial run logs do not establish the incumbent checkpoint's final cumulative
budget. As the [counter audit](../state_and_system_review_2026-09-06.md)
explains, Apex CLI steps and PQN hero transitions are different units. Compare actual
optimizer updates, replay/hero rows, and wall time separately; the current native PQN
line also inherits supervised food training and is not a from-scratch comparison.

CG is complete and audited, including the two preserved operational partials and
their successful saved-evidence verifications. Its H128 absolute milestone does not
erase CF's previous failed replication or establish a causal benefit from the longer
training cap. The subsequent [CH H256 confirmation](solo_food_pqn_h256_confirmation_2026-09-20/README.md)
completed only the scripted-anchor screen: H256 teacher food remained strong, while the
frozen survival-time and endpoint calibration checks failed. All six learned CH roles
were consequently not run. This is an H256 anchor limitation, not evidence that either
CG learned arm regressed. CH is complete and independently audited, with its failed calibration preserved. The separate CI diagnostic is being prepared to measure all six fixed learners on the retained CH bank while reusing the completed anchors.
