# DA: saved opponent-learning diagnosis

Completed and audited before the user's requested stopping point. DA reads the
completed CZ experiment; it performs zero new games, inference calls, or updates.
The diagnostic took 3.138118 seconds of its 60-second CPU budget. Its separate
provenance and arithmetic audit passed 5,203 file hashes in 2.857315 seconds of
its 30-second budget. Both jobs exited naturally under the existing resource guards.

## What the saved experience shows

Every run contains 512 updates and approximately 131,000 valid hero transitions.
Only 19–32 of those transitions are terminal. Policy-time squared TD residual is
concentrated on these deaths and the immediately preceding states:

| Seed | Arm | Valid transitions | Deaths | Death share of half-squared TD residual | Previous 1–4 steps' share |
| --- | --- | ---: | ---: | ---: | ---: |
| 2026096101 | blind | 130,828 | 32 | 51.79% | 34.24% |
| 2026096101 | visible | 130,932 | 21 | 48.13% | 31.89% |
| 2026096102 | blind | 130,932 | 19 | 49.03% | 32.60% |
| 2026096102 | visible | 130,922 | 21 | 48.79% | 33.83% |
| 2026096103 | blind | 130,878 | 23 | 49.78% | 28.67% |
| 2026096103 | visible | 130,949 | 19 | 48.95% | 30.23% |

These are residual shares, not parameter-gradient shares. They do not establish
an optimizer defect. The predecessor partition is restricted to the same saved
rollout; no missing prior-rollout states are inferred.

Enemy-aware final policies still had 18, 10, and 18 head-collision deaths in the
384-lane S2 evaluations. Their endpoint survivors declined from the midpoint to
the final checkpoint in every seed: 368→364, 380→373, and 368→364. Thus the short
learning curve does not establish a monotonic benefit from additional practice.

Food differences are largely present while both the initial and final policies
are still alive. For visible seeds 6101/6102/6103, total food-count changes across
all 384 lanes are −1,476 / +4,242 / +79; their both-alive-frame components are
−1,453 / +3,653 / −294. Changed survival is therefore insufficient to explain
the food changes. Final boost fractions are 0.025%, 34.153%, and 3.699%, compared
with initial values of 21.215%, 0.109%, and 41.472%. These associations do not
establish that boost alone caused the outcome.

The saved diagnostic also partitions live frames before and after each policy's
first close encounter, using the frozen frame-17-or-later and L1-distance≤8 rule.
Those cohorts depend on policy actions; they are descriptive, not randomized
threat exposure. Full-horizon food and survival remain the governing metrics.

## Boundaries and handoff

The audit independently checks JSON provenance, quarter/partition arithmetic,
producer-report joins, and all 12 paired food decompositions. It does not reopen
NPZ arrays or checkpoints; raw-array semantics are enforced in the diagnostic
producer and the previously qualified CZ pipeline. The diagnostic authenticates
all 3,159 frozen inputs before and after reduction.

The next proposed study is a visible-only continuation from all three CZ final
checkpoints for 1,024 unchanged updates, with fresh worlds and separate cumulative
and immediate-parent comparisons. This tests dose, not a clean enemy-input causal
effect. A controlled avoidable-encounter benchmark remains the alternative for
isolating local threat learning. **Neither next study was implemented or run.**
The user requested a stopping point before further work.

- [CZ behavioral comparison](../solo_food_pqn_enemy_access_2026-09-21/README.md)
- [DA frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-enemy-learning-diagnostic/intent.json)
- [DA saved-data analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-enemy-learning-diagnostic/analysis-attempt1/analysis.json)
- [DA independent audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-enemy-learning-diagnostic/audit-attempt1.json)
