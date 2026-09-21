# CR: paired H256 versus H384 continuation

**Status: in progress.** CR is a predeclared equal-dose continuation comparison
between an H256 rollout cap and an H384 rollout cap. Qualification is complete;
the six scientific learners and their fresh greedy evaluations are still active.
There is no CR policy result, treatment result, or promotion claim yet.
The recovery record identifies the active execution queue as session 42059; the
immutable intent digest is
`5e80df83e59c7ae487443ed90ae2c34b90685a18c406c10add4d73a86aa5fbe0`.

## Frozen study

Each CR seed restores its matching completed CO `half_mse` final checkpoint at
mark 3072, including model, full Adam state, counters, and odometer:
2026095901 from CO 2026095801, 5902 from 5802, and 5903 from 5803. The two
arms, `train_h256` and `train_h384`, use the same native Q(lambda=.65), half-MSE
objective, learning rate, sampler, masks, and fresh per-seed streams. They differ
only in their rollout cap, 256 or 384 frames. Each arm runs 512 updates and saves
marks 3072, 3328, and 3584. The parent family was selected from CO's completed
absolute screen, while CO's paired loss-effect result remains false.

This is a bounded continuation study, not an exact environment restoration or a
new teacher fit. The continuing-task treatment of time-limit truncation follows
the rationale in [Time Limits in Reinforcement Learning](https://arxiv.org/abs/1712.00378).
That source motivates retaining bootstrapping at a time limit; it does not predict
that the H384 cap improves behavior here.

CR uses fresh worlds 2026109000--2026109031, four headings, and three reachable
food placements per world: 384 lanes at H384. H64, H128, and H256 are exact
prefixes of each H384 trace. The fixed 22-condition absolute package is evaluated
separately for both treatments and each seed: the inherited H256 food, cell,
survival, endpoint, and paired-random conditions, plus the H384 conditions and
late-food windows 129--256 and 257--384. The 48 pose-cell subchecks are retained.
The prior CH calibration failure, CO relative failure, and CP all-three H384
confirmation failure remain recorded failures.

The primary treatment rule is also per seed, using the 32 paired world means
(df 31) without pooling lanes or seeds. At final mark 3584, `train_h384` must
have an endpoint lower confidence bound above zero, a food-versus-`train_h256`
lower bound above -5% of the control mean, and a food-versus-parent lower bound
above -5% of the parent mean. All three conditions and all three seeds are
required. Midpoint mark 3328 is descriptive only; no best checkpoint or seed is
selected. The study is not promotion eligible.

## Qualification evidence

Qualification passed all 42 tests in 0.968 seconds. The guarded smoke took
12.541131000034511 seconds, for 13.5091310000345 seconds against the 180-second
qualification budget. It made two discarded MPS optimizer updates, no CPU
optimizer updates, and 48 passive rollout batches (24 per arm); none belongs to
the study lineage. It also dispatched 1,536 CPU gameplay lane-frames.

The first paired update and CPU gameplay were exact, and the complete Adam state
at mark 3072 was restored. Passive-rollout evidence showed zero post-256 valid
rows for `train_h256` and 1,837 for `train_h384`. This proves the intended cap
exposure in the guarded qualification setup. It does not establish a behavioral
benefit or justify an active learner result.

## Execution and interpretation boundary

The scientific queue reserves 6,560 seconds within a 6,600-second science budget:
six serial learners at up to 600 seconds, three initial and twelve learned
evaluations at up to 180 seconds, Teacher at 60 seconds, RandomSafe at 80
seconds, and analysis at 120 seconds. The science run has six learners, each
with 512 updates, or 3,072 optimizer updates total. Actual valid transitions,
training deaths, resets, and post-256 exposure will be reported separately from
fresh greedy survival and food metrics.

CQ's completed death census records that the preceding CP H384 endpoint losses
were self-collisions. That describes the observed terminal event, without proving
that a longer rollout cap rescues it or identifying a causal action error. CR
changes the reset/truncation distribution as well as later-state exposure, so any
completed result will remain a package comparison. Apex remains the operational
incumbent; its larger historical training dose is a confound, not evidence of
inherent superiority.

- [Immutable CR intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-h384-training/intent.json)
- [Frozen CR design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-h384-training/design.md)
- [Qualification completion](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-h384-training/qualification-complete.json)
- [Qualification accounting](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-h384-training/qualification-accounting.json)
- [Qualified smoke record](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-h384-training/qualification-smoke/report.json)
- [Execution recovery record](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-h384-training/execution-recovery.json)
- [CO completed result](../solo_food_pqn_loss_shape_2026-09-20/README.md)
- [CP completed duration check](../solo_food_pqn_h384_confirmation_2026-09-20/README.md)
- [CQ completed death census](../solo_food_pqn_h384_death_census_2026-09-20/README.md)
