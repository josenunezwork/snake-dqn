# First-action diagnostic on frozen food policies — 2026-09-13

The first-action probe found strong teacher-first repairs in the five predeclared
failed held-out blocks, but the result is **mixed or insufficient coverage**. It
does not establish one universal failure mechanism or support selecting a
training change on that basis alone. The next step is a bounded, zero-SGD subgroup and coverage
diagnosis; no forced-first-action training has been selected.

This is an outcome-informed diagnosis of six existing final checkpoints from the
small solo food task. It ran no optimizer updates, changed no policy, and does not
show learner improvement, broad survival, opponent competence, fresh-seed
replication, or a reason to replace the Apex `vector61` incumbent.

## Method

For every completed task, the probe independently rebuilt three branches: force
left, straight, or right for the first turn, then retain the same frozen greedy
model. A separate frozen teacher continued each branch for calibration. The
natural branch had to reproduce the original completed episode and full trace; a
freshly measured masked argmax also had to equal the original first action.

Each original greedy episode received one mutually exclusive label, in this order:

1. **Original success** — the recorded greedy episode already succeeded.
2. **Teacher-first repair** — it failed, but the model succeeded after the teacher's first action.
3. **Other-first repair** — it failed, and another model first action succeeded.
4. **Continuation failure** — all model branches failed while a teacher continuation succeeded.
5. **Teacher unresolved** — every branch failed.

The [frozen intent](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-first-action-probe/intent.json>) and [protocol clarification](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-first-action-probe/protocol-clarification.json>) define the task, classification order, and predeclared mechanism checks. The source was frozen at `56e0e924`; the main repository was at `5b53d57` when this report was prepared.

## Closed execution

The teacher solved all 3,024 branches; the six model jobs ran 12,960 branches. All
4,320 natural model branches reproduced their original outcome and full trace,
with fresh masked-argmax parity. Every one of the seven jobs closed naturally.
The [analysis](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-first-action-probe/analysis/analysis.json>) binds 248 frozen inputs and records 98.1433 seconds summed job time, 427,016,192 bytes maximum observed RSS, and 27,281,227,776 bytes minimum available host memory.

Native probe qualification passed 8 tests in 0.92 seconds, and analyzer
qualification passed 6 tests in 0.02 seconds. The pre-execution analyzer repair
kept teacher branches grouped by training seed and bound each receipt to its
exact launch command before analysis; no execution failed and no policy was
rerun for that repair. See the
[probe qualification](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-first-action-probe/qualification/result.json>) and [analyzer qualification](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-first-action-probe/analysis-qualification/result.json>).

## Classification results

Each held-out panel contains 288 original episodes. The compact counts are
`original success / teacher-first repair / other-first repair / continuation
failure / teacher unresolved`. A dash in the repair rate means there were no
original failures in that panel.

| Checkpoint | Held-out split | Counts: O / F / A / C / U | Original failures | Teacher-first repair of failures |
| --- | --- | --- | ---: | ---: |
| Q constant 2301 | New world | 161 / 127 / 0 / 0 / 0 | 127 | 100.00% |
| Q constant 2301 | New placement | 159 / 126 / 0 / 3 / 0 | 129 | 97.67% |
| Q constant 2302 | New world | 288 / 0 / 0 / 0 / 0 | 0 | — |
| Q constant 2302 | New placement | 256 / 32 / 0 / 0 / 0 | 32 | 100.00% |
| Q constant 2303 | New world | 279 / 0 / 0 / 9 / 0 | 9 | 0.00% |
| Q constant 2303 | New placement | 276 / 0 / 0 / 12 / 0 | 12 | 0.00% |
| R late-half 2301 | New world | 285 / 0 / 0 / 3 / 0 | 3 | 0.00% |
| R late-half 2301 | New placement | 285 / 0 / 0 / 3 / 0 | 3 | 0.00% |
| R late-half 2302 | New world | 288 / 0 / 0 / 0 / 0 | 0 | — |
| R late-half 2302 | New placement | 288 / 0 / 0 / 0 / 0 | 0 | — |
| R late-half 2303 | New world | 95 / 190 / 0 / 3 / 0 | 193 | 98.45% |
| R late-half 2303 | New placement | 94 / 185 / 0 / 9 / 0 | 194 | 95.36% |

For the training tasks, Q2301 had 81 original successes and 63 first-turn repairs;
R2303 had 48 original successes and 96 first-turn repairs. Q2302, Q2303, R2301,
and R2302 each had 144 original successes. There were no other training classes.

The five previously failed held-out blocks each had at least 95.36% teacher-first
repair among original failures. That is useful descriptive evidence, but the
predeclared first-turn mechanism rule also required failures for both left and
right teacher actions in every failed block. Q constant seed 2302 on new placement
had 32 original failures, all with a straight teacher action; its left and right
failure strata were empty. The rule cannot be waived after observing the result.

Small continuation-failure groups also remain: Q constant seed 2301 had 3 on new
placement, Q constant seed 2303 had 9 and 12, R late-half seed 2301 had 3 in each
held-out split, and R late-half seed 2303 had 3 and 9. These groups, together with
the missing side strata, keep the declared result `mixed_or_insufficient_coverage`.

## Evidence and limits

The [classification rows](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-first-action-probe/analysis/classifications.jsonl>) retain every one of the 4,320 classifications. The primary analysis also carries forward the completed Q and late-half-LR learning curves without presenting them as new learning data. The prior [λ=1 report](../solo_food_lambda1_2026-09-13/README.md) and [late-half-LR report](../solo_food_late_half_lr_2026-09-13/README.md) contain the original fixed-checkpoint results.

The first renderer's classification labels overlapped. A versioned v2 repair changed
only labels and layout, retained the same recorded inputs, and generated the
following evidence without new gameplay. The [visual review](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-first-action-probe/visual-review.json>) passed the corrected PNG chart and all three trajectory grids; the latter were byte-identical across renders. SVGs were not individually inspected.

- [Classification chart, PNG](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-first-action-probe/evidence-v2/first-action-classifications.png>) and [SVG](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-first-action-probe/evidence-v2/first-action-classifications.svg>)
- [Seed 2301 recorded trajectories](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-first-action-probe/evidence-v2/seed2026092301-first-action-trajectories.png>), [2302](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-first-action-probe/evidence-v2/seed2026092302-first-action-trajectories.png>), and [2303](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-first-action-probe/evidence-v2/seed2026092303-first-action-trajectories.png>)
- [Render manifest](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-first-action-probe/evidence-v2/render-manifest.json>) and [v2 render receipt](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-first-action-probe/supervisor-runs/render-v2/receipt.json>)

A forced teacher action is an intervention on a frozen policy's first move. It can
show that an alternate first move led to success in this task; it does not show
that training the policy on that action would generalize or improve the policy.

## Posthoc distance-ladder supplement

An exploratory reduction of the same recorded data joined all 576 complete
world/heading/placement ladders across distances 2–7 and exactly recounted all
4,320 original classifications. It loaded no models and ran no new episodes or
optimizer updates. The [supplement intent](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-first-action-probe/distance-ladders-intent.json>) explicitly preserves the primary mixed decision.

For Q2302's straight-ahead food, all 32 matched cases selected straight and
succeeded at distances 2 and 3. All selected a side turn at distances 4–7:
left at 4, 5, and 7, and right at 6. All still succeeded through distance 6. At distance 7, none of the original left-first
branches succeeded; forcing straight or right succeeded in every case. Thus the
ranking difference appears before the gameplay failure boundary. This does not
establish that extending the 16-frame horizon would recover the failing branch.

All 42 continuation failures occurred in worlds `2026100204` (33) and
`2026100206` (9), including residual failures in passing checkpoints. This remains
a separate behavior to investigate. The paired distance profiles distinguish
these all-branch failures from failures avoidable by changing the first action.

The supplement completed in 0.819 seconds under the same resource guards. Its
[summary](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-first-action-probe/distance-ladders-analysis/summary.json>), [complete ladders](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-first-action-probe/distance-ladders-analysis/ladders.json>), and [distance-profile chart](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-first-action-probe/distance-ladders-analysis/distance-profiles.png>) are descriptive evidence. The PNG passed visual review; the SVG was not individually inspected. These results do not measure historical training exposure or establish a single training remedy.
