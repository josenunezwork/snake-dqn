# Solo food-source trace — September 13, 2026

**Status: completed same-world attribution; independent review passed.** The trace
replayed the two final S1 checkpoints on the original eight evaluation worlds
without training a model. Both replays reproduced every original final row exactly.
It explains the unusually large food count in seed 2026091601: 8,679 of 8,807
contacts (98.5466%) came from that snake's own boost trail. The other finalist
collected 115 ambient contacts and no own-trail contacts.

## What this resolves

The prior [solo food-and-survival report](</Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_survival_2026-09-13/README.md>)
showed a predeclared two-seed S1 diagnostic signal but could not assign food to a
source. This trace is a post-hoc diagnosis of the same checkpoints and worlds. It
does not change the original eight-condition decision, create a new primary gate,
or supply a population confidence interval or an independent replication.

| Training seed | Final ambient contacts | Final own-trail contacts | Own-trail share | Final ambient mean | Initial total-food upper bound | Conservative ambient-improvement bound |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026091601 | 128 | 8,679 | 98.5466% | 16.000 | 2.500 | mean +13.500; median +8.0; positive 8/8 worlds |
| 2026091602 | 115 | 0 | 0.0000% | 14.375 | 0.125 | mean +14.250; median +8.5; positive 8/8 worlds |

The initial runs did not record food source. Initial ambient food can therefore be
no greater than the original total food. Subtracting that upper bound from final
ambient food is conservative: both finalists still improve ambient food on every
fixed world. Seed 2026091601 does collect ambient food, then spends much of
its time boosting and re-consuming its own trail. It should not be described as no
growth or no food learning.

Survival was 1.0 at initial and final checkpoints for both seeds. This trace
confirms preservation at the ceiling; it does not show survival improvement.

## Method and evidence

The source and trace inputs were frozen at
`e083116182eb03888f5e82b7ea8f14cbb3f3bed1`. Each CPU trace uses the original
final checkpoint, its matching final call, the eight original worlds, and the
artifact trace of native food operations. The [protocol](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-source-trace/protocol.json>)
and [execution freeze](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-source-trace/execution-freeze.json>)
preserve those inputs. The [summary](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-source-trace/summary.md>)
and [machine-readable reduction](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-source-trace/summary.json>)
contain the full per-world arithmetic.

Nine native trace-qualification tests passed. The two traces each exited naturally
with return code 0 in about 18.1 seconds. Both preserve exact full-row parity across
all 16 finalist-world rows. The final resource details are in the
[seed-1601 supervisor receipt](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-source-trace/supervisor-runs/trace1601/receipt.json>)
and [seed-1602 receipt](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-source-trace/supervisor-runs/trace1602/receipt.json>).
The qualification result and output closure are also retained:
[result](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-source-trace/qualification-v1/result.json>)
and [closure](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-source-trace/qualification-v1/output-closure.json>).

The [reward and objective notes](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-source-trace/reward-objective-notes.md>)
are analytical context only. They do not alter the frozen recipe, result rule, or
promotion boundary.

## Remaining boundary

The independent [canonical review](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-source-trace/audit.md>)
passed after rehashing 309 frozen inputs and all three job closures, confirming the
nine-test qualification, all 16 exact replay rows, raw events, and conservative
bounds with no mismatch. Across qualification and replays, peak observed RSS was
323,076,096 B and minimum available memory was 28,061,253,632 B; the two traces
closed naturally in 18.125 s and 18.101 s. The [machine-readable review](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-source-trace/audit.json>)
preserves the status.

The next study is a fresh two-seed, same-recipe S1 replication with fresh worlds;
it is in preparation and has produced no learning result. This trace does not
justify a promotion, incumbent replacement, or S1-to-S6 claim.
