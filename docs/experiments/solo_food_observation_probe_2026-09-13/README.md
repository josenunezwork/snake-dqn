# Food-observation recoverability probe — 2026-09-13

An observation-only food-distance rule recovered the teacher's action on all **12,288
saved examples**: 3,072 from X training, 3,072 from Y training, and 6,144 held-out
teacher states. Every action class had 100% recall in each dataset; no row used the
no-food fallback or lacked a legal normal action. The frozen requirement was at least
99% agreement independently in every dataset.

This is a passive diagnostic. It did not train a model, run inference, generate new
states, or add gameplay. The prior [native food BC](../solo_native_food_bc_2026-09-13/README.md)
and [food diversity](../solo_native_food_diversity_2026-09-13/README.md) results and gates
remain unchanged, including Y's missing midpoint.

The decoder retains exact food cells in tactical channels 4–5 and the nearest
out-of-view food coordinates in scalar indices 12–14. It chooses the safe normal
action with minimum post-move food distance, breaking ties straight, left, right.
It uses the same externally supplied legal mask. It does not see hidden world food,
strategic pixel centers, teacher labels, or a model's Q values when choosing actions.

The six saved learned models still made 836–993 held-out errors each. Every one of
those errors occurred on a row where the observation-only rule matched the teacher.
All six models correctly classified every example in their own training archives.

| Study | Training seed | Learned held-out correct | Learned errors on decoder-correct rows |
| --- | --- | ---: | ---: |
| X | 2026092701 | 5151 / 6,144 | 993 |
| X | 2026092702 | 5200 / 6,144 | 944 |
| X | 2026092703 | 5270 / 6,144 | 874 |
| Y | 2026092801 | 5242 / 6,144 | 902 |
| Y | 2026092802 | 5308 / 6,144 | 836 |
| Y | 2026092803 | 5245 / 6,144 | 899 |

The evidence supports testing how the network learns the available food information.
It does not show that raster31v3 is fully observable in general, establish a universal
accuracy ceiling, or prove why the network generalizes poorly. A separate static
out-of-view-food ambiguity construction was recorded but was not executed as a
native reproduction; no such disagreement occurred in these saved examples.

The [frozen protocol](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-observation-probe/intent.json>) defined the rule, datasets, criterion, and 20-second CPU budget before execution.
The [result](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-observation-probe/analysis/report.json>) binds dataset row identities, decoder actions and distances, all mismatches,
per-world and per-quarter summaries, and the twelve training/held-out Q comparisons.
Its SHA-256 is `b5c4983d7723dceb37ebe456a02e59db95bff0ba6f7723809a54c0cc795b76a9`.

Qualification passed seven geometry, masking, aggregation, and row-join tests in
1.048 guarded seconds. The probe completed naturally in 0.835938 seconds, with
720,486,400 bytes peak RSS and 29,344,186,368 bytes minimum available memory.
The [independent review](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-observation-probe/independent-review.json>) verified 164 frozen inputs, receipts, and all twelve learned-correct counts against
saved fit reports. It did not independently decode raw NPZ arrays.

The next bounded learning screen will keep the same network and optimizer while
retaining only the food input channels used here. That will test whether restricting
input information helps generalization. It will use a separate protocol and three
fresh model seeds; the reused benchmark will remain diagnostic. Apex stays incumbent,
and only the shared tournament gate can support promotion.
