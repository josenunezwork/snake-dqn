# Solo training exposure analysis — 2026-09-13

Historical C2 telemetry shows that the failed ambient seed continued to receive
ambient-food contacts throughout training. Seed 1901 ended with 15.017 contacts
per 1,000 eligible hero steps, within its roughly 12–15 range, while the
fraction of rollouts with no ambient contact grew from 6.1% to 30.3%. This
rules out a simple account in which the failed endpoint stopped seeing food. It
does not identify why its held-out food score fell, establish a timing effect,
or support a model promotion.

This was a descriptive, post-hoc analysis of the completed
[three-seed budget study](../solo_budget_curve_2026-09-13/README.md) and its
[food-signal diagnostic](../solo_food_signal_2026-09-13/README.md). It read six
saved telemetry streams and created no environment steps, optimizer updates,
checkpoint inferences, or new policies.

## Inputs and method

The analysis included every control and ambient stream from C2: 4,703 telemetry
update rows in total. It assigned whole rollouts to eight 25k useful-step bins
from the **pre-rollout** agent-step clock; a rollout crossing a boundary remained
in its starting bin. Valid hero-step weights vary by rollout, so rates use the
recorded eligible hero steps rather than treating rows equally.

The frozen [intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-training-exposure/intent.json)
has SHA-256 `10571cdc47e664866dd9bd4bc21b0c3a067b3959731bc25f051a274ebba42a33`;
the [execution freeze](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-training-exposure/execution-inputs.json)
has SHA-256 `6fa1457d2c4bae111080b22d44fa769f8b0d35ce531d0e0c262409be3c88a512`.
The analyzer identity is [analyze.py](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-training-exposure/analyze.py),
SHA-256 `bc22912800ffb5d50602fa503329f89432b38d262c8bca9b1228a799aaac533d`.
It operated on frozen source revision `56e0e92434ae510a84ebaf6f58cf41c3e4421b04`.

The analysis recomputed ambient and corpse contact totals, ambient reward dose,
zero-contact rollout fractions and streaks, raw action counts, and prior-clock
epsilon. The C2 preflight aggregate join passed, as did the static review: it
found no fundamental data-contract blocker in these saved streams. Supplementary
checks confirmed `total_food_contacts ≤ eligible_hero_steps` in all 4,703 rows,
exact epsilon on the pre-rollout clock, and raw-action entropy reconstruction.
Those checks establish telemetry consistency; they do not establish the learning
pressure of individual food transitions.

## Observed exposure

The table reports all six streams. “Final-bin rate” is the 175k–200k nominal
bin; its exact usable-step endpoints vary by at most one rollout. The last column
is the previously reported held-out greedy-policy mean ambient-food count at initial, 50k, and final
C2 checkpoints. It is context for the exposure history, not a new evaluation or
causal comparison.

| Seed / arm | Ambient contacts | Overall rate / 1k | Final-bin rate / 1k | Zero-contact rollouts | Prior held-out food: initial → 50k → final |
| --- | ---: | ---: | ---: | ---: | --- |
| 1901 control | 2,881 | 14.402 | 11.627 | 15.8% | 0.125 → 2.375 → 0.125 |
| 1901 ambient | 2,756 | 13.765 | 15.017 | 18.9% | 0.125 → 2.875 → 0.000 |
| 1902 control | 2,861 | 14.295 | 12.651 | 19.3% | 1.500 → 18.375 → 0.125 |
| 1902 ambient | 3,558 | 17.783 | 19.312 | 11.5% | 1.500 → 0.000 → 22.625 |
| 1903 control | 2,759 | 13.784 | 13.440 | 18.9% | 0.125 → 0.125 → 9.500 |
| 1903 ambient | 3,443 | 17.213 | 21.146 | 9.9% | 0.125 → 1.250 → 110.750 |

For failed seed 1901 ambient, the last-bin contact rate was 15.017 per 1,000
eligible steps, compared with 13.393 in its first bin. Its zero-contact fraction
nevertheless grew from 6.1% to 30.3%, and its longest zero-contact streak was
18 rollouts (4,608 eligible steps). The old held-out score was 2.875 at 50k and
0 at the final checkpoint. The successful ambient seeds ended at 22.625
(seed 1902) and 110.750 (seed 1903). These histories show divergent behavior
under the same broad recipe; temporal co-movement is not proof that food
visitation, epsilon, or any other recorded quantity caused the final result.

There was no 100k held-out evaluation in C2. The eight telemetry windows must
therefore not be read as evidence for a timing regression at 100k or any other
unmeasured checkpoint. Survival was already at the masked initial ceiling in
the ambient arms, so this analysis cannot demonstrate learned survival.

The [nine-panel plot](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-training-exposure/plot/solo-training-exposure.png)
and [SVG](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-training-exposure/plot/solo-training-exposure.svg)
were visually reviewed as readable. The supporting
[summary](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-training-exposure/plot/summary.json)
contains each series and the additional consistency checks.

## Qualification and local execution

The analyzer qualification passed 11 tests in 0.02 seconds; its
[result](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-training-exposure/qualification/result.json)
records the exact interpreter, imports, and CPU-thread witness. The analyzer
finished naturally with exit code 0 in 0.2103 seconds and the plot job finished
naturally with exit code 0 in 0.8372 seconds. Both used the shared two-thread CPU job limit, a 4 GiB RSS ceiling, and a
12 GiB available-memory floor; neither guard tripped. The qualification witness
recorded one Torch interop thread; analysis and plotting did not use Torch.

The analyzer's reported 163,840-byte RSS sample is too coarse to establish
a useful peak-memory estimate for such a short job. The plot supervisor observed 103,071,744
bytes RSS. Minimum available memory was 28,467,167,232 bytes for analysis and
28,445,097,984 bytes for plotting. The complete receipts and output closures
are retained under [supervisor-runs](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-training-exposure/supervisor-runs).

## Interpretation and next question

PQN is replay-free but remains off-policy through greedy Bellman targets. The
literature does not make this project’s 30k-of-200k epsilon decay (15% of useful
steps) inherently too short: schedules are task specific, and added randomness
can harm performance. [PQN paper](https://arxiv.org/html/2407.04811v6) and its
[pinned Atari implementation](https://github.com/mttga/purejaxql/blob/47af6d7b35c89ddfe633aaf7341bdb8964cb7cce/purejaxql/pqn_atari.py#L105-L110)
provide scale and counter semantics, not a Snake budget recommendation.

The next proposed question is narrower: how much native Q(lambda) pressure do
fresh solo food transitions receive after the learner's actual masking and
returns are applied? A [frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-pressure/intent.json)
exists for that diagnostic, but it has not run. No conclusion here selects an
exploration change, claims a causal explanation, or warrants replacing the
incumbent.

## Evidence

- [Frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-training-exposure/intent.json), [analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-training-exposure/analysis/analysis.json), and [analysis closure](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-training-exposure/analysis/output-closure.json)
- [Plot summary](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-training-exposure/plot/summary.json) and [plot closure](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-training-exposure/plot/closure.json)
- [Qualification result](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-training-exposure/qualification/result.json) and [research notes](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-training-exposure/research-notes.md)
