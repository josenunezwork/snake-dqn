# Solo native-pressure diagnostic — 2026-09-13

The native Q(lambda) path preserved the positive ambient-food bonus and its
multi-step credit on all three captured endpoint trajectories. The failed seed
1901 nevertheless contained only six ambient contacts in this short capture,
compared with 176 and 243 for seeds 1902 and 1903. This is a descriptive,
same-trajectory observation. It does not identify whether epsilon, spatial
coverage, forgetting, or another factor caused the earlier performance split,
and it does not justify model promotion.

This diagnostic follows the [training-exposure analysis](../solo_training_exposure_2026-09-13/README.md), [food-signal probe](../solo_food_signal_2026-09-13/README.md), and [C2 budget study](../solo_budget_curve_2026-09-13/README.md). It ran no SGD and saved no candidate weights.

## Method

The capture used the three ambient C2 final checkpoints under frozen source
`56e0e92434ae510a84ebaf6f58cf41c3e4421b04`. It collected 8,192 valid hero
transitions per checkpoint (24,576 total) from fixed native rollouts, then
computed the recorded native target/overlay quantities with **zero optimizer
steps**. This makes the calculation a counterfactual inspection of the same
realized short trajectories, not a replay of historical learning. Each endpoint
started from the same 16 fresh diagnostic worlds and action RNG state.

The authoritative result is
[analysis-v2/analysis.json](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-pressure/analysis-v2/analysis.json),
produced by [analyze_v2.py](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-pressure/analyze_v2.py),
SHA-256 `ae17bec5188a2f3417baf9a3ae8cb6adc1902dbf0d31397b48a54d5d65522852`.
Its exact input freeze is
[analysis-v2-inputs.json](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-pressure/analysis-v2-inputs.json),
SHA-256 `e7ae4419787a65b036b72c1fcf9960e119855cdf8514cbb8a2f33caab087126d`.

## Observed native pressure

| C2 ambient endpoint | Valid transitions | Ambient contacts | Corpse contacts | Ambient rows with positive bonus | Deaths in capture |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026091901 | 8,192 | 6 | 0 | 6 | 0 |
| 2026091902 | 8,192 | 176 | 122 | 176 | 0 |
| 2026091903 | 8,192 | 243 | 3 | 243 | 0 |

The native `corpse_food_ate` category also includes boost-trail pellets. These
contacts therefore do not imply that another snake or a death was present.

Every rewarded ambient contact carried a positive target-overlay contribution.
The analysis also found overlay credit in earlier neither-food rows—18,
1,044, and 1,595 respectively—showing that the captured Q(lambda) return carried
future ambient reward back to actions before the food-contact frame. Chosen-Q,
overlay propagation, target arithmetic, and capture identities were checked
against the frozen records.

The short windows had no deaths. That fact describes these captures only; it
cannot establish a longer-horizon survival gain. Likewise, the sparse six-contact
seed1901 sample makes this result compatible with several explanations. It does
not show that epsilon is the specific problem, nor that raising exploration
would repair the endpoint.

There is one important audit limit: the final bootstrap Q was not retained, so
this work could not independently reconstruct the absolute target. It could
check the recorded overlay propagation and target arithmetic, but cannot make a
stronger claim about every absolute TD target.

## Versioned analysis repair

The first analysis had the same numerical result but incomplete identity and
configuration bindings. V2 binds the three exact intent endpoints, raw metadata
paths and hashes, frozen configuration digest, gamma, lambda, reward
coefficients, and actual zero living rewards. It did not change the raw capture
or scientific calculations. The
[v1/v2 parity record](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-pressure/analysis-v2-parity.json)
shows byte-identical analysis output and reduced plot data; only the output
closure differs because the provenance wrapper was repaired.

The [plot PNG](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-pressure/plot/solo-native-pressure.png)
and [SVG](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-pressure/plot/solo-native-pressure.svg)
were visually inspected as readable. They present this fixed capture and do not
supply a held-out behavioral evaluation.

## Qualification and local execution

The probe qualification passed five tests; its native smoke passed one test in
0.64 seconds. The v2 analysis qualification passed three tests. The receipts
preserve the actual interpreter, import witness, limits, and output closures.

All three jobs ended naturally with exit code 0 and no resource trip:

| Job | Wall time | Observed peak RSS | Minimum available memory |
| --- | ---: | ---: | ---: |
| Native capture | 21.762 s | 456,966,144 B | 27,976,220,672 B |
| V2 analysis | 0.407 s | 61,194,240 B | 28,741,861,376 B |
| Plot | 0.630 s | 95,682,560 B | 28,614,148,096 B |

## Decision boundary

A future, exact matched 200k-arm screen may test a higher epsilon floor,
`0.02 → 0.10`. That is a hypothesis for a new controlled experiment, not a
demonstrated fix from this capture. Any such study needs its own frozen inputs,
fresh rollouts, independent evaluation, and promotion evidence.

This result is therefore `AUDITED_DESCRIPTIVE_ONLY`: it verifies that the native
bonus and multi-step credit were present in the sampled trajectories, while
leaving the reason for uneven food behavior unresolved. It does not support an
incumbent replacement.

## Evidence

- [V2 analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-pressure/analysis-v2/analysis.json), [output closure](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-pressure/analysis-v2/output-closure.json), and [repair record](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-pressure/analysis-review-repair.json)
- [Capture record](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-pressure/capture/capture.json), [capture closure](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-pressure/capture/output-closure.json), and [execution inputs](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-pressure/execution-inputs.json)
- [Probe qualification](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-pressure/qualification/result.json), [native smoke](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-pressure/qualification-smoke/result.json), [v2 analysis qualification](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-pressure/analysis-v2-qualification/result.json), and [supervisor receipts](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-pressure/supervisor-runs)
