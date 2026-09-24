# Solo exploration-floor screen — 2026-09-13

**Stage A did not establish a reliable learning result.** The higher exploration
floor met 16 of 24 learning conditions and 6 of 12 C2-baseline nonregression
conditions, so the frozen screen failed. The independent v2 audit passed and
matched the same failure decision. The screen changes only `eps_end`, from
0.02 to 0.10. It cannot
promote a model; Apex remains the incumbent and the shared tournament gate
remains the promotion authority.

Stage A follows the [training-exposure analysis](../solo_training_exposure_2026-09-13/README.md), [food-signal probe](../solo_food_signal_2026-09-13/README.md), and [native-pressure diagnostic](../solo_native_pressure_2026-09-13/README.md). It asks whether keeping more late exploration can improve the same solo food screen.

## Frozen comparison

The frozen [intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-exploration-floor/intent.json)
sets a three-seed matched comparison for seeds 2026091901–2026091903. Both arms
use source revision `56e0e92434ae510a84ebaf6f58cf41c3e4421b04`, `E16/S1/T16`,
200k useful hero steps, epsilon start 1.0, a 30k decay, and ambient-reward
coefficient κ = 0.1. The treatment alone changes the final epsilon floor:

| Setting | Baseline | Treatment |
| --- | ---: | ---: |
| `eps_end` | 0.02 | 0.10 |
| `eps_start` | 1.0 | 1.0 |
| Epsilon decay | 30k useful steps | 30k useful steps |
| Other training, source, and recipe fields | frozen match | frozen match |

The treatment arms completed 600,153 useful steps: 200,074, 200,032, and
200,047 by seed. Their wall times were 173.539, 165.609, and 164.143 seconds.
Exact initial checkpoints, full native checks, and the first 47 telemetry fields
matched their stated comparisons for all three seeds. The frozen launch identity
and inputs are retained in [manifest-v5.json](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-exploration-floor/manifest-v5.json)
and [execution-freeze-v5.json](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-exploration-floor/execution-freeze-v5.json),
whose freeze SHA-256 is `988f983aaa09072b05d83364404cc7da61166e84e9f272c5ce6f482c97ca3eba`.

## Training manipulation evidence

The versioned [training diagnostics](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-exploration-floor/training-diagnostics-v2/diagnostics.json)
confirmed the pre-update epsilon schedules and reconstructed raw action counts.
Ambient food contacts were 2,756 → 3,731 for seed1901, 3,558 → 4,203 for
seed1902, and 3,443 → 3,955 for seed1903 when comparing the earlier C2 ambient
arm to this higher-floor treatment. These counts verify that the manipulation
changed recorded training exposure; they are not greedy evaluation and do not
by themselves show learning.

The late 150k+ descriptive window was defined after the first treatment run
completed, while the second was running, and before the late-window values were read. Diagnostics v1 omitted the native entropy
term `p * log(p + 1e-12)`. V2 repaired that formula and passed five
qualification tests; the raw histories and scientific training outputs were not
changed.

## Results

All six v5 calls completed and the frozen
[analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-exploration-floor/analysis/analysis.json)
reported `STAGE_A_SCREEN_NOT_ESTABLISHED` (SHA-256
`6105560e2b79ee8e1932e6c04ac04076261fad41e4929c8ace9239a10fb96164`).
All three treatment food means declined from 50k to the final checkpoint:
2.625 → 1.375, 7.500 → 5.000, and 9.750 → 2.375. The final treatment food
means were 1.375, 5.000, and 2.375; the corresponding earlier C2 `.02` finals
were 0.000, 22.625, and 110.750.

| Seed | Initial: food / mass / survival | .10 at 50k | .10 final | C2 .02 final |
| --- | --- | --- | --- | --- |
| 1901 | 0.125 / 1.125000 / 1.000000 | 2.625 / 3.092625 / 1.000000 | 1.375 / 1.761475 / 1.000000 | 0.000 / 1.000000 / 1.000000 |
| 1902 | 1.500 / 2.125925 / 1.000000 | 7.500 / 4.463900 / 1.000000 | 5.000 / 3.334900 / 1.000000 | 22.625 / 5.511250 / 1.000000 |
| 1903 | 0.125 / 1.125000 / 1.000000 | 9.750 / 3.328975 / 0.559575 | 2.375 / 3.341175 / 1.000000 | 110.750 / 6.557400 / 1.000000 |

Final treatment survival equalled the initial ceiling in every seed, so it is
preservation rather than survival improvement. Seed 1903's lower 50k survival
does not change that comparison. These are the reused matched seeds and worlds
specified in the frozen intent; they are not a population estimate.

The predeclared decision required all 24 learning conditions and all 12 C2
baseline-nonregression conditions across the three existing seeds. Each seed
needed eight learning conditions comparing final treatment with shared initial
and same-arm 50k states, plus four baseline nonregression conditions against the
prior C2 ambient final. Pooling, choosing a best checkpoint, and rescue by a
secondary metric were forbidden. The 16/24 and 6/12 result fails that
conjunction. It does not advance the recipe to a fresh replication, select an
intermediate checkpoint, or justify a budget increase.

The reducer ended naturally in 0.607 seconds (peak RSS 50,741,248 bytes; minimum
available memory 28,314,451,968 bytes). The six evaluator calls together used
120.131 seconds, with observed peak RSS 501,891,072 bytes and minimum available
memory 27,641,528,320 bytes. Their receipts are linked below. The source and
models remained unchanged during the evaluator repairs.

## Evaluator recovery and qualification

Four prior evaluator attempts are retained because none reached a world:

| Version | Stop point | Versioned repair |
| --- | --- | --- |
| v1 | Unit qualification referenced undefined `INTENT` | Bind immutable intent path |
| v2 | Preflight used the wrong runner-helper namespace | Route to the frozen helper |
| v3 | Preflight lacked a declared evaluation world | Bind the existing profile world |
| v4 | First call could not bare-import `source_metrics` | Import from its frozen declared path |

Each failure occurred before simulation, and the source, checkpoints, recipe,
and models remained unchanged. The [v2](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-exploration-floor/evaluator-v2-repair.json),
[v3](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-exploration-floor/evaluator-v3-repair.json),
[v4](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-exploration-floor/evaluator-v4-repair.json),
and [v5](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-exploration-floor/evaluator-v5-repair.json)
repair records preserve the exact causes.

The final evaluator qualification passed 20 tests with one retained legacy skip
and seven subtests in 1.72 seconds. [Preflight v5](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-exploration-floor/preflight-v5/preflight.json)
passed all 12 native checkpoint checks in 1.468 seconds. The canonical runtime
is [evaluate_v5.py](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-exploration-floor/evaluate_v5.py).

## Independent audit and plot closure

The [independent v2 audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-exploration-floor/analysis/independent-audit-v2.json)
passed. It independently matched all 780 frozen inputs, 36 snapshots, 12 calls,
96 rows, 36 conditions, and 216 paired deltas to the `16/24` and `6/12` failure
decision. It ended naturally in 0.631 seconds, with peak RSS 55,296,000 bytes
and minimum available memory 28,744,892,416 bytes. The auditor SHA-256 is
`38d2a41129cc37751a0ecfee8d624858b9cc0bc1f6c061e56e2bad0ce1f2f0e6`; the
analysis SHA-256 remained
`6105560e2b79ee8e1932e6c04ac04076261fad41e4929c8ace9239a10fb96164`.

Audit v1 incorrectly required `agent_steps` in C2 call specifications, which
omit that field. V2 validates the actual step count from the native snapshot
against its declared checkpoint role; its arithmetic
and the scientific result are unchanged. The supplementary before/after audit
inputs passed.

The original plot was rejected because shared-axis autoscaling clipped later
food and mass values. The [v1 visual review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-exploration-floor/plot-v1-visual-review.json)
preserves that failure. [Plot v2](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-exploration-floor/plot-v2/solo-exploration-floor.png)
uses independent per-panel axes whose explicit maxima cover every raw value. Its
[visual review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-exploration-floor/plot-v2-visual-review.json)
passed; its reduced plot data are byte-identical to v1. Scales vary by seed and
should not be compared by height across panels.

## What remains open

This screen did not support raising the epsilon floor as a reliable remedy. It
also cannot distinguish an epsilon mechanism from broader coverage or
forgetting explanations. No promotion, fresh replication, or recipe advance
follows from this result.

## Evidence

- [Frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-exploration-floor/intent.json), [v5 manifest](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-exploration-floor/manifest-v5.json), and [v5 execution freeze](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-exploration-floor/execution-freeze-v5.json)
- [Training diagnostics v2](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-exploration-floor/training-diagnostics-v2/diagnostics.json) and [diagnostic qualification](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-exploration-floor/training-diagnostics-v2-qualification/result.json)
- [Analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-exploration-floor/analysis/analysis.json), [analysis receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-exploration-floor/supervisor-runs/analysis/receipt.json), and [v5 call receipts](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-exploration-floor/supervisor-runs)
- [Evaluator v5 qualification](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-exploration-floor/evaluator-v5-qualification/result.json), [preflight](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-exploration-floor/preflight-v5/preflight.json), [independent v2 audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-exploration-floor/analysis/independent-audit-v2.json), and [v2 plot review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-exploration-floor/plot-v2-visual-review.json)
