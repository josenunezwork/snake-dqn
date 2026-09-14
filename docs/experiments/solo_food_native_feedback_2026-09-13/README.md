# Native-feedback diagnostic on completed solo-food endpoints — 2026-09-13

The native-feedback diagnostic found **no single-mechanism remedy**. The
predeclared initial-transition weighting pattern was not supported: the failed
constant-rate endpoint had positive first-row pressure with near-zero
continuation pressure, while the failed late-half-LR endpoint had near-zero
first-row pressure with negative continuation and full pressure. Three controls had no mixed-outcome reference tasks; Q constant 2302 had only
two, so its directional result has a narrow reference set.

No PQN model improved in this cycle: the six completed checkpoints were loaded,
observed, and left unchanged (`optimizer_updates = 0`). Earlier on the same small
food task, BC met the gate in 3/3 seeds and PQN in 1/3. This diagnostic does not
change that result, select an intervention, promote a model, or alter the Apex
`vector61` incumbent.

## What was measured

Each endpoint was sampled with three fresh rollout/action seeds (`2401`–`2403`),
using the qualified native task, λ=1 target calculation, and ε=0.02 behavior.
Each sampling unit contains nine native rollouts and caches all 144 prepared first
observations, rather than only tasks that failed in the earlier greedy evaluation.
The prior first-action probe supplies the fixed empirical success/failure action
sets for the primary Q-margin measurement.

For first, continuation, and full rollout components, the analysis reports
normalized plain-gradient directional pressure, `rho`. It is an infinitesimal
plain-gradient quantity, not an Adam-step prediction, not a measurement of the
historical training distribution, and not proof that a weighting change would
work. The [frozen intent](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-native-feedback/intent.json>) specifies the estimand and the gate. The previous greedy labels come from frozen counterfactual continuations; they are not guarantees of success under ε-greedy continuation.

## Directional pressure and endpoint classification

`Fst`, `Cont`, and `Full` show `rho` mean / classification for each fresh sampling
unit. `+`, `0`, `−`, `mix`, `inc`, and `NA` mean positive, near-zero, negative,
mixed within a seed, inconsistent across seeds, and missing fixed reference respectively. The endpoint class requires the same
component classification in all three units.

| Endpoint | Sampling seed | Fst rho / class | Cont rho / class | Full rho / class | Endpoint class: Fst / Cont / Full |
| --- | ---: | --- | --- | --- | --- |
| Q constant 2301 | 2401 | +0.10446 / + | −0.00810 / 0 | +0.09636 / + | + / 0 / inc |
| Q constant 2301 | 2402 | +0.12967 / + | −0.03425 / 0 | +0.09542 / mix | + / 0 / inc |
| Q constant 2301 | 2403 | +0.10575 / + | +0.00171 / 0 | +0.10746 / + | + / 0 / inc |
| Q constant 2302 | 2401 | +0.02302 / 0 | +0.16303 / + | +0.18605 / + | 0 / + / + |
| Q constant 2302 | 2402 | +0.02936 / 0 | +0.14174 / + | +0.17110 / + | 0 / + / + |
| Q constant 2302 | 2403 | +0.04584 / 0 | +0.14126 / + | +0.18710 / + | 0 / + / + |
| Q constant 2303 | 2401 | NA | NA | NA | NA / NA / NA |
| Q constant 2303 | 2402 | NA | NA | NA | NA / NA / NA |
| Q constant 2303 | 2403 | NA | NA | NA | NA / NA / NA |
| R late-half 2301 | 2401 | NA | NA | NA | NA / NA / NA |
| R late-half 2301 | 2402 | NA | NA | NA | NA / NA / NA |
| R late-half 2301 | 2403 | NA | NA | NA | NA / NA / NA |
| R late-half 2302 | 2401 | NA | NA | NA | NA / NA / NA |
| R late-half 2302 | 2402 | NA | NA | NA | NA / NA / NA |
| R late-half 2302 | 2403 | NA | NA | NA | NA / NA / NA |
| R late-half 2303 | 2401 | −0.02674 / 0 | −0.36677 / − | −0.39350 / − | 0 / − / − |
| R late-half 2303 | 2402 | −0.02989 / 0 | −0.37286 / − | −0.40275 / − | 0 / − / − |
| R late-half 2303 | 2403 | −0.02874 / 0 | −0.37091 / − | −0.39965 / − | 0 / − / − |

The weighting follow-up required both selected failed endpoints to be first-row
positive, continuation negative, and full negative or near-zero in every sampling
unit. Neither endpoint met that full pattern. The specific first-transition weighting hypothesis fails its declared support
gate. This does not establish that weighting could never work or identify
learning rate, Adam state, or training history as the cause.

The primary fixed-reference row counts were 144 for Q constant 2301, 2 for Q
constant 2302, 0 for Q constant 2303, 0 for R late-half 2301, 0 for R late-half
2302, and 144 for R late-half 2303. Missing reference sets remain controls, not
mechanism evidence.

## Conditional TD residuals from native behavior

Initial actions were sampled under ε=0.02 and grouped by their stored S
greedy-continuation outcome. The labels describe those counterfactual branches;
the fresh ε-greedy continuation can produce a different episode outcome. Every nonempty group has at least
ten rows and is therefore descriptive under the frozen rule. Each cell is
`count / status / median(target − Q_taken)`; `NA` denotes no sampled rows.

| Endpoint | Sampling seed | Empirical-success TD | Empirical-failure TD |
| --- | ---: | --- | --- |
| Q constant 2301 | 2401 | 80 / descriptive / +0.29510 | 64 / descriptive / −0.46173 |
| Q constant 2301 | 2402 | 81 / descriptive / +0.29305 | 63 / descriptive / −0.52927 |
| Q constant 2301 | 2403 | 82 / descriptive / +0.29573 | 62 / descriptive / −0.39392 |
| Q constant 2302 | 2401 | 144 / descriptive / −0.04236 | 0 / NA / NA |
| Q constant 2302 | 2402 | 144 / descriptive / −0.04236 | 0 / NA / NA |
| Q constant 2302 | 2403 | 144 / descriptive / −0.04203 | 0 / NA / NA |
| Q constant 2303 | 2401 | 144 / descriptive / −0.00655 | 0 / NA / NA |
| Q constant 2303 | 2402 | 144 / descriptive / −0.00601 | 0 / NA / NA |
| Q constant 2303 | 2403 | 144 / descriptive / −0.00569 | 0 / NA / NA |
| R late-half 2301 | 2401 | 144 / descriptive / −0.00740 | 0 / NA / NA |
| R late-half 2301 | 2402 | 144 / descriptive / −0.00756 | 0 / NA / NA |
| R late-half 2301 | 2403 | 144 / descriptive / −0.00782 | 0 / NA / NA |
| R late-half 2302 | 2401 | 144 / descriptive / −0.00183 | 0 / NA / NA |
| R late-half 2302 | 2402 | 144 / descriptive / −0.00283 | 0 / NA / NA |
| R late-half 2302 | 2403 | 144 / descriptive / −0.00184 | 0 / NA / NA |
| R late-half 2303 | 2401 | 49 / descriptive / +0.70047 | 95 / descriptive / +0.01577 |
| R late-half 2303 | 2402 | 49 / descriptive / +0.70154 | 95 / descriptive / +0.01704 |
| R late-half 2303 | 2403 | 49 / descriptive / +0.70154 | 95 / descriptive / +0.01704 |

## Verification and evidence

Seven qualification tests passed in 3.22 seconds. The six collection jobs closed
naturally in 79.94522 seconds; analysis took 1.03801 seconds. Peak observed RSS
was 983,875,584 bytes and the lowest available host memory was 27,392,327,680
bytes. The [completed analysis](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-native-feedback/analysis/analysis.json>) binds 304 frozen inputs and the [qualification result](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-native-feedback/qualification/result.json>) records the native proof.

The two recorded PNG plots were visually inspected: [gradient pressure, PNG](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-native-feedback/analysis/gradient-pressure.png>) / [SVG](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-native-feedback/analysis/gradient-pressure.svg>) and [reused greedy curves, PNG](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-native-feedback/analysis/learning-curves.png>) / [SVG](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-native-feedback/analysis/learning-curves.svg>). The curves are carried forward from completed Q/R checkpoint evaluation; no new learning curve was produced.

For the fixed greedy branches that provide the counterfactual labels, see the
[first-action probe report](../solo_food_first_action_probe_2026-09-13/README.md). The
original benchmark, λ=1 comparison, and late-half-LR comparison remain in the
[food microbenchmark report](../solo_food_microbenchmark_2026-09-13/README.md),
[λ=1 report](../solo_food_lambda1_2026-09-13/README.md), and
[late-half-LR report](../solo_food_late_half_lr_2026-09-13/README.md).

The next question is whether Q's regression can be reproduced on fixed teacher
experience relative to BC. That comparison is only a design at this point: it is
not frozen and has no outcome to report.
