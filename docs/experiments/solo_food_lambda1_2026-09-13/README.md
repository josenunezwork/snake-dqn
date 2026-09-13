# Solo food λ=1 target contrast — September 13, 2026

**Result: changing native PQN from λ=.65 to λ=1 did not establish reliable food
seeking across the three matched seeds.** The primary gate passed only seed 2303.
Seeds 2301 and 2302 acquired high success at earlier retained checkpoints but lost
a required final condition. The completed λ=.65 controls are reused as a matched,
outcome-informed contrast; this is not a fresh replication. No checkpoint,
recipe, or source is promoted. Apex `vector61` remains the incumbent.

## Question and boundary

The frozen question was whether changing **only** Q(λ), from .65 to 1.0, could
retain reliable greedy food-seeking in the completed short solo microbenchmark.
The task, teacher data, native `BatchSim` Watch path, raster observation,
optimizer, learning rate, epsilon schedule, seeds, and 50k useful-step budget
were fixed. The three earlier BC controls passed their own small fixed task and
were reused; BC was not rerun.

This remains a food-seeking diagnostic: length one and wall clearance remove
meaningful survival difficulty. Survival is a sanity check, not a learned-survival
result. The matched seeds were selected after the preceding λ=.65 result, so this
screen cannot estimate a fresh-seed effect or separate credit propagation from
online-bootstrap feedback.

The [PQN paper](https://arxiv.org/html/2407.04811v6#S4.E18) provides the
online, parallel Q-learning context and reports a λ ablation in its own tasks.
[Retrace](https://arxiv.org/abs/1606.02647) is relevant background on safe,
efficient off-policy multi-step returns. Neither paper sets a correct λ for this
Snake task or makes this three-seed contrast causal evidence.

## Frozen design

Each seed used the paired step-zero network from the completed λ=.65 PQN run.
The new λ=1 treatment retained the first checkpoints at or above 12.5k, 25k,
and 50k valid hero steps. Greedy evaluation used the same 144 training tasks and
288-task new-world and new-placement splits. The primary rule required every
seed to meet the fixed train, held-out, cell, improvement, and final-retention
conditions: train success ≥95%; each held-out split ≥90%; each heading-by-placement
cell ≥75%; each held-out split at least 30 percentage points above its initial
and calibrated random baseline; and no more than a five-point loss from midpoint
to final. Pooling seeds, choosing an earlier checkpoint, and secondary-metric
rescue were forbidden.

The frozen ceiling was three serialized MPS arms, each 50,000 valid hero steps
(with at most 255-step overshoot) and 600 seconds, plus nine CPU evaluations of
at most 180 seconds each. CPU/MPS smokes used the excluded seed 2026092309 and
512-step threshold. Guards remained two CPU threads, RSS ≤4 GiB, available memory
≥12 GiB, MPS driver memory ≤8 GiB, and heartbeat gaps ≤20 seconds. No budget
extension was used.

The [intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-lambda1/intent.json),
[execution freeze](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-lambda1/execution-freeze.json),
and [evaluation index](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-lambda1/evaluation-index.json)
record the only treatment change and the exact checkpoint/report closure.

## Primary gameplay result

The table reports greedy success as **train / new world / new placement**. `I` is
the reused matched initial. Clocks are actual valid hero steps. `C .65 final` is
the completed matched control endpoint, included for context and not used to
replace the treatment's own primary gate.

| Seed | λ=1 checkpoint / actual clock | Train / world / placement success (%) |
| --- | --- | --- |
| 2301 | I (shared 0) | 0.00 / 0.00 / 0.00 |
| 2301 | 12.5k / 12,630 | 100.00 / 100.00 / 100.00 |
| 2301 | 25k / 25,082 | 100.00 / 99.31 / 97.57 |
| 2301 | 50k / 50,060 | 56.25 / 55.90 / 55.21 |
| 2301 | C .65 final / 50,118 | 33.33 / 33.33 / 33.33 |
| 2302 | I (shared 0) | 0.00 / 0.00 / 0.00 |
| 2302 | 12.5k / 12,666 | 89.58 / 97.92 / 84.72 |
| 2302 | 25k / 25,062 | 100.00 / 100.00 / 100.00 |
| 2302 | 50k / 50,070 | 100.00 / 100.00 / 88.89 |
| 2302 | C .65 final / 50,106 | 100.00 / 100.00 / 100.00 |
| 2303 | I (shared 0) | 2.08 / 1.04 / 0.00 |
| 2303 | 12.5k / 12,536 | 92.36 / 91.32 / 79.17 |
| 2303 | 25k / 25,012 | 93.06 / 92.71 / 88.19 |
| 2303 | 50k / 50,056 | 100.00 / 96.88 / 95.83 |
| 2303 | C .65 final / 50,026 | 100.00 / 97.57 / 85.76 |

Only seed 2303 passed its entire treatment gate. Seed 2301 reached 100% on all
three splits at 12.5k and then declined to about 55–56% at the final. Seed 2302
reached 100% at 25k but ended at 88.89% on new placements, below the 90% floor
and its midpoint. These acquired-then-lost trajectories cannot be replaced by the
stronger earlier checkpoints.

The final matched treatment-minus-control deltas were positive in all three
splits for seed 2301 (+22.92, +22.57, +21.88 percentage points), zero/zero/−11.11
for seed 2302, and zero/−0.69/+10.07 for seed 2303. These outcome-informed,
reused-seed comparisons are descriptive; they do not repair the failed all-seed
treatment gate.

## Teacher fit versus actual gameplay

The final teacher-state accuracy is separate from greedy gameplay. It counts
correct labels on recorded teacher paths; most teacher-path actions are straight,
so macro recall is also shown. PQN teacher fit is descriptive, not a gate.

| Seed | Teacher accuracy train / world / placement (%) | Train macro recall (%) | Path efficiency train / world / placement |
| --- | --- | --- | --- |
| 2301 | 86.11 / 86.11 / 90.90 | 44.44 | .826 / .830 / .995 |
| 2302 | 78.30 / 77.78 / 68.68 | 91.32 | .861 / .861 / .801 |
| 2303 | 75.00 / 74.74 / 77.36 | 60.00 | .685 / .686 / .667 |

Efficiency is optimal distance divided by actual steps, **among successful episodes
only**. Seed 2301's high placement efficiency covers its surviving successes and
cannot compensate for failed episodes. Seed 2303 meets the success criteria while
using detours, illustrating why fit, success, and efficiency must remain separate.

## Secondary mechanism check

A secondary, post-primary check examined teacher action margins in six fixed
train-state groups. It passed one group (initial relative action 0 at distance 6)
and failed the other five. The first greedy actions were not all straight. This
is evidence against the particular predeclared all-six-group pattern, but it is
not a primary gameplay condition and does not establish why the trajectories
degraded. The [fit/mechanism audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-lambda1/fit-mechanism-audit.json)
keeps those measurements separate from the decision.

## Qualification, closure, and resources

The analysis completed with `bc_plays_well_pqn_fails`; the treatment primary gate
passed 1/3 seeds. The independent fit/mechanism audit passed all 27 fit blocks.
The V2 checkpoint audit passed all 12 checkpoint reloads. The frozen input closure
covered 736 files. Twelve scientific receipts exited naturally in 261.7204 s,
with maximum observed RSS 1.898 GB and minimum available memory 26.171 GB.

Checkpoint audit v1 made two invalid top-level `seed` assertions against a config
schema where seed is nested under `pqn_config`; it stopped with zero snapshots and
is retained as a failure record. V2 removed those two schema-invalid assertions,
then reloaded all 12 checkpoints successfully. It did not rerun learning or
change the task, seeds, criteria, or reported gameplay. See the preserved
[v1 audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-lambda1/checkpoint-audit.json)
and [v2 audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-lambda1/checkpoint-audit-v2.json).

## Recorded path evidence

The static learning curve and the three λ=1 final trajectory grids passed visual
review. The regenerated control grids are available for matched comparison but
were not separately visually inspected. In the fixed representative sample
(world `2026100200`, distance 5), λ=1 seed 2301's left and straight placements
succeeded in all eight cases in five steps while its right placements timed out
in all four at frame 16. Seed 2302 succeeded in all 12 cases with seven-step
detours. Seed 2303 succeeded in all 12 cases in seven or eleven steps. These are
bounded examples from predeclared cases, not an additional evaluation or a
population result.

- [Learning curves PNG](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-lambda1/evidence/lambda1-vs-control-learning-curves.png) and [SVG](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-lambda1/evidence/lambda1-vs-control-learning-curves.svg)
- λ=1 final trajectories: [2301](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-lambda1/evidence/seed2026092301-lambda1-final-trajectories.png), [2302](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-lambda1/evidence/seed2026092302-lambda1-final-trajectories.png), [2303](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-lambda1/evidence/seed2026092303-lambda1-final-trajectories.png)
- Regenerated completed controls: [2301](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-lambda1/evidence/seed2026092301-control-final-trajectories.png), [2302](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-lambda1/evidence/seed2026092302-control-final-trajectories.png), [2303](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-lambda1/evidence/seed2026092303-control-final-trajectories.png)
- [Visual review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-lambda1/visual-review.json), [render receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-lambda1/supervisor-runs/evidence-render/receipt.json), and [all recorded reports](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-lambda1/evaluation-index.json)

## Evidence and next step

- [Analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-lambda1/analysis/analysis.json), [analysis inputs](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-lambda1/analysis-inputs.json), [fit/mechanism audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-lambda1/fit-mechanism-audit.json), and [supervised receipts](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-lambda1/supervisor-runs)
- [Preceding microbenchmark report](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_microbenchmark_2026-09-13/README.md) and its completed λ=.65 [analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-microbenchmark/analysis/analysis.json)

The current evidence rejects λ=1 as a reliable replacement for λ=.65 in this
fixed diagnostic. It does not say whether the loss comes from longer credit
propagation, the removal of online bootstrap feedback, or another interaction.
The next bounded intervention keeps λ=1 and Adam at 5e-4 through the update
that writes the 25k checkpoint, then changes only its learning rate to 2.5e-4
through 50k. It preserves acquisition and tests later update magnitude. Each
run must match the completed control network, complete optimizer state, and
odometers at initial/12.5k/25k before the lower-rate update is allowed. Earlier
gameplay can be reused only after those comparisons pass. Full new arms are
needed because the checkpoints do not preserve simulator or rollout/SGD RNG
state and explicitly cannot resume this experiment. This next study is not
yet evidence; no fresh replication, promotion, or survival claim follows here.
