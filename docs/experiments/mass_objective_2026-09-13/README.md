# Corrected PQN living-mass experiment — September 13, 2026

Status: **INCONSISTENT_STOPPED_NOT_PROMOTED**. Four training arms, two complete paired
evaluations, and the movement diagnostic completed. The first reward gain reversed on
the fresh training seed, triggering the declared futility stop. No model has been
promoted. Apex remains the incumbent; the new objective remains disabled by default.

## Question and prior evidence

Does a small explicit reward for living mass improve corrected PQN's full-horizon mass
compared with the current reward? The preceding four-hour campaign found that additional
unchanged training made both independent training seeds worse. Its trajectory audit observed
near-exclusive right turns, period-four loops, and very little food collection. These
observations do not establish the cause.

The coefficient `0.003` was selected before this experiment. A September 6 legacy raster/v2
screen tested the same coefficient; its three-seed effect estimate was inconclusive. The
present test is a replication with corrected observations, Watch decision timing, full SGD
coverage, per-environment autoreset, and fixed opponents. It is not a first test of this idea.

## Change and research basis

PQN adds this optional training-only term to the existing simulator reward:

```text
reward = simulator_reward + beta * transition_valid * post_alive * post_logical_length
```

`beta=0` is the default. The nonzero term changes the objective deliberately; it is not
potential-based shaping. Death and invalid rows receive no overlay. Surviving rows at the
population floor or frame cap receive the overlay, retaining existing bootstrap semantics.
The simulator, gameplay, architecture, optimizer, and evaluation metric are unchanged.

[Ng, Harada, and Russell](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf) explain the
policy-invariance distinction for potential-based shaping. This work supports identifying
the objective change; it does not predict a benefit from this coefficient. With
`gamma=.997` and `beta=.003`, a persistent mass unit contributes approximately one unit
of discounted return. Rewarding survival at mass one might also reinforce the observed loops.

[Empirical Design in Reinforcement Learning](https://www.jmlr.org/papers/v25/23-0183.html)
motivates declaring the comparison, seeds, and decision rule before examining results.
One matched training pair is descriptive evidence, regardless of how many evaluation
worlds it visits. It is not a population confidence interval or a promotion result.

## Frozen screen

- Training: one control and one treatment, seed `2026091317`, each targeting 50,000 useful
  hero transitions with identical initial tensors and RNG initialization.
- Recipe: CPU, two numerical threads, E16/S6/T16, one hero against five fixed `random_safe`
  opponents, corrected-v3/raster31v3, Watch phase, derived per-environment reset.
- Optimization: one complete shuffled epoch, padded batches of 256, gamma `.997`, lambda
  `.65`, unchanged Adam/Huber, epsilon `1 -> .02` over 30,000 useful transitions.
- Evaluation: initial/control/treatment across four held-out seeds `2026091351–2026091354`,
  5,000 frames, separate `random_safe` and `greedy_food` opponent mixes; six complete calls.
- Advance only if treatment-minus-control mean mass integral is positive in both mixes,
  all artifacts close cleanly, and further independently seeded replication is justified.
  Do not tune the coefficient from this result. No production promotion is authorized
  by this diagnostic comparison.

## Resources and reproducibility

The inspected host is an M5 Pro MacBook Pro with 18 CPU cores, 20 GPU cores, and 64 GiB
memory. PyTorch 2.9.1 reports MPS available. About 28.7 GiB was available initially, and
other applications were using the GPU, so this first screen uses two CPU threads.
Training and evaluation run serially under an external supervisor: 4 GiB process RSS,
12 GiB available-memory reserve, 600-second per-arm/evaluation limits, and confirmed
process cleanup. Host CPU/GPU utilization, available memory, swap, and thermal reports
are sampled separately. These bounds reduce risk; they do not reserve system resources.

Implementation commit: `1366b227b85eb72964dfadc598e4f30921a8554e`.
The source baseline is `c69533e6daccaff209919d7630c4238e60bfe38a`.
The external artifact directory contains the preregistered protocol, runner, supervisor,
configurations, source hashes, receipts, checkpoints, coverage counts, and pre-SGD
per-action target/TD/Huber aggregates:

[/Users/josenunez/Projects/ml/snake-dqn-artifacts/mass-objective-20260913](</Users/josenunez/Projects/ml/snake-dqn-artifacts/mass-objective-20260913>).

## Verification

Focused objective and configuration tests: 32 passed. Zero mode compared exactly against
the baseline for three updates in each of legacy, corrected pre-transition, and corrected
Watch modes: actions, rewards, validity, deaths, Q values, masks, targets, loss, RNG state,
and network parameters matched. The non-slow suite passed **2,760 tests**, with five
skips and three slow tests deselected, in 96.06 seconds. Two skips concerned private
champion checkpoints absent from the isolated checkout; three concerned active matching
reward contracts. The supervisor confirmed natural exit and no source drift.

## First matched pair: completed

Both arms completed naturally and reloaded their final checkpoints for inference. Control
trained on 50,070 useful transitions in 205.86 seconds; treatment trained on 50,113 in
198.05 seconds. Every eligible transition appeared exactly once in SGD. Configurations
differed only in the living-mass coefficient. Initial network tensors were identical:
`9a691bf6dd0a460ec807be1e632c8efed5492a8719c296a9801b602838e25c86`.

All six evaluation calls completed: 24 world rows and 120,000 scored world frames.
The metric is mean post-alive logical mass over the full 5,000-frame horizon.

| Opponents | Initial | Control final | Mass-reward final | Treatment minus control |
| --- | ---: | ---: | ---: | ---: |
| Random-safe | 2.59795 | 0.89305 | 11.29810 | +10.40505 |
| Greedy-food | 2.08355 | 1.52815 | 2.08815 | +0.56000 |

This passed the descriptive advance rule, but the greedy-food treatment barely exceeded
initialization and four evaluation worlds do not substitute for independent training seeds.
The control's post-training regression repeats the general concern from the prior campaign.

A separately declared two-call movement diagnostic replayed both finals on the same four
random-safe worlds. It reproduced every mass score exactly. Food events totaled 70 for
control and 202 for treatment. Per-world unique head cells were `164/272/482/668` for
control and `2047/630/3810/493` for treatment. Neither arm had an exact period-four loop
through its last 100 valid decisions on these worlds. This fresh control favored left
turns during evaluation; the older campaign's right-loop finding must not be copied onto
these new trajectories. Treatment lost to control on one of the four random-safe worlds.

The first pair's training process peaks were 0.90 GiB (control) and 0.84 GiB (treatment).
The complete evaluation peaked at 0.41 GiB. The system retained at least 26.27 GiB
available during training, comfortably above the declared 12 GiB reserve.

## Fresh-seed replication

`replication-protocol.json` declares fresh seeds `2026091318` and `2026091319`, the same
unchanged paired recipe and four evaluation worlds, and a conservative sequential stop:
if the first fresh pair is nonpositive in either opponent mix, do not run the second.
This is a screen for repeatability with an explicit futility rule, not an experiment
supporting formal population confidence intervals. Retain all outcomes, including
failures. Do not tune beta or replace the champion from this diagnostic.

The first fresh pair completed with 50,092 control and 50,240 treatment transitions.
All six evaluation calls completed, with matching paired world identities and no source
drift. The result failed the advance rule in both mixes:

| Opponents | Initial | Control final | Mass-reward final | Treatment minus control |
| --- | ---: | ---: | ---: | ---: |
| Random-safe | 1.33075 | 1.46435 | 0.87820 | -0.58615 |
| Greedy-food | 1.97255 | 1.67440 | 1.47070 | -0.20370 |

The second fresh seed (`2026091319`) was therefore **not run**, as preregistered.
Do not average away the reversal or describe the coefficient as a reliable improvement.
The fresh treatment's random-safe rows all had mean alive mass exactly one. This is
evidence of survival without growth; the first pair's trajectory trace does not prove
an exact movement loop in the second pair.

Total learning: **200,515 useful hero transitions** across four completed arms, with
each transition used exactly once by the sampler. Total quality evaluation: 12 complete
calls, 48 world rows, and 240,000 scored world frames. The separate movement diagnostic
replayed another 40,000 world frames and exactly reproduced its eight metric rows.

![Paired experiment means](</Users/josenunez/Projects/ml/snake-dqn-artifacts/mass-objective-20260913/results.png>)

## Decision and handoff

Keep the optional objective implementation as a tested research control, with default
zero. Preserve the first pair's promising candidate at
[mass/final.pth](</Users/josenunez/Projects/ml/snake-dqn-artifacts/mass-objective-20260913/mass/final.pth>)
alongside the negative replication. It has no promotion authority and is not the champion.

The result narrows the next question: whether broader heading/action experience can
make growth learning repeatable. A bounded next experiment should isolate randomized
reset headings or teacher-provided action coverage, while holding the reward and other
settings fixed. Fixed east-facing starts remain a plausible bias source, not a proven
cause. Do not combine this change with a new optimizer, architecture, or reward tuning
in the same comparison.

Authoritative local evidence:

- [First screen audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/mass-objective-20260913/audit.md>)
- [Replication audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/mass-objective-20260913/replication-audit.md>)
- [Final decision](</Users/josenunez/Projects/ml/snake-dqn-artifacts/mass-objective-20260913/decision.json>)
- [Frozen protocol](</Users/josenunez/Projects/ml/snake-dqn-artifacts/mass-objective-20260913/protocol.json>)
- [Experiment runner](</Users/josenunez/Projects/ml/snake-dqn-artifacts/mass-objective-20260913/runner.py>)
- [Host resource samples](</Users/josenunez/Projects/ml/snake-dqn-artifacts/mass-objective-20260913/resources.jsonl>)

Every supervised phase retains exact command arguments in `supervisor-runs/<name>/launch.json`
and completion/source/resource evidence in the adjacent `receipt.json`. Reproductions must
use new output directories and supervisor names because artifacts are create-only. The
public CLI also exposes the optional setting as `--living-mass-reward-coefficient`, with
a matching YAML `pqn.living_mass_reward_coefficient` field, for corrected-v3 recipes.

The closed host log contains 381 samples with no logger errors or reported thermal warnings.
System CPU ranged from 1.6% to 98.3%; exactly one sample exceeded 80%, during the fresh-seed
evaluation. Its cause cannot be attributed from host aggregates. System GPU utilization ranged
from 0% to 80%, despite all experimental jobs using CPU. The host retained at least 26.33 GiB
available in the five-second samples, and swap ended lower than it began (6.79 to 6.66 GiB).
The supervisor's faster samples separately captured a 26.27 GiB training minimum. These
measurements distinguish bounded learner usage from other activity on the Mac.

All experiment processes and the resource logger have exited. No checkpoint was promoted
and no remote push was performed.
