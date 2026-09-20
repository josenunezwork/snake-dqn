# Shared safety head fails the training admission gate (BH)

The restricted shared correction did not fit the training intervention labels in any
of three fresh residual seeds. Its final exact accuracies were 66/111, 52/103, and
42/88, all below the frozen 90% per-arm, per-seed admission threshold. Phase B was
therefore skipped: this study produced **no held-data evaluation, new-world
gameplay, or promotion evidence**. The qualified CNN arm reached 100% training
intervention fit in every seed, but that is training fit only and does not establish
generalization or behavior.

This report is for readers following the solo food and parent-safety campaign. It
records the frozen comparison, its exposure controls, and the stopped decision; it
does not propose a deployment procedure.

## Question and frozen comparison

BH asked whether a small shared per-action space correction could learn the
parent-consistent safety targets that the larger residual had fitted on the same
training data. Both arms retained the frozen FoodGeometryNetwork parent, full native
parent observation, six relative normal/boost actions, external legal mask, and
masked greedy argmax. There was no teacher action substitution or hard action
override.

The `cnn` arm was the qualified BA `BodyWallResidualNetwork` with reachability
inputs. The `shared` arm added the output of a shared `Linear(3,16)`, ReLU,
`Linear(16,1)` head separately to each of the six frozen-parent Q values. Its input
for action *a* was the deterministic three-value summary `[deficit[i],
threshold/cap, boost]`, where `i = a % 3` and `boost = int(a >= 3)`. This is a
rescaling of existing reachability and scalar information, not a privileged label or
teacher/parent-action input. The shared head gives equally roomy normal directions
the same residual in real arithmetic; native float32 rounding and low-index ties
remain part of evaluation. Normal and boost actions retain their separate speed-bit
input.

This is a **model package** comparison: it changes model size, parameter sharing,
and the residual input restriction together. The result cannot isolate a cause,
show that the CNN uses a shortcut, or prove that the available information is
insufficient. The [frozen design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-safety-head/design.md)
and [intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-safety-head/intent.json)
contain the complete action and feature contract.

## Matched Phase A

Fresh training seeds 2026094301–2026094303 map respectively to BF seeds
4201–4203 and BA parent lineages 4101–4103. Each arm used the same frozen,
parent-consistent 3,072-row BF training split for its seed; held data were not
loaded. On every update, both arms used the same sampled indices: 128 common
supervision plus 128 preservation rows with replacement, then a shared final
shuffle. They used the unchanged BF masked CE/KL/ordinary-residual loss, 500 Adam
updates, and fixed marks 0, 250, and 500.

All six training reports record MPS float32, PyTorch 2.9.1, two CPU threads, one
inter-op thread, no backend fallback, exact zero residual/parent behavior at mark 0,
and unchanged frozen-parent hashes. The admission reducer independently reloaded
the checkpoints. It found no parent-backend action mismatch across any 3,072-row
training set. These checks establish matched Phase A exposure and parent lineage;
they do not compare policy quality.

`I` below is exact intervention oracle-target accuracy. `P` is ordinary,
non-intervention agreement with the frozen parent. Both are training-split measures;
the denominators vary because the parent-consistent intervention subset is 111, 103,
or 88 rows.

| Seed | Mark | CNN I | CNN P | Shared I | Shared P |
|---|---:|---:|---:|---:|---:|
| 2026094301 | 0 | 0/111 (0.00%) | 2961/2961 (100.00%) | 0/111 (0.00%) | 2961/2961 (100.00%) |
| 2026094301 | 250 | 111/111 (100.00%) | 2929/2961 (98.92%) | 60/111 (54.05%) | 2961/2961 (100.00%) |
| 2026094301 | 500 | 111/111 (100.00%) | 2939/2961 (99.26%) | 66/111 (59.46%) | 2961/2961 (100.00%) |
| 2026094302 | 0 | 0/103 (0.00%) | 2969/2969 (100.00%) | 0/103 (0.00%) | 2969/2969 (100.00%) |
| 2026094302 | 250 | 102/103 (99.03%) | 2945/2969 (99.19%) | 48/103 (46.60%) | 2969/2969 (100.00%) |
| 2026094302 | 500 | 103/103 (100.00%) | 2943/2969 (99.12%) | 52/103 (50.49%) | 2968/2969 (99.97%) |
| 2026094303 | 0 | 0/88 (0.00%) | 2984/2984 (100.00%) | 0/88 (0.00%) | 2984/2984 (100.00%) |
| 2026094303 | 250 | 87/88 (98.86%) | 2964/2984 (99.33%) | 38/88 (43.18%) | 2984/2984 (100.00%) |
| 2026094303 | 500 | 88/88 (100.00%) | 2961/2984 (99.23%) | 42/88 (47.73%) | 2984/2984 (100.00%) |

The [BH training curves](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-safety-head/admission/training-curves.png)
show these three marks and the per-update objectives. The curves are evidence about
this fixed training package, not a learning curve for behavior in the simulator.

## Frozen stop rule and outcome

Admission required every arm and seed to have at least 90% final exact training
intervention accuracy, together with finite values, parent, runtime, sampler, and
resource checks. CNN passed its three training fits. Shared failed all three, so the
every-arm rule failed and the [admission analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-safety-head/admission/analysis.json)
sets `admitted_to_phase_b` to `false`.

As declared before execution, there were zero held-out evaluations and zero gameplay
evaluations. The fresh held worlds 2026104000–2026104007, gameplay worlds
2026104100–2026104107, and Phase B quality gates were never prepared or run. No
learning rate, update count, or selected mark was changed after the failed fit.

The appropriate next question is narrower: a train-only exact shared-feature
feasibility audit of the saved Q values and data can distinguish representability
pressure from preservation-loss constraints. It is not a result yet, and it does not
authorize new training or gameplay.

## Compute, qualification, and boundaries

The Phase A reservation was 390 seconds within the 1,200-second whole conditional
study budget: six serialized MPS learners capped at 60 seconds plus one 30-second
CPU admission reducer. The passing [resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-safety-head/resource-rollup.json)
records 210.16510287299752 seconds of completed Phase A science work, seven logical
and physical jobs, and no failed attempts. The unspent Phase B reservation is not
charged work and did not permit another experiment.

Qualification passed 31 tests in 1.765 seconds of its independent 120-second cap.
Peak recursive RSS was 1,152,843,776 bytes, minimum available host memory was
34,647,293,952 bytes, and peak MPS driver allocation was 1,165,017,088 bytes. The
rollup verifies input freezes and unchanged source/driver receipts; the MPS trainer
jobs reported nonzero allocation telemetry throughout their resource-bearing
heartbeats.

BH does not change the operational status of any policy. Apex remains the incumbent
until the shared tournament gate supports a replacement. Its larger historical
training budget is a confound, not evidence of inherent architecture superiority.

## Evidence and historical context

- [Frozen design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-safety-head/design.md)
- [Frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-safety-head/intent.json)
- [BH admission analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-safety-head/admission/analysis.json)
- [BH training curves](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-safety-head/admission/training-curves.png)
- [BH qualification receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-safety-head/qualification-complete.json)
- [BH resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-shared-safety-head/resource-rollup.json)
- [BF learning report](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_parent_safety_learning_2026-09-19/README.md)
- [BF learning curves — historical context, not BH behavior evidence](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-learning-r2/analysis/learning-curves.png)
- [BF representative gameplay — historical context, not BH behavior evidence](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-parent-safety-learning-r2/analysis/representative-gameplay.png)
