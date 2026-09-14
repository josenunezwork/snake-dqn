# Behavior-anchored native PQN retains H64 food seeking (AP)

AP shows that a fixed teacher-action anchor can **reliably retain** the learned H64 food-seeking
skill during 32 native PQN updates. The anchored hybrid passes retention in all three fresh seed
pairs and every preregistered anchored-versus-native comparison. The unanchored native continuation
fails retention in all three pairs. None of the three anchored seeds has a final-versus-initial food
interval with a strictly positive lower bound, so AP does not show an improvement beyond the AN
behavior-cloning parent. This is a
short-horizon skill-preservation result. It does not establish longer-horizon reliability,
opponent performance, or promotion. Apex remains incumbent and the shared tournament gate remains
the only promotion authority.

This report is for researchers assessing the first successful retention intervention after
[AO's failed unanchored warm start](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_pqn_warmstart_2026-09-14/README.md).
It describes a fixed, paired hybrid configuration and what it can establish: under this exact
H64 task, a same-step cross-entropy anchor protects the pre-existing learned food skill better
than native PQN alone. It cannot identify the optimal anchor coefficient, prove a cause of AO's
failure, establish own-policy coverage, or support serving changes.

## Frozen question and matched treatment

The frozen question is: *Does a fixed teacher-action cross-entropy anchor inside the same native
PQN optimizer step preserve the demonstrated H64 food skill across three fresh matched seed pairs?*
The frozen intent uses source revision `56e0e92434ae510a84ebaf6f58cf41c3e4421b04`, a 960-second
scientific budget, and a 120-second qualification budget. It changes no canonical source file,
serving path, policy-selection contract, tournament gate, or Apex checkpoint.

Each pair starts from exactly matched AN raster behavior-cloning checkpoint-500 weights: parent
seeds `2026093401`–`2026093403` pair with fresh AP seeds `2026093601`–`2026093603`. Both arms use
fresh Adam state and the same independent 3,072-row original behavior-cloning examples, labeled
by GreedyFood `old_actions`, in batches of 256 for 8,192 presentations. Archived SpaceTeacher
source actions remain retained separately. The native trajectories are allowed to diverge after
their actions diverge.

The `native` control has CE weight 0: it runs unchanged native mean SmoothL1 Q(λ), computes the
anchor CE only for logging, and returns the original TD tensor. The `anchored` treatment has CE
weight 1: native TD plus masked six-action cross entropy over the fixed examples, followed by one
shared backward pass, gradient clip, and Adam step. The artifact-only subclass uses a temporary
proxy for canonical trainer-module `F`, restores it in `finally`, and leaves the canonical source
unchanged. Qualification establishes weight-zero exact native-update parity; it does not turn this
hybrid into pure PQN.

Both arms use the AO recipe: 16 one-snake environments, 16-frame rollouts, H64, 300 initial and
maximum food, epsilon 0.1, gamma 0.997, lambda 0.65, Adam learning rate 0.0005 and epsilon
0.00015, clip 10, one padded batch of 256, ambient coefficient 0.1, corrected-v3 watch/autoreset,
derived environment-episode seeds, and disabled opponent pool. Native random reset distributions
remain different from balanced evaluation placements. The demonstration motivation is related to
[Deep Q-learning from Demonstrations](https://arxiv.org/abs/1704.03732), but AP is not a DQfD
replication; its native learner basis is [PQN](https://arxiv.org/abs/2407.04811).

## Fresh task and fixed decisions

Evaluation uses fresh worlds `2026102100`–`2026102107`, four headings, and three reachable food
rays at distance six: 96 balanced lanes at H64 per call. It records greedy gameplay plus separate
report-only agreement with 3,072 retained original behavior-cloning labels and 1,536 fresh
GreedyFood-teacher observations. These fit references are separate from TD and CE losses and
cannot rescue failed gameplay.

The fixed anchors establish useful room over random while making H64 survival easy:

| Anchor | Food per H64 game | Time alive | Endpoint survival |
|---|---:|---:|---:|
| GreedyFood teacher | 12.7604167 | 100.00% | 96 / 96 |
| Random-safe | 1.3229167 | 100.00% | 96 / 96 |

Each arm must retain at least 95% of initial mean food, reach 75% of teacher food and 50% in every
cell, keep time alive at least 0.95 and endpoint survival at least 87/96, remain above random with
an eight-world food interval whose lower bound is strictly positive, and finish no more than 0.05
teacher food below the midpoint mark. Improvement beyond BC is a separate test: each anchored
seed would also need an eight-world final-minus-shared-initial food interval with a strictly
positive lower bound. The anchored-versus-native comparison additionally requires all three
anchored retention passes, a positive lower bound for final food in every eight-world pair, time
no more than 0.01 lower, and endpoint no more than two lanes lower. Each interval uses eight
paired world means and a t 95% interval (df 7); no best checkpoint selection is permitted.

## Completed gameplay and agreement

The three initial evaluations are shared aliases: native step 0 was evaluated once per seed after
an exact paired-tensor-hash check, then used as the step-0 result for both roles. This avoids
duplicating games. Together with native and anchored steps 8 and 32, the table contains all 15
fixed learned checkpoint roles. `Food` is mean ambient food per game; `time` is fraction of frames
alive; `end` is surviving lanes of 96.

| Seed | Checkpoint role | Updates | Food | Train agreement | Held agreement | Time | End |
|---|---|---:|---:|---:|---:|---:|---:|
| 3601 | Shared initial (native = anchored) | 0 | 11.5521 | 99.97% | 87.37% | 99.84% | 95 |
| 3601 | Native | 8 | 5.7188 | 71.84% | 75.13% | 100.00% | 96 |
| 3601 | Native | 32 | 5.9375 | 72.56% | 73.24% | 99.92% | 95 |
| 3601 | Anchored | 8 | 10.7500 | 94.89% | 88.02% | 99.74% | 94 |
| 3601 | Anchored | 32 | 11.5417 | 99.80% | 87.43% | 100.00% | 96 |
| 3602 | Shared initial (native = anchored) | 0 | 11.6458 | 99.97% | 88.41% | 100.00% | 96 |
| 3602 | Native | 8 | 8.3125 | 78.32% | 79.10% | 100.00% | 96 |
| 3602 | Native | 32 | 3.3438 | 75.29% | 59.57% | 100.00% | 96 |
| 3602 | Anchored | 8 | 10.9479 | 99.35% | 87.57% | 99.40% | 94 |
| 3602 | Anchored | 32 | 11.2813 | 99.48% | 86.46% | 100.00% | 96 |
| 3603 | Shared initial (native = anchored) | 0 | 11.4167 | 99.90% | 86.85% | 99.51% | 95 |
| 3603 | Native | 8 | 4.6563 | 58.11% | 70.44% | 100.00% | 96 |
| 3603 | Native | 32 | 1.3958 | 66.18% | 60.68% | 100.00% | 96 |
| 3603 | Anchored | 8 | 10.2708 | 94.53% | 86.07% | 99.76% | 93 |
| 3603 | Anchored | 32 | 11.3229 | 99.61% | 86.52% | 100.00% | 96 |

The anchored final food is 11.5417, 11.2813, and 11.3229. It retains the H64 skill in every seed;
the native final food is 5.9375, 3.3438, and 1.3958 and fails retention in every seed. The H64
time and endpoint checks alone are not enough to detect this: both arms mostly survive at the
short horizon, while food seeking separates them. The anchored arm misses the report-only held
agreement thresholds in every seed, which does not alter its behavioral retention decision.

## Paired comparison and boundary of the result

Every anchored-versus-native comparison passes. All 24 paired world deltas are positive, and each
seed's paired eight-world interval has a strictly positive lower bound:

| Seed | Anchored − native food | Eight-world 95% interval | Time comparison | Endpoint comparison | Pair gate |
|---|---:|---|---|---|---|
| 3601 | +5.6042 | [+5.16394, +6.04439] | Pass | Pass | Pass |
| 3602 | +7.9375 | [+6.87455, +9.00045] | Pass | Pass | Pass |
| 3603 | +9.9271 | [+9.47953, +10.37463] | Pass | Pass | Pass |

Anchored retention passes three of three, native retention fails three of three, and the
preregistered comparison passes three of three. The anchor therefore provides credible evidence
that this fixed hybrid retains food seeking better than the matched native update on this task.

It does **not** demonstrate improvement beyond the behavior-cloning parent. Anchored
final-minus-initial food intervals are `[-0.52297, +0.50213]`, `[-0.96958, +0.24041]`, and
`[-0.60900, +0.42150]` for seeds 3601–3603. Every interval includes zero, so all three
improvement decisions fail. AP can claim preservation across this update horizon; it cannot claim
that value updates added food-seeking performance beyond imitation or that CE weight 1 is optimal.
For comparison, the native final-minus-initial intervals are `[-5.99424, -5.23493]`,
`[-9.07872, -7.52545]`, and `[-10.55758, -9.48408]`; each is wholly negative.

## Completion, qualification, and remaining review

All 25 intended scientific jobs completed without numerical failure or repeat in
`180.46930208266713` of the 960-second scientific budget. Training recorded 49,085 valid agent
steps and 192 Adam steps. Qualification passed 27 tests and three smoke runs in
`15.245418958028779` of the 120-second budget, including exact weight-zero native-update parity.
The resource rollup passed: peak RSS was 1,133,264,896 bytes, minimum available host memory was
24,287,674,368 bytes, and peak MPS driver allocation was 1,224,376,320 bytes. Training itself
used `62.71626304043457` seconds. The campaign retained serialized heavy jobs and its Mac guards.

Independent review returned `PASS_AUDIT_STUDY_COMPARATIVE_SUCCESS_NO_IMPROVEMENT`: it validated
914 frozen analysis inputs and all 25 serialized scientific jobs, confirmed the three-of-three
retention and comparative outcomes, and separately confirmed no improvement beyond the shared BC
initialization. The independent raw-array audit passed with zero mismatches after decoding 17
frame and 30 Q archives, comparing every gameplay summary and train/held fit metric across 46,080
training and 23,040 held-out rows, and checking all three shared-initial aliases. It reuses the
separate study review for the 914 frozen hashes, 25 receipts, confidence intervals, and gates.
Visual QA passed both figures with no model or game rerun. The completed closeout is
`CLOSED_COMPLETE_COMPARATIVE_RETENTION_PASS_NO_IMPROVEMENT`; promotion remains ineligible and
Apex remains unchanged.

The next bounded probe is AQ: evaluation only at H128 for the existing AP anchored-32 checkpoints
against the AN behavior-cloning parents, over the same three seed cohorts and fresh worlds
`2026102200`–`2026102207`. Its design budgets 260 scientific seconds and 120 qualification
seconds. AQ intent SHA-256 is
`9fb40c31b5d3770d4869763c6f8e7a5aa13970981a2fb24221f49a2e0d8ac448`; qualification has started,
but no AQ science result exists. Its [frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-anchor-128/intent.json)
records the prospective evaluation-only boundary.

## Evidence

[Frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-behavior-anchor/intent.json)
· [Frozen design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-behavior-anchor/design.json)
· [Job map](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-behavior-anchor/job-map.json)
· [Qualification completion](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-behavior-anchor/qualification-complete.json)
· [Completed analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-behavior-anchor/analysis/analysis.json)
· [Resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-behavior-anchor/resource-rollup.json)
· [Visual QA](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-behavior-anchor/visual-qa.json)
· [Independent study review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-behavior-anchor/independent-review.json)
· [Raw-array audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-behavior-anchor/independent-array-audit.json)
· [Final closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-behavior-anchor/closeout.json)
· [Teacher anchor](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-behavior-anchor/anchors/teacher/report.json)
· [Random-safe anchor](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-behavior-anchor/anchors/random_safe/report.json)

The raw-array audit SHA-256 is
`322ab877fb21bec9eda4830dfdc97c8876908f4488273e89452ec818eb482862`; final closeout SHA-256 is
`90fa979d551a37bd3a510363ce0fa624c88b64898e6230f417eaa90fc2abed94`.

![AP learning curves](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-behavior-anchor/analysis/learning-curve.png)

![AP representative gameplay](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-behavior-anchor/analysis/representative-gameplay.png)

The learning curve shows all nine seed/role trajectories, separating native dashed from anchored
solid series and raw TD, CE, weighted, and total losses. The fixed gameplay panel presents the
first fresh world, upward heading, and all three initial food rays for the anchors, shared starts,
and both final policies per seed. These displays are fixed before outcomes and are not selected
best or worst examples.
