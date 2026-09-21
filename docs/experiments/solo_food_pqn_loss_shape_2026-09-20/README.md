# CO: native Huber versus half-MSE continuation

**Status: scientific execution is in progress.** CO compares the trainer's native
Huber objective with half mean-squared TD error after a matched continuation. The
qualification gate has passed, but no behavioral, reliability, or policy claim is
available until all three seeds finish and the saved evidence is independently
audited.

## Frozen comparison

Each arm restores the same CK `body_food` parent at mark 2560, including the full
Adam state, counters, and odometer, then performs 512 further updates. The three
fresh optimization seeds are 2026095801, 2026095802, and 2026095803. Both arms
use native Q(lambda=.65), gamma .997, the original sampler, masks, rewards,
clipping, action space, observations, network, and forward count. There is no
selector cut.

| Arm | Objective |
| --- | --- |
| `huber` | Canonical `smooth_l1_loss(q_taken, target.detach())`, beta 1, mean |
| `half_mse` | `0.5 * mse_loss(q_taken, target.detach())`, mean over the same real rows |

The study has six learners: both arms for each seed. Mark 2816 is descriptive;
mark 3072 is the decision checkpoint. The fixed H256 evaluation bank contains 32
fresh worlds, four headings, and three balanced reachable food placements per
world (384 lanes). The existing common CK parents are a Huber-trained warm start,
so this is a matched loss switch during continuation rather than training both
objectives from initialization.

The primary sources are the [PQN paper](https://arxiv.org/html/2407.04811v3) and
the authors' [MinAtar implementation](https://github.com/mttga/purejaxql/blob/main/purejaxql/pqn_minatar.py),
which uses a half-squared-error loss. They motivate a bounded local test; they do
not predict improved Snake behavior.

## Predeclared decision rule

At final H256, each policy must meet the inherited 16 absolute food and survival
conditions: food at least 75% of teacher at H64, H128, and H256; every one of the
12 pose cells at least 50% of teacher at those horizons; survival time at least
95%; at least 348 of 384 endpoint survivors; positive paired food-minus-random
lower confidence bound; and late-frame (129--256) food at least 75% of teacher.

For each seed, half-MSE must also beat Huber on all three paired conditions:
endpoint lower 95% confidence bound strictly above zero, food lower bound above
-5% of Huber mean food, and food-minus-initial lower bound above -5% of initial
mean food. The unit is the 32 paired world means (df 31); neither lanes nor seeds
are pooled. The final report will state Huber absolute reliability,
`mse_absolute_all_three`, and `loss_effect_all_three` separately. Overall success
requires the latter two conjunctions across all three seeds; Huber reliability is
reported independently.

## Qualification evidence

The qualification gate passed 41 tests in 1.687 seconds and its guarded MPS smoke
passed in 7.919393917 seconds, for 9.606393917 seconds of the 180-second
qualification budget. It charged three discarded MPS optimizer updates and zero
CPU optimizer updates, completed six checkpoint roundtrips, and dispatched 1,536
smoke gameplay lane frames.

The unwrapped canonical trainer and the Huber proxy were fully bit-exact. Their
first paired targets matched exactly, and each performed 18 learned forwards. The
first real batch had no saturated residuals; its maximum absolute residual was
0.453, so its actual Huber and half-MSE objectives coincided as expected. That
local batch cannot establish the intended difference. A separate MPS autograd
probe used residuals `[-2, -1, -0.5, 0, 0.5, 1, 2]`, detached targets, and no
optimizer updates to distinguish the two derivatives beyond beta 1.

A pre-freeze metadata mismatch was repaired before the intent froze by copying
the already hash-bound freshness scan into its expected metadata field. It started
no numerical process and made no optimizer update; the bank, criteria, and budget
were unchanged.

## Execution boundary

The scientific budget is 5,940 seconds. All six learners must complete their 512
updates, for 3,072 planned study updates, before final analysis. Every final
policy will be evaluated at H256, with H64 and H128 obtained as saved prefixes.
Jobs run serially with both compute locks, two CPU/BLAS threads, a 4 GiB process-tree
RSS cap, a 12 GiB available-memory floor, and an 8 GiB MPS driver cap.
No partial result, midpoint, or early seed result decides the study. The training
and evaluation evidence remains separate from historical teacher-fit references.

[CN's completed selector-cut result](../solo_food_pqn_selector_cut_2026-09-20/README.md)
did not support advancing to H384: its selector-cut absolute reliability was 2/3,
native was 1/3, and the complete paired benefit rule passed 0/3. CO is a new,
lambda=.65 loss-shape comparison, not a causal comparison with CN's lambda=.95
selector-cut continuation. Apex remains the operational incumbent; its larger
historical training dose remains a confound rather than evidence of inherent
superiority.

- [Immutable CO intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-loss-shape/intent.json)
- [CO frozen design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-loss-shape/design.md)
- [Qualification completion](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-loss-shape/qualification-complete.json)
- [Qualification accounting](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-loss-shape/qualification-accounting.json)
- [Primary-source note](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-loss-shape/research.md)
