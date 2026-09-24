# BP: saved-data stall census separates inherited zero-food lanes from changed long stalls

**Status: COMPLETE.** BP finds that every observed
zero-any-food lane is an exact inherited parent/candidate trajectory. It also finds
six candidate-only food-free runs of at least 128 live frames. In all six, the parent
dies earlier and the candidate collects at least as much food, so these records do
not support a paired food-performance regression claim. This is a diagnosis of saved
solo traces, not a new policy result.

BP reuses the completed BN and BO archives for seeds 2026094401–2026094403. It
performs no rollout, training, inference, checkpoint selection, or promotion:
`new_rollouts`, `new_training`, and `new_inference` are all zero. The fixed policy
and the prior behavioral record remain in the [BM learning and fresh-initialization
report](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_shared_head_fresh_init_confirmation_2026-09-19/README.md),
[BN H256 report](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_longer_solo_confirmation_2026-09-19/README.md),
and [BO H512 report](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_h512_confirmation_2026-09-20/README.md).

## Frozen evidence and definitions

The census authenticates twelve existing learned-policy archives and analyzes 96
lanes for each seed/slice: BN H256, BO H256, and BO H512. BO H256 is the prefix of
BO H512 and is reported separately, never pooled with it as independent samples.
Food uses transition-valid rows and counts ambient plus corpse events; **zero-any**
means neither kind of food occurred. A long stall is a contiguous live,
transition-valid, food-free run of at least 128 frames. The threshold selects a
diagnostic record and is not a behavioral gate.

Each result cell gives the lane counts as `neither/shared/parent-only/candidate-only`
for the paired outcome. “Shared” is exact inherited only after available valid prefix
fields agree through the relevant candidate run. A candidate-only run after an action
difference is associated with that changed policy trajectory; it does not establish
that the safety head caused the stall. The source does not retain body geometry, time-varying food maps, reachability/Q
values, RNG state, or boosted intermediate positions. The original BO/BN reports
retain initial food maps. A repeated recorded kinematic suffix is therefore not a
full-state cycle.

## Nine saved slice results

| Study | Horizon | Seed | Zero-any N/S/P/C | >=128 stall N/S/P/C | Lanes with exact shared >=128 run | Shared stalls after divergence |
|---|---:|---:|---:|---:|---:|---:|
| BN | 256 | 2026094401 | 96 / 0 / 0 / 0 | 95 / 0 / 0 / 1 | 0 | 0 |
| BN | 256 | 2026094402 | 95 / 1 / 0 / 0 | 95 / 1 / 0 / 0 | 1 | 0 |
| BN | 256 | 2026094403 | 95 / 1 / 0 / 0 | 93 / 3 / 0 / 0 | 3 | 0 |
| BO | 256 | 2026094401 | 95 / 1 / 0 / 0 | 91 / 4 / 1 / 0 | 4 | 0 |
| BO | 512 | 2026094401 | 95 / 1 / 0 / 0 | 88 / 4 / 2 / 2 | 4 | 0 |
| BO | 256 | 2026094402 | 95 / 1 / 0 / 0 | 95 / 1 / 0 / 0 | 1 | 0 |
| BO | 512 | 2026094402 | 95 / 1 / 0 / 0 | 94 / 2 / 0 / 0 | 1 | 1 |
| BO | 256 | 2026094403 | 95 / 1 / 0 / 0 | 95 / 1 / 0 / 0 | 1 | 0 |
| BO | 512 | 2026094403 | 95 / 1 / 0 / 0 | 92 / 1 / 0 / 3 | 1 | 0 |

Every zero-any event is in the shared category: eight reported slice-lane instances
are exact inherited, including the three BO H256/H512 prefix repetitions. There are
no parent-only or candidate-only zero-any outcomes. Long stalls have both exact
inherited/shared and candidate-only outcomes. The BO H512 candidate-only cases are
therefore a change in the long-stall classification, not a change in whether a lane
has any food at all.

## Candidate-only long stalls and unequal exposure

The six candidate-only >=128-frame stalls are below. `Food` is any-food events and
`live` is transition-valid live frames. Each parent ends earlier; each candidate has
equal or higher food. Consequently, a longer candidate food-free suffix here does
not demonstrate a paired decline in food collection.

| Study / horizon | Seed | Lane | Candidate food / live | Parent food / live | Candidate qualifying run | First action difference |
|---|---:|---:|---:|---:|---|---:|
| BN H256 | 2026094401 | 7 | 13 / 256 | 13 / 91 | 186 frames, 71–256 | 91 |
| BO H512 | 2026094401 | 12 | 46 / 512 | 16 / 91 | 196 frames, 317–512 | 91 |
| BO H512 | 2026094401 | 90 | 56 / 512 | 28 / 176 | 145 frames, 168–312 | 176 |
| BO H512 | 2026094403 | 1 | 33 / 512 | 33 / 276 | 254 frames, 259–512 | 276 |
| BO H512 | 2026094403 | 65 | 51 / 512 | 19 / 97 | 139 frames, 304–442 | 97 |
| BO H512 | 2026094403 | 70 | 62 / 512 | 32 / 171 | 128 frames, 385–512 | 84 |

The first action difference may occur before a candidate-only run begins or during
its first qualifying 128-frame interval. The frozen definition requires it by
completion of that interval. In either timing, it remains association evidence, not
causal proof.

## Deterministic representative traces

The frozen selection order chose the longest candidate-only, exact-inherited shared,
and parent-only records. They are illustrations only; the tables above contain the
full nine-cell census.

- [Candidate-only: BO H512, seed 2026094403, lane 1](representative-candidate_only.png)
  — first changed action at frame 276; candidate-only 254-frame run.
- [Exact inherited shared: BO H512, seed 2026094401, lane 66](representative-shared.png)
  — no action difference in the saved common trace.
- [Parent-only: BO H512, seed 2026094401, lane 73](representative-parent_only.png)
  — first changed action at frame 24.

## Guarded execution and status

The completed scientific reducer used 4.3528928750311024 of its 30-second CPU science
cap. The [resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-stall-census/resource-rollup.json)
reports 606,633,984 bytes peak RSS, 40,376,123,392 bytes minimum available memory,
no MPS request or use by this CPU-only workload, one completed science job, and no
scientific failure. The actual
qualification wrapper passed all 18 tests once in 1.061 seconds. A controller
acceptance call made before that wrapper failed because its receipt did not yet
exist; it launched no job, did no numerical work, and was not a retry.

The resource rollup, static review, visual QA, Luna independent audit, and Sol science
review all pass. The completed campaign closeout reconciles BP as `BP_COMPLETE` without
a numerical rerun.

## Boundary and next decision

BP does not replace any policy, establish a learner improvement, or change the
frozen BN/BO outcomes. Apex remains the operational incumbent until the shared
tournament gate supports replacement. Its larger historical training budget is a
confound and does not establish architecture superiority.

The next bounded BQ plan will inspect both inherited stalls and stalls made visible by
the candidate’s extra survival through exact saved-action replay parity, before any
learner change. It performs no new training.

## Primary records

- [BP frozen design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-stall-census/design.md)
  — SHA-256 `b75532890b1caf0a8634cbf238096de2a9fe78a23b817ac2634c71de7e731341`
- [BP frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-stall-census/intent.json)
  — SHA-256 `09c9014cfb1311d6880ee379afffe6b160e0a85e129c3ea75084cf5d4722895e`
- [Completed analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-stall-census/analysis/analysis.json)
  — SHA-256 `85cfb124043199a6449150c45c939de0001f6a864da9cf928cc6075758312890`
- [Full lane CSV](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-stall-census/analysis/census.csv)
  — SHA-256 `d3efa7ad085be36b7d9897e3e3c1460c02b926db9bab508a90a80e45d41a0fbd`
- [Qualification completion](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-stall-census/qualification-complete.json)
- [Passing resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-stall-census/resource-rollup.json)
- [Passing static review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-stall-census/static-review.json)
- [Passing visual QA](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-stall-census/visual-qa.json)
- [Passing Luna independent audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-stall-census/independent-audit.json)
- [Passing Sol science review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-stall-census/science-review.json)
- [Completed BP closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-stall-census/closeout.json)
  — SHA-256 `e7004e751acfc1de373b0030710730d0b54c9748188be1772909343196f562ea`
