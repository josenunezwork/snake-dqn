# BQ: saved-action state diagnostic finds no exact teacher-label conflicts

**Status: COMPLETE, independently audited.** BQ replays a fixed BO H512 panel using
archived actions and inspects the missing pre-action state evidence. All six replays
match the original trajectory exactly. Across the three seeds, no repeated exact
input received conflicting teacher labels. That absence does **not** prove the
observation is sufficient. The persistent zero-food lanes instead show food available
throughout while both diagnostic teachers often prefer a different action, without a
change to the parent’s archived action. This supports an own-action coverage and fit
investigation, not a learned repair or a causal claim about escape.

BQ is a diagnostic reuse of BO records, not a fresh score. It makes no optimizer
update, changes no archived policy action, and creates no rollout, promotion, or
opponent result. The preserved behavioral and learning evidence remains in the
[BP stall census](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_stall_census_2026-09-20/README.md),
[BO H512 confirmation](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_h512_confirmation_2026-09-20/README.md),
and [BM fresh-initialization report](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_shared_head_fresh_init_confirmation_2026-09-19/README.md).

## Fixed panel and replay parity

The panel uses the BO parent/candidate pairs for seeds 2026094401–2026094403. It
contains 12 deterministically selected lanes per seed: all 14 BP long-stall cases
(inherited, candidate-only, and parent-only) plus 22 fixed controls. That is 72
role-lane traces, or 36 parent/candidate lane pairs. It is a targeted diagnostic
panel, so its repeated rows and controls are not an independent score sample.

Each parent and candidate replay ran for 512 frames with archived actions. Every one
of the six replays exactly matched all 14 saved raw fields, the 12-lane initial food
state, and the shadow masked-argmax checkpoint action on all valid rows. The prepared
query witness also left its recorded fields unchanged. It witnesses alive/head/
direction/length/boost, masks, bodies, and native-order food; it does not separately
hash hidden RNG, hunger/frame, corpse identity, respawn, or event state. Exact raw
replay parity is the behavioral check; neither witness is an exhaustive hidden-state
claim.

The diagnostic compares the parent action with GreedyFood and the candidate action
with SpaceTeacher on the same prepared state and resolved six-action mask. These are
read-only labels, never rollout actions or training examples. The bounded state
capture preserves initial food maps but not a fresh random sample of all possible
states, so teacher disagreement alone cannot show an escape route or successful
imitation.

## Exact-input conflict audit

For each seed, rows are pooled only within that seed across its visited parent and
candidate states. `Unique` is the exact input-hash count; `repeat groups` counts input
values with more than one row. Every conflict count is zero for every stated target.
That rules out an exact deterministic label contradiction within this visited panel;
it does not establish capacity, optimizer, state-coverage, or generalization
sufficiency.

| Seed | Valid rows | Candidate input vs SpaceTeacher: unique / repeat groups / conflicts | Food input vs GreedyFood: unique / repeat groups / conflicts | Full observation vs SpaceTeacher: unique / repeat groups / conflicts |
|---|---:|---:|---:|---:|
| 2026094401 | 11,130 | 4,362 / 2,119 / 0 | 4,337 / 2,120 / 0 | 7,278 / 3,828 / 0 |
| 2026094402 | 10,460 | 5,844 / 2,998 / 0 | 5,841 / 2,999 / 0 | 7,034 / 3,410 / 0 |
| 2026094403 | 9,538 | 5,599 / 2,764 / 0 | 5,590 / 2,772 / 0 | 6,406 / 3,116 / 0 |

## All selected stall cases

The decision statistic is `valid / food-present / GreedyFood disagreement /
SpaceTeacher disagreement / candidate-action change from parent`, restricted to the
role’s within-stall valid frames. `role-only` is valid exposure after the other role
has no matching frame. It is reported separately so no candidate decision after a
parent death is treated as a paired parent comparison. A zero is an absent stall for
that role, not a zero-length observation.

| Seed | Lane | BP category | Parent within-stall statistic; role-only | Candidate within-stall statistic; role-only |
|---|---:|---|---|---|
| 2026094401 | 8 | shared | 438 / 438 / 240 / 240 / 0; 0 | 438 / 438 / 240 / 240 / 0; 0 |
| 2026094401 | 9 | parent-only | 353 / 353 / 214 / 249 / 0; 0 | 0 / 0 / 0 / 0 / 0; 0 |
| 2026094401 | 12 | candidate-only | 0 / 0 / 0 / 0 / 0; 0 | 196 / 196 / 100 / 99 / 32; 420 |
| 2026094401 | 59 | shared | 506 / 506 / 379 / 379 / 0; 0 | 506 / 506 / 379 / 379 / 0; 0 |
| 2026094401 | 66 | shared | 512 / 512 / 384 / 384 / 0; 0 | 512 / 512 / 384 / 384 / 0; 0 |
| 2026094401 | 73 | parent-only | 492 / 492 / 248 / 490 / 0; 0 | 0 / 0 / 0 / 0 / 0; 0 |
| 2026094401 | 85 | shared | 506 / 506 / 379 / 379 / 0; 0 | 506 / 506 / 379 / 379 / 0; 0 |
| 2026094401 | 90 | candidate-only | 0 / 0 / 0 / 0 / 0; 0 | 145 / 145 / 61 / 55 / 10; 335 |
| 2026094402 | 10 | shared | 512 / 512 / 384 / 384 / 0; 0 | 512 / 512 / 384 / 384 / 0; 0 |
| 2026094402 | 71 | shared | 356 / 356 / 101 / 151 / 0; 0 | 356 / 356 / 155 / 143 / 11; 0 |
| 2026094403 | 1 | candidate-only | 0 / 0 / 0 / 0 / 0; 0 | 254 / 254 / 94 / 93 / 26; 235 |
| 2026094403 | 19 | shared | 512 / 512 / 511 / 511 / 0; 0 | 512 / 512 / 511 / 511 / 0; 0 |
| 2026094403 | 65 | candidate-only | 0 / 0 / 0 / 0 / 0; 0 | 139 / 139 / 67 / 57 / 22; 414 |
| 2026094403 | 70 | candidate-only | 0 / 0 / 0 / 0 / 0; 0 | 128 / 128 / 61 / 56 / 11; 340 |

Within the three exact inherited zero-any lanes—seed 4401 lane 66, seed 4402 lane
10, and seed 4403 lane 19—food is present on all 512 valid frames for both roles.
GreedyFood and SpaceTeacher each disagree with the archived action on 384/384/511
frames respectively, while the parent-action-change count remains zero. This is
consistent with an own-action coverage/fit question. It does not demonstrate that a
teacher action escapes the path, that a safety head caused the path, or that a learner
can reproduce the teacher behavior.

## Representative state evidence

The fixed representatives show pre-action state and recorded-path context. They are
examples selected by the frozen BP order, not new evaluation plots.

- [Candidate-only state: seed 4403, lane 1, frame 287](state-candidate_only.png)
- [Exact inherited shared state: seed 4401, lane 66, frame 3](state-shared.png)
- [Parent-only state: seed 4401, lane 73, frame 26](state-parent_only.png)

## Guarded execution and boundary

Qualification passed 40 tests in 3.662 seconds. Six saved-action replay jobs and one
analysis job used 55.195727625105064 of the 300-second science cap. The passing
resource rollup records 981,663,744 bytes peak RSS, 39,245,135,872 bytes minimum
available memory, and a CPU-only workload with no MPS allocation measurement. No
new rollout or training occurred, and optimizer updates remain zero.

All numerical jobs completed on their first attempts. Independent Sol scientific
review and Luna receipt/report verification passed; root visually inspected all
three figures. The audit verified 2,823 frozen input paths and reconciled all 28
role-specific stall-table entries with the saved analysis. Audit-code development
required corrections to several field-location assumptions; those were checker
changes, not experiment retries or changes to the saved results.

BQ cannot establish observation sufficiency, teacher escape behavior, causal safety
benefit, learner progress, or promotion. Apex remains the operational incumbent until
the shared tournament gate supports replacement; its larger historical training
budget is a confound, not evidence of architectural superiority. The next bounded
work should test own-action coverage and fit under a separately frozen protocol.

## Primary records

- [BQ frozen design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-stall-state-diagnostic/design.md)
  — SHA-256 `95d205e129a0057ab6f5104b7c88c4d291b4ec6008b7e31452b007f48ebef4bf`
- [BQ frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-stall-state-diagnostic/intent.json)
  — SHA-256 `508ce68ab44769efb71f14af1a9a45614e6fc95fcc752b772eb2456c3f6c6dd3`
- [Completed analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-stall-state-diagnostic/analysis/analysis.json)
  — SHA-256 `014065ebc36e1142ebea36efc9cb9d336c6a00733f71cd36ccce61c810575793`
- [Panel audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-stall-state-diagnostic/panel-audit.json)
- [Qualification completion](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-stall-state-diagnostic/qualification-complete.json)
- [Passing resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-stall-state-diagnostic/resource-rollup.json)
- [Passing static review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-stall-state-diagnostic/static-review.json)
- [Independent receipt and report audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-stall-state-diagnostic/independent-audit.json)
- [Scientific review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-stall-state-diagnostic/science-review.json)
- [Visual inspection](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-stall-state-diagnostic/visual-qa.json)
- [Completed closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-stall-state-diagnostic/closeout.json)
  — SHA-256 `3166f066ddd1e6efd8cd28c49ed7ea8d4d94eb3c0ae3ddbf1612723ffb91d781`
