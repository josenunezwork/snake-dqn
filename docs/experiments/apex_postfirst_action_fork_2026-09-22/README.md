# Apex post-first-action fork qualification — 2026-09-22

This planned post-first-food action-fork diagnostic stopped at qualification
because its required strict novelty check found one overlap with prior saved
support. The sealed status is `COMPLETE_AUDITED_QUALIFICATION_FRESHNESS_FAIL`
and the decision is `STOP_NO_ENDPOINT_EVALUATION`. No checkpoints were loaded,
no endpoint games ran, and there is no learner Q-value or endpoint action-fork result.
The prior second-food study remains `FAILURE_GUARD` and Apex remains the
incumbent.

For completed learner evidence, see the [second-food continuation](../apex_second_food_continuation_2026-09-22/README.md) and its
[saved-record regression census](../apex_second_food_regression_census_2026-09-22/README.md).
This qualification failure adds no competing behavioral evidence.

## Planned question and qualified setup

The planned question was whether fresh post-first-food recurrences exposed a
local realized-return conflict, an action-value ranking mismatch, or a
continuation gap under all three first-action forks. It used 48 cases from two
starts, four headings, three bearings, and two second-food distances (8 and
14 moves).

For each case, a teacher naturally took two straight native actions, collected
the first food on frame 2, and grew from length 3 to 4. A controlled second
pellet was then inserted, with native replenishment maintaining one ambient pellet
at a time and a 32-frame continuation horizon. The teacher baseline completed 48/48 second-food
and survival outcomes. Each of the three forced first teacher actions also
completed 48/48 second-food and survival outcomes, for 144 forced fork games.
Fork-state identity was checked before any endpoint phase.

The planned fixed endpoint package was six checkpoints and 864 evaluation games.
It never ran because strict novelty was gated before **any** model use. The
criteria required zero exact overlap between post-first float32 state plus
six-bit selection mask and the frozen saved Apex optimization/game supports.

## Why qualification stopped

Case `s0-h1-b2-d8` overlaps one of 127,506 prior inputs. The matching record is
the training-arm replay-exposure artifact
[`results/2026099202-h64_growth-train/replay-exposure.json`](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/apex-second-food-continuation/results/2026099202-h64_growth-train/replay-exposure.json),
not a held evaluation path. That one legitimate overlap fails the fixed
zero-overlap rule. The study therefore stopped without resampling the bank,
relaxing criteria, or evaluating endpoints.

The independent saved-record audit passed the rejection: it checked all 192
qualification games, confirmed the teacher and forced-fork reachability facts,
and rejected the novelty overlap. It does not turn the invalid bank into fresh
evidence.

## Completed work and preserved failures

Qualification completed 192 games over 6,528 native frames, plus two frames
from incomplete observer prefixes. It recorded zero new model forwards,
training updates, or checkpoint loads. The completed native work is teacher
reachability and fork identity only; it is not an endpoint-policy evaluation.

Two observer failures are preserved separately from those 192 complete games:

| Attempt | Observer issue | Completed games / frames before failure | Updates / model loads |
| --- | --- | ---: | ---: |
| Original qualification | Absent lazy `_reward_prev_length` field was read before the first reward | 0 / 0 | 0 / 0 |
| Qualification v2 | Used `np.get_state` instead of `np.random.get_state` while capturing global RNG | 0 / 2 | 0 / 0 |

Both repairs changed observer handling only. They did not change the simulator,
reward function, bank, criteria, or RNG draws. No complete games were repeated.

## Limits and next authority

The bank is invalid as fresh evidence. The closed study permits neither
promotion nor an endpoint restart, bank resampling, criterion relaxation, or a
claim about learner performance across first-action forks. The root-owned next step is a new design
decision; it must not treat this failed bank as a training or held-path source.

Four guarded receipts charged 37.007 seconds. The observed peak
child RSS was 0.446 GiB and minimum available RAM was 32.083 GiB.

## Provenance

- Frozen source revision: `5f426debf6526eaa40c02be870f68aadd7fe147a`.
- [Frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/apex-postfirst-action-fork/intent.json)
  — SHA-256 `200ee47ebf887220cd29759246d5b4b16a53ec1f4355f8913e51bd8dff0ec758`.
- [Qualification record](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/apex-postfirst-action-fork/qualification.json)
  — SHA-256 `a6940ecb6dcdd6f665059db15bfa9166fc84c49a01853a757c4145d79a8f4ea9`.
- [Qualification-failure audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/apex-postfirst-action-fork/qualification-failure-audit.json)
  — `PASS`; SHA-256 `63a9f9bb762aa5ed34594c1af0e4fe3551fea6d925dc00ca10cc004d9bbaa863`.
- [Recovery amendment v1](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/apex-postfirst-action-fork/recovery-amendment-v1.json)
  — preserved the lazy-field failure; SHA-256 `a7195c9acb4b26335951ad04bd92cfda1124cc31db882148e38db65de2db2496`.
- [Recovery amendment v2](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/apex-postfirst-action-fork/recovery-amendment-v2.json)
  — preserved the RNG-observer failure; SHA-256 `08548b9c0f98f1f47e105a0b4d28f823220608bf5d11de8181b17ca8bf7bb408`.
- [Failure closeout admission](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/apex-postfirst-action-fork/failure-closeout-admission.json)
  — SHA-256 `92bd49ed852420cf6e12f6ecace79c26eca4864c420f73be40e5d9d7cdc1f817`.
- [Sealed closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/apex-postfirst-action-fork/closeout.json)
  — `COMPLETE_AUDITED_QUALIFICATION_FRESHNESS_FAIL`; SHA-256 `ca2cbc1c39b81ef60c054508d33c88125c0fccb460988ec8674c23af9a897816`.
