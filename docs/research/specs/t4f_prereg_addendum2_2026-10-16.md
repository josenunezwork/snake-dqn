# T4f RETURN_C pre-registration amendment — 2026-10-16

## Addendum 2 fix

This sibling amends `t4f_prereg_2026-10-16.md` under binding Planner addendum 2
of `t4f_return_contrast_spec_2026-10-16.md`. The original preregistration is
preserved byte-for-byte (SHA256
`dbf63e1043a6ab4b63245a1a7e60939fdb98c5a3ffe241b629a956930e2fdf48`).
The earlier repeated-relative-turn bank and its calibration are superseded.
All loss, sampling, recipe, seeds, tripwires, early-warning thresholds,
heldout-outside observations, fresh rounds 70/71, and GO/KILL/EXTEND rules remain
as registered. Five 8M runs remain UNSTARTED.

The corrected macro takes the recorded branch boost action on frame 1. On each
of frames 2 through k, B8 selects afresh at the prepared selector state; the
hero takes that relative direction with boost forced on if its boost variant
is legal, otherwise its normal variant. After frame k, ordinary B8 masked-greedy
continuation resumes until D=300 or death. Opponent dispatch, runtime preparation,
and observation occur on every frame. k=3 is read from the simulator's configured
`boost_length_cost_frames`. The corrected macro does not mechanically repeat a
left/right turn. It can still change survival and geometry relative to a single
boost frame, so observed target differences are not a pure metabolic-cost estimate.

Artifact paths below are relative to
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/`.
The new immutable bank is `t4f-return-bank-v3-20261016/banks/{train,holdout}`.
Its collection schema is `t4f-return-bank/v3`; field-layout schema remains 2.
The manifest binds the original T4e source bank and original collection plans.
The earlier source and repeated-turn banks remain preserved.

## Completed re-roll and target gaps

Task-owned `tmux t4f-collect2` completed successfully and exited naturally.
Twelve single-thread CPU workers took 2,463.705035 seconds (41.06 minutes).

| Split | Shards | States | Rows | Boost re-rolled | Non-boost unchanged |
|---|---:|---:|---:|---:|---:|
| Train | 330 | 18,924 | 97,846 | 47,269 | 50,577 |
| Holdout | 30 | 1,052 | 5,418 | 2,622 | 2,796 |
| Total | 360 | 19,976 | 103,264 | 49,891 | 53,373 |

Independent validation compared every original non-boost row field byte-for-byte
against both the source and superseded bank. All 53,373 pass. Boost initial-state
fields and row order, recorded B8 actions, exact shard sets/hashes, original plans,
source checkpoint hashes, and publication totals also pass. Evidence:
`t4f-addendum2-validation-20261016/macro-validation.json`.

Full targets use realized G plus gamma^T times masked source-B8 Q for survivors;
death has no bootstrap. Average boost and normal targets within each state, then
average those differences with equal state weights. Training retains moving-target
bootstrap as originally registered. These are descriptive targets, not policy results.

| Family | States with both classes | Original single-frame | Superseded repeated-turn | Corrected B8-direction macro |
|---|---:|---:|---:|---:|
| Training decisive | 9,457 | +31.9855 | -26.5725 | **+32.3813** |
| Training outside | 9,438 | -0.1588 | -61.3863 | **-0.9259** |
| Held-out decisive | 1,051 | +30.6650 | -25.2620 | **+31.9231** |

The corrected outside median paired gap is -0.0787 Q; the mean is -0.9259 Q.
Boost survival is 87.85% on training decisive, 96.34% on training outside, and
84.67% on held-out decisive rows. Normal survival remains 50.07%, 97.52%, and
47.89% respectively. The decisive boost advantage is retained. The outside mean
is within the addendum's expected order of magnitude, but includes changed
survival/continuation outcomes as well as burn cost. Complete distributions and
source hashes are in `t4f-addendum2-validation-20261016/{gap-table,macro-train-targets,
macro-holdout-targets}.json`.

The early-warning loader/evaluator smoke passed on the corrected holdout bank:
1,035 eligible decisive states and 17 explicit nondecisive exclusions, using
unchanged eligibility rules. Comparing B8_s0 to itself on all 40,559 frozen outside
observations yields zero gap change. This smoke uses synthetic 500k telemetry and
is an evaluator check only, not a training result or binding 1M outcome. Evidence:
`t4f-addendum2-validation-20261016/early-warning-smoke/full_bank_500k.json`.

## Calibration and prepared launch

Chosen **W=0.0012**, replacing the superseded W=0.00053. All other flags and
recipe settings remain unchanged. The first bracket measured equally pooled
initial gradient ratios 0.043733 at W=0.0001 and 1.708021 at W=0.01. Its log-log
interpolation selected candidate W=0.0021, but that candidate's confirmation
measured 0.864577, above the target. Narrowing to the measured bracket
W=0.0001/0.0021 gave interpolated W=0.001201083, rounded to **0.0012**.
A separate confirmation pair at the final weight measured pooled ratio
**0.545265**. The confirmation does not reselect the fixed weight.

| Initialization | Initial ratio, updates 1-50 | Active fraction 1-50 | Active fraction all 300 | Middle ratio 126-175 | Last ratio 251-300 | Max abs Q |
|---|---:|---:|---:|---:|---:|---:|
| B8_s0 | 0.636699 | 52.6468% | 66.2877% | 1.471601 | 1.218067 | 31.707859 |
| B8_s2 | 0.466962 | 53.5231% | 63.6061% | 1.031948 | 0.594727 | 49.557030 |

The initial ratio is the first-50-update geometric mean; seeds are equally pooled
geometrically. Active fractions use sampled non-reference rows as denominator.
Batch composition remains 256 anchors /128 return candidates /768 agent TD rows.
All eight explicit-MPS smokes completed exactly 300 updates below their 200k
transition ceiling, with no halt and actor exit codes [0,0]. Ratios drift after
initialization; the table retains this evidence. No gameplay, held-out outcomes,
or smoke-checkpoint performance entered W selection. These are calibration
smokes only, not 0.5M/1M early-warning or 8M campaign results.

Raw evidence: `t4f-calibration-addendum2-20261016/`; tracked calibration receipt:
`../t4f_calibration_addendum2_2026-10-16.json`. Both initial and refined bracket
receipts and all per-update measurements are retained, including the initial
candidate that exceeded the target.

The completed-bank audit `../t4f_seed_audit_addendum2_2026-10-16.json` passes:
rounds 70/71 have no collection-world or registered/evidence collisions. The
unfiltered sibling preserves the documentary transcript matches, and the qualified
audit records their exact point-in-time exclusion. Refresh before future screen
launch as originally required. No fresh evaluation worlds were played here.

Both tracked and artifact `redesign-t4f-mac-20261016/lane{0,1}.sh` now use
`research/t4f_return_20261016/launch_inputs_addendum2.json`, the corrected training
bank, calibrated W, and corrected holdout-bank evaluator configs. The receipt
binds this amendment alongside the unchanged original preregistration, input
checkpoints, bank manifests, code, scripts, and calibration/validation evidence.
`--check` verifies bound inputs and prints the full five commands without training.
Seeds, 8M transitions, explicit MPS, loss and epsilon, both boost tripwires, 500k
checkpoints, 0.5M/1M early warnings, shared arm halt, create-only output refusal,
and all GO/KILL/EXTEND rules remain unchanged.

The five 8M RETURN_C runs are **UNSTARTED**. Only the collection, evaluator smoke,
and capped calibration described here were executed. No unrelated tmux session
or process was manipulated. Once bound, preserve this amendment's bytes; further
changes require a sibling amendment.
