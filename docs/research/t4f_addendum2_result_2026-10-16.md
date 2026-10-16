# RETURN_C addendum 2 completion — 2026-10-16

The corrected boost macro, re-roll, recalibration, preregistration amendment,
and prepared launch inputs are complete. **No 8M RETURN_C training was started.**
The prior preregistration, source bank, and repeated-turn bank are preserved.

## Corrected macro and verification

Frame 1 executes the recorded boost branch. Frames 2..k select B8 afresh,
keep its current relative direction, and use its boosted variant when legal,
otherwise its normal variant. Ordinary B8 continuation follows to D300 or death.
The simulator burn period is **k=3**. Runtime preparation, opponent dispatch,
and observation remain in their normal per-frame lifecycle.

The integration regression test exercises `rollout_macro` and the real selector
helper: an initial boost-left followed by B8-straight produces `[3,4,4]`, with
heading changes `[-1,0,0]`. Other tests cover fresh changing B8 directions,
illegal-boost fallback, lifecycle, and rejection of the old macro by launch checks.
The broad focused suite passed 72 tests (3 slow tests deselected), followed by
all 5 macro tests including the added integration test. Independent read-only
review found no actionable issue and passed its 40 focused tests (overlapping
coverage, not an additional unique-test count). The learner/trainer and original
input identities were checked against their prior SHA bindings; default training
code was not changed. No full-repository suite was run.

Implementation commits: `51fc77f`, `9f3861b`, `bd8ee05`; each was fetched/merged
with origin/main and pushed to main and t4d-20261016 as requested.

## Re-roll and new target-gap table

Task-owned `tmux t4f-collect2` completed successfully and exited naturally.
**12 single-thread CPU workers; 2,463.705035 seconds (41.06 minutes).**
The new bank is `../snake-dqn-artifacts/t4f-return-bank-v3-20261016`.

| Split | Shards | States | Rows | Boost re-rolled | Non-boost byte-identical |
|---|---:|---:|---:|---:|---:|
| Train | 330 | 18,924 | 97,846 | 47,269 | 50,577 |
| Holdout | 30 | 1,052 | 5,418 | 2,622 | 2,796 |
| Total | 360 | 19,976 | 103,264 | 49,891 | 53,373 |

The independent validator verified every source/shard/checkpoint identity,
source/v2/v3 shard sets, all original non-boost row bytes against both older banks,
boost initial fields and row order, unchanged B8 actions, and publication totals.

Full bootstrapped boost-minus-normal targets, averaging within each state first
and then weighting states equally, in Q units:

| Family | Original single-frame | Repeated-turn macro | Corrected macro |
|---|---:|---:|---:|
| Training decisive | +31.9855 | -26.5725 | **+32.3813** |
| Training outside | -0.1588 | -61.3863 | **-0.9259** |
| Held-out decisive | +30.6650 | -25.2620 | **+31.9231** |

The corrected macro preserves the decisive boost advantage and removes the
large outside penalty from repeated turns. The outside paired median is -0.0787 Q.
These targets include changed geometry/survival and B8 continuation, so the outside
mean is not a pure isolated segment-burn estimate. Full target distributions and
survival fractions are retained in the evidence.

## Calibrated W and launch boundary

**W=0.0012**, pooled initial gradient ratio **0.545265**. B8_s0/s2 confirmation
ratios are **0.636699 / 0.466962**, with initial active-row fractions
**52.6468% / 53.5231%**. Active fractions over all 300 updates are 66.2877% /63.6061%.
The first broad interpolation produced W=0.0021, but its confirmation ratio was
0.864577. The measured narrower bracket selected 0.0012; its separate confirmation
was accepted without reselecting W. All eight MPS smokes completed exactly 300
updates at 78,080–78,208 transitions, with no halt and actor exits `[0,0]`.

The new [Addendum 2 fix](specs/t4f_prereg_addendum2_2026-10-16.md) is a sibling
amendment because the original preregistration is hash-bound. It preserves the
loss, recipe, seeds, tripwires, early warnings, heldout-outside set, rounds70/71,
and GO/KILL/EXTEND rules. The refreshed seed audit passes. The updated early-warning
loader/evaluator smoke reports 1,035 eligible decisive states /17 exclusions and
zero outside-gap change for identical B8 inputs; synthetic telemetry makes this
an evaluator check only, not a trained result.

`research/t4f_return_20261016/launch_inputs_addendum2.json` binds 93 evidence/input
files and exact bank shard hashes. Both artifact scripts passed `--check`, printing
all five full commands with corrected bank, W, explicit MPS, unchanged guards,
and 8M limits. The launch directory still contains only `lane0.sh` and `lane1.sh`:

- `/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t4f-mac-20261016/lane0.sh`
- `/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t4f-mac-20261016/lane1.sh`

No unrelated tmux session or process was manipulated. All task-owned collection,
statistics, evaluator, and calibration commands completed. The 8M campaign and
fresh screen remain unstarted.

Machine-readable [result](t4f_addendum2_result_2026-10-16.json) and
[calibration](t4f_calibration_addendum2_2026-10-16.json) bind the evidence.
Raw collection/validation artifacts are under
`../snake-dqn-artifacts/t4f-addendum2-validation-20261016/`; calibration artifacts
are under `../snake-dqn-artifacts/t4f-calibration-addendum2-20261016/`.
