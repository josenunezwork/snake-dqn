# Native death replay — 2026-09-14

The replay establishes that the 100 deaths recorded in the 256-frame screen occurred
when the advisory safe-action set was empty. Every replay was exact, every fatal action
was a legal fallback, and none was an action that the advisory mask had marked safe.
This rules out an advisory-safe action becoming fatal at the recorded decision point.
It does not show that an earlier action could not have avoided the later dead end.

## Scope and method

The [frozen intent](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-death-replay/intent.json>) replayed the saved action arrays from the
256-frame AD screen for teacher and the three fixed final AA policies. Each target used
96 lanes for 256 frames. It rebuilt the original native worlds, loaded no checkpoint,
and made no model decision. There were zero SGD updates, zero new policy decisions, and
no rescore of the preceding behavior result.

All four targets exactly reproduced the archived 14 state, event, action, and mask
fields and the initial-food state. A death case is every valid transition with `done`;
the replay inspected all such rows rather than selecting examples. `fallback` means the
chosen resolved action was legal, the advisory mask rejected it, and no action was both
legal and advisory safe. An advisory-safe fatal case would instead require an action
marked advisory safe to die on the exact recorded transition.

## Recorded decision-time finding

All 100 recorded deaths were self deaths and legal fallbacks. No target had an
advisory-safe fatal action.

| Target | Recorded deaths | Advisory-safe fatal | Legal fallback |
| --- | ---: | ---: | ---: |
| Teacher | 12 | 0 | 12 |
| Final AA seed 2026092901 | 41 | 0 | 41 |
| Final AA seed 2026092902 | 17 | 0 | 17 |
| Final AA seed 2026092903 | 30 | 0 | 30 |
| **Total** | **100** | **0** | **100** |

The [complete replay analysis](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-death-replay/evidence/analysis.json>) records
`ALL_RECORDED_DEATHS_FALLBACK`, four exact targets, no missing targets, and no new
learning work. The [first-death figure](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-death-replay/evidence/first-deaths.png>) is one
chronological death from each target; it illustrates the rule and is not a death census.
The preceding [AD learning curves](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-food-256/evidence/behavior-curves.png>) remain the
behavioral evidence: its three final policies failed the 256-frame survival gates. This
replay leaves those failures unchanged. AA held-out fit remains 0 of 3, and Apex remains
the incumbent.

## What this does and does not identify

The result localizes the recorded fatal transition to a state with no advisory-safe
legal action. It does not prove that the policy could not have reached a safer state
through an earlier turn, identify a unique body-layout or planning cause, or establish
that changing the advisory mask would improve survival. The original saved traces do not
supply counterfactual earlier alternatives. The teacher's 12 fallback deaths also show
that fallback is not unique to the AA policies.

The next learning question is whether a body-aware policy can choose earlier turns that
preserve an escape route while collecting food. AA deliberately suppresses own-body
observations, while the greedy teacher optimizes immediate food distance. A new
experiment must therefore distinguish the benefit of visible body state from the
quality of the teacher target; repeating this replay would not answer that question.

## Qualification and closure status

[Qualification closure](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-death-replay/qualification-complete.json>) passed in 3.986 seconds:
15 unique checks across 27 test executions. Two renderer-harness attempts are retained
as failures because their relative output paths prevented a required runtime witness;
the final absolute-output harness passed. Those failures were in the qualification
support path, not replay or product behavior, and no replay or product code changed.

The [resource reconciliation](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-death-replay/resource-reconciliation.json>) records five naturally completed serial jobs: 7.701 seconds for the four replays and
0.623 seconds for analysis and rendering, 8.324 seconds within the 60-second scientific
budget. Peak sampled RSS was 238,747,648 bytes (0.222 GiB), and minimum sampled available
memory was 31,385,616,384 bytes (29.230 GiB). Root directly viewed the first-death image
and recorded a [visual review](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-death-replay/visual-review.json>) PASS. The [independent artifact audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-death-replay/independent-review.json>) passed 8 of 8 checks. It verified frozen files, source and runtime provenance,
qualification history, serial receipts and budget, and decoded selected original NPZ
action, mask, valid, done, and death-cause payloads using the Python standard library.
Those raw rows independently account for every one of the 100 reported cases. The audit
did not independently rerun the simulator or inspect image pixels.
