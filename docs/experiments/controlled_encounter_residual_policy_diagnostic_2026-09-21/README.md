# Controlled-encounter residual-policy diagnostic — 2026-09-21

All 48 fatal final expanded-policy held games contain a first action outside a
viable stationary-H4 target. In 36 games, the first subsequently nonviable
choice occurs later, so the two findings should not be treated as one event or
as proof of a single cause. The diagnostic favors a bounded generalization or
experience-weighting test over a tie-only explanation. It did not train a
learner, run a new confirmation world, or establish a promotion result.

This diagnostic follows the completed
[global-teacher learning comparison](../controlled_encounter_global_teacher_learning_2026-09-21/README.md).
It reverified every fatal final-1536 held trajectory: 48 games and 168 native
frames, with no subsampling. The three lineages contributed 12, 16, and 20
fatal games respectively.

## Replayed decisions

For each saved game, the analysis located its first avoidable raw-target error,
its first nonviable action despite a viable alternative, and any earlier
allowed but non-minimum action. Every game has both the first avoidable error
and a nonviable choice; 12 share the same step, while 36 reach the first
nonviable choice later. Sixteen games have an earlier allowed non-minimum
action. That temporal association is not a branch-mechanism or causal result.

| Lineage | Fatal games / frames | First avoidable target errors | First nonviable choices | Earlier allowed non-minimum actions |
| --- | ---: | ---: | ---: | ---: |
| 2026098001 | 12 / 40 | 12 | 12 | 8 |
| 2026098002 | 16 / 64 | 16 | 16 | 4 |
| 2026098003 | 20 / 64 | 20 | 20 | 4 |
| **All lineages** | **48 / 168** | **48** | **48** | **16** |

The [residual-decision figure](residual-decisions.png) visualizes the saved
replays. The full 48-game record is available in the
[per-game CSV](per-game.csv).
It should be read with the earlier [global-teacher learning curves](../controlled_encounter_global_teacher_learning_2026-09-21/README.md), which report the matched learned-behavior comparison rather than this post-hoc diagnosis.

## Coverage and teacher-prefix context

Exact TRAIN coverage is zero for all 168 replayed frames, including every
first-error frame. The admitted training corpus contains 25,616 rows, but these
held inputs were intentionally excluded by the held/train split. Zero coverage
is therefore expected context; it does not establish that missing exact inputs
caused any fatal trajectory or identify a corrective target.

A separate post-hoc teacher-prefix join finds that 32 first errors remain on
the usual held teacher route and 16 are off that route. The split is 4/8,
12/4, and 16/4 same-route/off-route for lineages 2026098001, 2026098002, and
2026098003. This evidence argues against reducing the diagnosis to only
unusual-prefix coverage or tie breaking; it does not prove an alternative
mechanism.

The native replay drew on 100 EARLY collector-record occurrences and 52 HIDDEN
baseline occurrences. It made eight fresh stationary-H4 oracle calls, covering
16 fresh occurrences: eight unique calls and eight reuse occurrences. These
queries support the saved decision analysis; they are not learner updates or
policy inference.

## Completion, audit, and boundary

The probe and analysis both completed under guard with no failed numeric job.
They charged 30.495209499960765 seconds of the 240-second budget. Peak RSS was
4,031,266,816 bytes and minimum available memory was 30,285,742,080 bytes.
The closeout is `COMPLETE_AUDITED_DIAGNOSTIC`; the independent saved-data audit
returned `PASS_SAVED_DATA_CONSISTENT` after checking the 48 cases, 168 replayed
frames, counts, freezes, joins, and receipts.

No model inference, learner update, new confirmation-world gameplay, or
promotion evaluation occurred. Apex remains unchanged. The next question is a
prospective bounded generalization or experience-weighting learner change using
the admitted TRAIN data; this diagnostic does not claim that test has run or
that it will succeed.

## Source records

- [Frozen diagnostic intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-residual-policy-diagnostic/intent.json)
  — SHA-256 `29526705404e955963d62e70c8a9cc69af81d442f812f4435aaaf39af11bcfb8`
- [Completed native-replay probe](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-residual-policy-diagnostic/probe/report.json)
  — SHA-256 `bb00abca082206d47a0e3e7d7f8eb866ea2eafa8f8c62a2a958647a092526847`
- [Completed saved analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-residual-policy-diagnostic/analysis/report.json)
  — SHA-256 `259a3fc0914b31d6e18c1e4c0f4619530bed0f34168e4423cda1617c82b9e110`
- [Independent saved-data audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-residual-policy-diagnostic/independent-audit.json)
  — SHA-256 `34c00b33b376ccfa7bc3f6914013bfb917bbe334dd635bc4ceb1845f3200b39e`
- [Audited closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-residual-policy-diagnostic/closeout.json)
  — SHA-256 `da807f35b1b678b8ca49d587db94dc851d860f23489827dcae0d8822f13ee4d9`
- [Saved residual-decision figure](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-residual-policy-diagnostic/analysis/residual-decisions.png)
