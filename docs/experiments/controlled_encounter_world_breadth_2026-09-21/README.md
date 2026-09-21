# Controlled-encounter world-breadth collection — 2026-09-21

This completed collection admitted a 7,184-row broader earlier-food teacher
corpus from 48 additional training worlds. All 1,152 new four-frame teacher
games survived and collected 1,464 food contacts. It is a data and
teacher-label result, not a learner result: it made zero learner updates and
zero policy inferences, did not change the incumbent, and is not promotion
eligible.

The collection extends the [early-food target admission](../controlled_encounter_early_target_2026-09-21/README.md) after its
[finite-horizon diagnosis](../controlled_encounter_horizon_diagnostic_2026-09-21/README.md).
The guarded closeout is `COMPLETE_ADMITTED`. The later learning comparison
remains a separate study and is not implied by this admission.

## Frozen data scope

The three shards each contain 16 new worlds, giving 48 worlds and 1,152 new
four-frame cases. The existing 2,576 training rows are retained as an exact
prefix, including their observations, non-label arrays, and teacher masks. The
new shards add 4,608 rows, yielding a 7,184-row broad training corpus.

| Corpus | Rows | World source | Role |
| --- | ---: | --- | --- |
| Existing early-target training set | 2,576 | Existing train worlds | Exact preserved prefix and small-data arm |
| New breadth shards | 4,608 | 48 predeclared worlds, 16 per shard | Additional broad-data rows only |
| Broad training corpus | 7,184 | Existing prefix plus all three shards | Candidate data for a later comparison |

The world and food RNG varies across the new worlds while the task retains the
same six encounter geometries, four headings, and interior anchor strata. It
does not claim broader hazard geometry. Every actual step is labelled by the
native stationary early-H4 teacher: survive four frames first, then maximize
total food, then prefer earlier food with binary timing weights 8/4/2/1, and
finally use the minimum raw allowed first action.

## Admission boundary

The frozen admission requires all new teacher games to survive, each shard to
be complete and balanced, nonempty raw-label intersections over the full
union, normal actions 0–2 to retain support, no exact overlap between training
and held inputs, and no change to the old prefix. Any scientific failure must
be preserved and rejected; rows may not be dropped and thresholds may not be
changed to obtain admission.

The existing 768 held rows and worlds `2026097417`–`2026097424` are overlap
checks only. This collection does not request new held oracle data. The eight
confirmation worlds `2026097801`–`2026097808` remain reserved.

The provenance check is recorded separately in
[early-audit-provenance-check.json](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-world-breadth/early-audit-provenance-check.json).
It binds the outer world-breadth record to the earlier audit inputs before any
future learning comparison can rely on the assembled corpus.

## Completed collection and admission

Each completed shard contains 384 games, 1,536 native frame facts, and 1,536
new state rows. All shard gates passed, including four native frames per case,
nonempty raw early targets and within-shard intersections, normal-action
support, balanced geometry and headings, and survival of every teacher game.

| Shard | New worlds | Games / survivors | Food contacts | New rows |
| --- | ---: | ---: | ---: | ---: |
| 0 | 16 | 384 / 384 | 484 | 1,536 |
| 1 | 16 | 384 / 384 | 488 | 1,536 |
| 2 | 16 | 384 / 384 | 492 | 1,536 |
| **Total** | **48** | **1,152 / 1,152** | **1,464** | **4,608** |

Independent admission found 1,965 unique inputs in the assembled training
corpus and 1,782 alias groups. There were zero empty full-train intersections,
alias conflicts, raw-action removals, or exact train/held input overlaps. The
old 2,576-row prefix and its effective labels are byte-identical, while the
new rows complete the admitted 7,184-row broad corpus. The provenance check
passed and the admission status is `ADMITTED_FOR_PROSPECTIVE_LEARNING`.

## Separate prospective learning study

A [separate learning intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-world-breadth-learning/intent.json) was frozen after admission. It uses fresh
model seeds `2026098001`–`2026098003`, 512 equal updates of 128 rows, and the
same actual-greedy fixed-held evaluation plus separate fit measurements for the
2,576-row small and 7,184-row broad arms. It cannot aggregate more data during
that comparison.

Learning outcomes will be reported separately. Corpus admission does not imply
a held-world benefit or alter the promotion decision.

## Execution record and boundary

Four guarded attempts completed naturally with no failed attempt: three serial
collection shards and the admission check. They charged 442.9899654168985
seconds of the 900-second budget. Peak RSS was 2,058,993,664 bytes and minimum
available memory was 31,444,959,232 bytes.

The closeout records zero learner updates and zero model inferences. Although
the data corpus is admitted for prospective learning, no policy has yet been
trained or evaluated by this collection, so it supplies no learned-policy,
held-performance, or promotion conclusion.

## Frozen source records

- [World-breadth intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-world-breadth/intent.json)
- [Input freeze](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-world-breadth/input-freeze-initial.json)
- [Outer early-audit provenance check](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-world-breadth/early-audit-provenance-check.json)
- [Shard 0 report](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-world-breadth/collection/shard0/report.json), [shard 1 report](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-world-breadth/collection/shard1/report.json), and [shard 2 report](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-world-breadth/collection/shard2/report.json)
- [Independent admission report](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-world-breadth/admission/report.json)
- [Audited closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-world-breadth/closeout.json)
- [Shard 0 receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-world-breadth/supervisor-runs/collect-0/receipt.json), [shard 1 receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-world-breadth/supervisor-runs/collect-1/receipt.json), [shard 2 receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-world-breadth/supervisor-runs/collect-2/receipt.json), and [admission receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-world-breadth/supervisor-runs/admission/receipt.json)
