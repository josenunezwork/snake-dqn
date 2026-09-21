# Controlled-encounter train-trajectory aggregation — 2026-09-21

This feasibility study admits an expanded **training** dataset for a separately
declared continuation. It does not report a learned-policy improvement, run
scientific training, or make a promotion claim. The [coverage diagnostic](../controlled_encounter_coverage_diagnostic_2026-09-21/README.md) found that the completed food-label policies frequently visited familiar-world states outside the original teacher dataset; this study asks whether the existing food-only H4 teacher can label those train-only prefixes consistently.

## Dataset and admission result

The collection replayed and verified 4,600 saved policy frames from all three
completed greedy lineages on their training worlds. It retained all 1,536
original training rows byte-identically, then added 1,040 teacher-labelled
native-prefix rows, for 2,576 training rows and 709 unique exact train inputs.
The original held NPZ was preserved byte-identically and was used only for the
overlap audit; no held trajectory collection or optimizer access occurred.

| Check | Recorded result |
| --- | --- |
| Native frames replayed and verified | 4,600 / 4,600 |
| Original / new / total training rows | 1,536 / 1,040 / 2,576 |
| Unique exact train inputs | 709 |
| Empty teacher targets | 0 |
| Conflicting exact-input teacher intersections | 0 |
| Exact train/held input overlaps | 0 |
| Normal-action support | passed |
| Original training prefix and held split | unchanged |

Every feasibility gate passed, so the dataset is `ADMITTED_FOR_PROSPECTIVE_STUDY`.
That admission is a data-quality result only. The closeout records zero training
updates, and the admission report leaves scientific training disallowed until a
separate protocol and budget are frozen.

## Teacher boundary

The teacher labels are the food-only H4 action sets: all initial normal actions
that reach the maximum realized food contacts among complete surviving H4
branches, without clearance tie-breaking. The teacher observes full state and
realized future. Passing the finite exact-input and teacher-set checks therefore
shows empirical consistency for this collected dataset; it does not prove that
the targets are globally observable from the learner input.

The [food-label study](../controlled_encounter_food_labels_2026-09-21/README.md)
is the source of the completed lineages and original data. Its saved learning
outcome is unchanged by this collection.

## Execution boundary and next prospect

The audited closeout records two receipt hashes, one for collection and one for
admission. It completed in 116.676 seconds of the 630-second budget, with peak
RSS 1,126,301,696 bytes and minimum available memory 31,811,174,400 bytes.

The next prospect is not launched or evaluated here: freeze a matched,
equal-update original-data control against an aggregated-data continuation
across the three existing lineages, restore the full optimizer state, and retain
the original held-gameplay criteria. Only that separately declared comparison
could test whether the admitted data changes learned behavior.

## Source records

- [Frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-aggregation/intent.json)
- [Collection report](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-aggregation/collect/report.json)
- [Admission report](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-aggregation/admission/report.json)
- [Audited closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-aggregation/closeout.json)
