# Solo food-response diagnostic — September 13, 2026

**Status: independently audited descriptive probe (PASS, 12 checks).** This CPU-only
diagnostic asks whether the six existing initial/30k/final solo networks change
their action values when one ambient pellet is relocated immediately left, straight,
or right of the hero in the same prepared initial Watch states. It ran no training,
executed none of the altered actions, selected no checkpoint, and cannot establish
a reward, optimizer, or behavioral cause.

## Frozen intervention

The probe uses frozen source `e083116182eb03888f5e82b7ea8f14cbb3f3bed1`, the
fresh worlds from the preceding [solo ambient replication](</Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_ambient_replication_2026-09-13/README.md>),
and its six native checkpoints. It captured the first prepared native Watch state
at frame 1, before movement. Seven worlds met all physical eligibility rules; world
2026091714 was retained but excluded because food was already on an adjacent target
(`target_on_food` and `food_within_one_cell`). No replacement world was sought.

For every eligible state, the probe copies the physical observation and moves only
the farthest ambient donor pellet to the adjacent left, straight, or right target
cell. It preserves food count, food slots, mass, source flags, all non-food arrays,
and the resolved action mask. The baseline plus three variants makes 28 conditions;
six checkpoints yield 168 raw-Q rows. All three normal actions remained mask-allowed.
The [protocol](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-response/protocol-draft.json>),
[execution freeze](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-response/execution-freeze.json>),
and [probe result](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-response/probe-v1/result.json>)
preserve the contracts and raw rows.

For food direction \(d \in \{L,S,R\}\), the reported directed normal margin is
\(Q_d - \max_{a \in \{L,S,R\} \setminus \{d\}} Q_a\). The targeted relocation
contrast is \(Q_d(\text{food at }d) - \operatorname{mean}_{p \ne d} Q_d(\text{food at }p)\).
The first asks whether the food-directed normal action leads the other normal
actions; the second asks whether the same action value changes when only food
placement changes. Raw Q is retained for all six actions, while action summaries
use the normal actions and the unchanged resolved mask.

## Descriptive response summary

Every raw-Q vector changed by a nonzero amount across the three physical food
placements. The independent audit also found all 42 condition-vector pairs at each
checkpoint distinct: six pairs among four conditions in each of seven worlds. This
establishes local input sensitivity in the measured states; it does not establish
global food awareness or a causal explanation for the earlier replication outcome.

`L/S/R baseline` counts normal greedy actions over seven unmodified states.
`Directed` is the fraction of 21 relocation variants whose normal (and, here,
masked) greedy action points toward the placed food. `Switch` is the fraction of
worlds whose action changes across placements.

| Seed | Checkpoint | L/S/R baseline | Directed normal / masked | Switch | Fixed left across all placements |
| ---: | --- | --- | ---: | ---: | ---: |
| 1701 | Initial | 0 / 0 / 7 | 1/3 / 1/3 | 0/7 | 0/7 |
| 1701 | 30k | 3 / 0 / 4 | 8/21 / 8/21 | 1/7 | 2/7 |
| 1701 | Final | 0 / 6 / 1 | 1/3 / 1/3 | 0/7 | 0/7 |
| 1702 | Initial | 1 / 0 / 6 | 1/3 / 1/3 | 0/7 | 1/7 |
| 1702 | 30k | 1 / 3 / 3 | 8/21 / 8/21 | 1/7 | 1/7 |
| 1702 | Final | 7 / 0 / 0 | 1/3 / 1/3 | 0/7 | 7/7 |

The final 1702 network selects left in every one of the 21 relocation variants,
whereas the final 1701 baseline is stable at 0/6/1 (left/straight/right). Neither
final checkpoint changes its chosen normal or masked action across the three food
placements. These selected-action facts coexist with nonzero raw-Q responses, so
they do not support a dead-input or global-food-blindness claim.

The [independent audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-response/audit-v4.json>)
passed 12 checks with no failures. It preserved 371 input hashes, independently
rebuilt all 28 eligible native observations, and re-reduced all 168 Q rows. The
excluded 2026091714 record has coherent listed reasons, but its raw physical state
was not archived, so the exclusion cannot be independently recomputed. The audit
closed naturally in 0.405 s; its [output closure](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-response/audit-v4-output-closure.json>)
records the result hash. Three earlier checker failures are retained as versioned
audit history (wrong protocol key, undefined local requirement, and wrong
baseline-food tuple indexes), rather than as a debugging chronology.

## Qualification, resources, and limits

The native probe qualification passed 6 tests in 0.08 s. The guarded probe itself
closed in 1.43 s with 385,548,288 B peak RSS and 28,850,978,816 B minimum available
memory. The [qualification result](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-response/qualification-v1/result.json>),
[outer closure](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-response/qualification-v1-outer-closure.json>),
[probe closure](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-response/probe-v1/output-closure.json>),
and [outer probe receipt](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-response/probe-v1-outer-closure.json>)
provide the execution evidence.

All initial states faced right and had dense ambient food. Moving a donor pellet
also removes it from its original location. The altered observations were never
stepped, so no trajectory, food collection, survival, or policy-quality conclusion
can follow. The probe does not claim a causal cause of the failed replication.

An ambient-only objective implementation candidate is now in an isolated worktree
at source revision `0657ef6`; no study using it has trained or evaluated. This
diagnostic does not change the preceding replication decision or authorize
promotion.
