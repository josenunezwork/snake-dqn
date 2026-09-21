# Controlled-encounter target-sufficiency diagnostic — 2026-09-21

The preceding paired continuation improved familiar-world food with the
aggregated dataset, but coverage and teacher-set membership did not reproduce
the archived executed teacher's food on every fully covered trajectory. This
read-only diagnosis establishes that finite target membership is insufficient
for that reproduction on the observed cases; it does not establish the cause
of a deficit or an effective repair.

## Reconstructed saved gameplay

The diagnostic replayed the saved actions from both paired arms and all three
seeds: 2,304 training-world games (`2 arms × 3 seeds × 384 games`) and 9,208
native frames. It reconstructed exact transformed observations and resolved
masks while checking native frame facts and outcomes. It made zero policy
inferences, optimizer updates, or new held-game replays.

The table separates aggregate-arm game-level membership witnesses from every
aggregate-arm food-deficit game. A fully covered/member game has every observed
input in the 2,576-row aggregated dataset and every saved selected action in
the associated teacher intersection.

| Seed | Covered and teacher-member frames | Fully covered/member games | Food deficits in that subset | All aggregate-arm food-deficit games |
| --- | ---: | ---: | ---: | ---: |
| 2026097501 | 1,368 / 1,536 | 260 / 384 | 17 / 260 | 57 / 384 |
| 2026097502 | 1,392 / 1,536 | 288 / 384 | 39 / 288 | 55 / 384 |
| 2026097503 | 1,368 / 1,536 | 256 / 384 | 23 / 256 | 41 / 384 |
| **Total** | **4,128 / 4,608** | **804 / 1,152** | **79 / 804** | **153 / 1,152** |

Every one of the 4,128 aggregate-covered frames selected a teacher-member
action. Nevertheless, 79 of the 804 fully covered/member games collected less
food than the executed teacher for the same case. Those 79 are a strict subset
of the 153 aggregate-arm food-deficit games. The remaining 74 deficits are not
therefore assigned to one cause by this diagnostic.

## What the witness supports

The [paired continuation](../controlled_encounter_paired_2026-09-21/README.md)
contains the learning curves and gameplay that showed the aggregate arm's
familiar-world food increase. This report adds no learner, checkpoint, or new
gameplay evidence. Its fully covered/member witnesses show that matching the
recorded finite teacher action sets need not match the executed teacher's food
score, even where the trajectory leaves neither the aggregate dataset nor its
teacher sets.

That is not a proof that all remaining deficits have the same mechanism, that
H4 is the cause, or that a different target will improve learning. The next
prospective diagnostic would evaluate stationary H4, remaining-horizon, and
earlier-food-tie targets on these same 79 witnesses. It has not been launched
here and its result is not implied by this audit.

## Execution record

The audited closeout binds two completed receipts. The work used 44.276042916986626
seconds of its 150-second budget, reached peak RSS 2,511,224,832 bytes, and
recorded minimum available memory of 31,598,002,176 bytes. It is not promotion
eligible.

## Source records

- [Frozen target-diagnostic intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-target-diagnostic/intent.json)
- [Saved native replay report](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-target-diagnostic/replay/report.json)
- [Saved witness-game records](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-target-diagnostic/replay/games.json)
- [Admission report](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-target-diagnostic/admission/report.json)
- [Audited closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-target-diagnostic/closeout.json)
- [Replay receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-target-diagnostic/supervisor-runs/replay/receipt.json) and [admission receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-target-diagnostic/supervisor-runs/admission/receipt.json)
