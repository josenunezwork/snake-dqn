# Controlled-encounter horizon diagnostic — 2026-09-21

All 79 conditional familiar-world witness games contain finite-horizon regret
despite their saved actions being native H4-optimal members. An earlier-food
tie target excludes all 87 regretted decisions, and its allowed actions are
remaining-horizon optimal at all 221 queried native nodes. This makes the early
tie a target candidate only; full-data admission and actual training are still
required before it can support a learned-policy claim.

## Conditional native-oracle result

The diagnostic replayed the 79 train-only aggregated-arm games selected by the
[target-sufficiency diagnostic](../controlled_encounter_target_diagnostic_2026-09-21/README.md): every input was covered, every saved action was a teacher
member, and each game collected less food than the executed teacher. It queried
221 unique native nodes, representing 316 per-lineage decision occurrences.
It reverified the existing 79 saved-action trajectories and computed new native
H4 and remaining-horizon oracle branches at those nodes. No policy inference,
learner update, new held-world replay, or new learned-policy gameplay ran.

| Seed | Witness games | H4-member decisions | Regretted decisions | H4 member but remaining-horizon nonmember |
| --- | ---: | ---: | ---: | ---: |
| 2026097501 | 17 | 68 | 17 | 17 |
| 2026097502 | 39 | 156 | 47 | 47 |
| 2026097503 | 23 | 92 | 23 | 23 |
| **Total** | **79** | **316** | **87** | **87** |

The first-column game count and later decision counts intentionally differ:
some saved games contain more than one regret. H4 labels cover all 316 saved
decision occurrences, while only 229 are remaining-horizon members. Regret
appears at steps 1, 2, and 3 (16, 32, and 39 decisions); no step-0 regret was
observed.

The earlier-food tie target excludes each of the 87 regretted choices. At every
one of the 221 queried nodes, every action that it allows is remaining-horizon
optimal. Across A4, A4-early, and remaining-horizon target groups, the audit
found 67 exact input groups, 54 repeated groups, and zero disagreements,
compatible disagreements, or conflicts in all three target families.

## A concrete saved timing witness

For seed 2026097501, cross-right case 32, the saved learner collected one food
contact while the executed teacher collected two. At step 3, the saved action
was `0`, which belonged to the stationary H4 set `[0, 1, 2]`. The earlier-food
tie and remaining-horizon sets were both `[1]`; choosing `0` had one remaining
food contact of regret. This is a saved native-gameplay timing example, not a
counterfactual learned-policy result.

The prior [paired continuation](../controlled_encounter_paired_2026-09-21/README.md)
contains the existing learning curves and gameplay. This diagnostic adds no
learner result and does not claim that the early target fixes every failure,
generalizes to unseen worlds, or removes the teacher's future-information
dependence.

## Execution record and boundary

The audited closeout is `COMPLETE_AUDITED` (SHA-256
`5c8a473afb8f738eac19da8aba319cc133d9b901d25eee431bff49c9bf54fa34`).
It records three receipts: two completed jobs and one preserved saved-checker
failure, repaired after its masked-CI action-key issue. Charged execution was
52.41458162490744 seconds of the 270-second budget, with peak RSS
1,424,900,096 bytes and minimum available memory 32,549,355,520 bytes.

The early target remains a hypothesis. It needs full-data admission and actual
training plus greedy gameplay before any performance or promotion conclusion.

## Source records

- [Frozen horizon-diagnostic intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-horizon-diagnostic/intent.json)
- [Native oracle probe report](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-horizon-diagnostic/probe/report.json)
- [Saved native-node records](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-horizon-diagnostic/probe/nodes.json)
- [Audited analysis-v2 report](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-horizon-diagnostic/analysis-v2/report.json)
- [Audited closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-horizon-diagnostic/closeout.json)
- [Probe receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-horizon-diagnostic/supervisor-runs/probe/receipt.json), [preserved checker-failure receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-horizon-diagnostic/supervisor-runs/analysis/receipt.json), and [completed analysis-v2 receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-horizon-diagnostic/supervisor-runs/analysis-v2/receipt.json)
