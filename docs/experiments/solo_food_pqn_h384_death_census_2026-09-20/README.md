# CQ: recorded deaths in the completed H384 study

Status: complete and independently audited.

This posthoc census reads the five saved CP gameplay archives. It adds no games,
policy inference, training, or checkpoint selection. The unchanged CP all-three
criterion still fails because seed 5801 ends with 344/384 survivors.

| Fixed CO half-MSE seed | Deaths through H256 | Deaths in frames 257–384 | H384 survivors | Recorded cause of every death |
| --- | ---: | ---: | ---: | --- |
| 2026095801 | 10 | 30 | 344/384 | Self-collision |
| 2026095802 | 12 | 14 | 358/384 | Self-collision |
| 2026095803 | 7 | 9 | 368/384 | Self-collision |

The teacher has 97 self-collision deaths, 49 after frame 256. RandomSafe has 28
self-collision deaths and one wall death; 23 deaths occur after frame 256.
The reducer verifies one death per lane, valid terminal transitions, no revival,
and the survivor counts at every CP prefix. Its source death-code mapping is
authenticated against the pinned simulator.

The independent standard-library audit verified 142 file hashes and reconciled
the saved death rows, cause/window counts, CP endpoints, input closure, and
supervisor receipt. The single guarded analysis took 0.206 seconds of a declared
60-second budget. The job was too short for the supervisor to capture a useful
process peak; its 147,456-byte sampled RSS is not presented as peak memory.
The supervisor recorded at least 37.9 GB available. Both shared compute locks,
two-thread limits, and the existing memory guards remained in effect.

These records identify the terminal event. They do not prove that a collision
was inevitable earlier, that the native advisory mask was empty, or that a
different action would have rescued it. They support the next bounded question:
whether equal-dose continuation with H384 rather than H256 episode caps improves
late survival while retaining food. That comparison changes the reset and
truncation distribution as well as exposure to later states.

- [CP behavior and criteria](../solo_food_pqn_h384_confirmation_2026-09-20/README.md)
- [Frozen census intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-h384-death-census/intent.json)
- [Complete census with every death row](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-h384-death-census/analysis/analysis.json)
