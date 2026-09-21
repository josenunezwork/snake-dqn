# CS: H512 fixed-control confirmation

**Status: frozen, qualified, and running; results pending.** CS asks whether the completed CR H256
control cohort retains its food and survival package farther into fresh H512
gameplay. It performs evaluation only: no new learning, optimizer step, training
seed, fit inference, checkpoint substitution, or tournament inference.

## Fixed cohort and question

CS evaluates exactly the three completed CR `train_h256` final checkpoints at
mark 3584: seeds 2026095901, 2026095902, and 2026095903. They were chosen as an
outcome-informed family because all three passed CR's final 22-condition H384
absolute package. CR's primary H256-versus-H384 treatment comparison remains
false, as do its all-seed H384 treatment package, CO's relative loss result,
CP's all-three H384 confirmation, and the earlier CH calibration. CS does not
rescue, rerun, or reinterpret any of them.

The question is duration retention for this fixed cohort: whether its food and
survival continue on longer fresh worlds. It does not ask whether a longer rollout
cap trains a better policy, whether a prior loss choice caused the cohort's
behavior, or whether any policy should replace Apex. Apex remains incumbent; its
larger historical training dose is a confound rather than evidence of inherent
superiority.

## Fresh evaluation and criteria

Each fixed policy runs once to H512 on 32 fresh worlds, seeds
2026109200--2026109231. Four headings and three reachable food placements per
world produce 384 lanes. H64, H128, H256, H384, and H512 are exact prefixes of
one H512 trace, and the CS native extension admits only that additional horizon.
The smoke world is separate. Teacher and RandomSafe are descriptive anchors.

Every fixed policy must meet all 29 conditions, with 60 pose-cell subchecks. CS
keeps the 22-condition H384 package, including both inherited late windows, and
adds the analogous H512 food, 12-cell, survival-time, endpoint, and paired-food
conditions. It also requires late food in frames 385--512 to reach at least 75%
of Teacher and mean H512 ambient food of at least 64. The food floor is a
one-pellet-per-eight-frames absolute floor against a weak teacher reference.
The 32 paired world means use df 31 confidence intervals. There is no lane or
seed pooling, best-mark choice, or checkpoint substitution; all three policies
must pass.

## Operations and evidence boundary

The scientific budget is 1,080 seconds: Teacher and RandomSafe at 80 seconds
each, the three fixed-policy evaluations at 240 seconds each, and analysis at
120 seconds, for 1,000 seconds of scheduled job caps. Qualification was capped at
180 seconds: tests 60 and smoke 120. All 32 tests passed and the H384 raw14 prefix
matched exactly; qualification used 9.42 seconds. It makes zero optimizer updates and records
10,752 discarded qualification gameplay lane-frames only.

Jobs are serialized on the Mac with two CPU and one interop thread, a 4 GiB
process-tree RSS cap, a 12 GiB available-memory floor, an 8 GiB MPS-driver cap,
and heartbeat monitoring. Freeze requires CR's audited closeout, authenticates
its fixed training lineage, and requires CS worlds, lanes, and native seeds to
be disjoint from prior evidence.

CS has no training curve because it performs no optimization. CR's saved curves
are historical evidence for the chosen cohort, not CS evidence or a decision
substitute.

- [Frozen CS design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-h512-control-confirmation/design.md)
- [CS protocol](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-h512-control-confirmation/protocol.py)
- [CR audited result](../solo_food_pqn_h384_training_2026-09-20/README.md)
- [CR historical training curves](../solo_food_pqn_h384_training_2026-09-20/training-curves.png)
- [CR independent audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-h384-training/independent-audit.json)
