# Reset-heading directional-exposure study — September 13, 2026

Status: **INCONSISTENT_NOT_ADVANCED**. Randomizing the learner's initial heading produced a large gain in the first matched training seed and a loss in the second. It failed the frozen repeatability rule. No checkpoint was promoted, no reserved confirmation world was used, and the Apex incumbent remains unchanged.

## Question and intervention

This study asked whether broader initial cardinal-heading exposure improves repeatable growth learning for corrected PQN. The control starts all snakes facing east. The treatment changes only hero slot 0: at episode zero and each selected derived reset, it samples one of the four cardinal headings from a run/env/episode/slot-keyed seed. Fixed opponents remain east-facing. The reward coefficient remains `beta=0`; the simulator, network, optimizer, action space, and unassisted evaluation are unchanged.

This is a start-state-distribution intervention, not symmetry augmentation. The `raster31v3` observation is heading-canonical locally, but the rectangular arena and absolute world-position scalars preserve directional context. The protocol therefore does not claim a symmetry or predict improvement.

The implementation is an artifact-only experimental trainer subclass. It does not change a production environment, default configuration, serving path, or champion.

## Frozen design

The active protocol was declared before numerical runs on source `a19b13ca3d7d3983fe7c7f083f3e1f395084a220`:

- Two matched training seeds: `2026091321` and `2026091322`. Within each pair, control and treatment share the exact initial network tensors and unchanged world, food, fixed-opponent, action, and SGD streams until trajectories diverge.
- CPU recipe: two threads, E16/S6/T16, one hero against five fixed `random_safe` opponents, corrected-v3/raster31v3, Watch pre-move decisions, derived per-environment autoreset, one exact shuffled SGD epoch, padded minibatches of 256, gamma `.997`, lambda `.65`, and epsilon `1.0 -> .02` over 30,000 useful hero steps.
- Each arm targets 50,000 useful hero transitions. It finished with 50,002/50,163 steps for seed 2026091321 control/treatment and 50,087/50,179 for seed 2026091322 control/treatment: **200,431** in total.
- Unassisted evaluation compares initial, control-final, and treatment-final checkpoints on validation worlds `2026091361`–`2026091364`, 5,000 frames each, against `random_safe` and `greedy_food`. That is six calls and 24 world rows per training seed, or **12 calls / 48 rows / 240,000 scored world frames** overall.

The predeclared descriptive advance rule required a positive treatment-minus-control mean full-horizon mass in both opponent mixes for **both** independent training seeds, with finite evidence and no reward-free minimum-mass-survival diagnostic. A pass would authorize held-out confirmation and more training-seed replication only; it would not authorize promotion. The four worlds in each mix are paired diagnostics, not a population confidence interval.

## Results

Each cell is the mean across the same four validation worlds. Mass is full-horizon post-alive logical mass with dead frames scored as zero; food and survival are shown as diagnostics. Delta is treatment minus control.

| Training seed | Opponents | Control mass | Treatment mass | Delta mass | Delta food | Delta survival |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2026091321 | Random-safe | 0.80640 | 8.09180 | +7.28540 | +18.50 | +0.10425 |
| 2026091321 | Greedy-food | 1.27880 | 9.26645 | +7.98765 | +13.00 | -0.03335 |
| 2026091322 | Random-safe | 2.00820 | 1.64505 | -0.36315 | +9.75 | -0.31605 |
| 2026091322 | Greedy-food | 2.21420 | 1.82855 | -0.38565 | +11.50 | -0.29015 |

The first pair satisfies the descriptive mass condition, but the fresh pair is negative in both mixes and has lower survival. The second outcome therefore fails the two-training-seed criterion. The score means are influenced by a small four-world diagnostic set, including large individual-world gains and losses; they are not enough to establish a global conclusion about heading exposure. The result only shows that this isolated intervention did not replicate under the declared screen.

![Paired mean mass by training seed and opponent mix](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/direction-coverage/results.png>)

## Integrity and resource evidence

The independent standard-library audit passed. All four arms completed naturally, had finite telemetry, exact eligible/drawn/unique/useful coverage, matching pair initialization, and clean source/driver closure. Initial network state hashes were `3b0d295f78470f3cbbf701cd32cdb8ea267fc7e76adb0698756f2eace607a375` for seed 2026091321 and `24accb9ae3be8f578c11880cd7c57d38b0120f7f1de1b3e40b8b662804963cdf` for seed 2026091322. The frozen runner SHA-256 is `56c1f345618aeedcaefc314461938909078ba0497bf85585e54d235c1a63d99a`; the frozen protocol SHA-256 is `7e6901efefa6f075a4a82c6353a6693f3152502330c8d8771cdca73614f70a8d`.

Nine focused qualification tests passed in 1.13 seconds. Two earlier qualification attempts each exposed a test-fixture boundary error; their failed receipts are retained, the test fixtures were corrected, and the frozen runner did not change. They cover three-update control/base exact parity for actions, rewards, targets, loss, and weights; initial heading visibility; selective reset and RNG/world isolation; mask consistency; and reset preconditions. The two explicit 512-step smokes are marked `SMOKE_NOT_EVALUABLE`; they only verify bounded execution and inference reload.

The serialized CPU experiment kept within its stated limits. Across 448 host samples, CPU use was 4.4–46.8%, available memory 27.57–30.22 GiB, and no thermal or performance warning was recorded. These host-wide measurements cannot attribute GPU activity or other load to the learner. Supervisor receipts also record natural child exit, no source/driver drift, per-run RSS, and the available-memory floor.

## Evidence and limits

Training and evaluation outputs are retained in create-only directories outside the repository; the audit and plots are reproducible reductions:

- [Frozen protocol](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/direction-coverage/protocol-v2.json>) and [runner](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/direction-coverage/runner.py>)
- [Independent audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/direction-coverage/audit.md>) and [machine-readable audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/direction-coverage/audit.json>)
- [Evaluation plan and calls, seed 2026091321](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/direction-coverage/seed1321/evaluation/plan.json>) and [seed 2026091322](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/direction-coverage/seed1322/evaluation/plan.json>)
- [Qualification test driver](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/direction-coverage/test_runner.py>) and [closed host resource log](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/direction-coverage/resources.jsonl>)

No additional training seed beyond the two predeclared seeds, no reserved confirmation world, and no cross-stack tournament was run for this study. Its outcome does not rule out every possible use of directional context; it rules out treating this particular hero-only reset-heading intervention as a repeatable improvement on this screen.

