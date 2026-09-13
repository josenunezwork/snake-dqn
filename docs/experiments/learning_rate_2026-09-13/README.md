# Learning-rate retention study — September 13, 2026

Status: **NOT_ADVANCED**. Lowering fixed Adam learning rate from 5e-4 to 1e-4 passed every required comparison for the first training seed but failed five of six comparisons for the fresh second seed. The treatment lost strongly to both its 150k control and shared initial checkpoint in both second-seed mixes, and its random-safe retention from 50k to 150k was negative. No checkpoint advanced, no incumbent comparison ran, and no model was promoted.

## Question and frozen method

The study asked whether fixed 1e-4 Adam learning rate, rather than 5e-4, preserves or improves unassisted full-horizon mass from 50k to 150k useful hero transitions. Each seed paired the two learning rates from the same initial tensors, all non-LR configuration, world derivation, fixed canonical random-safe opponents, and native random streams until trajectories diverged.

Training used source a19b13ca3d7d3983fe7c7f083f3e1f395084a220 with MPS, corrected-v3/raster31v3, native PQNTrainer, beta zero, native east starts, E16/S6/T16, one exact padded SGD epoch at batch 256, and epsilon 1.0 to .02 over 30k useful hero transitions. Each arm saved 50k midpoint and 150k final checkpoints. Evaluation was unassisted CPU Watch play on the same four worlds, 2026091441–2026091444, for both random-safe and greedy-food mixes.

The predeclared rule required treatment 150k to strictly exceed control 150k and the shared initial checkpoint, and to be at least treatment 50k, in both mixes and both seeds. Food and survival are secondary outcomes without a post-hoc threshold. The four worlds per mix are paired diagnostics, not a population confidence interval.

## Full-horizon mass

Every value is the mean over four fixed worlds. All five checkpoint roles are shown to make the 50k-to-150k retention comparison visible.

| Seed | Opponents | Initial | Control 50k | Control 150k | Treatment 50k | Treatment 150k | T150 - C150 | T150 - T50 | T150 - Initial | Result |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026091341 | Random-safe | 1.45940 | 2.06315 | 2.60905 | 2.80150 | 3.92005 | +1.31100 | +1.11855 | +2.46065 | Pass |
| 2026091341 | Greedy-food | 0.73920 | 1.65785 | 1.18200 | 1.78150 | 2.75375 | +1.57175 | +0.97225 | +2.01455 | Pass |
| 2026091342 | Random-safe | 6.93155 | 10.20410 | 15.58865 | 2.58600 | 2.05110 | -13.53755 | -0.53490 | -4.88045 | Fail |
| 2026091342 | Greedy-food | 1.89665 | 2.62105 | 6.17740 | 0.96560 | 1.12510 | -5.05230 | +0.15950 | -0.77155 | Fail |

The first seed’s complete pass does not replicate. The second seed fails against control and initial in both mixes; random-safe also loses from treatment 50k to 150k. That produces the frozen NOT_ADVANCED outcome.

![Fixed-world mass by checkpoint role](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/learning-rate/plots/v2/learning-rate-fixed-world-mass.png>)

## Food and survival diagnostics

Food and survival are descriptive. The entries below use the same role order as the mass table: initial / control 50k / control 150k / treatment 50k / treatment 150k.

| Seed | Opponents | Mean food by role | Mean survival by role | Treatment final minus control final: food / survival |
| ---: | --- | --- | --- | --- |
| 2026091341 | Random-safe | 0.50 / 6.00 / 27.75 / 45.25 / 142.50 | 0.97970 / 0.84390 / 0.84885 / 0.34540 / 0.69850 | +114.75 / -0.15035 |
| 2026091341 | Greedy-food | 1.25 / 7.75 / 12.75 / 26.25 / 61.00 | 0.48620 / 0.53660 / 0.34440 / 0.20830 / 0.35655 | +48.25 / +0.01215 |
| 2026091342 | Random-safe | 49.50 / 44.25 / 36.00 / 82.75 / 19.75 | 0.39380 / 0.44255 / 0.74705 / 0.57535 / 0.33815 | -16.25 / -0.40890 |
| 2026091342 | Greedy-food | 26.00 / 26.00 / 12.25 / 34.50 / 16.00 | 0.25610 / 0.20355 / 0.73575 / 0.23745 / 0.24000 | +3.75 / -0.49575 |

Both ordinary 5e-4 controls exceeded their own initial checkpoint in both mixes on these fixed worlds. This is a selected four-world, two-seed descriptive result that warrants a separate generalization check; it does not identify why control improved, establish a new incumbent, or authorize promotion.

## Qualification, provenance, and execution

The original evaluator qualification passed eight tests in 1.30 seconds but was review-rejected before any evaluation world because its fixture relabeled native CPU/E1/T1 descriptors as MPS/E16/T16. The archived v1 evaluator remains evidence of an inadequate fixture, not a valid evaluator qualification.

The revised evaluator reconstructs expected native descriptors from the declared configuration, verifies LR-aware optimizer groups and nested provenance, and rechecks checkpoint hashes around loading and calls. It passed eight focused tests in 1.54 seconds, including wrong-LR and relabeled-config rejections. No v1 evaluation world was run.

All 12 full training checkpoints reloaded, and each seed's no-world preflight validated all five checkpoint roles before its ten scored calls. The final independent standard-library audit passed 123 checks with no failures or pending items. It confirmed all 20 calls, 80 unique finite rows, exact coverage, matching pair initialization, frozen source/driver closure, serial execution, and clean supervisor receipts.

The MPS 50k stability gate first completed 50,206 useful transitions in 92.211 seconds and qualified only the declared MPS learning arms. The four 150k MPS arms then completed in 261.633–274.953 seconds, with 600,643 useful hero transitions in total. Earlier short matched CPU/MPS profiling measured 2.361x and 2.534x MPS useful-step advantages, but it does not establish a long-run CPU comparison for this campaign.

The closed host log has 551 samples: CPU 0.0–43.8%, available memory at least 26.62 GiB, swap 6.575–6.591 GiB, and GPU 0–77%. No sample exceeded 80% CPU or GPU utilization. These are host aggregates, not learner-only attribution; the raw thermal report recorded no thermal or performance warning.

## Evidence and limits

The frozen protocol SHA-256 is 3e7ec845fb4531707c6a869f2f0046c7fe4ec6133cb24e853b552472151a6e03. The training driver SHA-256 is 60fc8edcde6ee0971305e1b889753a6fc982b37d3ff0b78591126d3ecd534455; the final evaluator SHA-256 is a29318a752e3478d2c2cf86c7d63d3244269568def3bd6b2d19ee73e119a024f.

- [Frozen protocol](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/learning-rate/protocol.json>) and [research rationale](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/learning-rate/research-notes.md>)
- [Final independent audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/learning-rate/audit.md>) and [machine-readable audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/learning-rate/audit.json>)
- [Seed 2026091341 comparison](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/learning-rate/seed1341/descriptive-comparison.json>) and [seed 2026091342 comparison](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/learning-rate/seed1342/descriptive-comparison.json>)
- [V1 review-rejected evaluator archive](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/learning-rate/evaluate-v1-qualified-but-review-rejected.py>), [V2 qualification notes](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/learning-rate/qualification-notes.md>), and [V2 evaluator](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/learning-rate/evaluate.py>)
- [Final plot SVG](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/learning-rate/plots/v2/learning-rate-fixed-world-mass.svg>) and [closed host resource log](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/learning-rate/resources.jsonl>)

This screen does not determine the cause of the first/second-seed reversal or reject every smaller learning rate. Any next question needs a separate frozen protocol, fresh evidence, and the same distinction between operational execution and policy quality.
