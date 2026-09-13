# Scripted exploration advice study — September 13, 2026

Status: **NOT_ADVANCED**. Replacing half of actual hero epsilon-exploration choices with a deterministic food-seeking suggestion did not meet the frozen two-seed mass criterion. The treatment lost to its matched control in the random-safe mix for both seeds. No checkpoint advanced, no reserved confirmation world was used, and the Apex incumbent remains unchanged.

## Question and intervention

At each eligible hero decision, the native sampler first made its normal epsilon-exploration draw. A shadow replay over deep-copied real per-environment generators identified actual exploration events without consuming extra native randomness. For treatment only, a keyed deterministic coin selected half of those events for the canonical scripted:greedy_food suggestion. An illegal suggestion retained the already-sampled native action. Non-exploration hero choices and every fixed-opponent choice remained native.

The teacher is a deterministic, normal-speed, one-step food-distance anchor. It has no long-horizon anti-trapping strategy and is not an expert demonstration. There is no supervised imitation loss: executed behavior alone changed, while the replay-free Q(lambda) update, reward, network, optimizer, and unassisted evaluation were unchanged.

This artifact-only intervention left native east-facing resets, the production environment, defaults, serving path, and champion unchanged. The checkpoint sidecar labels the behavior contract NON_PROMOTABLE_NATIVE_BEHAVIOR_CONTRACT_MISMATCH. This records study eligibility, not a native enforcement mechanism: the weights were explicitly loaded for unassisted diagnostic evaluation. They are not qualified for promotion or native training continuation under this extra behavior contract.

## Frozen method

The protocol was declared before numerical runs on clean source a19b13ca3d7d3983fe7c7f083f3e1f395084a220:

- Two matched training seeds, 2026091331 and 2026091332. Within a pair the arms shared exact initialization, configuration, world derivation, fixed opponents, and native RNG streams until trajectories diverged.
- CPU recipe: two threads; E16/S6/T16; one hero against five fixed random_safe opponents; corrected-v3/raster31v3; Watch pre-move decisions; derived per-environment autoreset; beta zero; one shuffled SGD epoch; padded minibatches of 256; gamma .997; lambda .65; epsilon 1.0 to .02 over 30,000 useful hero transitions.
- The four arms completed 50,160/50,155 useful hero transitions for seed 2026091331 control/treatment and 50,048/50,231 for seed 2026091332 control/treatment: **200,594** total.
- Every evaluation was unassisted native Watch play: initial, control-final, and treatment-final checkpoints; 5,000 frames on worlds 2026091371–2026091374; random_safe and greedy_food mixes. The completed campaign has **12 calls, 48 world rows, and 240,000 scored world frames**.

The predeclared rule required treatment mean full-horizon mass to exceed both its matched control and the shared initial checkpoint in both mixes for both seeds, with finite evidence and recorded food/growth. Food and survival are secondary descriptive outcomes; no post-hoc threshold was added. A pass would only have authorized first-class behavior-contract implementation and independent confirmation, never promotion.

## Paired evaluation

Each row is the mean across the same four fixed worlds. Mass is full-horizon post-alive logical mass, with dead frames scored as zero. Food and survival deltas are treatment minus control.

| Seed | Opponents | Initial mass | Control mass | Treatment mass | Delta mass | Delta food | Delta survival |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026091331 | Random-safe | 1.00765 | 2.73240 | 0.49710 | -2.23530 | -32.0 | -0.21470 |
| 2026091331 | Greedy-food | 2.83695 | 0.47025 | 0.75025 | +0.28000 | -16.0 | +0.48310 |
| 2026091332 | Random-safe | 1.96185 | 8.54625 | 6.94900 | -1.59725 | -49.5 | -0.00295 |
| 2026091332 | Greedy-food | 3.56860 | 6.01935 | 6.08045 | +0.06110 | -24.5 | +0.20935 |

Both seeds fail the predeclared rule because treatment regressed against control in random-safe play. Treatment also fell below its own initial checkpoint in both mixes for seed 2026091331; for seed 2026091332 it exceeded initial in both mixes, while the ordinary control itself also exceeded initial in both mixes. That single-seed control result is descriptive evidence only and makes no champion or causal claim.

The four worlds in each mix are paired diagnostics, not a population confidence interval. No confidence interval was computed. The mixed individual-world outcomes and reduced food outcomes do not establish why the treatment failed. They only show that this advice fraction and teacher did not produce a repeatable advantage on this frozen screen.

![Paired mass results](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/scripted-advice/results.png>)

## Evaluation recovery and qualification

The original evaluation runner is retained as a pre-simulation failure. Its fake-fixture assumption expected a top-level checkpoint source_revision field; real checkpoints store provenance differently, so preflight raised RuntimeError: checkpoint source_revision drift before any validation world was stepped. This failure consumed no validation world, changed no checkpoint, and triggered no retraining.

A separate read-only evaluate_v2.py adapter corrected the native provenance reader. Its freeze binds the adapter SHA-256 a8225fa36236b24c17c92b2a66da5c204e5b6f1450ca059a164f862437a81e43, the original runner, protocol, and adapter tests. It is not a training or intervention change. The v2 preflights and both six-call evaluations completed and were bound to matching supervisor receipts, adapter digest, plans, calls, and terminals.

Eleven focused training/behavior qualification tests passed, including zero-fraction control/base parity; actual-epsilon-only application; shadow-generator agreement; teacher/native accounting; illegal-teacher fallback; fixed-policy invariance; and bounded captures. Nine real-checkpoint v2 evaluator-adapter tests passed with return code zero. Smoke artifacts remain explicitly non-evaluable.

## Integrity, resources, and evidence

The independent standard-library audit passed: 42 PASS, 0 FAIL, and 0 PENDING checks. It verified four completed training arms, all 12 evaluation calls and 48 rows, matching recorded initialization hashes, finite telemetry, correct coverage, frozen v2 adapter lineage, and eight completed supervisors with source/driver drift false.

The CPU2 campaign closed with 482 host samples: CPU 4.3–63.8% with no samples over 80%; available memory 27.23–29.15 GiB; and swap 6.599–6.606 GiB. GPU use was 0–81%, with two host samples above 80%, but this host aggregate cannot be attributed to the CPU learner. The resource log recorded no thermal warning, no performance warning, and no CPU power status. These observations support bounded local execution; they do not reserve or prove exclusive system resources.

The frozen original runner SHA-256 is 1a7df64da912b25570a6e549d372f3d1d905a6c7b9ef6f3a14b3c015b4a171e6. The protocol SHA-256 is ac54f70cdf69d888815142a1c59cebd5f38d55edf38c3d6e54c8d8e4e012ddee.

- [Frozen protocol](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/scripted-advice/protocol.json>) and [research rationale](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/scripted-advice/research-notes.md>)
- [Independent audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/scripted-advice/audit.md>) and [machine-readable audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/scripted-advice/audit.json>)
- [V2 evaluator freeze](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/scripted-advice/eval-v2-freeze-v1.json>), [original failed receipt](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/scripted-advice/supervisor-runs/seed1331-evaluation/receipt.json>), and [v2 evaluation receipt](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/scripted-advice/supervisor-runs/seed1331-evaluation-v2/receipt.json>)
- [Results SVG](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/scripted-advice/results.svg>) and [closed resource log](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/scripted-advice/resources.jsonl>)

No additional advice fraction, training seed, reward setting, or confirmation world was selected after these scores. A future hypothesis requires a new frozen protocol and separate artifacts.

