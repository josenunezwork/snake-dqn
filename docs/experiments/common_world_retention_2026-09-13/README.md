# Common-world retention diagnostic — September 13, 2026

Status: **HETEROGENEOUS_RETENTION / HETEROGENEOUS_NET_CHANGE.** A complete common fresh-world evaluation does not support either a universal late-collapse claim or reliable learning/generalization claim. It trained no model and added no useful agent steps: it re-evaluated 12 existing checkpoints retained from four earlier training runs of roughly 150k steps each.

The diagnostic evaluated initial, 50k, and 150k checkpoints from four fixed-5e-4 controls. Each role ran on eight fresh worlds (2026091461–2026091468), against random-safe and greedy-food scripted mixes, for 5,000 frames per world. The resulting 24 calls produced 192 finite world rows. The independent audit passed 52 checks with no failures or pending items.

![Common-world phase means](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/common-world-retention/plots/v1/common-world-retention-mass.png)

## What the common worlds show

The table reports mean total-horizon logical mass across the eight worlds. `50k` is the retained midpoint. The two delta columns are paired within each world; the leave-one-world-out range recomputes the final-minus-50k mean after omitting one world.

| Run | Cohort | Mix | Initial | 50k | 150k | 150k − 50k | LOO range | Median late delta | 150k − initial |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Older run 1 | Older | Random-safe | 1.308300 | 1.986150 | 0.875425 | -1.110725 | -1.370029 to -0.792314 | -0.589100 | -0.432875 |
| Older run 1 | Older | Greedy-food | 1.473050 | 1.683750 | 0.967500 | -0.716250 | -0.848400 to -0.528200 | -0.473400 | -0.505550 |
| Older run 2 | Older | Random-safe | 2.519050 | 11.314550 | 2.139825 | -9.174725 | -10.485400 to -6.480543 | -8.510900 | -0.379225 |
| Older run 2 | Older | Greedy-food | 1.191400 | 3.026900 | 1.793500 | -1.233400 | -1.647600 to -0.511543 | -0.090800 | +0.602100 |
| Current seed 1341 | Current | Random-safe | 0.821825 | 2.000925 | 2.209025 | +0.208100 | -0.102457 to +0.684114 | +0.792200 | +1.387200 |
| Current seed 1341 | Current | Greedy-food | 0.832225 | 3.379650 | 1.159100 | -2.220550 | -2.424457 to -2.062029 | -2.301300 | +0.326875 |
| Current seed 1342 | Current | Random-safe | 5.672275 | 5.753225 | 10.410625 | +4.657400 | +2.636029 to +5.924486 | +4.005000 | +4.738350 |
| Current seed 1342 | Current | Greedy-food | 2.443200 | 2.927650 | 4.776575 | +1.848925 | -1.745657 to +3.086400 | -0.898300 | +2.333375 |

Both current controls finish above their own initial mean in both mixes. That descriptive result is not enough to establish learning beyond this selected diagnostic set.

The older controls decline from 50k to 150k in every mix. Older run 2 still finishes above initial in greedy-food, separating late retention from net change. Current seed 1341 declines sharply in greedy-food despite finishing above initial. Current seed 1342 has positive mean late and net deltas in greedy-food, but its late leave-one-world-out range crosses zero and its median late delta is negative; its positive average is sensitive to the world block. Those patterns give the frozen heterogeneous labels.

## Boundaries of the result

The four runs come from two outcome-informed, nonexchangeable cohorts. They differ in source revision, deterministic opponent stream, initialization and seed namespaces; the protocol forbids pooling them into one exchangeable sample or attributing differences to one adapter detail. Eight worlds are matched diagnostic blocks, not a population sample or powered promotion test.

The result rejects a universal collapse premise on these selected checkpoints and worlds. It does not establish reliable generalization, identify a cause, select a new checkpoint, or change the Apex/vector61 incumbent. No promotion comparison ran.

## Execution and resource evidence

The frozen source was `a19b13c`; the protocol SHA-256 is `44919f567ff5522bd846c3f058649724e6ce42400132c495c7fff804878abd3b`; the analysis SHA-256 is `8921f92737b59e372600a9150d8d4b82d2165e618336501b72e21c71dcabad81`. All 12 checkpoints passed native preflight before scoring. Four serial CPU2 batches completed under their 600-second caps.

The closed host resource log contains 755 samples. Aggregate CPU peaked at 57.6%; minimum available memory was 28,563,046,400 bytes (26.60 GiB); and the raw thermal strings reported no warning. These host readings do not attribute utilization to a learner. The logger exited cleanly and all numerical jobs are closed.

The readable plot was rendered from the final analysis and passed its clean receipt: 0.840723 seconds, 155,303,936-byte peak RSS, and no source or driver drift. See the [PNG](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/common-world-retention/plots/v1/common-world-retention-mass.png>) or [SVG](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/common-world-retention/plots/v1/common-world-retention-mass.svg>).

## Evidence and next decision

- [Frozen protocol](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/common-world-retention/protocol.json>), [checkpoint inventory](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/common-world-retention/checkpoint-inventory.json>), [analysis](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/common-world-retention/analysis.json>), and [independent audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/common-world-retention/audit.md>)
- Full world rows: [batch 0](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/common-world-retention/evaluation/batch-0/calls.jsonl>), [batch 1](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/common-world-retention/evaluation/batch-1/calls.jsonl>), [batch 2](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/common-world-retention/evaluation/batch-2/calls.jsonl>), and [batch 3](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/common-world-retention/evaluation/batch-3/calls.jsonl>)
- [Resource closure](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/common-world-retention/resources-closure.json>), [resource log](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/common-world-retention/resources.jsonl>), [qualification receipt](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/common-world-retention/supervisor-runs/qualification-v4/receipt.json>), and [final audit JSON](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/common-world-retention/audit.json>)
- [Primary-research rationale and proposed next screen](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/common-world-retention/next-decision-options.md>), including [Gallici et al.'s PQN paper](https://arxiv.org/abs/2407.04811) and [Patterson et al.'s empirical-design guidance](https://www.jmlr.org/papers/volume25/23-0183/23-0183.pdf)

The proposed next exploration-floor screen is draft and code preparation only on source `c777b04`. It has not trained, run qualification, or started evaluation. Its proposed matched pairs keep the 30k decay while comparing epsilon floors .02 and .20, both with the canonical anchor. That changes the whole 0–30k exploration curve, so it is a new intervention requiring its own frozen protocol. No paid compute, remote push, incumbent change, or promotion is authorized.

