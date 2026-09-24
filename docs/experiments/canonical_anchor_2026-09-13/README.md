# Canonical Watch scripted anchor — September 13, 2026

Status: **engineering integrated; opt-in only.** Main at `87cb858` contains the canonical Watch adapter introduced in `ce5253c` and the mechanics-v2 fixture correction in `87cb858`. It gives future corrected-v3/raster31v3 training a fixed scripted opponent whose action stream matches the shared Watch anchor. It does not change an archived or current study, establish model quality, compare policies, or promote a checkpoint.

## What was delivered

The adapter joins the training fixed-policy interface to the existing shared `scripted-anchor/v1` rule.

- [`BatchSim.get_world_seeds()`](../../../src/simd_env/batch_sim.py) returns each lane's current unsigned seed. A selected reset can replace one lane's seed, so the policy reads it at action time.
- [`watch_anchor_frame()`](../../../src/evaluation/anchors.py) converts the prepared Watch frame to the shared zero-based decision index. A prepared frame of 1 maps to anchor frame 0.
- [`CanonicalWatchScriptedPolicy` and its factory](../../../src/training/rollout_policies.py) build a `FixedPolicySource` with identity `scripted-anchor/v1:<kind>`, where `<kind>` is `random_safe` or `greedy_food`.
- The policy accepts sparse `(environment, slot)` rows in caller order. Its tests cover first and continuing Watch decisions, selected-lane reseeding, sparse/reordered lanes, and the shared live/SIMD clock conversion.

## Required future use

This source is for a newly declared run only. Build it explicitly and bind the exact identity in the run configuration:

```python
from src.training.rollout_policies import canonical_watch_scripted_source

source = canonical_watch_scripted_source("random_safe")
fixed_policy_identity = source.identity  # scripted-anchor/v1:random_safe
```

The complete `PQNConfig` must select `rollout_policy_mode="fixed"`, set `fixed_policy_identity` to that value, and use `decision_phase_mode="watch_pre_move_v1"`. The Watch phase is restricted to the corrected-v3/raster31v3 contract. Construct `PQNTrainer(config, fixed_policy=source)`; the trainer rejects a missing, mismatched, or changed identity and stores the fixed source in checkpoint and sampler provenance.

A protocol must freeze the source revision, complete configuration, world and training seeds, source identity, and evaluation procedure before any measurement. Do not retrofit this adapter into earlier experiments or silently resume their checkpoints with it.

## Qualification

A root-supervised focused suite covered six related test modules. The retained first attempt ran 58 tests in 1.35 seconds and produced 57 passes plus one failure: the live Watch-profile fixture had mechanics v1 while the production profile requires mechanics v2. It was a fixture setup failure, not a model or policy result.

The fixture-only correction produced 58 passing tests in 1.10 seconds. The supervisor recorded a natural exit in 1.891 seconds, maximum RSS 532,103,168 bytes, and no source or driver drift. This qualifies the integration boundary. It is not a training run, throughput benchmark, policy-quality result, or promotion gate.

## Context and evidence

The common-world retention rationale identified the frame-key difference between historical training adapters and the production Watch anchor. It called for an explicit production-compatible helper with a named decision index, while preserving archived waves. This delivery supplies that helper; it does not remove the need for a separately frozen fresh-world study.

- The implementation is local commit `ce5253c`; the fixture-only correction and current main are `87cb858`. Inspect either with `git show <commit>` in this repository.
- [Adapter source](../../../src/training/rollout_policies.py), [shared anchor](../../../src/evaluation/anchors.py), [lane-seed accessor](../../../src/simd_env/batch_sim.py), and [regression coverage](../../../tests/test_canonical_watch_scripted_anchor.py)
- [Qualification receipt](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/canonical-anchor/supervisor-runs/canonical-focused-v2/receipt.json>) and [retained failed-attempt receipt](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/canonical-anchor/supervisor-runs/canonical-focused-v1/receipt.json>)
- [Common-world retention research notes](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/common-world-retention/research-notes.md>) and [decision-phase contract note](../../pqn_decision_phase_2026-09-12.md)

The Apex/vector61 incumbent remains unchanged. Promotion still requires a separately frozen request, calibrated common-world evaluation, the strict promotion rule, and serving validation.
