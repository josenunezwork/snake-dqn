# Fresh-runtime continuation of the fixed 6102 endpoint

This module implements the direction recorded in `task-aligned-continuation-direction-20260924T133044Z.json`, after reading campaign metadata first. It preserves the original completed 6102 candidate and collects new experience. It does not pool the previous lineages, reconstruct their games, reset Adam, change the architecture or objective, or claim exact runtime resumption. The root owns prospective admission, resource limits, held-world exclusion, evaluation, and audit. Implementation and source compilation are not permission to start numerical work.

## Source and interfaces

`source_binding()` returns the exact source identity without filesystem access:

- Lineage seed: 2026096102.
- Source global update: 35,613; preceding mixed-task rollouts: 31,005.
- Source path: sibling artifact tree `task-aligned-cz-integration-20260924/admission-v2/run/train_2026096102/output/checkpoint_rollout_31005.pth`.
- SHA-256: `8d67915ef1ea5dbb0804ffd8bf9874dedb27c95878bacdd5b5e5ae77682aa46f`.

Public APIs are:

```python
source = load_source(device="mps")
trainer = build_continuation(source, runtime_seed, device="mps",
                             initialize_runtimes=True, qualification=False)
evidence = trainer.restoration_evidence()  # Before any updates; numerical inspection.
row = trainer.update()
trainer.save_checkpoint(path)             # Science: only after exactly 31,005 updates.
seeds = trainer.used_world_seeds()
trainer.close()

endpoint = load_endpoint(path, expected_sha256, device="cpu")
```

`load_source` authenticates the fixed source through the qualified adapter's `load_candidate`, including its schema, model/Adam layout, step ages, completed-dose clocks, input/task contracts, and original CZ ancestry. It performs one checkpoint hash and one deserialization. Its return is the existing adapter's `LoadedParent`, preserving the actual candidate payload and identity. It is not fabricated as an original CZ parent.

`build_continuation` restores the source using the qualified adapter's full-state constructor. The underlying constructor checks exact model and optimizer tensor digests, restores global update/hero clocks and action-collapse evidence, and refreshes numerical recovery snapshots after restoration. The wrapper's `restoration_evidence()` additionally verifies those digests and tripwire values before updates. Fresh physical counters include the source load; inherited training counters are recorded separately in the child payload.

## Preserved numerical and world contract

The only training update called is the original adapter's qualified `TaskAlignedTrainer.update`. It continues native PQN Q(lambda) with the unchanged restricted CZ network, half-MSE, Adam, gamma 0.997, lambda 0.65, epsilon 0.1, one exact-coverage epoch, padded B256, and gradient clipping 10. It preserves the existing native reward plus the 0.1 ambient-food overlay. No enemy-input weights are reset.

Collection remains E16/T16 over persistent solo/S2/S6 runtimes scheduled `[solo, s2, s2, s6, s6]`, with H1024 hero episodes, the same scripted opponents, Watch pre-move decisions, terminal hero death, live-cap bootstrap, allocation-only body storage, and original scalar normalization. Every profile shares one learner and Adam state. The numerical kernel's singleton hero tensor dimension remains separate from actual physical roster metadata.

Qualified source references: `research/task_aligned_20260924/adapter.py::build_trainer`, `collect_rollout`, `_half_mse_sgd`, `task_contract`, `model_type`, `load_candidate`; `src/training/pqn_trainer.py::_compute_targets`, `_sgd`, and `refresh_numeric_recovery_state`. The new module never invokes the old checkpoint writer or edits these files.

## Fresh streams, clocks, and final-only identity

A declared nonnegative integer seed is converted to a new effective seed using the first eight SHA-256 bytes, big endian, of:

```
task-aligned-continuation/v1/{science|qualification}/runtime/{declared_seed}
```

That effective seed enters the unchanged qualified world, per-episode action, and SGD permutation derivations. Both declared and effective seeds are recorded. Qualification and science have distinct domains, and collision with the source runtime seed is rejected. This does not prove exclusion against all prior or held world IDs: the harness must bind and check the declared world namespaces/actual seed ledger. No saved RNG position or interrupted episode is restored.

The science dose is exactly 31,005 fresh rollouts, with final global update **66,618**. The additional valid hero-transition count is measured, not assumed; cumulative hero steps equal source hero steps plus this new count. The wrapper checks update, optimizer-step, completed-rollout, sampled-draw, and hero clocks after each successful update.

The new `task-aligned-continuation-full-state/v1` payload records actual source-checkpoint identity and state digests, original CZ provenance separately, child lineage, model, Adam, cumulative clocks, new/inherited physical counters, tripwire evidence, per-profile collection totals, new runtime derivation, and episode counters. It explicitly records no environment restoration, no historical replay, and no promotion. It does not reuse the old `4608 + completed_rollouts` schema assumption. `save_checkpoint` creates one file exclusively, rejects the source path and partial endpoints, and stops further updates after publishing. A failed write is not an automatic retry.

`load_endpoint` authenticates the new file and validates its full-state layout/ages and cumulative-clock relationships. It accepts only the completed science schema; serving receives the unchanged wrapped model and source configuration. The evaluation harness must additionally compare that configuration and serving-contract digest with its frozen intent.

## Disposable qualification and budget boundaries

`build_continuation(..., qualification=True)` fixes a separate ten-rollout disposable dose. Only this mode exposes `snapshot_state()`, using `task-aligned-continuation-disposable-state/v1`. The root may serialize that payload in its qualification directory and call `load_qualification_snapshot(path, sha, device="cpu")` for the bounded model/Adam/clocks roundtrip. Science serving rejects this schema; the science writer rejects qualification mode. Qualification performs no selection and its updates are discarded.

Science maximum scheduled hero capacity is 31,005 × 256 = **7,937,280** transitions. With this unchanged one-step kernel, successful science entails 31,005 optimizer steps, 558,090 model invocations, and 16,370,640 evaluated model rows (16 E16 collection calls, one E16 target call, one padded B256 update per rollout). Native active frames and valid hero rows are measured; opponent-transition capacity is 19,049,472 for the fixed schedule. Preview calls are 31,005. Initial runtimes construct 48 worlds; a conservative episode-start bound is 48 + 16 × 31,005. Counts cover actual physical work; a later tripwire can leave more successful optimizer steps than completed accepted rollouts, which must remain a failed run rather than an endpoint.

The disposable ten-rollout probe has at most 2,560 scheduled hero rows, ten optimizer steps, 180 training forwards and 5,280 forward rows, plus any separately admitted inspection fixtures. Each explicit checkpoint loader adds one hash and one deserialization. Wall-time, memory, checkpoint-load, fixture, evaluation, audit, and handoff reservations remain the root protocol's responsibility; no unbounded extension or automatic old-run resumption is supplied.

Only source formatting/compilation was performed while implementing this module. No checkpoints were opened or hashed, no numerical libraries were imported by executing this module, and no fixtures, model calls, or games were run.
