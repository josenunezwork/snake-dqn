# TRAIN-01 second-seed replicate — sandbox-a-train01-20261028-s2-01

Prepared only. Operator dispatches through the qualified broker. Development evidence only,
labelled `simulator-D`; no F5 inputs/outcomes, serving changes, or changes to running studies.
Owner authorization: 2026-10-28 22:05 PT, approvals yes for eight hours; H100 first, A100 fallback.
Standing sandbox authority: `docs/research/governance_a100_sandbox_20261026/ADOPTED.md`.

## Frozen runtime and recipe

The accepted running packet-15 continuation uses `dccf7e25a5fb3ebc2fe303b2dc072aeb83efd213`,
which is newer than the original packet-15/16 runtime and is absent from this worktree's main
base `baa620a3f58b25e7d16cd1f00cf778131a7a31e1`. This packet imports that accepted runtime and
continuation machinery. `train.py`, model, learner, proxy, task, preparation, compiler, rosters,
acceptance screen, CPU reference, simulator, and observation modules are byte-identical to the
accepted runtime. The retained main launcher adds `TORCH_CUDA_ARCH_LIST='8.0;9.0'`.

Runtime pin: `98018ea4d962dfb55ca184c9f4d03c7234c6dada`. The integration preserves main's
reviewed H100 preference emitter, adds the explicit `packet-train01-s2-01` continuation family,
and emits compact-format and source-packet bindings. Preferred-GPU continuations validate the
executed primary/fallback identity and use the same whole-timeout price builder. Existing
historical packets remain immutable. The initial alternative compact implementation was reverted;
it is absent from the final runtime.

Recipe: pinned B8 s3 start, 384 updates, 1024 worlds, rollout horizon 256, task horizon 5000,
100,663,296 actor rows, `compile_mode=default`, E24 calibration, E360 critic preparation,
8,388,608 preparation-row cap, snapshots every 16 updates, E96/H5000 proxies at 0,16,64,128,
192,256,320,384. The temperature, actor/critic/RNG initialization and all learning parameters
are computed independently by the unchanged frozen recipe.

## Seeds and evidence boundary

Fresh training allocation index 2, offset 2,000,000,000. This avoids both packet-15's offset 0
and packet-16's previously used offset 1,000,000,000. The actual registry audit covers
114,299,008 roots, with canonical uint64 digest
`828020fee364c45cb64f6a932be045d2d14885b40282ddaa07efeb0dbdf8450c` and no aliases or exclusions.
Registry SHA256: `39b80bcd4cca6a3e137206958fced3292f5aa5cb77e4f5f22cb9e810757ffc2c`.
An exhaustive union audit covers 342,894,912 distinct roots across all three domains, finding
zero index-interval overlaps and zero effective low63 collisions. The 288 monitor and 768
nomination roots remain intentionally identical across runs. No efficacy outcome was consulted.

## Finite envelopes and resume

A100 forecast uses the owner's matched 4.06 min/update observation: 1559.04 minutes for 384
updates, plus 280 proxy minutes, 9 startup-screen minutes, 20 calibration minutes, 115 preparation
minutes and 26 minutes for allocation/staging/snapshot/export/cleanup = 2009.04 minutes.
Base estimate is 2040 minutes (34 hours); the effective 2x runaway guard is 4080 minutes
(68 hours). No H100 speed improvement is assumed. H100 is provisionally priced at 12 u/hour:
408 units base, 816 effective. A100 fallback is 8 u/hour: 272 base, 544 effective. Both price
all setup, work, idle, transfers and cleanup inside the entire timeout. Base/effective reserves
are both zero. These are operational runaway guards, with no scientific unit rationing.

The guard exceeds Colab's roughly 24-hour VM lifetime; it does not extend that lifetime.
Compact full-state snapshots are exported at source and Mac-verified incrementally. The first
attempt uploads an empty ledger. Only the latest eligible verified full-state snapshot may
continue after a verified mechanical stop and verified release, through the accepted one-command
`training_continuation` builder. Keep original N, RNG state, counters, preparation, all seed/proxy
banks and the existing at-most-two outcome-blind completion attempts. Every continuation uses
fresh manifests, state/output paths and receipts, a committed compact allowlist, and its own
2x remaining-work envelope. No automatic workload retry or post-failure scientific tuning.

## Verification and handoff

The final local evidence and exact dispatch command are in
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/astra-impl-train01-s2/SUMMARY.md` and
`dispatch-command.sh`. The checklist distinguishes local preparation/dry-run evidence from
live accounting, allocation, GPU execution and stop verification, which remain operator steps.
Native CUDA compilation/execution cannot be measured on this Mac. The tiny actual launcher uses
PyTorch's architecture-flag generator and verifies `compute_90,code=sm_90` plus `sm_80`, then
runs three real PPO updates and verifies a compact full-state export on CPU. Its device shim
is confined to the retained test harness, outside the sealed packet and shipped source.
