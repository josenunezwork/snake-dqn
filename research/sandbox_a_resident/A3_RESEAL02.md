# A-3 reseal 02: Argon compile repairs

Prospective engineering correction to A-3 after Argon's 2026-10-26
ACCEPT-WITH-FIXES review. The original packet-a3 and every older seal remain
unchanged. Only research/sandbox_a_resident changes. The matrix, frozen inputs,
seed ranges, measured cycles, timeout, simulator-D scope and release boundary
are unchanged. This packet is prepared for operator dispatch; no allocation is
performed by this repair.

- Validate each tensor-shape/gradient specialization with both CUDA matmul and
  cuDNN TF32 disabled around **both** compiled and eager calls. Restore the
  caller's switches even on failure. Retain exact integer comparisons and
  floating rtol=atol=2e-5. Timing continues under TF32 enabled.
- Compile v8 feature planes and observation construction from current tensors.
  Resolve R2 occupancy/live windows, TTL, component sizes, food and head tensors
  outside the graph. Exact CPU/CUDA flood work enters a sandbox-local custom op
  with fake registration; native pybind loading and mutable simulator/cache
  access are outside Dynamo tracing. The feature-only v8 computation omits final
  action reranking, which has no effect on the five observation feature planes.
- Keep separate ActorCritic_collect and ActorCritic_train blocks, so learner
  validation and backward fallback do not disable collector inference.
- Reuse the same compiled blocks and model parameter storage across cells with
  matching device/mode/arena/storage geometry. World count and the R2 live
  window are tensor-shape specializations. Copy each fresh cell's initial
  model values into that storage and create a fresh optimizer; simulator, RNG,
  buffers and optimizer state are never reused. A run-local cache pays the
  compilation cost once for each specialization, including the common learner
  minibatch across world counts. Scope Dynamo recompile_limit=64 and
  accumulated_recompile_limit=256 to the finite assay; restore caller settings.
- Report Torch's named compiler timer deltas separately for setup/warmup and
  measured cycles, including lazy backward and TF32-specialization work. Named
  timers may nest and must not be summed. Wrapper first-specialization wall
  time is an inclusive setup/validation diagnostic, not pure compiler time.
  Any compiler activity during a measured cycle invalidates its steady-state
  timing instead of subtracting nested compiler timers from the rate.
- Deep-copy compile reports at export; mark unused collector blocks in cached
  cells and unused learner blocks in collection cells not_applicable. Eager
  mode emits its explicit requested-mode/status schema.

CPU admission runs real unmocked torch.compile with the aot_eager backend and
fullgraph=True for all five blocks; default Inductor remains the deployment
backend. Byte-exact source parity checks exercise v8 features and observation
planes/scalars across simulator steps and resets. Additional coverage checks
TF32 restoration, independent backward fallback, graph/model reuse without
optimizer reuse, specialization timing and immutable report snapshots.

A100 execution, CUDA timings and live provider accounting remain operator-run
checks. The new packet has a fresh job/output identity, an empty allocation-bound
ledger, exactly ten physical sandbox manifest lines, and the same base 14,400 /
effective 28,800 second runaway guard. See packet-a3-02/DISPATCH.txt and
VERIFICATION_A3_RESEAL02.json for the final pin, seals and admission evidence.
