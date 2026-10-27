# A-3 reseal 03: one numerical mode, prewarmed R2 buckets

Prospective compile repair after Argon REJECTED packet-a3-02. The separately
sealed packet-a3-eager-01 was published and passed preflight plus dispatch
dry-run before this repair began. All older seals remain immutable.

Compiled forward and eager validation now share the caller's TF32 switches;
there is no TF32-off graph compiled for validation. Each first specialization
executes the compiled callable once with the caller's gradient state and checks
a no-grad eager reference in the same numerical mode. The returned training
graph remains available for backward in that mode. Tensor strides/offsets,
grad/inference modes and numerical globals are validation-key inputs. Exact
integer/bool comparison and float rtol=atol=2e-5 remain. Compiled first-call wall
time excludes the separate reference/comparison timer; Torch's nested compiler
metrics remain authoritative diagnostics, not additive timing deductions.

R2 already quantizes the live window in 256-slot increments. Production cap 1024
therefore has exactly four static buckets: 256/512/768/1024. Before each active
collection cell's ordinary excluded warmup, warm_compile_buckets invokes the
compiled v8 boundary for all four shapes at that E. Extra ring slots retain
the original validity masks. This setup does not step/prepare/reset the sim,
draw RNG, update the actor/optimizer, or advance assay counters. Normal warmup
still executes training backward. Any compiler work in a measured cycle still
makes its timing INVALID. Tiny cap 32 tests alone are not bucket qualification.

The anchor callable is explicitly static (already so in the working source).
Cached TensorBlocks owners contain only configuration metadata and pure
functions; no simulator or assay reference. A real two-assay aot_eager test
populates Adam and proves the first assay, simulator and optimizer are garbage
collected while the shared actor/graph cache survives. Inactive per-cell report
blocks omit prior validation/specialization fields; active counters are labelled
run_cache_cumulative, with independent snapshots.

The original 24-cell matrix, H128, one warmup, three measured cycles, minibatch 2048, ep64/512 and
TF32 timing / deterministic proof separation remain fixed. This distinct packet
uses roots 360,000,000 with 3,000,000 spacing, proof 440,000,000, and fresh output
a3-attempt-03/output. Base/effective runaway guard remains 14,400/28,800 seconds.
The ledger is empty and allocation-bound. Source, seals, tests, preflight and
exact operator command are recorded separately. Simulator-D timing/parity only;
no efficacy, serving, F5 or transport change. CPU real aot_eager admission is
not an A100 Inductor, TF32 tolerance, memory-capacity or throughput result.
