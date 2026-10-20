# T34 A5: qualify singleton processes before buying more GPU time

2026-10-20 · Python/Numba performance advice · source `e04107010c5a74f63eb2ef17bd08949b5567937d`.

**Use independent, single-thread processes with one resident world each. Profile the host path, then optimize its largest exclusive cost. The present evidence supports a host bottleneck and excessive cold-fill latency; it does not establish where every Colab millisecond goes or guarantee A5 GO.** Advice only: no jobs launched, registration changed, or compute committed.

Artifact paths below are relative to `/Users/josenunez/Projects/ml/snake-dqn-artifacts/`.
Verified A5 #1 exports report **49.79 row-steps/s/lane**, zero decisions/worlds
(`t34-a5-20261020/l4-{0,1}-output/exports/benchmark-lane-aggregate.json`).
A5 #2's final polls show **38,400–38,912 rows in 1,373–1,385 s/worker**, approximately
28/s/worker, 56/lane, zero worlds. Both terminal receipts are INCOMPLETE, with no
exports and `stop_verified=true` (`t34-shadow-batch-20261020/final-l4-*-output/`).

The reported 35 ms is amortized worker wall time, including initialization,
not a kernel latency. W256 × D300 requires up to **76,800 cold-fill rows/worker**:
about **46 minutes at 28/s**, before the first root decision, exceeding both the
1,500-second collection and 30-minute session limits. Increasing timeout alone
does not improve throughput. Early shadow deaths can shorten this bound.

The matched Mac receipt, [SHADOW_BATCHING.md](../../../research/t34_exit_b8/SHADOW_BATCHING.md),
confirms 86,008 rows/32 **H1000** worlds: 135.81 serial versus 109.88 batched rows/s,
96 matching files and 32 matching replays. Hero forward/transfer consumes only
43.99/782.73 seconds (5.6%); shadow/root transitions consume 708.59 seconds.
These are nested stage timers, not a Colab call profile. H1000 work cannot be
treated as measured H5000 work; the stated 13–31k rows/world remains a planning range.

The source explains why five threads fail to imply five useful cores:

- [policy.py:144](../../../research/t34_exit_b8/policy.py) constructs rasters separately, one world/row at a time. Several native kernels release the GIL but remain serial; Numba thread counts do not parallelize Python loops ([Numba flags](https://numba.readthedocs.io/en/stable/reference/jit-compilation.html)).
- [lockstep.py:258](../../../research/t34_exit_b8/lockstep.py) rebuilds merged simulator/runtime/roster state and copies/deepcopies it back every tick. Its W1 path bypasses that assembly.
- [vector61_policy.py:935](../../../src/simd_env/vector61_policy.py) still invokes Python v8 live hooks; T34 does not use T22's compiled `v8_decide` path. Recursive [full_digest](../../../research/t34_shadow/shadow.py) also traverses mutable carry. Their exclusive shares are unmeasured.

**Cheapest confirming measurement:** one future CPU-only runtime, one process,
one native thread, W1, CPU inference; a ≤60-second warmed profile on excluded
scripted/frozen/mixed snapshots spanning short and long bodies. Record exclusive
v8, digest, raster, simulator and inference time, wall/process-CPU ratio, RSS,
affinity/cgroup quota and provider rate. At owner-supplied ≤0.5 u/h, the measured
minute costs ≤0.0084 u; propose a 12-minute whole-session cap and 0.12-u envelope
including 20% reserve, through the qualified supervisor. This tests the cheapest
candidate and identifies its binder; it cannot causally decompose the original
W256/CUDA/5-thread run. CUDA launch/transfer attribution requires a matched L4 profile.

Proposed dependency order: profile → qualify runtime recipe/optimize → A5 → Round 0.
The six-process A5 proposal requires prospective recipe, validator and aggregator
amendments: current A5 enforces two CUDA workers, five threads and W256. A CPU
diagnostic cannot supply A5 GO.

| Work | Exact next layout after qualification | Decision/cost |
|---|---|---|
| Teacher A5 on known L4 | 6 spawned processes; 1 resident world/process; CPU inference; torch intra/inter-op, Numba, BLAS/OpenMP all 1 | At unchanged 28 rows/s/process: 168/lane, only **20–47 worlds/h** at 31–13k rows/world; no promised GO. |
| CPU teacher production candidate | Initially 2 processes × 1 thread × 1 world, contingent on two usable cores/quota | At 0.5 versus 2 u/h, CPU wins units/world above **one quarter** of L4's completed-world rate, including overhead. Cores/rate remain unverified. |
| Unpaced Path B | Same L4 host: `--nproc 7` gives 6 processes, one world each; CPU inference, numerical pools 1; preserve each controller's main/search threads | Existing process-pool implementation; measure contention because each world permits two CPU threads. Never price it from teacher row rates. |

At 31k rows/world, 50 worlds/h requires **431 rows/s/lane**, or **72/process**
with six workers: approximately 2.6× the observed worker rate. If v8 dominates,
port its compiled decision kernel while reproducing all nested counters/carry;
the league adapter omits diagnostics that T34 hashes. Otherwise optimize the
measured digest/assembly term. Preserve fp32, masks, RNG, ordering and labels.

Round 0 currently enforces CUDA/five threads and serial pairs
([runner.py](../../../research/t34_round0/runner.py)); amend prospectively through
the delegated judge and repeat A6. Keep each pair in one process; prove
serial/parallel hashes at the same inference recipe. W1/CPU versus W256/CUDA
can change Q ties and needs separate qualification.

A5 retains its 1,200-second completed-world window excluding cold fill. Export
each finished world; budget setup, fill, window and export before admission.
At 50 teacher worlds/h, 240 pairs cost **at least 4.8 lane-hours/9.6 units**,
plus bare execution and metric regeneration. Measure that paired entrypoint.
Path B's 362.5-second forecast comes from 64-frame prefixes, not complete worlds
([cost receipt](../../../research/t34_pathb/VERIFICATION_RETENTION_FIXES2.md)). With the
manifest's assumed 3× cloud slowdown and ideal six-worker scaling, 176 rows need
**8.9 lane-hours/17.7 L4 units**; two CPU workers need 26.6 hours/≤13.3 units.
These exclude setup/export/reserve and may underprice late search and Watch
sessions. The manifest's doubled timeout plus reserve can refuse six-worker L4
liability; reprice complete-world costs and have the judge review any cap amendment.
Paced timing, strict gates and serving qualification remain Mac-only.
