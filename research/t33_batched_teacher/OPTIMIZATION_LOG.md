# T33 round-two optimization log

Development-only. Controlling brief: `docs/research/specs/2026-10-19_batched_teacher_round2.md`
and `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t33-reviews/fable-hotpath-review-1.md`.
Baseline worktree HEAD was `f76820776e39a6a53c13197f6f0948aa7c6a95f4`; all frozen
comparators came from Git objects in this checkout. No other worktree was used.

## Measurements and evidence boundaries

Every microbenchmark uses three warmed rounds on this Mac, nice 10, single-thread
BLAS/Torch; Numba pool sizes are explicit below. Rows mean simulator environments
except where a metric is labeled source decisions or staging calls. Parent timers
include their named child timers; the lockstep tables use exclusive teacher stages,
never sum runtime internals into them. The machine also hosts an independently
operated calibration; these are observed local rates, not quiet-host qualification.

| Affected stage | Before | After | Interpretation / receipt |
|---|---:|---:|---|
| Full factorial ablation, H16/N4 smoke | 2.683 s | 1.208 s | 2.22x total; 8.95 -> 19.86 equivalent full rows/s; exact scores/choices, `ROUND2_PREFIX_BENCH.json` |
| Body rings at H32/L800 | 998,400 bytes/row | 79,968 bytes/row | capacity 10,400 -> 833; int64 retained; ~12.5x smaller body allocation |
| Merge assembly, 48 resident rows | 1,610 rows/s | 5,872 rows/s | medians, `T33_ROUND2_TEACHER_MICROBENCH_20261019_V2.json` |
| Short teacher, two scripted sources | 5.69 decisions/s | 6.83 decisions/s | H8/N4/epsilon .5, documented RNG schedule change; same V2 receipt |
| Vector61 warm ordered-food staging | 22.14M rows/s | 215.93M rows/s | tiny unchanged-food cache, ~9.75x; excludes feature kernel, `ROUND2_V8_V61_MICROBENCH_MAC_20261019.json` |
| Observation build, 64 rows | 132,594 rows/s | 138,830 rows/s | earlier bounded-chunk implementation, 1 thread, `ROUND2_OBSERVATION_MICROBENCH_MAC_20261019.json`; persistent scratch added later |
| Grid step, small 48x3 fixture | 67,278 rows/s | 68,027 rows/s | final matched medians, ~1.01x; `ROUND2_FINAL_MICROBENCH_V3.json` |
| Corpse event, 448 incoming cells | 1,443 rows/s | 15,850 rows/s | final ~10.99x, O(1) grid membership and ordered queue, V3 receipt |
| v8 L800, painted serial -> direct owner four threads | 35,398 rows/s | 84,430 rows/s | final ~2.39x pool scaling; direct owner serial was 33,757 rows/s, V3 receipt |
| random-safe anchor dispatch, 12,288 rows / 256 unique triples | 767,534 rows/s | 3,290,856 rows/s | ~4.29x, exact choices, `ROUND2_ANCHOR_MICROBENCH.json` |

The earlier grid V1 receipt compared `jit=False` to `jit=True` and is superseded.
V2 corrected that confound. `ROUND2_FINAL_MICROBENCH.json` includes compiled food
consume but precedes incremental spawn mirroring and persistent geometry scratch.
Historical receipts remain unchanged; their hashes identify exactly what ran.

## Implemented behavior

1. Merge re-caps rings to maximum source segment count + H+1, keeps C-contiguous
   int64, re-bases ring pointers and repairs slot grids. It clones list/set/RNG
   state by type, pads unequal ordered-food capacities, validates source grids
   and rejects live body overlaps. Admission counts unique allocations, resident
   bytes/row, clone headroom and a conservative fixed-shape GPU activation bound.
2. Lazy equivalence leaders advance until exploration diverges. Forks occur before
   lifecycle preparation/movement and copy frame, RNG, food, simulator arrays,
   V61 carry, trend memory and accumulated mass. A quiet leader is retained when
   the previous representative explores. Group work is O(N) per exploration event.
3. Pre-drawn epsilon tensors preserve source/action/rollout identity, CRN across
   first actions, and exact H32/H64 prefixes of the canonical H128 schedule. This
   is a new draw order relative to Phase 0; it is not called Phase 0 RNG identity.
4. V61 uses the ordered numeric food mirror and active leader masks. Frozen
   checkpoint weights stay on the selected device; forward groups are fixed-size
   with inactive states zero-padded. Initial hero Q is evaluated once per source
   through a separate adapter, never all 6N replicas.
5. Observation output slabs are reused. Only selected rows and stale tails are
   cleared; global writers sum addressed coarse blocks instead of a whole E×H×W
   temporary. Local flood and region union-find scratch persists in disjoint
   per-partition caches belonging to each Python caller. Canonical scalar NumPy
   helpers still allocate temporaries; the entire simulation is not allocation-free.
6. v8 owner BFS reads owner pads directly, including landing/release overlays,
   and uses maintained coarse counts for coverage. The painted oracle stays the
   default for all shared callers. Row chunks retain private workspaces without
   allocating small action scratch once per row. Grid head/tail coarse updates,
   corpse queue compaction, and numeric noncontested
   train-mode food updates use Numba. Ordered Python collection maintenance,
   contested/nontraining consume and CPython rejection spawning preserve order.
   The spawn mirror appends incrementally and consumes compact the mirror in place.
   Inactive action-mask kernels skip work; internal mask views avoid full copies.
   Random-safe anchors hash unique seed/slot/frame triples and preserve exact uint64
   modulo, each row's lagged frame, and normal-action legal ordering.
7. CUDA towers/fusion use fp16 and dueling value/advantage/mean/subtraction use fp32.
   Qualification requires masked agreement only, while reporting raw agreement and
   legal margins <.01. Submit/collect tickets fence slab reuse; two resident A/B
   lanes own separate buffers/streams. CPU A/B orchestration matches isolated lanes.
   CUDA compile/precision/overlap remains unverified without hardware.
8. GPU crop step 5 is dropped: Fable's packed full-board transfer is larger than
   the raster slab. No GPU simulator or observation port was added.

## Prefix and leaf calibration

`ROUND2_PROTOCOL.md` precedes new data. One full CRN rollout yields cumulative
mass and next-boundary max/V values at all requested H prefixes; N uses its first
streams. The independent-action legacy oracle remains a separate full run.
Uniform reservoir sampling captures selected seams in one bare trajectory pass,
with optional source/checkpoint/world-bound eligible/selected index caching.

Leaf scale units are **segment-frames per raw network-Q unit**. The fixed fit is
nonnegative through-origin least squares from prefix Q to remaining full-horizon
mass. The report compares scales 0, 1 and fitted, in-sample only; it never selects
a winning production recipe. On three synthetic L800 decisions, fitted scales
were 5,269–8,725 depending on H/N/leaf; fitted agreement varied and sometimes
worsened. This confirms scale 1 has a unit mismatch, not that a fitted leaf or
H32 is good. `ROUND2_ABLATION_SMOKE.json` took 57.66 seconds. No result from the
separately running 300-decision calibration-v1 was accessed.

## Lockstep profile and L4 projection

Final measured cells, per-stage microseconds per resident/active row-step, and
conditional projections are recorded in the source-bound lockstep JSON receipts.
The profile uses N4/H32 strictly as a timing shape, not a selected label budget.

CPU evidence in `ROUND2_MAC_LOCKSTEP.json` used three Numba threads, Torch/BLAS
one, and the earlier source closure (before direct-owner/anchor final changes).
Its 190 captured Python hashes exactly match commit `5e8bb7a1`, verified from
this checkout's Git objects; its recorded HEAD was the preceding commit with
that implementation still dirty.

| Sources | Resident rows | CPU wall | CPU decisions/s | Active row-steps |
|---:|---:|---:|---:|---:|
| 256 | 6,144 | 122.107 s | 2.0965 | 44,844 |
| 1,024 | 24,576 | 504.223 s | 2.0308 | 179,500 |

The first CPU-only attempt was stopped during warmup when an accidentally included
slow training test was noticed. That test's owned process tree was stopped; its
68 preceding passes are partial evidence, not a completed gate. The clean non-slow
focused rerun passed 68 tests with 2 slow deselections. The retained CPU profile
above then completed both cells (the second finished just before a stop request,
which found no running PID). No partial second-cell rate is used.

Final-source MPS evidence is `ROUND2_MAC_MPS_LOCKSTEP.json`, bound to code
commit `7a5d208dea257d983e015c3081b6f659e2651436` and complete Python source hashes.
Torch/BLAS used one thread, Numba three; precision was eager fp32.

| Sources | Resident rows | Wall | MPS decisions/s | Active row-steps |
|---:|---:|---:|---:|---:|
| 256 | 6,144 | 10.982 s | 23.3114 | 44,844 |
| 1,024 | 24,576 | 42.641 s | 24.0145 | 179,500 |

Per-stage microseconds per row-step (assembly/initial-Q/leaf costs are included):

| Exclusive teacher stage | 256 resident | 1,024 resident | 1,024 active |
|---|---:|---:|---:|
| Assembly | 2.714 | 2.815 | 12.333 |
| Observation host build | 6.396 | 6.123 | 26.824 |
| Hero MPS forward | 12.567 | 12.297 | 53.875 |
| Opponent dispatch, including device forward | 5.590 | 2.750 | 12.050 |
| V61 post-step carry | 2.570 | 2.293 | 10.046 |
| V61 selection prepare | 0.258 | 0.261 | 1.142 |
| Simulator step, selector time removed | 23.986 | 25.828 | 113.158 |

Resident row-steps are 196,608 / 786,432. Active movement row-steps are 44,844 /
179,500; this difference exposes the lazy-fork savings rather than counting
followers as executed work. Residual wall overhead (hashing, fork bookkeeping,
initial setup and guards) is retained in wall rates and the L4 model. Simulator
work is now the largest host item; MPS inference is about 23% of measured wall.

Using the 1,024-world cell, conditional L4 projections are:

| CPU-core slowdown assumption | One L4, serial | One L4, ideal A/B | Three L4s, serial | Three L4s, ideal A/B |
|---|---:|---:|---:|---:|
| 1x Mac core speed | 59.63/s | 64.43/s | 178.88/s | 193.30/s |
| Historical 3.7x slower core | 17.04/s | 17.41/s | 51.13/s | 52.24/s |

These scenarios do **not** establish the target of 100–200 decisions/s per L4.
Even the optimistic one-host estimate is below 100/s at this timing budget.
No new budget was selected and no hardware job is authorized by this table.

Large-shape allocations/first-forward costs are included after a three-world warmup.

The L4 model replaces measured Mac inference wall time with Fable's 0.8 us B8
compute +0.6 us H2D per row, and a conditional 0.086 us/vector row at 10 TFLOP/s.
It assumes six physical cores for a 12-vCPU host and brackets per-core slowdown
at 1x and the historical 3.7x. Perfect host scaling and ideal A/B overlap are
optimistic scenarios, not confidence intervals. Three-host figures assume linear
scaling and exclude orchestration/setup. There is no L4 measurement or target claim.

## L4 handoff and remaining verification

The draft benchmark now requests 6,144 / 24,576 / 49,992 resident rows at N4/H32,
with admission before allocation. The ~50k host-memory goal does not prove the
same GPU batch fits: the conservative 512-KiB-per-row activation reserve can reject
49,992 rows on a 24-GB L4. Actual CUDA workspace/shape qualification is pending.
All six public checkpoint URLs remain null; the updated manifest stays publication
blocked and grants no allocation authority. No cloud commands or publishing ran.

`L4_MANIFEST_PUBLICATION_BLOCKED_ROUND2.json` pins the published code commit
`7a5d208dea257d983e015c3081b6f659e2651436`, with `T33_RESIDENT_ROWS=6144,24576,49992`,
N4/H32 and the previous L4-only resource/cap envelope. It was generated locally
without calling the supervisor or any cloud CLI. `transport.dispatch_ready=false`
and all six URLs are null. It was deliberately not passed to a dispatch validator.

The actual `bench.sh` entry point also completed a local CPU preflight with six
resident rows, N1/H2, one thread, one timed iteration and no CUDA qualification.
`ROUND2_ENTRYPOINT_CPU/benchmark.json` and its payload/hash/completion sidecars
record the result. Its tiny-shape rate is not used in the lockstep comparison or
the L4 projection. Both final-source receipts contain 191 Python file hashes;
all matched the current source closure, and every payload sidecar was checked.

Final verification counts are recorded in `ROUND2_VERIFICATION.json`. Independent
read-only integration reviews found and resolved fixed opponent shapes, unequal
food capacities, source overlap admission, epsilon H-prefix identity and the
remaining v8 map staging. The final numeric queue, anchor hashing, scratch
partitioning and projection arithmetic received a second read-only audit with no
concrete defect found. The 100–200 search decisions/s L4 target, CUDA fp16 legality
agreement/compile, true A/B overlap,
full 300-decision budget selection and any label-quality conclusion remain unverified.

### Test-gate boundary

The final affected-path gate passed **527 tests**, with five skips and three slow
deselections, in 166.12 seconds. Its 28 files cover T33, GridBatchSim stepping
parity, V61 and veto compatibility, v8 body maps, raster bytes and anchor choices.
The two skipped parallel checks were then run explicitly with a four-thread
Numba ceiling and both passed in 1.15 seconds. The remaining three skips are
opt-in SIMD gameplay episode tests. Black/isort/flake8, the repository lint target,
shell syntax and whitespace checks passed.

The full repository non-slow gate is **not certified green**. Its first attempt
reached the offline CLI fault properties at 23%, with one visible failure in the
older V7 strict-run module, before being stopped as too broad/expensive. That
module passed all 110 tests in isolation and again passed in the diagnostic broad
prefix; the earlier failure's cause was not established. SIGINT/SIGTERM did not
stop the first run, so only its identified owned process tree was killed. A
second diagnostic broad run was stopped in offline CLI supervisor tests after
the V7 module passed. Neither interrupted run is reported as a completed gate.

The remaining local non-cloud run then passed 1,879 tests before stopping at
`test_perf_compile.py::test_compiled_loss_terms_close_on_fixed_batch[bv8-1.0]`.
The failure was `BackendCompilerFailed` / `ImportError`: an existing cached
Inductor `.so` could not load `@rpath/libc++.1.dylib` from Torch's library directory.
It occurred while loading compiled code, before the numerical comparison. No
shared venv, external checkout or global compiler cache was repaired. Raw logs
and final affected-path results are retained under `ROUND2_TEST_LOGS/`.
