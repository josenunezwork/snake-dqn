# Actor throughput plan — 2026-10-17

**Recommendation:** retain CUDA actor forwards; first qualify the existing compiled actor
path, then remove repeated v8 body painting within each prepared frame. Optimize ego output
allocation and integer geometry next. Start the integrated throughput sweep at **3–4 actors,
one CPU thread each**. Four optimized actors are a plausible target only if long-body cost
inflation stays modest; there is no evidence that any actor count alone will deliver
21,600 transitions/s on the present A100 learner.

This is a performance design and bounded local measurement, not a production change,
training result, promotion, or claim of achieved speed-up. No production source, running
job, CUDA/MPS service, or tmux session was changed. Baseline checkout: `42dae602e1e8a6bf4030a17d12cc2d82bc48f081`,
branch `persona-actor-20261017`, initially clean. Current release is bare B8, per
[`docs/STATUS.md`](../../STATUS.md#L7); historical hero-only actor results are not equivalent
to six-slot self-play.

## 1. Evidence and accounting

The retained extraction is
[`smoke03_summary.json`](../../../research/actor_perf_20261017/smoke03_summary.json).
It includes source hashes, final records with original JSONL line numbers, arguments,
resource summaries, and an explicitly defined final approximately 60-second telemetry
window. Source is Drive `My Drive/t16_colab/smoke-03/`, implementation
`ef32fb99e7691fa3e716083a0abb94adf3eea2a0`. Each row below is
`<row>/attempt-001/{summary.json,log.jsonl,telemetry.jsonl}`. `summary.json`, not
`summary.jsonl`, is the terminal receipt. All listed actor exits were zero.

| Row | Actors / gather threads / batch scale | Whole-run tr/s | Last ~60s telemetry delta tr/s | Final `learner_wait_s` (queue + ingest) |
|---|---|---:|---:|---:|
| `sp_w128_a2_raw_g0_bs2_fp32` | 2 / 0 / 2 | 12,294 | 11,504 | 62.311 s |
| `sp_w128_a3_raw_g3_bs2_fp32` | 3 / 3 / 2 | 14,701 | 15,019 | 14.447 s |
| `sp_w128_a4_raw_g3_bs2_fp32` | 4 / 3 / 2 | 15,462 | 16,179 | 6.870 s |
| `sp_w128_a4_raw_g3_bs4_fp32` | 4 / 3 / 4 | 14,544 | 15,776 | 6.182 s |

Whole-run rate is terminal transitions / trainer wall time, including warm-up. The window
rate is `(last.transitions-first.transitions)/(last.t-first.t)`, selecting the first
telemetry record closest to 60 seconds before the last. These are different estimators;
the task's approximately 15.9k steady figure is consistent with the interval data, not
identical to the whole-run rate. The a2→a3 comparison changes gather threads as well as
actor count. There is **no measured a6 row**; the runner omitted it. Batch scale 4 doubles
transitions per update, so compare fixed-recipe work as well as raw updates/s.

The a4 fp32 K2 final log and telemetry both refer to 4,601,856 transitions (line 46).
At that matching boundary:

| Summed actor timer | Seconds | Fraction of `env_s` | Microseconds / reported transition |
|---|---:|---:|---:|
| v8 feature construction + redecision | 256.686 | 40.6% | 55.78 |
| ego featurizer | 187.882 | 29.7% | 40.83 |
| forward including transfer / host synchronization | 61.498 | 9.7% | 13.36 |
| remaining work inside actor frame | 126.011 | 19.9% | 27.38 |
| **inclusive actor `env_s`** | **632.077** | **100%** | **137.35** |

`env_s` is **inclusive**, not simulation-only: it spans `step_with_policy(choose)`, v8,
forward, reward calculation, replay-ring writes and n-step processing. It excludes the
following queue put/backpressure and preceding weight refresh. See
[`train_m3c.py:2751`](../../../research/redesign_m3c_20261008/train_m3c.py#L2751),
[`2811`](../../../research/redesign_m3c_20261008/train_m3c.py#L2811),
[`2850`](../../../research/redesign_m3c_20261008/train_m3c.py#L2850), and
[`2985`](../../../research/redesign_m3c_20261008/train_m3c.py#L2985).
The residual is not a pure simulator measurement. In this all-slot run `opponents_s=0`;
the table must not be generalized by summing overlapping opponent timers in hero mode.
The v8 timer also includes the second `q.float().cpu()` transfer after greedy collection.

Mean reported GPU utilization was **83.5%**. This is sampled device activity, not proof
of 83.5% SM arithmetic utilization or a trace of context-switch time. Four CUDA actors and
the learner compete, but the 61.5 s forward total cannot be attributed wholly to context
switching. Capture device traces before assigning that cause a numerical fraction.

Despite its name, `learner_wait_s` includes queue reads **and** replay/label ingestion and
statistics updates ([`train_m3c.py:3358`](../../../research/redesign_m3c_20261008/train_m3c.py#L3358)).
The a2 62.311 s is therefore not 62 s of demonstrated pure actor starvation; a queue-only
timer is needed to isolate that. At four actors this entire queue/ingest interval is only
6.87 s of 296.58 s. Eliminating it all could recover at most about **2.4%** at unchanged
learner cost, and eliminating pure wait would recover less.
`learn_s / updates = 250.589 / 17,916 = 13.99 ms/update`; even stripping other costs leaves
about 18.3k tr/s at K2. The target requires **84.375 updates/s, 11.852 ms/update** at the
unchanged 256-transition recipe. The older smoke-02 **23–25k** A100 ceiling was a planning
estimate, not a smoke-03 observation
([results:30](../t16_colab_smoke02_results_2026-10-17.md#L30)). Actor optimization must be
paired with a measured reduction in learner/device contention or learner work.

**Upstream update reviewed before publication:** commit `a22c03a9` adds an isolated
same-A100 learner measurement: raw/g3/fp32/K2 takes **12.82 ms/update (~78 updates/s,
~20k tr/s)** without actors ([smoke-03 append](../t16_colab_smoke02_results_2026-10-17.md#L187)).
That is still below 84.375 updates/s; removing actor interference alone is therefore
insufficient at the measured current learner configuration. It needs at least another
~7.6% reduction from 12.82 ms to 11.852 ms, plus end-to-end overhead headroom. The raw
`p0_fixed/bench_raw-rings_g3.json` receipt is hash-bound and retained inside this task's
`smoke03_summary.json`: five rounds of 30 updates, random initial weights, `/dev/shm`
rings, GPU leg 12.665 ms measured in separate serialized diagnostics. Its metadata still
carries the harness's default noisy-MPS note
([`bench_learner.py:136`](../../../research/perf_learner_20261013/bench_learner.py#L136));
the no-actor condition is reported by the upstream results note, not inferred from that
generic string. This is not a proof that all possible A100 learner optimizations fail.
The same append reports
that `compare.py` marks rows INVALID because `policy_lag_mean` was not lifted into its
summary; underlying telemetry includes lag, but these descriptive numbers do not override
that automated qualification result.

## 2. Local CPU measurements

Artifacts:

- [`bench_kernels.py`](../../../research/actor_perf_20261017/bench_kernels.py): fixed-state kernel and CPU-forward harness.
- [`mac_cpu_kernels.json`](../../../research/actor_perf_20261017/mac_cpu_kernels.json): every sample, profiles, versions and source SHA-256s.
- [`bench.log`](../../../research/actor_perf_20261017/bench.log): completed scenarios.
- [`extract_smoke03.py`](../../../research/actor_perf_20261017/extract_smoke03.py): read-only receipt extraction.
- [`verification.txt`](../../../research/actor_perf_20261017/verification.txt): focused reference-test result.

Apple M5 Pro, 18 physical/logical cores; Python 3.12.3, Torch 2.9.1, Numba 0.68.0.
**CPU only**, `nice -n 10`, one Torch intra/inter-op, Numba, BLAS and OpenMP thread.
The worktree has no `venv`; the main checkout's project venv was used explicitly. No GPU
was initialized for the benchmark. Background jobs were left alone; a system daemon
occupied roughly one core at the initial process snapshot. Numbers are descriptive of
this shared Mac, not calibrated Mac→Colab scaling or a production actor capacity test.

The harness injects six non-overlapping serpentine bodies per world, lengths 5/150/600/1200,
into the 145×83-cell arena, with ring capacity 1600. It calls the real compiled ego and
v8 APIs over **all live slots**. It warms each operation, then takes seven repetitions;
JIT/setup is outside each timed sample. Worlds do not advance, so results depend on this
geometry and hot caches. These are not draws from the smoke's length distribution.

Rounded milliseconds per full all-slot batch:

| Worlds × slots | Length per snake | v8 scalars | ego-b raster | Cached v8 redecision | CPU CNN forward, random weights |
|---|---:|---:|---:|---:|---:|
| 16 × 6 | 5 | ~0.16 | ~0.8 | ~0.008 | ~38 |
| 16 × 6 | 150 | ~0.87 | ~0.7 | ~0.008 | ~38 |
| 16 × 6 | 600 | ~2.3 | ~0.8 | ~0.008 | ~38 |
| 16 × 6 | 1200 | ~4.5 | ~0.8 | ~0.008 | ~38 |
| 128 × 6 | 5 | ~1.3 | ~5.7 | ~0.019 | ~340 |
| 128 × 6 | 1200 | ~36 | ~5.4 | ~0.020 | ~340 |

Use the JSON for exact samples. CPU forwards use the real `Ego2sNet(7,60)` architecture
and observation shapes, with seeded random weights solely to measure cost; no checkpoint
quality or CUDA/CPU action parity is claimed. At 768 rows, ~340 ms implies only ~2.26k
forward rows/s/core before simulation. Moving all CNN forwards to these CPUs is therefore
an unattractive default, especially on the six physical Colab cores.

The per-action leaf probe executes 100 repetitions of six real `_landing_area` calls,
with opponents painted once outside the timer. A second Numba probe copies its exact
release-map paint prefix, without BFS, to isolate that cost. At length 1200 the six-action
sequence takes about **32–34 µs**, of which the paint-only probe takes **7–8 µs** (~23%).
At length 150 it is roughly 5 µs versus 1 µs. These separately timed probes are a cost
estimate, not a cycle-accurate additive decomposition; only slot zero is used in the leaf
probe, whereas the full v8 measurement includes all slots. They support removing repeated
painting but also show that eliminating painting cannot eliminate tail-aware BFS cost.

Ego component calls at 128 worlds/length 1200 are roughly **2.7–3.0 ms** for local/global
planes, **0.4 ms** for enemy-next, and **1.3–1.5 ms** for b-scalars, including **~1.1 ms**
for region sizes. At short length the region call is only ~0.26 ms. Component medians are
separate calls with their own allocations and cache effects, not an exact sum. cProfile
attributes Numba dispatcher work to Python wrappers; it does not resolve the internal BFS.

Repeated observations/features matched bytes in all ten cases; four one-world length
cases also matched the independent NumPy ego backend byte for byte. Focused existing ego
tests passed **11/11** (30 deselected), including b-scalar/reference and dense-region
comparisons. This validates the harness inputs/reference path, not any proposed optimization.
Full grid/live/v8 long-body qualification is reserved for implementation; no claim that
those entire suites ran in this documentation task.

Reproduce from the repo root (set `PY` to the project venv Python):

```bash
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
nice -n 10 "$PY" research/actor_perf_20261017/bench_kernels.py \
  --repeats 7 --out /tmp/actor-kernels.json
"$PY" research/actor_perf_20261017/extract_smoke03.py \
  '/Users/josenunez/Library/CloudStorage/GoogleDrive-josemnunez.jn@gmail.com/My Drive/t16_colab/smoke-03' \
  --out /tmp/smoke03-actors.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
SNAKE_DQN_DEVICE=cpu nice -n 10 "$PY" -m pytest -q tests/test_ego_raster.py \
  -k 'b_numba_equals_numpy_and_reference or numba_equals_numpy_on_rows_subset_and_paused_worlds or b_region_kernel_equals_reference_on_dense_random_views'
```

## 3. Ranked kernel changes

Expected gains below are **engineering hypotheses for planning**, not measured candidate
speed-ups. They describe their affected component; gains overlap and must not be multiplied
blindly. Effort is focused engineer-days including differential tests, excluding GPU access.

| Priority | Change | Expected component gain | Effort | Why / boundary |
|---|---|---|---|---|
| 1 | Share retained-body templates across actions | v8 ~1.1–1.3× | 2–4 days | Eliminate most of six release paints and three boost blocker paints; preserve BFS. |
| 2 | World-shared v8 membership and unique counts | additional v8 ~1.05–1.25×; combined priorities 1–2 target ~1.2–1.5× | 2–4 days | Scan bodies once/world, not all opponents again for each learner slot. |
| 3 | Preallocated ego outputs, integer enemy-next kernel | ego ~1.1–1.3× | 1–3 days | Reduce zeroing/concatenation and temporary-array work without changing float expressions. |
| 4 | Ring traversal and narrower/layout-aware scratch | affected paint/BFS ~1.05–1.2×, possibly no gain | 1–3 days | Remove dynamic modulo, reduce scratch bandwidth; benchmark branch/cache tradeoffs. |
| 5 | Optional two-worker world partition | kernel ~1.3–1.7× when CPU cores are available | 2–4 days | Private scratch required; could reduce total throughput with four actors. |
| 6 | Persistent cross-frame membership cache | unquantified incremental benefit after priorities 1–2 | 4–8+ days | Larger invalidation surface; do last, after profiling remaining body-paint share. |

### 3.1 v8 retained-body templates: keep the search identical

[`v8_nb.py:388`](../../../src/simd_env/v8_nb.py#L388) paints all other live bodies for each
observer; [`403`](../../../src/simd_env/v8_nb.py#L403) paints its own body. Eager features
then call three `_landing_count` and six `_landing_area` operations
([`431`](../../../src/simd_env/v8_nb.py#L431)). The latter rescan the post-move body at
[`235`](../../../src/simd_env/v8_nb.py#L235), with a max release time for duplicate cells.
Use one retained-old-body map per distinct post-move class, computed by the existing
`_post_move(n,L,bf,boost,min_boost,boost_cost)`:

- For old segment index `j` (0 = current head), normal movement retains release `lpost-j`;
  boost movement retains `lpost-j-1`. Include exactly the retained `ln-nh` old segments.
- Overlay the boost intermediate cell with release `lpost`; take `max` when it overlaps
  an existing old cell. The new landing head has **no release restriction**, even on an
  old-body overlap: the reference removes that cell after constructing the map.
- Static boost landing blockers use retained old cells plus the intermediate cell;
  preserve their distinct semantics from tail-aware area release.
- If boost cannot fire, the corresponding area equals normal movement and can be reused;
  the landing diagnostic must still be `_LANDING_NO_BOOST`, not a normal count.

Leave `_bfs` neighbor order `+x,-x,+y,-y`, FIFO discovery distances, passability checks,
seen-on-passable rule and caps unchanged
([`v8_nb.py:98`](../../../src/simd_env/v8_nb.py#L98)). Ordinary components or a new
wait-for-tail shortest-path algorithm are not interchangeable with this capped search.
The first implementation should reuse paints, not reuse flood results across directions.

### 3.2 Share world occupancy, with exact overlap semantics

Build per-world cell membership once, with one bit per snake for the six-slot mode.
For observer `s`, other-body blocking is `(membership & ~(1 << s)) != 0`; own-body
blocking is its bit. Compute unique covered cells exactly, because
`area_cap=max(1,min(max(32,2L),4096,gw*gh-covered))` depends on that count
([`v8_nb.py:423`](../../../src/simd_env/v8_nb.py#L423)). Here `covered` counts unique
**other-body** cells after excluding the observer's bit, including cells shared with
another snake. A sparse touched-cell list or
stamps can avoid clearing the entire board for every requested world. Group rows by world
but scatter outputs back to their original order, preserving duplicates.

Do not substitute `owner_pad` without a proof: its single owner and newest-segment
precedence cannot encode the body-union semantics of synthetic overlapping snakes
([`grid_sim.py:193`](../../../src/simd_env/grid_sim.py#L193)). A uint8 mask is sufficient
for six snakes; validate the limit and retain a general fallback for larger populations,
BatchSim, and live adapters. Never silently truncate membership bits.

### 3.3 Ego work: preserve what is already fast

Existing optimizations are not new work:

- Global coarse counts are maintained incrementally by the simulator and consumed at
  [`ego_raster_nb.py:443`](../../../src/simd_env/ego_raster_nb.py#L443).
- Region components are already computed once per requested world by run-length
  union-find and reused across its rows
  ([`ego_raster_nb.py:526`](../../../src/simd_env/ego_raster_nb.py#L526)); four empty cells
  are skipped at a time. Its TTL>1 blocker differs from v8's all-body blocker, so these
  component sizes cannot replace v8's counts.
- Local reach already uses an O(window) bucket relaxation, not repeated full-window
  sweeps ([`ego_raster_nb.py:44`](../../../src/simd_env/ego_raster_nb.py#L44)).
- `sim.bodies` is already contiguous int64; `_i64(sim.bodies)` is normally a view, not
  a full body copy to eliminate.

Allocate the final seven-channel local plane and 60 scalar outputs once per output
buffer, fill the existing six channels plus enemy-next directly, and avoid the intermediate
concatenations at [`ego_raster.py:496`](../../../src/simd_env/ego_raster.py#L496) and
[`train_m3c.py:2824`](../../../research/redesign_m3c_20261008/train_m3c.py#L2824).
Compile integer enemy geometry/occupancy and stable nearest-enemy selection if profiles
justify it. Preserve scalar NumPy float64 division, sqrt/log1p and operation order:
**no fastmath, reassociation, precision reduction, or assumed libm equivalence**. Even
Numba without fastmath can change last bits when it replaces NumPy transcendental paths
([`ego_raster.py:360`](../../../src/simd_env/ego_raster.py#L360),
[`ego_raster_b.py:234`](../../../src/simd_env/ego_raster_b.py#L234)).

Scratch can be reused; observation storage cannot be overwritten until the ring/n-step
consumer has taken ownership. Test retained observations over multiple calls and weight
updates. A coarse total maintained alongside existing counts is a small later option;
do not add another mutation invariant before measuring its current sum cost.

### 3.4 Layout, parallelism and cross-frame validity

Replace descending-ring `% ring` with decrement-and-wrap while preserving traversal.
Try int32 cell indices/queues and a single linear-cell queue only after proving bounds
for supported grids/caps; retain int64 stamps or a tested rollover reset. Compare current
`x*gh+y` against row-major indexing as a separate experiment; changing neighbor order
to obtain cache locality is forbidden. Performance is uncertain because layout changes
can exchange division work for extra branches.

`v8_nb._WORKSPACE` is process-global and keyed only by grid dimensions
([`v8_nb.py:739`](../../../src/simd_env/v8_nb.py#L739)). Adding `prange` to the current
row loop races paint/release/seen stamps, queues and row scratch. Partition **worlds**
into bounded chunks with private scratch per executing worker, and disjoint output rows.
Keep each world's BFS and floating-point operation order serial. Benchmark 1/2 threads
with 2/3/4 actors under the six-physical-core budget; 12 vCPUs are not 12 full cores.
Account for the learner, three gather workers and library/runtime threads as well.

Persistent caches need an explicit **prepared-state generation**, not just `frame`.
`step_with_policy` performs food/respawn/mask preparation before `choose`
([`batch_sim.py:655`](../../../src/simd_env/batch_sim.py#L655)); ego views expire at the
next step ([`grid_sim.py:957`](../../../src/simd_env/grid_sim.py#L957)). Cover normal and
boost pushes/pops, growth, boost burns, death, respawn, reset/reset_envs, snapshot restore,
injection, inactive worlds, and new segment construction. TTL crossing 1 can change ego
connectivity even without a changed occupied-cell set. Release times change each move.
Do not cache full feature values, head-risk flags, masks or connectivity across frames
without separately proving their dependencies unchanged.

## 4. Actor-side GPU contention decision

### First experiment: existing compiled actor CUDA graphs

Smoke-03 used `compile_actor=off`; it compiled the learner only. The actor implementation
already pads to `worlds*6=768` rows in all-slot mode
([`train_m3c.py:2641`](../../../research/redesign_m3c_20261008/train_m3c.py#L2641)) and
provides pinned staging, graph replay, in-place weight refresh, output cloning and
fallback receipts in [`compile_m3c.py:239`](../../../research/redesign_m3c_20261008/compile_m3c.py#L239).
Test `--compile-actor reduce-overhead` before writing another forward wrapper. It may
reduce launch/transfer overhead and avoids eager mode's repeated CPU/device model moves
on refresh ([`train_m3c.py:2751`](../../../research/redesign_m3c_20261008/train_m3c.py#L2751)).
It **does not merge CUDA contexts or remove the host synchronization needed for actions**.

Planning range: **0–30% shorter actor forward leg**, only **0–3%** direct actor-frame
gain at the observed 9.7% share, plus an unmeasured learner/contention effect. Historical
4090 hero-only results found **no actor compile gain within ±5%**
([quiet results:56](../../../research/perf_quiet_20261013/RESULTS.md#L56)); do not promise
a large win. Four actor compilations also consume startup time and memory; serialize
warm-up/limit compile workers and include startup in deployment economics.

PyTorch documents that reduce-overhead uses CUDA graphs where eligible and does not
guarantee applicability; graph outputs/storage need lifetime care
([torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html),
[CUDA graph trees](https://docs.pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html)).
Use the installed Torch 2.9.1 receipts as runtime authority, not an assumption that current
online docs describe every pinned-version detail.

The existing actor probe accepts Q tolerance `1e-3*(1+scale)` and records argmax flips,
but does not fail specifically on flips
([`compile_m3c.py:289`](../../../research/redesign_m3c_20261008/compile_m3c.py#L289)).
That is not a bit-identity gate. Require byte-identical features and explicit masked
greedy/v8 label identity on saved real observations, ties, changing live-row counts and
weight refreshes. If compiled Q changes actions, do not call it transparent: retain eager
or use a separately authorized algorithm/numerics qualification. Eager-kernel graph
capture without Inductor fusion is a follow-up option if compiler numerics fail.

### Second experiment: CUDA MPS on a fresh, dedicated Colab runtime

NVIDIA MPS permits concurrent work from cooperating processes and can reduce context
switching; on Volta-and-newer hardware clients retain separate GPU address spaces
([NVIDIA architecture](https://docs.nvidia.com/deploy/mps/latest/architecture.html),
[when to use MPS](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html)). This addresses
the multi-process scheduling issue more directly than graph replay. It adds no CPU feature
capacity and cannot create GPU compute or bandwidth headroom.

First check binary availability (`nvidia-cuda-mps-control`), GPU/driver support, container
permissions and ownership. This task did **not** establish that the Colab runtime permits
the daemon. In a new isolated benchmark runtime, start a job-owned daemon with dedicated
pipe/log directories **before any CUDA clients initialize**, verify that all actors and
the learner actually connect, run the fixed workload, then stop owned clients and daemon
in `finally`. Do not modify compute mode or start/stop a service in a runtime with other
jobs. Never change a shared user's daemon as cleanup. If unavailable, record unsupported
and retain the graph-only path.

Start with no active-thread quota and benchmark against MPS off. Later quota experiments
must distinguish an execution-resource limit from guaranteed exclusive SM ownership.
Planning range: **0–15% end-to-end improvement, with regression possible**; this is a
test range, not an NVIDIA guarantee or prediction from the 84% busy counter. Graphs and
MPS can be combined, but measure the 2×2 matrix separately. Record CUDA-event kernel time,
host launch gaps, D2H wait, learner update latency and policy lag; device busy alone is
insufficient.

### CPU forwards and central inference

`--actor-device cpu` already exists
([`train_m3c.py:1669`](../../../research/redesign_m3c_20261008/train_m3c.py#L1669)). It removes
actor CUDA contexts but the local ~340 ms/768-row CNN cost is much larger than its feature
work. At that measured CPU cost, ~10 dedicated forward cores alone would be needed for
21.6k rows/s. Colab has only six physical cores for the entire pipeline. Do not extrapolate
Mac speed as an x86 guarantee, but require a much better measured Colab CPU result before
considering this option. CPU/GPU Q is not guaranteed bit-identical; keep feature and action
qualification separate. No quantization or precision change is part of this plan.

If graphs/MPS fail to reduce contention and traces demonstrate context scheduling loss,
consider a single GPU inference service (EnvPool/Sample-Factory-style centralized batching)
as a larger architectural follow-up. Keep actor RNG/epsilon on the originating process,
tag responses with request and weight versions, bound batching delay and preserve the
weight freshness contract. Shared memory/IPC, queueing and learner scheduling make this
several days of design and testing, not a free speed-up. Do not pipeline stale actions to
hide the synchronization and call it equivalent self-play.

## 5. How many actors for 21.6k at long bodies?

Use actor **service cost**, not learner-consumed throughput divided by four. At the matched
smoke boundary, `4,601,856/632.077 = 7,280 transitions/s` per aggregate actor busy-second.
This is a proxy using learner-counted transitions against consumed actor messages; n-step
flushes/queue accounting and excluded refresh/put costs prevent treating it as a precise
standalone actor maximum. Full queues `[4,4,4,4]` and little queue/ingest time reinforce that
four actors already supply more than the present learner consumes.

For planning define:

```text
C0 = 7.28k tr/s per actor busy-second (smoke proxy)
d  = long-body service-cost inflation relative to that smoke mix
s  = measured full-actor service speed-up from proposed changes
eta = 0.80 planning efficiency for refresh, transport and contention (assumed)
A_required = ceil(21.6k * d / (C0 * s * eta))
achieved throughput <= min(A * C0 * s * eta / d, learner sustained capacity)
```

| Long-body inflation d | Current kernels s=1 | Optimized s=1.3 | Stretch s=1.6 |
|---:|---:|---:|---:|
| 1.0 | 4 actors | 3 | 3 |
| 1.5 | 6 | 5 | 4 |
| 2.0 | 8 | 6 | 5 |

**Practical estimate:** plan for **4–5 optimized actors** at moderate long-body inflation,
with four sufficient at `s=1.3` only if `d<=1.40`. Six or more actors on six physical cores,
plus learner/gather work, is an oversubscription warning rather than a capacity solution.
Three optimized actors require almost no cost inflation (`d<=1.05` in this model).
If every snake reaches the 1200-cell synthetic regime, local v8+ego cost is about six
times the short-body case; a modest inflation assumption is not defensible for that case.
The telemetry has no body-length histogram, so **d is currently unknown**. The table is a
sensitivity estimate, not a validated actor-count recommendation for all long bodies.

An illustrative combined service budget—v8 1.5×, ego 1.25×, forward 1.5×—reduces 137.35 µs
to about **106 µs/transition**, a **1.29×** service speed-up at the smoke mix. Forward 1.5×
is a stretch beyond the initial graph planning range; use actual A/B numbers. This model
does not imply a 1.29× learner-rate increase under backpressure. The historical 128-world
long-body 2,987 tr/s CUDA and 7,324 tr/s Mac benchmarks are hero-only with opponent work;
dividing 21.6k by those values is not a self-play actor-count calculation.

A 3→4-actor whole-run gain was only 761 tr/s. Linear extrapolation would misleadingly
suggest about 12 total actors. The 2→4 slope is additionally confounded by gather threads.
Neither extrapolation is a credible deployment plan. Keep K2, replay ratio, label fraction,
precision and update cadence fixed; no `--update-every 2` throughput shortcut.

## 6. Codex implementation packets and acceptance gates

These are future implementation assignments, not changes made by this task. Capture the
new baseline/dirty state before each packet, assign one owner per overlapping module,
and retain a legacy oracle/rollback backend until qualification finishes.

1. **Measurement packet — actor harness/telemetry only.** Add bounded component timers
   (paint, v2/v5/v7 BFS, planes, region, packing, refresh, queue-put), body-length/cap
   histograms, live rows, valid/emitted transitions, and per-actor service time. Time each
   phase with profiling disabled in the scored run. Create identical saved short/long
   prepared-state corpora at lengths 150/600/900/1200/1500 with crowded, open and trapped
   shapes, all headings, boost phases and row subsets. Separate warm-up, scoring and
   shutdown; preserve input/source hashes. Extend this task's harness instead of changing
   a bound governance document.
2. **v8 kernel packet — sole owner of `v8_nb.py` and v8 differential tests.** Implement
   per-speed templates behind an opt-in backend first. Compare every `V8Result` field and
   every feature byte with legacy and Python/live references. Then add world membership
   sharing as a separate measurable change. No float/reward/mask/BFS semantic changes.
3. **Ego packet — sole owner of `ego_raster_b.py`, `ego_raster_nb.py` and output integration.**
   Compile integer enemy-next work and fill final outputs; preserve NumPy scalar math and
   stable enemy tie order. Measure allocation and copy costs before adding persistent
   output pools. Coordinate the actor integration with the v8 owner; no overlapping edits.
4. **Inference packet — `compile_m3c.py`/actor integration and dedicated Colab runner.**
   Benchmark existing `--compile-actor reduce-overhead` first, require real graph replay
   and no silent fallback, add action/label identity gates, then test MPS off/on in an
   isolated runtime. If nonzero action differences occur, stop equivalence claims. This
   packet must not change the training recipe to pass a speed target.
5. **Verification packet — independent read-only reviewer plus bounded verifier.** Run
   legacy/new lockstep, audit scratch ownership and invalidation, then a fixed-seed 2/3/4
   actor sweep (5 only if telemetry justifies it). Optional 2-thread worlds are a separate
   axis. Commit only after reference parity, lifecycle and performance evidence agree.

Required parity coverage:

- `tests/test_v8_nb.py`, including the **slow** long-body test at line 216. It requires
  pinned checkpoints; a skip is missing evidence, not a pass. Compare area/count/caps,
  landing sentinels, head-risk/waiver, eligibility, reasons and decisions, not just argmax.
- `tests/test_ego_raster.py`, `test_ego_raster_bv8.py`, `test_ego_live_identity.py`,
  `test_ego_live_bv8.py`, `test_grid_sim_parity.py` and `test_grid_sim_nb.py`.
- `tests/test_perf_actor.py`, `test_perf_compile.py`, plus existing all-slot trainer tests:
  retain RNG draws/order, n-step windows, emitted transition bytes, replay labels and
  checkpoint metadata. Exercise empty/all-dead, repeated/shuffled/subset rows and masks
  with no valid actions; exact Q ties and near ties; both mechanics versions; impossible
  boost and boost-burn boundaries; growth where `n<L`; wrapped rings and repeated cells;
  out-of-grid synthetic segments; overlap; reset/respawn/death/restore/inactive worlds.
- New lifetime/concurrency tests: prior output unchanged after next call, thread-private
  scratch, repeated 1-vs-2-thread output bytes, same feature results under different world
  scheduling, stamp rollover, and all-slot compiled weight updates without stale tensors.

Example full targeted qualification command (future implementation, no `-m 'not slow'`):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  ./venv/bin/python -m pytest -q \
  tests/test_v8_nb.py tests/test_ego_raster.py tests/test_ego_raster_bv8.py \
  tests/test_ego_live_identity.py tests/test_ego_live_bv8.py \
  tests/test_grid_sim_parity.py tests/test_grid_sim_nb.py \
  tests/test_perf_actor.py tests/test_perf_compile.py
```

Use the actual project venv path if the implementation worktree also lacks `./venv`.
Run each proposed kernel against its current oracle before broader suites. Byte
comparisons must include float32 representation, not tolerance-only equality. Numerical
GPU-forward checks are a distinct gate; CPU feature parity cannot qualify CUDA numerics.

Performance acceptance: on identical input corpora, aim for ≥15% v8 improvement at long
bodies without >5% median regression in short/other geometry cells, and ≥10% ego gain for
its packet. Require repeated unprofiled measurements with the same thread budget. Keep
changes that fail these thresholds experimental even if they pass parity. The final
capacity test needs at least three fixed **10-minute post-warm-up** runs per finalist,
with genuine long-body coverage or a separately labeled restart corpus, unchanged K2
recipe, no overruns, clean exits and bounded policy lag. Report both aggregate and
length-binned actor service, sustained updates/s, learner wait and device timings.

Call **21.6k achieved** only if the fixed-recipe learner sustains ≥84.375 updates/s and
≥21,600 valid consumed transitions/s in every scored finalist repeat, with a predeclared
stationary window and stable queue/lag behavior. If CPU fixes pass parity but the learner
stays below that threshold, publish the actor improvement and the remaining learner
bottleneck separately. Strict evaluation and serving qualification remain on the Mac;
this plan grants no promotion or release authority.
