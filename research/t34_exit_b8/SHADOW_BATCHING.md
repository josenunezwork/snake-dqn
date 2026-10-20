# T34 shadow batching implementation and verification

Scope: `docs/research/specs/2026-10-20_t34_shadow_batching.md`.
This is a performance/identity change, not a campaign result or serving release.
Mac execution uses the primary checkout's Python environment, CPU, and one
thread per process. Phase 0 remains active, so concurrent work stays within two
CPU threads. No Colab allocation or live tmux operation is part of this work.

## Reproduce the matched profile

Use fresh paths for every invocation. The profiler excludes a separate warm-up
bank, loads the frozen serial policy/engine/teacher independently, and records
source identities, platform, shape, stage times and all world-export hashes.
The defaults are 32 worlds, 1,000 frames, H300, D300, all three mixes, fp32 CPU.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-shadow-batch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SNAKE_DQN_DEVICE=cpu
export T34_PROFILE_ROOT=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-batch-20261020
export T34_PROFILE_PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
export T34_PROFILE_CKPT=/Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/inputs/B8_s3_final.pth

nice -n 10 "$T34_PROFILE_PY" -m research.t34_exit_b8.profile_shadow_batching \
  --implementation serial --checkpoint "$T34_PROFILE_CKPT" \
  --out "$T34_PROFILE_ROOT/serial-32x1000-r1.json"
nice -n 10 "$T34_PROFILE_PY" -m research.t34_exit_b8.profile_shadow_batching \
  --implementation batched --checkpoint "$T34_PROFILE_CKPT" --replay \
  --out "$T34_PROFILE_ROOT/batched-32x1000-r1.json"
"$T34_PROFILE_PY" -m research.t34_exit_b8.compare_shadow_profiles \
  --serial "$T34_PROFILE_ROOT/serial-32x1000-r1.json" \
  --batched "$T34_PROFILE_ROOT/batched-32x1000-r1.json" \
  --out "$T34_PROFILE_ROOT/comparison-32x1000-r1.json"
```

The serial oracle modules are frozen verbatim under
`tests/fixtures/t34_shadow_batching/`; their README and loader pin all three
SHA-256 identities. The comparison refuses incomplete or mismatched profile
shapes and reports export/sidecar hash differences and replay mismatches.

## A5 benchmark windows

`--seconds` is the total collection wall cap. `--steady-state-seconds` requests a
post-cold-fill measurement window. The first applicable wall/unit/transport cap
still wins; requesting a longer window does not authorize a larger budget.
`batch_timing` separates cold fill, first executed decision, observed steady
state duration, work, rates, stages, and whether the requested window completed.
The inclusive A5 go rule remains unchanged. Post-fill rates are descriptive.

For an operator-selected 1,200-second window inside the existing 1,500-second
maximum collection wall, build with `--benchmark-seconds 1500
--steady-state-seconds 1200`. A cold fill longer than 300 seconds can censor
that requested window. The manifests retain the 30-minute transport timeout,
15-minute estimate and one-unit cap per L4 lane. Neither setting bypasses the
supervisor or authorizes dispatch/retry.

## Implementation boundaries

`lockstep.py` assembles one environment per resident world, preserving the
original body-ring layout, environment RNG, runtime selection/carry, ordered
food and each row's independently grown food-mirror capacity/tail. It scatters
post-step state back to the captured world owners. Finished/dead shadows leave
the active list before the next round. Root transitions and two-root doom-cache
reconstruction use the same transition seam; search retains its existing
bounded branch buckets.

CPU root inference retains the historical 64-row bucket recipe so exported Q
values retain their original bytes. Shadow actions recover a batch-one forward
near a scale-aware fp32 tie margin. CUDA hero inference uses the full resident
bucket. CUDA Vector61 batching is explicitly enabled by v2: immutable networks
share device forwards, one Q slab moves to the host, and each independently
carried policy receives its Q slice through its native v8 action hook. Default
v1 dispatch remains on its existing recipe.

`model_forward_seconds` is nested inside the parent host-stage timers. Its hero
counter includes tensor transfer, fp32 forward and synchronized result copy;
it excludes raster construction. Add neither that counter nor the teacher's
other nested policy/simulator counters to the parent stages. `doom_cache_s`
separately attributes prepared-cache capture in the new path. The serial
oracle's uninstrumented remainder includes that capture.

## Measured Mac result

The full bank preserves all export bytes and replay outcomes. It does not
produce a Mac CPU speedup: throughput falls from **135.81 to 109.88 active
row-steps/s** (19.1% lower), with wall time **633.29 to 782.73 seconds**.
Both arms execute exactly **86,008 row-steps and 32,000 real decisions**.
All **96 payload/sidecar files match** and **32/32 candidate replays pass**.

Measured source commit: `dfaecebf3f92b612298722e70648dc3eb60cbd2c`. The serial oracle is the
three SHA-pinned fixtures from `a832e281`. Both runs use Mac CPU, one thread,
32 worlds × 1,000 frames, H300/D300, all three mixes, fp32 and identical bank
namespace. Warm-up covers all three mixes at W32 and is excluded. Phase 0
remains active; these are loaded-host diagnostics, not serving latency evidence.

| Parent stage | Serial seconds | Batched seconds |
|---|---:|---:|
| Build | 1.67 | 1.63 |
| Shadow fill/refill | 342.64 | 427.50 |
| Root inference | 30.21 | 31.12 |
| Root transition + confirmation | 216.72 | 281.09 |
| Branch search | 29.70 | 31.53 |
| Search root clone/evaluation | 0.03 | 0.03 |
| Prepared doom-cache capture | in remainder | 3.10 |
| Export | 3.39 | 3.34 |
| Unattributed remainder | 8.93 | 3.39 |

The batched run reaches its first real decision at **76.27 s** after an
initial cold fill of **76.17 s**. Its observed post-fill interval is **706.56 s**
with **76,408 active row-steps** and **32,000 real decisions**. Nested hero-model
forward/transfer time totals **43.99 s** (5.6% of end-to-end time).

Three additional matched W32 cold-fill comparisons (two real frames, D300)
all preserve 96 files and pass 32 replays each. Serial median: **69.14 s**
(range 67.70–69.50); batched median: **77.38 s** (range 76.94–77.54). The same
32 development worlds are reused; repeated replays are not independent
campaign evidence.

The initial `serial-32x1000.json` is retained as a diagnostic, excluded from
the comparison because its warm-up covered only one scripted world.

After profiling, one timing-only change moves the doom-cache timer into
`finally` so an interrupted cache contributes time as well as work. Complete
profile execution and exported data are unchanged by that correction; the
final regression adds a partial-cache timeout case. The measured bytes and
source hashes remain bound to the commit above.

Artifact root: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-batch-20261020`.

| Evidence | SHA-256 |
|---|---|
| `serial-32x1000-r1.json` | `b90489d0c346a56799e9d8c1ab0cdb2b39a95309a674f74b1dbcb71b93fcff85` |
| `batched-32x1000-r1.json` | `7dfc084ceb40c12ce93b0b2d4ea96c3fd74d8b0ad6d37ae20b180aa3e36cf0f9` |
| `comparison-32x1000-r1.json` | `ae22a13e4683c1c411abc7ec811096465d16e4577897bf5958ea88e8756fa57b` |
| `PROFILE_SUMMARY.json` | `4d0942c1247c2195f126224a7bcc5ccce5794c9a77148410ccc07d24f65e2a5b` |

## Conditional L4 sensitivity

These are assumptions applied to the W32 Mac profile, not L4 measurements or
a W256 forecast. Let F=43.99 s be synchronized hero-model work and H=738.75 s
be the remainder. Assume an L4 worker has five useful CPU threads, each core
is 3× slower than the measured Mac thread, and hero-model work is 64× faster.
For serialized fraction s of H, use `T_L4 = H*3*(s+(1-s)/5) + F/64`.

| Assumed serialized fraction of H | Row-steps/s per worker | Two-worker lane sum |
|---|---:|---:|
| 0% | 193.7 | 387.5 |
| 25% | 96.9 | 193.9 |
| 100% | 38.8 | 77.6 |

The remaining host term includes native CPU Vector61 inference. Actual CUDA
Vector61 batching, W256 amortization, five-thread scaling, setup, export and
GPU occupancy have not been measured here. Those omitted effects can change
these numbers; the table does not establish the **3,000 row-steps/s A5 GO**.
The Mac measurements identify root/shadow host work as the remaining cost.
L4 acceptance and any dispatch decision remain with the operator.

## Verification and file ownership

Commands ran from `/Users/josenunez/Projects/ml/snake-dqn-shadow-batch` with the
primary checkout interpreter. CPU test/profile commands set
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
SNAKE_DQN_DEVICE=cpu`.

```bash
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -n 2 tests/test_t33*.py tests/test_t34*.py
make PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python lint
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m black --check --line-length 100 research/t34_exit_b8/{teacher_v2,policy,lockstep,benchmark,make_manifest,profile_shadow_batching,compare_shadow_profiles}.py tests/test_t34_{batch_benchmark,batch_vector_device,batch_timing_collector,lockstep_projection,shadow_batching,shadow_batch_oracle,shadow_profile_receipts}.py
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m isort --check-only --profile black research/t34_exit_b8/{teacher_v2,policy,lockstep,benchmark,make_manifest,profile_shadow_batching,compare_shadow_profiles}.py tests/test_t34_{batch_benchmark,batch_vector_device,batch_timing_collector,lockstep_projection,shadow_batching,shadow_batch_oracle,shadow_profile_receipts}.py
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m flake8 research/t34_exit_b8/{teacher_v2,policy,lockstep,benchmark,make_manifest,profile_shadow_batching,compare_shadow_profiles}.py tests/test_t34_{batch_benchmark,batch_vector_device,batch_timing_collector,lockstep_projection,shadow_batching,shadow_batch_oracle,shadow_profile_receipts}.py
bash -n research/t34_exit_b8/bench.sh
git diff --check
```

- First complete gate: **486 passed, 2 skipped in 265.29 s**. The optional
  two-thread Numba observation/parallel checks are skipped under the one-thread
  per-worker ceiling.
- A second gate covers the additional independent retention changes merged
  from main; **513 passed, 2 skipped in 264.23 s** at source commit
  `b32f88f1c751f02c7f8b758d9a7647446e31fa69`.
- Frozen-oracle/core/receipt selection (`-n 1`): **15 passed in 164.33 s**.
- Edge, native-v8, benchmark and receipt selection (`-n 1`): **11 passed**.
- Native vector adapter + existing v2 policy selection: **8 passed**.
- Benchmark/manifest/shell selection: **27 passed**.
- The partial-cache timing regression first reproduced **1 failed, 2 passed**
  (`partial-cache-before-fix.log`), then passed in the complete gate after the
  timer moved into `finally`.
- `make lint`: **149 source/backend files** pass Black, plus isort and flake8.
  The focused **14 Python files** also pass Black/isort/flake8; Bash syntax and
  `git diff --check` pass.

The real Mac entrypoint ran with checked local input hashes and materialized
symlinks in `entrypoint-inputs`; it made four real decisions and exported both
worlds. The report verified the cold-fill boundary, first decision, zero build
time in steady stages, and a correctly incomplete requested 60-second window
because the two-frame fixture ended early:

```bash
PATH=/Users/josenunez/Projects/ml/snake-dqn/venv/bin:$PATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu T34_INPUT_DIR=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-batch-20261020/entrypoint-inputs T34_OUTPUT_DIR=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-batch-20261020/entrypoint-output T34_KERNEL=v2 T34_WORLDS=2 T34_FRAMES=2 T34_SECONDS=180 T34_STEADY_STATE_SECONDS=60 T34_NAMESPACE=t34-shadow-batch-entrypoint/v1 bash research/t34_exit_b8/bench.sh
```

Owned implementation files:

- `lockstep.py`: exact one-row-per-world assembly, transitions and projection.
- `teacher_v2.py`: lockstep shadows/roots/cache capture, identity guards and timing.
- `policy.py`: opt-in device Vector61 batching and nested forward counters.
- `benchmark.py`: cold/first-decision/steady receipt fields and rates.
- `bench.sh`: operator-selected steady-window forwarding.
- `make_manifest.py`: bounded workload/steady-window controls.
- `profile_shadow_batching.py`: pinned serial loader and immutable Mac profiling.
- `compare_shadow_profiles.py`: matched-shape/hash/replay comparison.
- `SHADOW_BATCHING.md`: results, commands, limits and operator handoff.
- `tests/fixtures/t34_shadow_batching/{teacher_v2_serial,policy_serial,engine_serial}.py`
  and its README: independent, hash-pinned original modules.
- `tests/test_t34_batch_benchmark.py`: benchmark/manifest timing contracts.
- `tests/test_t34_batch_vector_device.py`: native hook routing, isolation and v1 default.
- `tests/test_t34_batch_timing_collector.py`: collector cold/steady/partial-cache accounting.
- `tests/test_t34_lockstep_projection.py`: unequal food capacities and stale-tail writes.
- `tests/test_t34_shadow_batch_oracle.py`: all-mix and H300 doom/rescue byte oracles.
- `tests/test_t34_shadow_batching.py`: scale-aware near-tie recovery.
- `tests/test_t34_shadow_profile_receipts.py`: reject incomplete/vacuous proof.

## Final L4 drafts and operator commands

Final source pin: `b32f88f1c751f02c7f8b758d9a7647446e31fa69` (published).
The earlier `l4-{0,1}.json` drafts pin the profiling source; use the **final**
drafts below for any later operator dispatch. All drafts were validation only.
No runtime was allocated, retried or queried by this task.

| Final manifest | SHA-256 | Validation |
|---|---|---|
| `final-l4-0.json` | `3cddb7d4a9f1e1e5fb12b2886434c114647511617c125bf448fa0d37f7965f67` | PASS |
| `final-l4-1.json` | `c23310d6d6adea7e9e9f9bca9ff0f02e6a834263aa00e78cf6aa40a353292210` | PASS |

Both are in the artifact root above. Each draft requests L4 only, min 12 vCPU,
two workers × five Numba threads × W256, fp32 CUDA, all three mixes, H300,
5,000 frames, **1,500 s total workload / 1,200 s requested steady interval**,
15-minute estimate, **30-minute transport timeout / one-unit cap** at the
retained 2 units/hour planning rate. The first applicable cap wins. Cold fill,
setup or export can censor the interval; no cap increase or retry is implied.

Validation performed, once per shard with `--shard 0` and `--shard 1`:

```bash
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.t34_exit_b8.make_manifest --kind benchmark --kernel v2 --commit b32f88f1c751f02c7f8b758d9a7647446e31fa69 --shard 0 --attempt 3 --benchmark-seconds 1500 --steady-state-seconds 1200 --output-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-batch-20261020/final-l4-0-output --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-batch-20261020/final-l4-0.json --validate
```

Prepared dispatch commands for the operator (not executed here). The qualified
supervisor owns account-wide admission, leases, budget, exports and release:

```bash
set -C
/Users/josenunez/Projects/ml/snake-dqn-shadow-batch/research/colab_cli/launch.sh --check --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-batch-20261020/final-state-0 --colab /Users/josenunez/.local/bin/colab run /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-batch-20261020/final-l4-0.json
/Users/josenunez/Projects/ml/snake-dqn-shadow-batch/research/colab_cli/launch.sh --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-batch-20261020/final-state-0 --colab /Users/josenunez/.local/bin/colab run /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-batch-20261020/final-l4-0.json > /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-batch-20261020/final-l4-0-supervisor.log 2>&1 &
t34_a5_pid0=$!
/Users/josenunez/Projects/ml/snake-dqn-shadow-batch/research/colab_cli/launch.sh --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-batch-20261020/final-state-1 --colab /Users/josenunez/.local/bin/colab run /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-batch-20261020/final-l4-1.json > /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-batch-20261020/final-l4-1-supervisor.log 2>&1 &
t34_a5_pid1=$!
wait "$t34_a5_pid0"
wait "$t34_a5_pid1"
```

`--check` prints the launch plan; actual manifest validation was performed by
`make_manifest --validate`. After any operator run, inspect terminal supervisor
receipts, both worker reports, immutable exports and confirmed release. A CLI
exit code alone does not prove A5 acceptance or runtime release.

No live Colab/tmux operation, released checkpoint write, serving-pin edit or
change to frozen campaign/spec/receipt bytes was made by this implementation.
Independent upstream main changes were merged without editing those files.

Final verification artifact: `VERIFICATION_FINAL.json`, SHA-256 `c0f7e5ce529ca41b2643b6df422a686307f7b1ea3a9beeba9d962de5a5e782bf`.
