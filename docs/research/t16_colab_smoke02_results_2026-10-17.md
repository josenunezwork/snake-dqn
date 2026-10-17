# T16 Colab smoke-02 results and diagnosis (2026-10-17)

Throughput evidence only; not gate, quality or serving evidence. Runtime: Colab Pro+ A100-SXM4-40GB,
12 vCPU / 6 physical cores, Python 3.12 venv, torch 2.9.1+cu126, runner pin `6b7e280`. Parity tests
passed (2/2). Raw data: Drive `t16_colab/smoke-02/` (results JSON, per-config telemetry and logs).
Diagnosis by two local Fable reviewers (GPU/H100 and RL systems; reports reconciled and agreed).

## Results (150 s per config, steady-state window)

| config | tr/s | vs Mac | upd/s | cpu% | gpu% mean/max | bound by |
|---|---|---|---|---|---|---|
| hero_w128_a2 | 4,062 | 0.77x | 15.9 | 21 | 29/61 | actors |
| hero_w128_a4 | 8,014 | 1.5x | 31.3 | 39 | 45/73 | balanced |
| hero_w128_a8 | 9,138 | 1.7x | 35.7 | 60 | 53/75 | learner (+CPU contention) |
| selfplay_w64_a2 | 11,086 | 2.0x | 43.3 | 24 | 54/75 | learner |
| selfplay_w128_a2 | ~11,500 | 2.1x | ~45 | 17 | 42/70 | learner (row invalid: actor did not exit cleanly) |
| selfplay_w128_a4 | 11,341 | 2.1x | ~44 | 15 | 34/71 | learner (row invalid: same) |

Mac baseline: 5,283 hero / 5,459 self-play tr/s, learner-bound at ~21 upd/s.

## Diagnosis

- **Right metric is updates/s, not transitions/s.** `learner_m3c.py:655` fixes 256 transitions per
  update (replay ratio 3 gradient samples per transition). Honest 4x = >= ~84 upd/s. Colab = 2.1x.
- **Self-play is learner-bound by the learner's single host thread, not the GPU.** ~19-23 ms/update:
  rejection sampling, per-ring numba `unpack_into` from file-backed mmap rings into pinned staging
  (~22-25 MB/update), Python ingest, and a 100k-deque bincount every loop iteration
  (`train_m3c.py:3305-3308`). GPU work is ~8-11 ms and hides under it. CUDA graphs replay fine (1 graph,
  0 breaks).
- **More actors do not help** (a2 = a4; a4 doubles policy lag 4.4 -> 9.0). Hero configs are actor-bound
  (5 opponent evals + sim per hero transition) and cannot reach 4x on 6 cores.
- **H100 would not help today** (same vCPU class; GPU is a minority of each step). After the host fix the
  A100 ceiling is ~23-25k tr/s; H100 maybe 28-32k. Switch only if a post-fix profile shows kernel-busy > 80%
  with tr/s < 21.6k.

## Plan

1. Move gather off the learner thread: thread-pool the nogil per-ring unpacks / copies (cheap), GPU-resident
   packed ring with on-device gather + unpack as follow-up. Byte-identical batches required.
2. Ingest cleanup: collapse bincount to log cadence; no `tolist()` + deque per message.
3. Fix "actor did not exit cleanly" on w128 self-play.
4. Re-smoke on A100: selfplay_w128_a2/a3, `--actor-queue-frames 2-4`, >= 10 min, with `M3C_PROFILE`;
   report upd/s and policy lag beside tr/s.
5. Lane A league mode (N learner slots + frozen pool) is a prerequisite and must be re-measured.

Rejected as a "speed-up": `--update-every 2` (halves the replay ratio). `--batch-scale 4 --lr-scale sqrt` only
as a pre-registered arm after steps 1-2.

## T17 implementation and smoke-03 pin (2026-10-17)

Smoke-03 implementation pin: `fd3f0fc69214a6f42db591061ee9f5142f1d6e3a` (rebased on
`origin/main`; the notebook pins this implementation commit). P1–P4 are implemented:
opt-in persistent gather workers, exact running action-window counts, stop-aware actor
puts with continuous queue draining, and the four 600-second smoke-03 configurations.
The default gather remains serial and the T16 suite remains the runner default. Sampling
draw order, replay ratio, batch sizes, optimizer, precision, and recipe devices are unchanged.
The profile attempt records 60 active seconds after a 60-second loop warm-up and requires
hash-verified learner profile artifacts; truncated captures cannot be resumed as complete.

CPU micro-benchmark on Apple M5 Pro, Python 3.12.3: synthetic **anonymous mmap** rings with
nonzero pack-valid observations, `--obs-pack`, 768 agent + 256 on-policy label rows,
30 warm-up calls and 300 measured `_sample_fast` calls per setting. These are host-gather
times, including serial sampling, with no GPU transfer or training. Existing background
jobs were left running; timings are descriptive, not an isolated-machine throughput claim.

| Gather threads | 2 actors median ms/update | 2 actors mean | 3 actors median ms/update | 3 actors mean |
|---:|---:|---:|---:|---:|
| 0 (serial) | 3.113 | 3.128 | 3.448 | 3.486 |
| 2 | 1.679 | 1.686 | 1.962 | 1.977 |
| 3 | 1.791 | 1.827 | 1.389 | 1.423 |
| 4 | 1.205 | 1.252 | 1.403 | 1.411 |

Reproduce with `SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
./venv/bin/python research/redesign_m3c_20261008/bench_gather.py --packed --actors 2
--warmup 30 --updates 300` (repeat with `--actors 3`).

Validation: the requested `pytest tests/ -k "obs_pack or m3c or t16" -q` selection passed
**172 tests**, with two archived-module collection skips and the explicit MPS test deselected
to keep execution on CPU. Final focused validation passed **97 tests**, including all **40 T16
tests**, the new gather/shutdown tests, and profiler regressions (one MPS test deselected).
The frozen pre-T17 oracle checks replay and
label indices, RNG state, full staging bytes (including padding), and compaction across
packed/unpacked, wrapped 1/2/3-actor rings and 2/3/4 gather threads. Tiny-queue actors exit
with code 0; collapse decisions match the deque reference at the same transition counts.
Black, isort, flake8, and diff whitespace checks pass for changed Python files.

Supplemental learner/actor/bank/compile checks produced **71 passed, 13 failed**: all failures
were CPU Inductor compilation/loading checks in `test_perf_compile.py`. The local toolchain
cannot resolve `@rpath/libc++.1.dylib` (no LC_RPATH), independently reproduced by compiling
`lambda x: x + 1`, including with a fresh private cache. This does not qualify CUDA compilation.
No shared environment was modified, no input tarball was rebuilt, and no tmux jobs were touched.
The notebook is ready for the owner-run Colab smoke-03; **84 updates/s is not yet demonstrated**.


## T17b advisor merge, row chunks, and Mac reference (2026-10-17)

T17b supersedes the four-row T17 smoke-03 plan above. The advisor branch
`origin/claude/upbeat-allen-uvxqvd` was merged on `t17b-20261017`, followed by a
merge-preserving rebase on `origin/main`. Merge conflicts in the trainer shutdown,
smoke runner, comparison, notebook and comparison tests were resolved by retaining
one bounded drain/join implementation and one suite/matrix implementation.
The trainer defaults and batch bytes remain unchanged; opt-in workers now partition
contiguous destination rows across rings. The frozen pre-T17 oracle checks full staging
bytes, RNG state, rejected-row compaction and label re-checks for two rings with g3/g4.

The notebook's final implementation pin is
`ef32fb99e7691fa3e716083a0abb94adf3eea2a0`, with `RUN_NAME = 'smoke-03'`.
Its seven registered rows match B4. Under the 3,600-second runner cap, the conservative
3,785-second seven-row allowance selects the permitted six-row fallback (3,335 seconds),
omitting only row 5 (a6). Reconnects retain the chosen row set. Row 4 also has a separate
unscored 60-second profile after 60 seconds of warm-up. P0 runs first under its own
300-second deadline: K2/K4, fp32/bf16, reduce-overhead, raw/packed, g0/g3 and
`/dev/shm`/local-disk rings. Runtime-bound Drive receipts and owned temporary ring
directories preserve results and clean up interrupted P0 attempts. No CUDA/Colab execution
was performed in this task; 84 updates/s remains unproven.

B2 synthetic host-only gather on this Apple M5 Pro (two rings, 300 measured samples):

| Rings | g0 median ms | g3 median ms | g4 median ms |
|---|---:|---:|---:|
| Raw | 3.222 | 1.682 | 1.744 |
| Packed | 3.024 | 1.363 | 1.214 |

B5 full MPS learner, fp32 K2, actual local file-backed rings (40,000 rows/ring,
20 updates x 3 rounds, five separate device-event diagnostic updates):

| Rings | Threads | Host gather ms | Device-event leg ms | Wall ms/update | Updates/s |
|---|---:|---:|---:|---:|---:|
| Raw | 0 | 5.459 | 50.024 | 50.49 | 19.807 |
| Raw | 3 | 2.848 | 50.944 | 50.53 | 19.790 |
| Packed | 0 | 4.250 | 50.270 | 49.92 | 20.031 |
| Packed | 3 | 2.376 | 47.071 | 47.81 | 20.917 |

Host gather is measured in the overlapped update loop; device-event legs are separate,
serialized diagnostics, so they are not additive. The matrix also measures fp32 K4 and
bf16 K2/K4 for each ring/thread variant. Requested MPS reduce-overhead cells hit the
compiled/eager numerical verification guard and fell back; they are explicitly classified
as `fallback_eager`, not compiled evidence. The table uses the separately measured eager
reference cells. These shared-machine timings do not establish a change to the ~21 upd/s
Mac baseline. Existing tmux jobs were untouched. macOS has no `/dev/shm` equivalent in this
run, so B5 measures local file-backed rings only; CUDA P0 retains both storage backends.

Reproducible B2 commands and B5 raw JSON receipts are under
[`research/t17b_20261017/results`](../../research/t17b_20261017/results/).
The original implementation and advisory measurements above remain historical evidence.


Final validation on the rebased implementation: **229 passed, 2 archived-module collection
skips**, with the broad related selection below; **60 additional learner/actor/benchmark
regressions passed**, including the real two-actor CPU fast-learner smoke (12,000 transitions,
clean actor exits and final/checkpoint writes). The standalone MPS unit test was deselected;
B5 separately exercised MPS. The 34 gather tests preserve byte identity, and the notebook
orchestration tests cover all eight P0 invocations, timeout cleanup and stable resume choices.
All changed Python files pass Black, isort and flake8; notebook cells compile and whitespace
checks pass. Independent review found no outstanding actionable defect. No tmux commands
were issued.

```bash
./venv/bin/python -m pytest tests/ -q -m 'not slow' \
  -k '(obs_pack or m3c or t16 or t17) and not mps'
./venv/bin/python -m pytest -q tests/test_perf_learner.py tests/test_perf_actor.py \
  tests/test_perf_podbench.py -k 'not mps'
```

Full B5 numbers, compile-fallback receipts, source hashes and final test transcripts are
in the results directory linked above. The publication commit only adds these receipts,
this results append and the notebook pin; executable research code matches the pinned
implementation commit.

## smoke-03 results (A100, 2026-10-17, runner commit ef32fb9, 300 s rows)

P0 in the notebook crashed (packed bench read a Mac-only path; fixed on main 42dae60, notebook now non-fatal),
so the matrix was launched by hand on the same runtime; the fixed P0 bench ran afterwards (commit 7733aeb).

| row (self-play, 128 worlds) | tr/s | upd/s | vs Mac | GPU % |
|---|---|---|---|---|
| a2 packed g0 bs2 fp32 (control) | 12,249 | 47.8 | 2.3x | 63 |
| a2 raw g0 bs2 fp32 | 12,294 | 48.0 | 2.3x | 61 |
| a3 raw g3 bs2 fp32 | 15,071 | 58.9 | 2.8x | 81 |
| a4 raw g3 bs2 fp32 | 15,927 | 62.2 | 3.0x | 84 |
| a4 raw g3 bs4 fp32 | 15,606 | 30.5 (512 tr/upd) | 2.9x | 78 |
| a4 raw g3 bs2 bf16 | 12,189 | 47.6 | 2.3x | 53 |

compare.py marks every row INVALID only because the runner does not lift `policy_lag_mean` into the row (bug);
lag is in telemetry (a3 ~1, a4 ~4.8). No row reaches 84 upd/s.

Isolated learner bench on the same A100 (no actors; median ms/update, upd/s):

| rings / gather threads | bs2 fp32 | bs2 bf16 | bs4 fp32 |
|---|---|---|---|
| packed g0 | 17.8 ms, 56 | 17.7 ms, 56 | 31.9 ms, 31 |
| packed g3 | 12.9 ms, 78 | 10.1 ms, 99 | 24.4 ms, 41 |
| raw g0 | 12.9 ms, 77 | 11.6 ms, 86 | 24.6 ms, 41 |
| raw g3 | 12.8 ms, 78 | 8.5 ms, 117 | 24.3 ms, 41 |

Reading: the A100 learner at the unchanged fp32 bs2 recipe caps at ~78 upd/s even alone, and ~62 in the full run
(actor forwards share the GPU). bs4 is linear in cost (no gain). bf16 is 117 upd/s alone but only 47.6 in the full
run (unexplained; GPU 53%, so something else binds — investigate before any bf16 conclusion; bf16 is a numerics
change anyway). Conclusion: 4x at the fixed fp32 recipe is not reachable on the A100; options are an H100 smoke
(same recipe), and taking actor forwards off the learner's GPU timeline.

## Mac N=3 static-league baseline (2026-10-17, main 650217f, `research/t16_colab/mac_n3_baseline.sh`)

MPS learner, 2 actors, 128 worlds, `--learner-seats 3`, static six-policy pool, 120 s warm-up, 268 s window:
**5,214 learner transitions/s, 20.4 updates/s, 256.0 transitions/update, policy lag mean 0.75 / p95 1**.
Learner-bound (learn_s 382 of 406 s wall; 46.5 ms/update). 4x bar for N=3 = 20,855 tr/s = 81.5 updates/s.
Receipt: `snake-dqn-artifacts/t16-mac-n3-baseline/n3_w128_a2_mps_static_pool/throughput.json`.
H100 pod runs paused: the pod-side self-delete probe returns 403 (owner decision pending); no pods running.

## A100 N=3 league throughput (Colab A100-SXM4-80GB, 12 vCPU, 2026-10-17, run a100-n3-01, commit a049e92)

Isolated learner (P0, raw rings, 3 gather threads): fp32 bs2 **82.7 upd/s** (11.9 ms GPU leg, 5.0 ms host),
bf16 bs2 118 upd/s, fp32 bs4 43.8 upd/s (512/update).

| N=3 row (128 worlds, static six-policy pool) | tr/s | upd/s | vs Mac N=3 (5,214) | lag mean/p95 | GPU % | bound |
|---|---|---|---|---|---|---|
| a3 | 4,396 | 17.2 | 0.84x | 0.43 / 1 | 27 | actors |
| a4 | 5,938 | 23.2 | 1.14x | 0.57 / 1 | 36 | actors |
| a6 (extra 240 s cell) | ~7,720 | 30.2 | 1.48x | 0.88 / 1 | - | actors |

The frozen pool is the binder: frozen_forward_s 709 s of 1,063 s actor time at a4, of which GPU 6.2 s, i.e.
CPU-side opponent features / veto chains. Fix handed to Codex (t22: batch pool policies across worlds,
bit-identical). The A100 learner itself sits just under the fixed-recipe 4x bar (82.7 vs 84.4 upd/s isolated).

## T22 frozen-pool CPU optimization and independent action parity (2026-10-17)

**Implemented, CPU-qualified, and throughput-only. The 20,900 tr/s A100 target is
not established.** Branch `t22-frozen-pool-perf-20261017` started at `4c1e1b2` and
was rebased on main after the T23 registration/launcher additions. Only bounded
benchmarks and CPU tests (including tiny training fixtures) ran; no long training,
Colab execution, pod operation, serving change, or tmux interaction was performed.

### Attribution and implementation

The pre-change dispatcher already grouped every policy's live rows across all
worlds into one network call. The expensive part was the live Python v8 wrapper
inside the two wrapped vector61 policies. It rebuilt decision-time vector61
free-space features, converted each row to live snake views, and ran the veto
chain separately. The N=3 runtime also featurized all frozen seats, including B8
and scripted rows that never consumed vector61 states.

The pool now uses the existing Numba v8 kernel for those two wrappers, retains
its carried advisory masks and the original network batch shapes, and limits
vector61 carry/feature construction to vector61-assigned seats. Learner and B8
base rasters are gathered together and sliced into their original row orders;
B8's v8 scalars remain separately computed with its original selection mask.
The transient raster cache checks simulator identity, exact rows and frame,
and is cleared on empty calls and segment rebinding. Runtime pruning is also
applied after snapshot restore. The random-safe policy caches its immutable
key prefixes and batches canonical SHA-256 selection in Numba, with a legacy
fallback for values outside the compiled integer contract. Greedy food already
used a compiled kernel. Source hashes include the new execution paths.

The final instrumented Mac run uses CPU and one thread for Torch, Numba and
BLAS: 128 worlds, N=3, 200 unmeasured warm-up frames and 30 measured frames.
Each of the six fixed groups has 64 live rows per measured frame. Learner
seats move straight; this is a bounded actor microbenchmark, not a training
throughput run or an isolated-machine/late-game performance claim.

| Pool policy | Before ms/frame | After ms/frame | Attribution |
|---|---:|---:|---|
| B8-s3 | 28.307 | 27.491 | CPU network forward 26.651 / 26.847 ms; runs on GPU in Colab |
| FRP3-s12 bare | 0.230 | 0.237 | Forward 0.181 / 0.188 ms; no meaningful change |
| FRP3-s12 + v8 | 6.449 | 0.397 | Old veto 6.256 ms includes 0.457 ms decision features; new complete veto kernel 0.195 ms |
| Champion + v8 | 5.846 | 0.337 | Old veto 5.606 ms includes 0.411 ms decision features; new complete veto kernel 0.150 ms |
| Scripted greedy food | 0.062 | 0.063 | Existing compiled path retained |
| Scripted random safe | 0.082 | 0.036 | Canonical hash and tie order retained |
| Entire frozen dispatch | **41.109** | **28.687** | **30.2% less CPU dispatch time**; includes CPU B8 forward |

B8's old raster construction costs 1.510 ms, including 0.415 ms of v8 scalars.
The new B8 v8 scalars cost 0.421 ms. Separate learner base-raster construction
costs 3.980 ms; the combined learner+B8 gather costs 4.294 ms. Thus the B8 work
moved outside `frozen_forward_s` is charged back in the projection below; it is
not treated as eliminated. Production telemetry records the combined gather
under `featurizer_s`. Phase measurements are nested wall times, so inclusive
parents and their children must not be added together.

Shared vector61 preparation/mask retrieval costs 0.143 → 0.106 ms/frame;
post-step carry costs 2.276 → 1.739 ms/frame with controlled rows reduced
384 → 192. This work is outside dispatch and is not arbitrarily divided among
policies. No extra credit for this saving is included in the A100 estimate.
The two vector policies retain separate network forwards; there is no change
in BLAS batch shape, precision, masks, learner math, or network weights.

### A100 projection and its limit

The projection uses the saved `a100-n3-01` Drive data, matching the telemetry
and log counters by transition count for each row:

| Actors | Baseline tr/s | Actor env seconds | Frozen seconds | Frozen GPU seconds | Host-pool fraction | Projected tr/s |
|---|---:|---:|---:|---:|---:|---:|
| a4 | 5,937.953 | 1,063.044 | 709.054 | 6.199 | 66.12% | **13,578** |
| a6 | ~7,720 | 1,189.492 | 677.422 | 7.159 | 56.35% | **14,833** |

The a4 baseline is the saved scored result. The a6 baseline is the owner's
approximate reference rate, not the summary's end-to-end rate or a newly
selected steady-state window; its projection scales linearly with that input.
Using a4's frozen fraction for a6 would overstate a6's gain.

The host-only Mac pool estimate removes B8's CPU network forward and charges
incremental shared raster cost:

```text
old_host = 41.108593 - 26.650590 = 14.458003 ms
new_host = 28.687489 - 26.846687 + max(0, 4.293569 - 3.980440) = 2.153931 ms
r = new_host / old_host = 0.148978
h = (frozen_seconds - frozen_GPU_seconds) / actor_env_seconds
projected_rate = baseline_rate / (1 - h + h*r)
```

This assumes the relative Mac host speedup transfers to x86, cumulative actor
time fractions approximate the scored window, and GPU/other actor costs stay
unchanged. It includes no setup, transfer or GPU-contention penalty. These are
conditional estimates, not A100 measurements. Even zero host-pool cost caps
this restricted model at about 17,525 / 17,686 tr/s for the stated a4/a6 rates.
The remaining actor work therefore matters. The task's rounded target of
20,900 tr/s is 81.64 updates/s at 256 transitions/update (4×5,214 is 20,856),
not the older 84.4-updates/s target referenced above. The isolated 82.7
updates/s learner has only about 1.3% capacity above the rounded target before
shared-GPU overhead. A new Colab throughput run is needed to establish any
actual end-to-end gain.

### Parity, verification, and reproduction

The independent test oracle vendors the unedited pre-change pool module from
`4c1e1b2`, SHA-256 `18c3b2e7257af0410069d0f344f3bbe37dbee0f08d5a8e53938b87667575919c`.
Two separate simulators/runtimes run 96 frames in actual prepared-step callbacks
with all six real pinned policies. Every identity is forced through death and
normal respawn. The test compares actions, learner raster/v8 bytes, relevant
selection/carry/trend memory, world state, and world RNG state. The pre-change
runtime controls all frozen seats while the optimized runtime controls only
vector61 seats. The recorded 96×12×6 action stream has SHA-256
`f7aa8c96878a428a483704b13c1654056ea8d10a16c20d9502fd29a0037d8be9`.
Additional checks cover tied Q values on every slot, stale frame caches,
snapshot delegation, random-safe masks/SHA padding/integer limits, and rebinds.
This establishes the tested Mac CPU action identity; it is not cross-architecture
or CUDA action-parity qualification.

The final bounded CPU suite passed **224 tests, six slow tests deselected, no
artifact-dependent skips**, covering T16/T19/T21/T22, actor resume, the trainer,
existing v8/vector61 parity, and raster contracts. Changed Python files pass
Black, isort, flake8 and whitespace checks. The notebook already contained no
`files.download` call at the starting commit; a regression assertion and a
Drive-results comment now keep that requirement explicit.

Commands, exact timing/source/checkpoint receipts, the projection script and
hashed Drive inputs, and the test transcript are in
[`research/t22_frozen_pool_20261017`](../../research/t22_frozen_pool_20261017/README.md).
The independent recorded fixture is in
[`tests/fixtures/t22`](../../tests/fixtures/t22/recording.json).
