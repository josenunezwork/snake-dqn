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
