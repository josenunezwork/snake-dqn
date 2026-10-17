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
