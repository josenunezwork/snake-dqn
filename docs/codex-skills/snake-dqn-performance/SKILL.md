---
name: snake-dqn-performance
description: |
  Use for any throughput, profiling or speed-up work in /Users/josenunez/Projects/ml/snake-dqn: making the
  M3-C trainer, actors, frozen pool, v8 kernels, featurizers, evaluation or the live simulator faster; reading
  or producing bench receipts (research/perf_*_20261013, perf_sim_20261007, actor_perf_20261017, t17b, t22,
  t24, t16_colab); deciding whether a GPU/pod/Colab run is worth it (4x rule). Encodes the byte-identical
  optimization discipline, the measured baselines, and the speed-ups that were rejected.
---

# Snake DQN: performance work (measured, byte-identical, honest)

New Colab performance work is a `research/colab_cli` supervisor job: retain the manifest, measured limits,
telemetry, immutable exports, and terminal release evidence. Current Drive/notebook workers remain Q1-only legacy;
do not make a new performance launch through them.

Rule zero: a speed-up that changes training semantics is a **new recipe**, not a speed-up. Rule one:
**measure before and after on the same host, same thread limits, warm-up excluded**, and write the
receipt (JSON with argv, platform, source SHA-256s) next to the bench. Rule two: prove byte identity
against the frozen pre-change code (see snake-dqn-parity-determinism).

## Where time goes (measured)

- **Mac (M5 Pro, MPS learner), B8 recipe, 128 worlds x 2 actors:** ~5.2-5.5k tr/s, **~21 upd/s**,
  learner-bound at every actor/world count (docs/research/t16_mac_baseline_2026-10-17.md,
  research/perf_quiet_20261013/RESULTS.md). ms/update: 35 baseline -> 30.4 `--learner-fast` -> 44.7 per
  256 tr at K=2 (1.57x) -> 85 per 512 tr at K=4 (1.64x: linear cost, no gain).
- **A100 (Colab, 12 vCPU/6 cores):** isolated learner fp32 bs2 **78-83 upd/s** (raw rings g3); full
  run a4 g3 ~62 upd/s = 3.0x Mac; a2 ~48. GPU work 8-11 ms hides under 19-23 ms of host work
  (sampling, numba `unpack_into` from mmap rings into pinned staging, Python ingest). Hero configs are
  actor-bound (5 opponent evals + sim per hero transition). N=3 league on A100 was **actor-bound by the
  frozen pool** (709 of 1,063 actor-s CPU-side v8 wrappers) -> T22 cut host dispatch 41.1 -> 28.7 ms/frame
  (frp3+v8 veto 6.45 -> 0.40 ms), projected a4 13.6k tr/s; not yet re-measured on A100.
- **L4 (Colab):** owner-reported GPU-bound at ~7.6k tr/s for N=3; **no receipt in the repo** as of
  2026-10-18 (Q0b L4 r5 cells pending, research/q0q1_20261017/Q0_PARALLEL_R5_STATUS_20261018.md). Cite only
  with the receipt once it lands.
- **RTX 4090 pod (x86 EPYC, torch 2.5.1):** learner 14.9 ms baseline -> 8.4 ms with `--learner-fast
  --compile-learner reduce-overhead` (1.74x; needs the *devel* image for a C compiler); EPYC cores ~2.4x
  slower than M5 Pro per actor -> pods want 3-4 actors.
- **Live simulator (perf-sim, merged 2026-10-07):** 2.1-2.4x per strict H5000 episode, byte-identical;
  pure-Python geometry (collision checks, per-action danger, flood fill) was the cost, not torch.
- **Gather:** `bench_gather.py` packed rings 3.0 ms serial -> 1.2-1.4 ms with 3-4 threads (host-only).
- RAM per run (pods): 16.2 GB default, 6.6 GB `--obs-pack`; bv8 12.4 / 5.5 GB.

## Tools (verified flags)

- `M3C_PROFILE=cprofile|pyinstrument` (+ `M3C_PROFILE_DIR`, `M3C_PROFILE_WARM_S`,
  `M3C_PROFILE_DURATION_S`, `M3C_PROFILE_WARM_FROM_FIRST_TICK=1`, `M3C_PROFILE_INTERVAL`): every
  train_m3c process dumps `<role>.pstats|.pyisession`, `.txt`, `.meta.json`
  (research/redesign_m3c_20261008/profile_m3c.py); analyse with
  `research/perf_profile_20261013/analyze_profiles.py`. py-spy needs root on macOS.
- Learner: `research/perf_learner_20261013/bench_learner.py --devices cuda|mps,cpu --configs
  baseline,t1,t1_bf16,t1_k2 [--compile default,reduce-overhead] --updates N --rounds R --raw-rings|
  --obs-pack --gather-threads G --ring-dir DIR --out X.json`; `--drift` measures precision/compile
  drift vs the fp32 1-ulp chaos floor. Rounds interleave configs; report the median.
- Gather only: `SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./venv/bin/python
  research/redesign_m3c_20261008/bench_gather.py [--packed] --actors 2 --warmup 30 --updates 300`.
- Actor: `research/perf_actor_20261013/bench_actor.py --configs baseline,vectorized,numba --worlds
  64,128,256 --devices cpu,mps --b8 [--grow-to L] [--profile]`.
- Frozen pool: `research/t22_frozen_pool_20261017/bench_frozen_pool.py --checkpoint-dir saved_snakes
  --out X.json [--reference]` (`--reference` loads league_pool.py from commit 4c1e1b2 via `git show`).
- v8 kernels: `research/t24_v8_bodymap_20261017/bench_bodymap.py --out X.json`,
  `research/actor_perf_20261017/bench_kernels.py --out X.json` (synthetic serpentines: length sensitivity).
- Live sim: `research/perf_sim_20261007/{episode_harness,ab_bench,compare_identity,analyze_profile}.py`.
- Pods: `research/perf_podbench_20261013/pod_bench.py` (spends money: `--confirm` + owner approval only).
- Always: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`,
  `SNAKE_DQN_DEVICE=cpu` for CPU cells, `nice -n 10`, note whether the host was quiet.

## Discipline for an optimization PR

1. Profile first; name the binder from telemetry (`learn_s` vs `learner_wait_s`, `frozen_forward_s`,
   `v8_s`, `featurizer_s`; nested timers, do not add parents to children).
2. **Skip before scale**: derive from the target definition what work can never affect the decision
   and remove it (missed 15x on T3 labels; T23 eval `T23_DESCRIPTIVE=0` runs only deciding checkpoints).
3. Freeze the oracle: vendor the pre-change module into `tests/fixtures/<tag>/` with its SHA-256, or load
   it with `git show <rev>:<path>` (bench_frozen_pool.py:60). Compare bytes, not tolerances; floats by
   `repr` (tests/test_perf_sim_identity.py). Cover the regimes the proof depends on (long bodies, wraps,
   duplicates, dead rows, ties).
4. Keep the serial path as the default and the oracle (`--gather-threads 0`, `--actor-impl reference`,
   `--opp-forward per-group`); workers partition contiguous destination rows in a fixed order so
   serial == parallel bytes.
5. Process pools for CPU-bound episode loops (eval worlds), one world per worker, merged in a fixed order,
   with a serial==parallel SHA test (docs/research/q0_mass_worker_parity_20261018.json pattern).
6. numba is approved: `@njit` kernels with NumPy reference twins pinned equal in tests (obs_pack `pack_ref`
   vs `pack`, v8_nb vs live v8). Thread-local scratch (f7e74417), stamps cleared before rollover.
7. Measure after, same host, >= 3 rounds; publish a receipt table with argv and source hashes; state
   clearly what is a conditional projection vs a measurement.
8. Decide platform with the **4x per-step rule** (projected wall-clock >= 4x the Mac for that step, incl.
   setup/transfer) and "cheap pods first"; strict gates and serving qualification stay on the Mac.

## What NOT to do (already tried or rejected)

- `--update-every 2` as a speed-up: halves the replay ratio; recipe change (T16 diagnosis).
- `--batch-scale 4`: linear in cost at fixed replay ratio (Mac 85 ms/512, A100 24 ms/512) - no gain;
  K=2 was adopted only after a paired A/B showed no quality loss (perf_quiet RESULTS.md).
- bf16/fp16 without a pre-registered screen: 117 upd/s isolated on A100 but 47.6 in the full run
  (unexplained); slower than fp32 on MPS and 4090; numerics change regardless.
- `torch.compile` on MPS (torch 2.9.1): wrong conv-bias grads -> auto-fallback; `fallback_eager`
  receipts are not compiled timings. `max-autotune-no-cudagraphs` failed the first-steps check on CUDA.
- More actors when learner-bound (a2 == a4 upd/s; policy lag doubles). More worlds likewise on the Mac.
- `--opp-forward merged-device` on MPS: slower; `merged-cpu` saves ~0.2 ms of a 17 ms frame.
- One fused online(next)+online(current) forward, `torch.func.vmap` over stacked nets, `channels_last`:
  measured losses on MPS (learner_m3c.py docstring "Not done").
- Batched batch-N torch forwards inside pinned veto/eval paths: Q differs from batch-1 by ~6e-6, flips
  ties; NumPy `arctan2` vs libm `atan2` 1-ulp sector flips (perf_plan 2026-10-07 §2).
- H100 "because GPU": today GPU is a minority of each step; switch only if a post-fix profile shows
  kernel-busy > 80% with tr/s still below the 4x bar.
- Counting tr/s in self-play rows (every driven snake) against a hero-only baseline.
- Running a 4 h Colab job on one core (Q0 audit) or a per-iteration `bincount` over a 100k deque.
