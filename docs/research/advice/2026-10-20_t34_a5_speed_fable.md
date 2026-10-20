# Fable systems answer: where the 35 ms/world-step goes (2026-10-20 ~14:05 PT)

Verified: `final-l4-*-output/poll-*.json` (row-steps 256 -> 38,912 in 1,381 s, in steps of exactly 14x256 per ~113 s:
one W256 lockstep round ~8.1 s = 31.5 ms/world-step, done 0/256); `t34-a5-20261020/.../benchmark-lane*.json` (serial:
24.8 row-steps/s/worker, `policy_s` = 4.35 ms per CUDA batch-1 evaluate, GPU util 2-8%, RSS 2.0 GB);
`PROFILE_SUMMARY.json`; the teacher/lockstep/engine/policy/shadow sources, `bench.sh`, `benchmark.py`.

Two read-only Mac probes (1 thread, nice 10, host load ~12, 8 worlds x 30 steps, <6 s each; not kept):

| serial path, Mac M5 Pro (loaded) | ms/row-step | share |
|---|---:|---:|
| `full_digest` per ShadowFrame (`shadow.py:142`) | 3.73 | 64% |
| `teacher.evaluate` (bv8 raster + batch-1 CPU forward 0.46) | 0.93 | 16% |
| opponent dispatch (v8 `_apply_veto` hooks, Python) | 0.76 | 13% |
| `sim.step` (numba) | 0.27 | 5% |
| prepare + observe | 0.16 | 3% |
| **total** | **5.84** | |

Lockstep path at W=8: 6.78 ms = digest 3.79 + `step_prepared_worlds` 2.20 + evaluate 0.72. The merge/scatter seam
(`_merge_prepared_worlds`, `_copy_environment`, `_restore_policy_carry`: ~50k `deepcopy` calls per round) costs more
than the serial dispatch+step it replaces (2.2 vs 1.0 ms): hence Mac -19% and Colab "28 vs 25". Batching buys nothing
because <6% of the work is simulator and <10% is network.

Digest composition: 4.1k-5.1k recursive `_update_value` visits, ~10k `hashlib.update` calls per digest. Frozen/mixed
opponents' carry 663-1,091 visits (2.1-3.5 ms), `food_cells` ~900, `sim` dict 1,678, two MT RNG states 1,259 (625-int
tuples hashed one `repr` at a time). The 42 sim ndarrays (1.28 MB) are single `tobytes` updates: cheap.

## Where the Colab 35 ms goes (estimate)

The job is ~95% single-threaded CPython (hash walk, deepcopy, v8 hook Python, raster glue). Nothing uses the L4
(2-8% util); CUDA makes `evaluate` slower (4.35 ms with H2D/D2H sync vs 0.93 ms CPU). Per world-step on L4: digest
~14-18 ms, W256 merge/scatter ~8-10 ms, CUDA evaluate ~4 ms, dispatch/hooks ~3 ms, sim <1 ms. Multipliers over the Mac
5.8 ms: (a) Colab x86 vCPU ~2.5-3x slower per Python op than an M5 Pro core; (b) 2 workers x 5 Numba threads + torch
on 6 physical cores oversubscribe the one thread that matters, ~1.3-1.5x; (c) CUDA batch-1 sync, +3.5 ms. That
reproduces 30-35 ms with no exotic cause. Only 2 of 12 vCPU do useful work.

Structural: W=256 worlds in lockstep need 256 x ~13,400 row-steps (86,008/32 per 1,000 frames x 5) / 28 per s = ~34 h
before the first world completes. The A5 "completed worlds/h in 25 min" rule is unattainable at W=256 at any plausible
rate; "0 done" is a layout artifact, not a speed signal.

## Fixes, by payoff per line of code

1. **Process parallelism, not threads or GPU.** One single-thread serial worker per vCPU (12 on L4), `--device cpu`,
   `NUMBA/OMP/MKL=1`, `W=4` worlds per process (48 resident per lane; each process completes a world every ~10-25 min
   and refills from the row iterator). Expected 6 cores x ~1.25 HT x 40-50 row-steps/s (single process, CPU forward,
   no contention) = 300-370 row-steps/s per lane -> 80-100 completed worlds/h per lane (now 56 row-steps/s = 15
   worlds/h). Needs `bench.sh` to loop over `T34_WORKERS` instead of `for lane in 0 1`, and relaxing the A5 guards at
   `benchmark.py:224-238` (threads==5, W==256, device==cuda). RAM ~0.7 GB/process; share `NUMBA_CACHE_DIR`.
2. **Make `full_digest` O(bytes), not O(Python objects)** (export-byte-neutral: exports hash via `exports.state_hash`
   blake2b; `full_digest` is only compared with itself). Encode `food_cells`/`corpse_cells` as one
   `np.asarray(...).tobytes()`, MT states as `np.asarray(state[1], uint32).tobytes()` + index, opponents'/runtime carry
   via a flat leaf encoder (or memoize per policy on a mutation counter). Target 4-5 ms -> <0.7 ms; serial row-step
   5.8 -> ~2.7 ms (2.2x). The digest is also paid on every root confirm (`root_step_s` 217 of 633 s on Mac) and every
   doom-cache frame. Verify with the existing matched `profile_shadow_batching` comparison (96 files, 32 replays).
3. **Route CPU workers through the serial per-world stepper** (skip `_catch_up_lockstep`'s merge when `device=cpu`;
   the serial fixtures are the verified oracle): +20% and removes the W-superlinear deepcopy.
4. **Do not pay for the GPU.** A CPU-only runtime with >=4 vCPU beats the L4 per unit (L4 ~1.6-2.0 u/h for 12 vCPU;
   CPU <=0.5 u/h). Keep the L4 only if `nproc` there is 2.

Combined 1+2+3 per L4 lane: ~650-800 row-steps/s -> ~175-215 worlds/h -> Round 0 (240 worlds) + Path B in ~1.5-2.5
lane-hours, ~3-5 units. Fix 1 alone probably clears 50/h, not comfortably.

## Single cheapest confirming measurement (<0.3 unit)

One CPU-only Colab session, 15 min: record `nproc` and `lscpu` model; run the v2 kernel `T34_RUNTIME_VARIANT=cpu
--threads 1 --worlds 4 --frames 1000 --seconds 420` once alone, then K=`nproc` copies concurrently (after the bench.sh
fan-out). Read `active_row_steps_per_second` per process. Expected: alone 25-50/s (Mac 136); K-way sum 4-8x the single
rate. This pins the x86 factor, the contention factor and the per-vCPU-hour value of CPU runtime vs L4 (fix 4). If
alone is <20/s, add a 60 s `cProfile`; the top entries will be `_update_value`/`hashlib.update` or `deepcopy`.

## Exact runtime layout to dispatch next

- Runtime: Colab **CPU-only (high-RAM) if nproc>=8**, else **L4**; never A100/H100 for this job.
- Processes: `K = nproc` (12 on L4) independent `benchmark.py --kernel v2 --device cpu --threads 1 --worlds 4
  --max-worlds <rows/K>`, each on a disjoint row shard (per-worker seed namespace), `OMP/MKL/OPENBLAS/NUMBA=1`.
- Session wall 60 min (manifest timeout 75), `T34_SECONDS=3300`; exports are already per completed world.
- GPU: none. Hero and Vector61 forwards batch-1 on CPU (0.5 ms, below the raster cost). Same layout after fix 2.
