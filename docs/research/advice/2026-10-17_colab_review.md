# Advice: review of the T16 Colab throughput notebook, changes made, and the runtime to pick (2026-10-17)

Advisory note with code (cloud advisor), requested by the owner through the local session.
Branch `colab-opt-20261017`; main untouched. Inputs: `notebooks/colab_throughput_smoke.ipynb`,
`research/t16_colab/{run_smoke,bundle,compare}.py`, the T16 runbook and spec, the trainer's
multi-actor loop (`train_m3c.py`: `started`, `log.jsonl` every 100k transitions, lazy learner
compile on the first update, synchronous checkpoint saves), `compile_m3c.py`, the perf notes.

## What was right

The bundle verifier, hash-pinned checkout, isolated venv, fresh-process runner, owned process
groups, resumable per-config rows and the conservative comparison are all sound. Nothing in
the self-play trainer path looked wrong for a throughput measurement: every live slot is
driven by one batched forward, each (env, slot) row has its own n-step window, rewards and
done flags are flattened per slot, the ring is sized for `worlds x 6`, and the unused vector61
runtime is skipped. The 6-per-world constant assumes `num_snakes: 6`, which is the config.

## What would have wasted credits or misled the number, and what I changed

1. **Compile and replay-fill sat inside the measured window.** `measured_rate` opened at 30 s of
   wall clock. The learner compiles lazily on its first update, and `reduce-overhead` on a cold
   inductor cache can take a minute, during which actors either run free or stall on
   back-pressure. Either way the 30-150 s window was not the trained rate. The window now
   opens at the later of 30 s and 20 s after the first telemetry row with a learner update,
   needs 30 s of both transitions and updates, and the rate over the last 60 s is reported
   beside it so a residual transient is visible. (`run_smoke.measured_rate`)
2. **Every scored run paid first-process compilation.** Numba kernels (`cache=True`) and the
   inductor FX graph are compiled per process on a fresh machine; eight actors starting
   together compile eight times. An unscored 60-second warm-up run now precedes the matrix
   and the runner sets `NUMBA_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR` and
   `TORCHINDUCTOR_FX_GRAPH_CACHE=1` to one local directory shared by all trainer processes.
   A marker file skips the warm-up on a resume within the same VM and redoes it after a reset.
3. **Drive FUSE was in the hot path.** The trainer's `--out` was the Drive run directory, so
   `log.jsonl` appends, `telemetry.jsonl`, the manifest and every checkpoint went through the
   Drive mount from inside the learner loop. Trainer outputs now go to local scratch
   (`/content/t16-colab-scratch`, `--scratch-dir`) and the attempt directory is copied to Drive
   after each run; the small results JSON still goes to Drive directly.
4. **A checkpoint every 100k transitions.** At 20k transitions/s that is a synchronous
   `torch.save` plus manifest write every five seconds. The smoke now uses
   `--checkpoint-every 10000000`; the hash-receipted final checkpoint is unchanged and remains
   the resume seed.
5. **The trainer bound left 45 s for startup and shutdown.** CUDA contexts for the learner and
   K actors, numba imports, ring allocation and the final save are outside
   `--max-wall-seconds`; with eight actors they can exceed 45 s, and the run would have been
   killed and marked failed after a full 150 s of timing. The grace is now 150 s and the
   budget check reserves it.
6. **The budget could not fit the matrix.** 1,200 s minus venv, pip and parity left room for
   about three of five timings, forcing a second "Run all" that repeats the whole setup. The
   notebook budget is 3,000 s (the runner caps at 3,600) for setup, parity, warm-up and six
   timings, about 30 minutes.
7. **A forced 2.5 GB torch reinstall.** The probe demanded exactly 2.9.1; Colab's preinstalled
   CUDA torch is often another 2.9.x. Any CUDA 2.9.x is now accepted and recorded; the
   cu126 2.9.1 wheel is installed only when the preinstalled build is unsuitable.
8. **No way to tell actor-bound from GPU-bound.** The decision the smoke feeds (which runtime,
   whether self-play is 4x) needs utilization, not only a rate. A small sampler
   (`research/t16_colab/sampler.py`) records `nvidia-smi` utilization and memory, CPU percent
   and load average every 5 s beside each timed run; mean and max land in the row, and
   `cpu_affinity` (`sched_getaffinity`) plus an `oversubscription` ratio (actors + 1 over
   vCPUs) are recorded so the 8-actor row can be read correctly on an 8-vCPU runtime.
9. **The self-play matrix had no actor scaling.** Self-play is the lane the smoke exists for;
   it ran only at 2 actors. `selfplay_w128_a4` is added (six timings in all), and the compare
   report carries the self-play rows' caveat: their transitions count every driven snake while
   the 5,400/s baseline is the Mac's hero-only recipe, so the ratio is wall-clock per training
   transition until a 150-second `--all-snakes-learner` run on the Mac gives the matching
   baseline. That Mac run is the one follow-up I would do before reading the self-play rows.

Tests: the T16 test files are updated for the new window logic, the warm-up, scratch
publication and the sampler (two new tests), and all T16 tests pass locally with black, isort
and flake8 clean. No trainer code changed. The notebook pins the reviewed runner commit in
`PINNED_COMMIT`; the notebook itself is fetched from the branch the owner opens it from.

## Which runtime

- **For this smoke: A100.** The learner is about 1.5M parameters and its update is
  launch-overhead-bound (8.4 ms on a 4090 with CUDA graphs); an H100 buys little there. What
  binds is vCPU count and per-core speed for the numba actors, and the A100 and H100 Colab
  runtimes have the same vCPU class. The A100 rate is roughly two thirds of the H100's. Read
  the live rate in Colab's resources panel; recent public figures are about 2 units/hour for
  T4, 2.5 to 5 for L4, 9 to 12 for A100 and about 13 for H100, so this smoke costs on the
  order of 5 to 6 units on an A100 and 7 on an H100.
- **Move to H100 only if** the A100 result shows GPU utilization above about 70% on the
  self-play rows with CPU utilization under about 80%; that is the signature of a GPU-bound
  batched forward, which an H100 relieves. If CPU is pinned and GPU is low, no GPU upgrade
  helps and the only lever is fewer actors per run with more runs per session.
- **L4** is fine for cheap hero-only runs if the smoke shows the learner never binds, but its
  runtime has fewer vCPUs, so it is a worse choice for self-play.
- **High-RAM CPU and TPU: no.** The actors need a CUDA device for the batched forward and the
  simulator is numba on x86 cores that are slower than the Mac's; a TPU runtime has no CUDA.
- **For the league runs later:** the same rule, decided by the smoke's utilization columns:
  A100 unless self-play is GPU-bound on it. Expect a 5-seed, 20M-transition league member set
  to cost tens of units on an A100 at the rates above.
