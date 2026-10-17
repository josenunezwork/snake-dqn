# T17 spec: take the replay gather off the learner thread, then re-smoke Colab (2026-10-17)

Context: `docs/research/t16_colab_smoke02_results_2026-10-17.md`. Colab A100 self-play is learner-bound at
~45 updates/s (2.1x the Mac) because one host Python thread does replay sampling, obs-pack unpack and
ingest (~19-23 ms/update) while the GPU's share is ~8-11 ms. Goal: >= 84 updates/s at the UNCHANGED recipe
(256 transitions per update, batch 768 agent + 256 anchor rows). Throughput work only: no change to what is
learned.

Work in this worktree (`../snake-dqn-t17`, branch `t17-colab-gather-20261017`). Commit often; when tests
pass, merge to main and push (`git push origin HEAD:main` after rebasing on origin/main). Do not touch
running tmux jobs. You (Codex) run on CPU for tests; do not change any recipe device defaults.

## Invariants (hard)

1. **Batch identity.** For a fixed seed and replay contents, every sampled batch (indices, all tensors
   uploaded to the device, labels, masks) must be byte-identical to the current single-threaded path. The
   RNG draw order of `Replay.sample` / rejection sampling must not change. Add a test that runs the old and
   new gather on the same rings (obs-pack on and off, several ring layouts incl. multi-actor rings) and
   asserts byte equality of the staging buffers.
2. No change to replay ratio, batch sizes, lr, target sync, n-step, loss, compile scope, precision.
3. Defaults stay bit-identical to today unless a new flag is set; the Mac recipe must be unaffected unless
   the flag is passed (then it must also be identical in output, just faster).

## P1. Parallel gather (`research/redesign_m3c_20261008/learner_m3c.py`)

- `_gather` (around :1073) and `_sample_fast` (around :1101): split the per-ring work (fancy-index reads
  from the mmap rings and numba `unpack_into`, which is `nogil=True`) across a persistent
  `concurrent.futures.ThreadPoolExecutor` sized by a new flag `--gather-threads N` (default 0 = current
  serial path). Each worker writes a disjoint slice of the pinned staging block. Also parallelize the
  current/next/anchor/labels gathers against each other where they write disjoint destinations.
- Make sure numba is not limited to one thread by env (`NUMBA_NUM_THREADS`) in a way that blocks this; the
  kernels themselves stay single-threaded per call.
- If profiling shows rejection sampling itself is material, keep it serial (RNG order) but vectorize it
  without changing draws.

## P2. Learner-loop bookkeeping (`train_m3c.py` around :3251-3310)

- The action-collapse check converts a 100k deque to an array and bincounts it every loop iteration. Keep a
  running per-action count (add on ingest, subtract on eviction) or evaluate only at the log cadence; the
  halt decision must fire at the same transition counts it fires today (or document and test any shift of
  at most one log row; prefer exact).
- Avoid `acts.tolist()` + deque.extend per message (use a numpy ring).

## P3. Clean actor shutdown

smoke-02 w128 self-play rows reported `actor_exitcodes` [-15, ...]: in the `finally` block (~:3575) the
queues are drained once, actors refill them and block on `put`, then get SIGTERM after the 30 s join. Fix:
keep draining all queues in a loop while joining (until every actor exits or a deadline), and make actors
check `stop_evt` with a `put(timeout=...)` loop. Exit codes must be 0 on a normal max-wall stop. Add a test
with tiny queues that would previously hang.

## P4. Colab smoke-03 (`research/t16_colab/run_smoke.py`, `notebooks/colab_throughput_smoke.ipynb`)

- New config list behind `--suite smoke03` (keep the T16 list as the default suite for reproducibility):
  self-play only, 128 worlds: `{actors 2, gather_threads 0}` (control), `{2, 3}`, `{3, 3}`, `{3, 4}`; all
  with `--actor-queue-frames 4`. Timing per config 600 s (flag), one config additionally with
  `M3C_PROFILE=cprofile` for 60 s after warm-up, profile copied to Drive.
- Report per row: transitions/s, **updates/s**, transitions per update (must be ~256), policy_lag_mean,
  learn_s per update, GPU/CPU utilization. Comparison (`compare.py`) gains an updates/s verdict:
  PASS if updates/s >= 4 x 21.0 = 84 with transitions/update within 2% of 256.
- Notebook: RUN_NAME `smoke-03`, pin the merged commit, budget sized for warm-up + 4 x 600 s + setup
  (raise the runner cap accordingly; Colab Pro+ background execution allows it).
- Rebuild the input tarball only if its contents must change (they should not).

## Tests

`./venv/bin/python -m pytest tests/ -k "obs_pack or m3c or t16" -q` plus the new identity and shutdown
tests, and the existing T16 suite (35 tests). Add a micro-benchmark script (not a test) that times
`_sample_fast` serial vs `--gather-threads 2/3/4` on synthetic rings and prints ms/update.

## Out of scope (later specs)

GPU-resident replay ring with on-device gather (only if P1 falls short of 84 updates/s); league mode
(N learner slots + frozen pool); `--batch-scale 4 --lr-scale sqrt` arm (pre-registered, recipe change).

## Deliverable

Merged to main + pushed; a short note appended to `docs/research/t16_colab_smoke02_results_2026-10-17.md`
with the micro-benchmark numbers and the pinned commit for smoke-03.
