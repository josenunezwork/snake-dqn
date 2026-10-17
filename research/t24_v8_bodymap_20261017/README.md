# T24 v8 body-map benchmark

`bench_bodymap.py` is a bounded CPU-only measurement for shared per-frame v8
diagnostics in the N=3, six-policy static league. It never starts a learner,
GPU service, pod, or long training run.

The fixed-state rows use real 128-world `GridBatchSim` states. The short case
injects six length-5 bodies; the long case injects six length-1200 serpentine
bodies. Construction, injection, Numba warmup, and profiling are excluded from
each timed sample. The benchmark retains every unprofiled `perf_counter` sample
and hashes the v8, bv8, pool, and actor sources it executes.

The fixed-state report has three scopes:

- `learner_v8_direct_baseline` is the direct learner-only v8 invocation;
  `learner_plus_b8_v8_shared_candidate` is the actor candidate call that also
  primes the B8 slice. They are host feature work, not B8 forward time.
- `b8_frozen_dispatch_*` measures frozen dispatch with and without its primed
  v8 slice. CPU B8 forward is intentionally included and is reported as a
  synthetic host forward, not a device-throughput result.
- `whole_fixed_state_decision_frame_*` is a decision-pipeline proxy. It does
  not advance the simulator and therefore must not be described as a whole
  actor-frame result.

`whole_actor` is the actual actor-frame row. It calls `train_m3c.actor_main`
through the existing bounded message-sink harness, and covers simulation step,
runtime carry, replay/n-step ring writes, learner observation/forward/labels,
and six-policy frozen dispatch. It excludes learner updates and queue pressure.
An outer, temporary `build_segment` wrapper injects the body fixture before the
first decision and is restored after every sample.

The retained `baseline_23c1ba73_t22_short.json` is an exploratory main-23c1
receipt, recorded before the candidate harness. It is host-noisy and is not an
A/B comparison or performance claim. Final before/after rows must be paired in
alternating order on one host session and preserve both raw samples and source
hashes.

Run from this checkout with the project venv:

```bash
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
nice -n 10 "$PY" research/t24_v8_bodymap_20261017/bench_bodymap.py \
  --repeats 7 --actor-frames 8 --actor-warm 4 \
  --out research/t24_v8_bodymap_20261017/candidate.json
```

For a fast harness-only check that intentionally omits whole actor frames, add
`--without-whole-actor`. This is useful for checking cache/feature/action byte
identity but is not a substitute for the actor row.
