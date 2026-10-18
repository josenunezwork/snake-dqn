# T24 v8 body-map benchmark

`bench_bodymap.py` is a bounded CPU-only measurement for shared per-frame v8
diagnostics in the N=3, six-policy static league. It never starts a learner,
GPU service, pod, or long training run.

The fixed-state rows use real 128-world `GridBatchSim` states. The short case
injects six length-5 bodies; the long case injects six length-1200 serpentine
bodies. Construction, injection, Numba warmup, and profiling are excluded from
each timed sample. The benchmark retains every unprofiled `perf_counter` sample
and hashes the v8, bv8, pool, and actor sources it executes.

The fixed-state report measures four directly comparable host feature fields:

- `learner_v8_direct_baseline` is the direct learner-only v8 invocation;
  `learner_plus_b8_v8_shared_candidate` is the actor candidate call that also
  primes the B8 slice. They are host feature work, not B8 forward time.
- `b8_v8_direct_baseline` is the B8-only v8 call under its advisory mask.
- `b8_policy_standalone` is the unprimed B8 CPU policy total: raster, v8,
  forward, and action selection. It is reported separately because forward
  cost dominates this CPU-only value.
- `learner_plus_b8_v8_shared_candidate` is either the old pair of independent
  calls in the pinned baseline or the candidate's one shared call. It is the
  primary comparable combined-v8 field.

`whole_actor` is the actual actor-frame row. It calls `train_m3c.actor_main`
through the existing bounded message-sink harness, and covers simulation step,
runtime carry, replay/n-step ring writes, learner observation/forward/labels,
and six-policy frozen dispatch. It excludes learner updates and queue pressure.
An outer, temporary `build_segment` wrapper injects the body fixture before the
first decision and is restored after every sample.

## Final paired receipt

The final receipt alternated baseline/candidate/candidate/baseline (ABBA), with
three unprofiled repetitions per leg, `warm=4`, and `frames=8`. It ran only
after the qualification suite completed and other CPU test work stopped. The
pinned baseline production source is `23c1ba73`; the candidate production
source is `4b4723f`. Both receipts retain their leg files, raw samples, source
hashes, and the common benchmark-script SHA-256:

- [Before receipt](before_23c1ba73.json) and its two named leg files.
- [After receipt](after_4b4723f.json) and its two named leg files.

Across the six raw fixed-state samples, combined learner+B8 v8 work was
0.720 ms before and 0.694 ms after at length 5; it was 22.155 ms before and
16.280 ms after at length 1200, a 26.5% long-body reduction. The actual
whole-actor long-body median was 246.738 ms/frame before and 234.482 ms/frame
after. Its end-of-window median body length was 1201 in both receipts (initial
length 1200; end mean 1179--1188 in the exploratory retention check), so the
long row did not devolve into a short-body respawn measurement.

These are single-host CPU microbenchmark results, not A100 throughput or
training performance evidence. The projection document applies its own
conservative accounting to the recorded components.

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
`--without-whole-actor`. This is useful for checking feature/action byte
identity but is not a substitute for the actor row.
