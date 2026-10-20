# T34 engine fanout implementation and Mac receipt — 2026-10-20

Implemented the engine-fanout spec and the newer worktree addendum. The latter's
W1 resident layout and 60-minute A5 workload supersede the original W4 / 45-minute
sizing. The frozen dispatch order is also recorded in `CAMPAIGN_LOG.md` at
2026-10-20 ~14:20. All execution in this implementation task stayed on the Mac;
numerical jobs used at most two single-thread CPU processes. No Colab job or tmux
session was started, inspected, stopped, or modified.

## Execution contract

- `bench.sh`, `label.sh`, and their Python CLIs default to CPU inference and one
  thread. `--nproc`, `T34_NPROC`, or `T34_WORKERS` overrides concurrency; CPU
  defaults use available CPUs, capped at two on the Mac. Small banks reduce the
  effective process count to their world count.
- The parent materializes the world bank once and shards it in deterministic
  round-robin order. `T34_MAX_WORLDS` is the total bounded bank; benchmark
  `T34_WORLDS` is the resident capacity per process. Label chunks default to one
  resident world per process. Global export indices survive sharding, including
  filenames and checksum sidecars; exported row dictionaries are unchanged.
- CPU stepping bypasses lockstep merge/scatter for shadows, roots, doom-cache
  reconstruction, and branch rollouts. W1 inference is batch-one. Existing CPU
  W>1 root inference retains its prior batching recipe for export compatibility.
  Explicit CUDA retains its historical batched path and two-worker launcher.
- Lane reports contain completed worlds/hour, active/resident row-steps,
  completed versus unfinished work, host-stage timings, actual process count,
  and each process's report/RSS. A failed process ends the fanout; cleanup targets
  only owned children with bounded joins and kill escalation. There is no retry.
- Rollout-policy adapters are refused before preparing K12 manifests. The pinned
  B8 input path is supported; later tower adapters need explicit worker support.
- `full_digest` uses typed buffers, MT bytes, ordered bodies/food, and per-call
  runtime content fragments. Its schema is `t34-shadow-full-digest/v3`.
  Path B binding diagnostics include `digest_schema`. Historical receipts are
  unchanged. See `research/t34_shadow/DIGEST_V2.md` for the schema history and
  source-bound Path B preparation implications.

## Measured evidence

Artifact root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020`.
The frozen pre-change source is commit
`c97e6d7d7add2a7bb4253983d9069c7117327553`, archived under `baseline-source`.
Final measured implementation source is contained in published commit
`3ebf3991a0b37f2bcd20bbad872ec1df38fb2e26`.

| Check | Result |
|---|---|
| Warmed serial profile, three worlds × 120 frames, H300; three rounds each | Median **6.1605 → 3.5945 ms/active row-step**, **1.7139×** observed speedup |
| Profile work and exported bytes | 1,617 active row-steps per round; all nine world/sidecar files identical in every before/after round |
| Frozen implementation vs new W1 serial vs two-process fanout | Six worlds across scripted/frozen/mixed; **18/18 payload and sidecar files byte-identical** |
| M1 independent live/Grid/rolling-shadow fixture, seed 2026101934 | **15,000 frames**, 5,000 per mix; **zero** physical, carry, v8, action, or shadow differences; source hashes unchanged |
| M1 confirmed shadow frames | Scripted 4,999; frozen 4,997; mixed 4,999; death/respawn gaps remain covered by physical/action comparisons |
| Real local shell entrypoints | Benchmark and label each completed two W1 processes, two-frame worlds, H300 |
| Excluded CPU diagnostic smoke, six-second window | Alone 208.24 active row-steps/s; two processes 398.57 aggregate; all mixes and short/long bodies exercised |

The final timing samples are `before-serial-r{1,2,3}.json` and
`after-v3-serial-r{5,6,7}.json`, summarized with hashes in `MAC_PROFILE.json`.
Warm-up was excluded, thread limits were one, and no other task-owned numerical
job overlapped the final after samples. Phase 0 and other owner jobs could run:
these are contended-host engine measurements, not quiet-host Path B timing.

Export proof is `identity-attempt02/EXPORT_IDENTITY.json`. M1 receipts are
`m1-{scripted,frozen,mixed}-5000.json`. CPU diagnostic proof is
`cpu-profile-smoke-02/profile-cpu.json`, including exclusive digest/v8/raster/
sim/inference stages, process CPU/wall ratio, RSS, hardware and source identity.
Its throughput denominator includes post-warm-up snapshot setup/reset and wrapper
overhead. The diagnostic is excluded from campaign estimates and A5 admission.

Earlier attempts remain preserved. The first buffer-only digest yielded little
end-to-end benefit; the final runtime-fragment memo avoids repeated carry-array
hashing. `after-v3-serial-r1.json` is invalid for timing because source changed
during collection. The first CPU-profile smoke exposed an unprepared cloned
snapshot, corrected by preparing before cloning and rebinding injected geometry.
The first fanout smoke exposed missing mix aggregation and inconsistent default
bank materialization; both were fixed before the successful immutable proof.

## Prepared manifests and operator commands

Only the final `manifests-final-3ebf3991` directory is intended for operator use. The
earlier `manifests-8ce81f2e` and `manifests-3ebf3991` drafts are superseded. All three final manifests passed
the qualified transport's pure local validator, with full published source pins
and the six immutable public checkpoint inputs. None was dispatched.

| Manifest | Layout / useful work | Whole-session bounds |
|---|---|---|
| `a5-l4-s0.json`, `a5-l4-s1.json` | L4 assignment, CPU inference, K12 × one thread, W1, total refill bank 256, work ≤3,600 s, post-fill window 1,200 s | 120 min; 4.0 u cap/lane at 2 u/h; 0.8 u account reserve/lane |
| `cpu-profile.json` | CPU assignment; hardware/affinity/cgroup first; warmed W1 alone then effective K; ≤60 s profile window | 12 min; ≤0.12 u envelope at 0.5 u/h ceiling; 0.02 u reserve; 64 MiB export bound |

The 120-minute A5 timeout satisfies the supervisor's 2× estimate requirement and
exceeds the addendum's 25-minute setup/export margin. The CPU profile prefers an
observed positive rate at or below 0.5 u/h; its builder uses that ceiling when the
rate is unknown. Profile hardware determines effective K rather than assuming
the CPU-only runtime has twelve vCPUs. No K6 fallback is selected without the
addendum's prospective comparison evidence.

Run these manually in the prescribed order: CPU profile, inspect its verified
terminal receipt and placement evidence, then the A5 lanes. The launcher performs
AC/lid guarding and delegates allocation, exports and release verification to the
qualified supervisor. These are dispatch commands, not commands executed here.

```bash
/Users/josenunez/Projects/ml/snake-dqn-engine-fanout/research/colab_cli/launch.sh --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/broker --colab /Users/josenunez/.local/bin/colab run /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991/cpu-profile.json

/Users/josenunez/Projects/ml/snake-dqn-engine-fanout/research/colab_cli/launch.sh --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/broker --colab /Users/josenunez/.local/bin/colab run /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991/a5-l4-s0.json

/Users/josenunez/Projects/ml/snake-dqn-engine-fanout/research/colab_cli/launch.sh --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/broker --colab /Users/josenunez/.local/bin/colab run /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991/a5-l4-s1.json
```

Use `launch.sh --check` before the remaining arguments for a print-only preview.
Future operator dispatch must use a durable caller-owned execution context and
respect account-wide assignment/campaign caps. The work here proves Mac identity
and local transport preparation, not x86 identity, A5 GO, promotion, or serving
qualification. Released checkpoints, serving pins, running jobs, legacy Q1
workers and existing hash-bound documents were not edited by this task.

## Verification closeout

The complete gate at published source `3ebf3991` passed **884 tests, 2 skipped**
in 1,325.99 seconds. After merging the independent retention fixes from main,
the complete T33/T34 selection at `b15f80e2` passed **678 tests, 2 skipped**
in 188.30 seconds. Both runs used two workers with the numerical libraries
limited to one thread per worker. The two skips require separately authorized
two-thread fixtures. The integrated merge did not change this task's engine,
digest, profile, launcher or manifest files; all 230 M1 source hashes per mix
still match. The Colab tests use the local fake provider, with no live dispatch.

Exact final test and core lint commands:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -n 2 tests/test_t33*.py tests/test_t34*.py tests/test_colab_cli*.py

env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -n 2 tests/test_t33*.py tests/test_t34*.py

make PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python lint

git diff --check
```

Changed research/test Python files also passed Black, isort and flake8; launchers
passed `bash -n`. Actual tiny benchmark and label shell entrypoints both completed
with two W1 workers. The independent receipt audit passed after recomputing
source/artifact hashes, comparing exported bytes, checking CPU stage accounting,
and locally validating the final manifests.

The create-only summary [VERIFICATION_ENGINE_FANOUT.json](VERIFICATION_ENGINE_FANOUT.json)
records the complete file inventory, source hashes, exact saved profile
invocations, test counts, log hashes, immutable evidence hashes and manifest
bounds. Historical receipt bytes remain unchanged.
