# Path B v2 implementation verification — 2026-10-20

Implementation and smoke checks are complete for Amendment 2. No efficacy bank,
full T3 pre-check, operating-characteristic simulation, strict gate or release
was run. Historical M2/M3 NO-GO findings are unchanged.

Implementation source was merged with incoming main at
`2d7069811678f97d7b85b175a8335a5ce00b4c5d`. The following documentation-only
commit does not change the executable source closure. The initial clean baseline
was `45ecb92665265491dd9380be5cd1f301ff5426c9` on
`t34-v2-pathb2-20261019`.

## Commands and results

Commands below use these exact working-directory and environment settings:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-v2-pathb2
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
E=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-pathb-v2/verification-20261020
```

Final merged-source focused check:

```bash
"$PY" -m pytest -q tests/test_t34_pathb_*v2.py tests/test_t34_shadow.py
```

**94 passed in 7.28 seconds**, comprising 86 v2 tests and eight shared-shadow
tests. Evidence: `focused-final.log`. The earlier integrated check passed
85 tests in 7.21 seconds before the last registration guard was added;
`focused-premerge.log` preserves that result separately.

Coverage includes real B8 scripted/frozen/mixed branch continuations against the
independent T3 reference, chronological native windows with differing horizons,
early completed-source publication, full-state/action traces, exact Q ties,
trapped fallback, wall death, moving-frame/epoch rejection, one-request bounds,
confirmation overflow, checkpoint/roster binding and P7 decision boundaries.
Real Watch checks compare physical/RNG, carried features and opponent v8 state.
These are short development cases, not long-body performance qualification.

Bounded equivalents of repository lint, plus the new research/test files:

```bash
"$PY" -m black --workers 1 --check --line-length 100 research/t34_pathb/*v2.py tests/test_t34_pathb_*v2.py src web/backend
"$PY" -m isort --profile black --check-only research/t34_pathb/*v2.py tests/test_t34_pathb_*v2.py src web/backend
"$PY" -m flake8 --jobs 1 research/t34_pathb/*v2.py tests/test_t34_pathb_*v2.py src web/backend
git diff --check
```

All passed; Black checked 166 files. No frontend files changed.

Required broad non-slow check, started before the final integration fixes and
the incoming-main merge:

```bash
"$PY" -m pytest -q -n 2 -m 'not slow'
```

**8,455 passed, 15 failed, 29 skipped**, two warnings, in 2,214.86 seconds.
This is **not a green full suite** and is not represented as a frozen-source
post-merge run. All failures were in unchanged modules:

- Thirteen `tests/test_perf_compile.py` cases failed with a TorchInductor
  `@rpath/libc++.1.dylib` loader error or its downstream assertions.
- Two `tests/test_t22_frozen_pool.py` cases failed because the frozen v8 oracle
  rejects the newer `membership` keyword.

Both modules were checked in a new, clean, detached worktree at the initial
baseline, retained at `$E/baseline`. Exact baseline commands, with the same base
environment above:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-pathb-v2/verification-20261020/baseline
TORCHINDUCTOR_COMPILE_THREADS=1 MAX_JOBS=1 "$PY" -m pytest -q -m 'not slow' tests/test_perf_compile.py tests/test_t22_frozen_pool.py
"$PY" -m pytest -q -n 2 -m 'not slow' tests/test_perf_compile.py tests/test_t22_frozen_pool.py
```

The serial baseline check returned **34 passed, 2 failed, 2 deselected in
37.38 seconds**. The original-environment two-worker baseline check returned
**34 passed, 2 failed in 15.86 seconds**. Both reproduce the two T22 failures;
the compiler tests pass in isolation. The thirteen broad-run loader failures
remain unresolved environmental/context-sensitive suite results. They are not
claimed fixed or independently reproduced on baseline. The compiler tests,
T22 tests and `src/`/`web/backend/` remain byte-identical to the starting commit.

## Real entrypoint and timing smoke

The original T3 materializer was exercised through its real entrypoint:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-v2-pathb2
"$PY" -m research.t34_pathb.precheck_v2 --out "$E/t3-materialized-64.json"
```

Exit 0: **64 original trajectories, 32 unique physical seeds**, preserving
`b8_s0/b8_s3 x env0..31`, original checkpoint bytes, receipts and rosters. This
constructs an input inventory; it does not execute or pass the >=50/64 pre-check.

The diagnostic mixed roster uses seed 991827 and `planned_rosters(gid._context(1),
[991827], ["mixed"])`; its exact JSON is retained as `smoke-input.json`.

```bash
"$PY" -m research.t34_pathb.retention_v2 smoke \
  --input "$E/smoke-input.json" --frames 2 \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-pathb-v2/smoke-20261020-final
```

Exit 0: terminal/Watch x bare/paced, **four runs of two delivered decisions**,
all identity/parity `PASS`, no unconditional fault. Output: the sibling
`smoke-20261020-final/smoke.json`; log: `smoke-final.log`.

Standalone actual T15 ON/OFF probes were exercised after parity passed:

```bash
"$PY" - <<'PY'
import json
from pathlib import Path
from research.t34_pathb.adapter_v2 import open_watch
from research.t34_pathb.timing_v2 import run_probes
result = run_probes(lambda arm, fps: open_watch({'seed': 991827}, fps, arm), ticks=2, warmup=0, contended=True)
path = Path('/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-pathb-v2/verification-20261020/timing-smoke.json')
with path.open('x') as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
print(json.dumps({'path': str(path), 'probes': list(result['probes']), 'contended': True}))
PY
```

Exit 0. All probes are **contended, cold, n=2** and cannot qualify timing:

| Probe | Update p95 ms | Full-tick p99 ms | Observed overruns |
|---|---:|---:|---:|
| bare 12 fps | 77.912 | 80.871 | 0 |
| bare 24 fps | 9.345 | 9.438 | 0 |
| paced 12 fps | 31.224 | 31.371 | 0 |
| paced 24 fps | 28.825 | 28.918 | 0 |

The ON observations exceed the 11.688 ms update bar. First-result latency is
`NOT_OBSERVED` in all four probes: no natural doom completion was observed in
two ticks. The <=2-second engineering target is unmeasured. These values do not
establish a quiet-host result, cadence acceptance or absence of longer-run
overruns. The actual Watch/Grid bridge and snapshot/digest cost are included.

## Evidence identity and operator boundary

`$E/manifest.json` records SHA-256 for twelve evidence artifacts. The create-only
`$E/trace-receipt.json` binds the 94-test result to **752 source/config/governance
files** and explicitly excludes efficacy, timing and serving qualification.
Its SHA-256 is
`f29b554874ad47d4dc381703fd9be871eee2b3891ec0b92201eacf6eb5300e4d`.

The full pre-check and registered study remain operator-run exactly as in
[README_V2.md](README_V2.md). A study-specific fixed-N survival operating-
characteristic receipt, fresh collision-audited banks, committed full
pre-registration, source/checkpoint/profile identities and quiet-host load
ratification remain mandatory. No such efficacy study or OC receipt was
fabricated by this implementation task. There is no ONNX backend or release.

No live services/tmux sessions, released checkpoints, serving pins, score store,
release-monitor data or historical hash-bound documents were changed. No cloud
runtime or teacher label round was started. Numerical pools were limited to one;
the broad test run used two workers, and actual assisted smoke sessions ran
serially with one foreground thread and one background worker.
