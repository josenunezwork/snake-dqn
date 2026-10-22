# V0 implementation verification and operational handoff

Registration committed/published before V0 data:
`813d93d8f6b143b09563cbe50405444b231e8820`.
Initial implementation: `e7eabd3590fb3cd27c7419aef05a33602041c60c`.
This follow-up fixes the live-process exclusion defect observed at the real
entrypoint. The registration bytes and scientific rules remain unchanged.

## Targeted verification

All commands from `/Users/josenunez/Projects/ml/snake-dqn-wp2`:

```bash
env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  nice -n 10 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  -m pytest -q research/bl_v0/test_contract.py research/bl_v0/test_operator.py
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m black --check --line-length 100 research/bl_v0
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m isort --check-only --profile black research/bl_v0
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m flake8 research/bl_v0
bash -n research/bl_v0/run_mac.sh
git diff --check
```

**37 passed**, 0 failures (0.65 s pytest body); formatting, imports, lint, shell
syntax and whitespace checks passed. No broad suite. The synthetic suite covers
exact stop/all-die/ties, H64 forced/fatal accounting at transitions 1 and 64,
carry isolation, C1 oracle Q/legal binding, one real dummy invocation and C4
padding, cell/branch/leaf attestation, lease interoperability, window edges,
full-population decisions, cold exclusion, nonfinite INVALID classification,
raw canary inventory and immutable invalidation rejection.

Independent reviewer inspected both the operator and execution/admission paths.
Its pre-smoke review found no blocker; the real boundary revealed the process
listing defect described below. The corrected guard also refuses the actual
live K-1 parent process in a read-only invocation.

## Real bounded smoke: execution proof, INVALID for admission

Sol-K invoked the actual `run_mac.sh --smoke --operator Sol-K`, one thread,
nice 10, four frozen-800 batches, after publishing the implementation. It ran
5.834 s under the V0 lease. No full ladder was started.

| Cell | Actual branch frames | Kernel rows | Stock rows | Chosen / bare |
|---|---:|---:|---:|---:|
| C1 | 968 | 4,840 | 0 | 4 / 1 |
| C2 | 561 | 2,805 | 0 | 4 / 1 |
| C3 | 256 | 1,280 | 0 | 4 / 1 |
| C4 | 128 | 640 | 0 | 4 / 1 |

C3 and C4 each executed exactly one 12,806-parameter CPU FP32 dummy call at
batch size 4, torch intra/inter-op threads 1. All four chosen actions matched.
These counters prove execution only; no speed or scientific admission follows.

**Deviation:** the K-1 audit was active. The intended smoke refusal failed
because macOS `ps -o comm` truncates the Python path to `/Library/Framewo`.
The guard had filtered on that column before inspecting the command arguments.
The smoke's success metadata is revoked by a create-only `INVALIDATION.json`;
no historical receipt was edited. The corrected guard checks both COMM and the
executable at the beginning of ARGS, then checks again after each batch. Tests
include the observed truncated Mac row and a Codex prompt false-positive case.
`verify_canary` explicitly refuses an invalidation marker. No automatic retry.

Artifacts under
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-v0/`:

- `prepared-01/manifest.json`: SHA256
  `5d06072ccaf031af164e5f880ffcfefd507c7511f9f242d4f8f228cbc81f8a90`;
  obsolete after the guard source change.
- `smoke-01/smoke.json`: SHA256
  `2111e81c7452e3df5e3747818accbfda8c3b43f4d4a31fd24558dc6ef5f49832`,
  retained with raw batches, idle/load evidence, lease and stage timers.
- `smoke-01/INVALIDATION.json`: authoritative **INVALID / admits NONE**.
- `smoke-preflight-01.log`: SHA256
  `db9ee5494011481d23fbad607330bb98cb2143fcf57415bc37071f42e46212e2`.

Claude must obtain a fresh source-bound smoke-02 once K-1/X5 exit, then use the
prepared-02 commands in README.md. Source changes require another fresh plan
and canary. The two frozen judgment conflicts also need a prospective ruling
before any scientific PASS; the implementation never silently exempts scripted
zero-row batches or reverses the literal NO_CONSUMER inequality.

## Forecast and preserved boundaries

Full V0: round-13 projection 35–50 min, 60-min cap, 1,080 batches, 0 units and
0 upload bytes. This remains **unmeasured**, not inferred from the smoke.
Builder 0.5–1 day and review 0.25 day are planning estimates, not timesheets.
Real timing is Claude-only Fri 2026-10-23 22:00–23:00 PT or Sat 2026-10-24
08:00–10:00 PT, under the common lease; neither K-1 nor X5 may overlap.

No Colab calls, real V0 ladder, serving change, release, process/daemon kill,
tmux modification, checkpoint write, or protected T34 source edit. The sealed
artifact directory was never opened. Observed sessions left untouched:
`git-sync`, `k1-attempt-03`, `o2-v2-00`, `o2-v2-01`.
