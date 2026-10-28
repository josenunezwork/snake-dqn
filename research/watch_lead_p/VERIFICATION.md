# P packet-04 verification

Implementer Astra, 2026-10-28. This note replaces packet-03's verification claims
after the operator supplied Argon's REJECT verdict (one blocker, three major,
three minor). No independent acceptance review of the corrections has been
received. No dispatch, allocation, main-bank session, push or release is performed.

The implementation fixes all seven reported boundaries:

- Completion sealing accepts a verified ledger and unit root, ships every sealed
  payload/sidecar with its study tag, and requires exports only for missing units.
  Empty and all-complete completion ledgers refuse before packet creation.
- Scientific identity uses science-sources.json; attempt source pins separately
  bind operational code. Operational fixes preserve completed units; scientific
  dependency changes refuse the old recipe. This boundary is frozen before data.
- Interrupt handlers do no I/O, disable repeated stop signals, and defer unwinding
  while submit is registering a child. Cleanup discovers all owned processes,
  terminates them, lets the executor manager reap, verifies PID disappearance,
  and only then publishes one interruption record. No process-group kills occur.
- Twelve arm workers replace ten whole-session workers. The explicit indivisible
  task forecast includes tail capacity and setup/export; only full matched
  three-arm sessions become completion units. The base/effective gates stay 4/8 h.
- Tests exercise bare death, rescue, descending-Q/ascending-index alternatives,
  all-die fallback and the H300 fatal transition without post-respawn credit.
- Admission verifies committed seal bytes, its exact six-member inventory, ledger
  location, upload destinations and missing-unit exports. Dispatch verifies and
  loads the committed runtime inventory once, retaining traversal/prefix checks.
- LiveWorld preserves caller RNG on construction, refuses negative/out-of-range
  and masked forced actions, and refuses forced actions on non-acting ticks.

Final test counts, forecast receipt, detached checkout identities, sparse-source
proof and exact dry-run command follow below. Packet-03 and its
hash-bound files remain unchanged. No new judgment or approval is claimed.

## Final frozen preparation

- Runtime commit: `8d4333bebd182fe61b0dbea6babbf2f61745e8d6`.
- Broker/packet commit: `c47c1e4b8a68bab7e6306fe688ef0bcd582c8f90`.
- Identity: `sandbox-p-watch-lead-20261028-04`; packet `packet-04`.
- Clean detached clones: `/Users/josenunez/Projects/ml/snake-dqn-p-watch-lead-04-runtime.noindex`
  and `/Users/josenunez/Projects/ml/snake-dqn-p-watch-lead-04-broker.noindex`.
  Their own local Git excludes hide read-only venv/checkpoint dependency links.
- Qualified transport tree is unchanged: `5833e54e9c5396121cbbf99a5cd4e52006e564fd`.
- Seal SHA256: `efb48e3d9979aa3a690f60cc874575c401cfe5a8f9c6d096395b36fca3e829be`.

## Final verification evidence

All Python checks below used CPU, `nice -n 15`, all seven thread caps set to 1,
and no xdist. Exact final combined suite:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1 SNAKE_DQN_DEVICE=cpu nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -o addopts='' -p no:xdist research/watch_lead_p tools/sandbox/tests research/sandbox_b_heads/tests tools/ops/tests
```

**239 passed, 2 skipped in 60.59 seconds.** The two skipped operations tests
require absent historical R2/R6 artifact receipts. No P/sandbox test was skipped.
This includes real shell launch/exec argument, input, CPU, nice and thread binding;
spawned active-worker SIGUSR1+SIGTERM teardown; interruption before submit returns;
committed completion sealing/preflight/qualified-validator admission; nested upload
staging; exact verified-session skip; synthetic 32-session terminal union; missing
upload, complete ledger, uncommitted/incomplete seal and foreign ledger refusal.
The completion test uses synthetic outcomes, with gameplay run_arm replaced only
at its worker boundary. It is not a scientific or L4 completion result.

A preceding full P/sandbox run was **216 passed** before the last two sealer
regressions were added. Earlier development runs exposed and corrected the
executor-manager reaping race and test-fixture routing issues; final success is
based on the final combined command above and the real dry-run below.

All three commands below **PASS**, 16 files. `bash -n research/watch_lead_p/run_job.sh`
and `git diff --check` also **PASS**.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1 SNAKE_DQN_DEVICE=cpu nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m black --check --line-length 100 research/watch_lead_p tools/ops/dispatch.py tools/sandbox/manifest.py tools/sandbox/preflight.py tools/sandbox/registry.py tools/sandbox/supervisor.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1 SNAKE_DQN_DEVICE=cpu nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m isort --check-only --profile black research/watch_lead_p tools/ops/dispatch.py tools/sandbox/manifest.py tools/sandbox/preflight.py tools/sandbox/registry.py tools/sandbox/supervisor.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1 SNAKE_DQN_DEVICE=cpu nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m flake8 research/watch_lead_p tools/ops/dispatch.py tools/sandbox/manifest.py tools/sandbox/preflight.py tools/sandbox/registry.py tools/sandbox/supervisor.py
```

Excluded-fixture benchmark:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1 SNAKE_DQN_DEVICE=cpu nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.watch_lead_p.benchmark --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-watch-lead-1028/engineering-04/benchmark-final.json
```

Natural warmup is 2161 frames in 13.124051 seconds.
Six natural bare-surviving roots plus two explicit first-transition bare-death
engineering cases execute real Watch/H300 alternatives. Arm forecast costs are
P0 3517.482630, P48 447.906132, bare 45.342230
seconds. Explicit 96-task / 12-worker list scheduling gives **11908.898773
seconds = 198.481646 minutes = 3.308027 hours**,
including 1,200 seconds setup/export. Estimate rounds to 199 minutes, fitting the
unchanged 240-minute base gate. The old receipt correctly quantized as four whole
session waves would need 283.81 minutes and would be rejected. The sealer regression
refuses a forecast even one second above the base guard before packet creation.

This remains a Mac-derived forecast. Twelve-worker L4 CPU contention, longer
bodies and late multibranch doom roots are unqualified. The conservative six-full-
branch sensitivity is **17.731821 hours**; do not call
this an observed L4 runtime or assume a measured rescue frequency. Mac-equivalent
base forecast and 2x/3x slower CPU-work sensitivities (setup allowance held fixed)
are 3.308027/6.282722/9.257416 hours.
The fixed 8-hour per-attempt guard can require manual completion. Base/effective
wall guards 14,400/28,800 seconds, unit guards 6.16/12.32 at planning rate 1.54 u/h,
reserves 0/0, N=32 and at most two manual completions are unchanged.

Qualified source.prepare and install_fallback verified **303 files**, **1,307,469
compressed / 5,334,636 unpacked bytes**, pinning the exact runtime commit. The real
entrypoint also ran from that extracted archive:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-watch-lead-1028/engineering-04/source-plan/repo
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1 SNAKE_DQN_DEVICE=cpu nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.watch_lead_p.run smoke --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-watch-lead-1028/engineering-04/sparse-smoke.json
```

**PASS**, excluded seed 32, 60 real ticks per arm; no main-bank session. The archive
contains both requirements files, dynamic serving schema and science-sources.json.

Exact sealed dispatch dry-run, executed from the original worktree:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1 SNAKE_DQN_DEVICE=cpu nice -n 15 /Users/josenunez/Projects/ml/snake-dqn-p-watch-lead-04-broker.noindex/tools/ops/dispatch.sh sandbox-p-watch-lead-20261028-04 /Users/josenunez/Projects/ml/snake-dqn-p-watch-lead-04-broker.noindex/research/watch_lead_p/packet-04/dispatch-manifest.json --broker /Users/josenunez/Projects/ml/snake-dqn-p-watch-lead-04-broker.noindex --cwd /Users/josenunez/Projects/ml/snake-dqn-p-watch-lead-04-runtime.noindex --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-watch-lead-1028/attempt-04/state --dispatch-txt /Users/josenunez/Projects/ml/snake-dqn-p-watch-lead-04-broker.noindex/research/watch_lead_p/packet-04/DISPATCH.txt --dry-run
```

**PASS_DRY_RUN sandbox-p-watch-lead-20261028-04 allocations=0
live_accounting=NOT_CHECKED**. It traversed the actual detached broker's committed
sandbox preflight and qualified supervisor validate route, including the focused
P/sandbox tests. Both checkout HEADs remain detached and clean; attempt-04 state
and output remain absent. No live accounting/hardware validation, allocation,
provider execution, remote export/release verification or acceptance review is
claimed. Operator dispatch remains separate.

Evidence files are under `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-watch-lead-1028/engineering-04`:
`suite.log`, `benchmark-final.json` and sidecars, `benchmark.log`,
`dispatch-dry-run.log`, `checkouts-before.json`, `final-audit.json`,
`sparse-proof.json`, `source-plan/plan.json`, and `sparse-smoke.json` with sidecars.

No existing tmux/job, exp/mac-lease.json*, released checkpoint, serving pin, protected
scientific outcome, older packet bytes or qualified transport was modified. The
handed worktree's original untracked saved_snakes link remains. Every commit is
local; no push or dispatch occurred. No independent review of these fixes has
been received from the operator.
