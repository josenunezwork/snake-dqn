# P packet-05 verification — Astra, 2026-10-28

Prepared locally; no push, provider allocation, dispatch, main-bank session, release, or independent final Argon acceptance is claimed. This sibling note supersedes packet-04's verification for the requested fixes. Older packets and qualification bytes remain unchanged.

## Frozen identities

- Runtime pin: `ab935a25cde7fe84ca216e19a7fed9fd0bf77028`.
- Broker/packet commit: `76803c8b1b40c0f4bf389a542f86f766b823fbb8`.
- Identity: `sandbox-p-watch-lead-20261028-05`, packet `packet-05`.
- Runtime checkout: `/Users/josenunez/Projects/ml/snake-dqn-p-watch-lead-05-ready-runtime.noindex`.
- Broker checkout: `/Users/josenunez/Projects/ml/snake-dqn-p-watch-lead-05-ready-broker.noindex`.
- Both final checkouts are detached and clean; local Git excludes hide only the read-only venv/checkpoint links.
- Qualified transport tree unchanged: `5833e54e9c5396121cbbf99a5cd4e52006e564fd`.
- Seal SHA256: `7e912a87591bc2abf3e383fef26f51d629c5058e04066d7464852167c3f754fc`.
- Recipe SHA256: `acb5d0a6492c50a4ee382837f2bf4f115a305bd4779282626da5d2815b30170c`; final sparse smoke matches exactly.
- Evidence root: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-watch-lead-1028/engineering-05-resume-20261028-r1`. No partial evidence from the interrupted prior run was used for qualification.

## Five fixes and their evidence

1. `run.task_schedule` groups pending sessions in cohorts of 12 (final 8), with P0/P48/bare inside each cohort. The coordinator publishes each complete matched three-arm session immediately. `benchmark.project` uses the same schedule and records all 32 publication times. Regressions reject global arm-first scheduling and verify 12 units are banked before later-cohort results.
2. Stop handlers defer unwinding across spawn and complete publication/accounting, ignore repeated requests by flag, retain caught dispositions until teardown, and reset worker SIGTERM to default. Real spawned-worker tests cover active, before-submit, and after-submit stops; before-submit must finish below the two-second kill fallback. Four publication phases verify marker/accounting consistency under paired SIGUSR1/SIGTERM.
3. Admission validates upload destinations on first and completion packets, binds the actual ledger source/destination, enforces outcome-free exact schemas and integer indices/IDs, binds adjacent sidecars, and invokes `read_ledger` offline. Tests cover every reported malformed boundary and real staged completion/skip/union behavior.
4. Admission/tests require unique inventories, science subset of runtime, shipping the science inventory itself, and manifest/runtime equality. Sparse `smoke` now executes `recipe_fingerprint`.
5. Sealing normalizes all CLI paths and refuses working, staged, or committed science drift from the requested pin before packet creation. Tests cover inventory/dependency drift and relative-path normalization, including completion inputs.

## Final verification

All Python checks used CPU, nice 15, seven numerical thread caps of 1, and no xdist.

- Final full P/sandbox/operations suites: **274 passed, 2 skipped in 70.15 seconds**. The two skips require absent historical R2/R6 receipts; no P/sandbox test was skipped.
- Expanded run including matching supervisor/source suites: **369 passed, 2 skipped in 937.09 seconds**, before the final test-fixture cost reductions. Those reductions were subsequently covered by the final 274-pass run and final sealed dry-run.
- Final runtime checkout launch/exec binding test: **1 passed in 0.19 seconds**; actual shell exec, argv, uploaded inputs, CPU, nice and all thread caps verified.
- Black/isort/flake8: **PASS, 16 files**; shell syntax and whitespace checks PASS.
- Qualified `source.prepare` and `install_fallback`: **303 files**, **1,311,540 compressed / 5,354,226 unpacked bytes**, exact runtime pin and inventory verified. Final sparse `run smoke`: PASS, excluded seed 32, 60 actual ticks per arm, fingerprint equal to sealed ledger. No main-bank outcomes.
- Final real wrapper result: **PASS_DRY_RUN sandbox-p-watch-lead-20261028-05 allocations=0 live_accounting=NOT_CHECKED**. Attempt-05 state/output remain absent. No live hardware/accounting/provider execution, export, or release proof is claimed.

The initial expanded glob over every current transport test hit three collection errors: bundle, retention and retention_acceptance import modules absent from this frozen qualified tree. The qualified tree was preserved; the matching supervisor/source suites above passed. An initial nonexistent test-directory selection and a sparse-fixture checkout setup error were corrected; their logs remain in the fresh evidence root.

Two undispatched packet-05 preparation drafts hit the unchanged 180-second qualified-validation timeout. They are preserved byte-for-byte under `failed-dryrun-packet-05` and `failed-dryrun-packet-05-second`, with archive receipts and original local commits. Final fixtures use sparse checkouts, cleanup their owned clones, compute immutable fixture identity once, and create one synthetic unit for metadata-only cases. Explicit 31-unit resume/union and 32-unit refusal cases remain. The timeout, science, allocation N and attempt envelopes were not raised or weakened. The final successful packet was created fresh and sealed against its new runtime pin.

## Fresh forecast

Warmup: 2161 real frames in 13.931458 seconds. Six natural root samples and two explicit first-transition bare-death engineering fault cases; real H300 alternative continuations. No natural rescue-frequency claim.

Arm maxima projected per session: P0 **3488.654256 s**, P48 **507.754137 s**, bare **45.250705 s**. Shared cohort list scheduling projects **12771.972450 s = 212.866208 min = 3.547770 h**, including 1,200 seconds setup/export. Estimate rounds up to **213 minutes**, below the unchanged 240-minute base gate. Base/effective wall guards remain 14,400/28,800 seconds (4/8 hours), technical unit guards 6.16/12.32 at planning rate 1.54 u/h, reserves 0/0, N=32, one original plus at most two manual completions.

Twelve paired units bank at **4041.659097 s**, 24 at **8083.318195 s** of worker time. At 3x CPU-work slowdown these are 3.368049/6.736098 hours before setup/export. Whole-wall 2x/3x sensitivities with setup fixed are 6.762207/9.976644 hours. Six-full-branch stress: 19.180017 hours. These are Mac-derived forecasts; L4 speed, contention, long bodies and branch frequencies remain unqualified. The complete fresh benchmark receipt was reused across test-only reseals; scientific dependencies and forecast code did not change.

## Exact successful verification commands

Commands and working directories are also recorded in `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-watch-lead-1028/engineering-05-resume-20261028-r1/verification-commands.json`. Outputs go to the named immutable evidence logs. Reproducing create-only outputs requires fresh destinations.

Final full P/sandbox/operations suite; cwd `/Users/josenunez/Projects/ml/snake-dqn-falsifier-p`; log `suite-bounded.log`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1 SNAKE_DQN_DEVICE=cpu nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -o addopts='' -p no:xdist research/watch_lead_p tools/sandbox/tests research/sandbox_b_heads/tests tools/ops/tests --basetemp=/private/tmp/astra-build-p-fix2-r1-bounded-suite --durations=8
```

Expanded run including matching supervisor/source tests, before final fixture cost changes; cwd `/Users/josenunez/Projects/ml/snake-dqn-falsifier-p`; log `suite-qualified.log`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1 SNAKE_DQN_DEVICE=cpu nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -o addopts='' -p no:xdist research/watch_lead_p tools/sandbox/tests research/sandbox_b_heads/tests tools/ops/tests tests/test_colab_cli_supervisor.py tests/test_colab_cli_source.py --basetemp=/private/tmp/astra-build-p-fix2-r1-resume-qualified
```

Final source-bound launch/exec test; cwd `/Users/josenunez/Projects/ml/snake-dqn-p-watch-lead-05-ready-runtime.noindex`; log `launch-exec-ready.log`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1 SNAKE_DQN_DEVICE=cpu nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -o addopts='' -p no:xdist research/watch_lead_p/test_admission.py::test_actual_launch_exec_binds_python_module_and_uploaded_inputs --basetemp=/private/tmp/astra-build-p-fix2-r1-ready-launch
```

Fresh excluded-fixture benchmark; cwd `/Users/josenunez/Projects/ml/snake-dqn-falsifier-p`; log `benchmark.log`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1 SNAKE_DQN_DEVICE=cpu nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.watch_lead_p.benchmark --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-watch-lead-1028/engineering-05-resume-20261028-r1/benchmark-fresh.json
```

Final sparse-archive entrypoint; cwd `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-watch-lead-1028/engineering-05-resume-20261028-r1/source-plan-ready/repo`; log `sparse-smoke-ready.log`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1 SNAKE_DQN_DEVICE=cpu nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.watch_lead_p.run smoke --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-watch-lead-1028/engineering-05-resume-20261028-r1/sparse-smoke-ready.json
```

Sealer; cwd `/Users/josenunez/Projects/ml/snake-dqn-falsifier-p`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1 SNAKE_DQN_DEVICE=cpu nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.watch_lead_p.seal --pin ab935a25cde7fe84ca216e19a7fed9fd0bf77028 --broker /Users/josenunez/Projects/ml/snake-dqn-p-watch-lead-05-ready-broker.noindex --runtime /Users/josenunez/Projects/ml/snake-dqn-p-watch-lead-05-ready-runtime.noindex --benchmark /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-watch-lead-1028/engineering-05-resume-20261028-r1/benchmark-fresh.json --attempt 5
```

Final dispatch dry-run; cwd `/Users/josenunez/Projects/ml/snake-dqn-falsifier-p`; log `dispatch-ready-dry-run.log`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1 SNAKE_DQN_DEVICE=cpu PYTEST_ADDOPTS=--basetemp=/private/tmp/astra-build-p-fix2-r1-ready-dryrun nice -n 15 /Users/josenunez/Projects/ml/snake-dqn-p-watch-lead-05-ready-broker.noindex/tools/ops/dispatch.sh sandbox-p-watch-lead-20261028-05 /Users/josenunez/Projects/ml/snake-dqn-p-watch-lead-05-ready-broker.noindex/research/watch_lead_p/packet-05/dispatch-manifest.json --broker /Users/josenunez/Projects/ml/snake-dqn-p-watch-lead-05-ready-broker.noindex --cwd /Users/josenunez/Projects/ml/snake-dqn-p-watch-lead-05-ready-runtime.noindex --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-watch-lead-1028/attempt-05/state --dispatch-txt /Users/josenunez/Projects/ml/snake-dqn-p-watch-lead-05-ready-broker.noindex/research/watch_lead_p/packet-05/DISPATCH.txt --dry-run
```

black; cwd `/Users/josenunez/Projects/ml/snake-dqn-falsifier-p`; log `black-ready.log`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1 SNAKE_DQN_DEVICE=cpu nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m black --check --line-length 100 research/watch_lead_p tools/ops/dispatch.py tools/sandbox/manifest.py tools/sandbox/preflight.py tools/sandbox/registry.py tools/sandbox/supervisor.py
```

isort; cwd `/Users/josenunez/Projects/ml/snake-dqn-falsifier-p`; log `isort-ready.log`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1 SNAKE_DQN_DEVICE=cpu nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m isort --check-only --profile black research/watch_lead_p tools/ops/dispatch.py tools/sandbox/manifest.py tools/sandbox/preflight.py tools/sandbox/registry.py tools/sandbox/supervisor.py
```

flake8; cwd `/Users/josenunez/Projects/ml/snake-dqn-falsifier-p`; log `flake8-ready.log`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1 SNAKE_DQN_DEVICE=cpu nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m flake8  research/watch_lead_p tools/ops/dispatch.py tools/sandbox/manifest.py tools/sandbox/preflight.py tools/sandbox/registry.py tools/sandbox/supervisor.py
```

Shell syntax; cwd `/Users/josenunez/Projects/ml/snake-dqn-falsifier-p`:

```bash
bash -n research/watch_lead_p/run_job.sh
```

Whitespace; cwd `/Users/josenunez/Projects/ml/snake-dqn-falsifier-p`:

```bash
git diff --check
```

Sparse source proof executed `research.colab_cli.source.prepare(Path(runtime_checkout), manifest, Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-watch-lead-1028/engineering-05-resume-20261028-r1/source-plan-ready"))` followed by `source.install_fallback` in that fresh directory. The complete immutable plan, archive and receipt are retained there.

## Operator command — not executed

Append `--dry-run` to repeat offline validation. Exact dispatch command from the sealed DISPATCH.txt:

```bash
/Users/josenunez/Projects/ml/snake-dqn-p-watch-lead-05-ready-broker.noindex/tools/ops/dispatch.sh sandbox-p-watch-lead-20261028-05 /Users/josenunez/Projects/ml/snake-dqn-p-watch-lead-05-ready-broker.noindex/research/watch_lead_p/packet-05/dispatch-manifest.json --broker /Users/josenunez/Projects/ml/snake-dqn-p-watch-lead-05-ready-broker.noindex --cwd /Users/josenunez/Projects/ml/snake-dqn-p-watch-lead-05-ready-runtime.noindex --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-watch-lead-1028/attempt-05/state --dispatch-txt /Users/josenunez/Projects/ml/snake-dqn-p-watch-lead-05-ready-broker.noindex/research/watch_lead_p/packet-05/DISPATCH.txt
```

## Cleanup and boundaries

`cleanup.json` records removal of this run's owned `/private/tmp/astra-build-p-fix2-r1-*` directories and four superseded draft clones. Final ready checkouts and all evidence remain. The handed worktree retains its original untracked `saved_snakes` symlink. No existing tmux/job, released checkpoint, serving pin, older packet, protected outcome or qualified transport bytes were changed. All commits are local by explicit instruction; final Argon review and operator dispatch remain separate.
