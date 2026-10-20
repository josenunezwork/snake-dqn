# T34 v1 infrastructure verification

This is implementation evidence only. No v1 efficacy estimate, strict gate,
serving qualification or release is claimed. The owner-approved
[design v2](../../docs/research/final_run_20261019/DESIGN_V2_SHADOW.md) supersedes
v1 efficacy. M1–M4 and the v2 implementation were not run in this task.

## Source and chronology

- Worktree: `/Users/josenunez/Projects/ml/snake-dqn-pathb`.
- Branch: `t34-pathb-20261019`.
- Original protocol committed and pushed first at
  `b5fff59ecd9902d7122adafe577d9603094748a7`.
- Prospective timing clarification at
  `8bbf0d0bafab5d7534edbfe14b5096c1f18375d5`: the next actual action boundary
  remains the cutoff; nominal deadline accounts for the engine's full sleep.
- Prospective analysis clarification at
  `d95e34390f8cc021168e630827deba9ce1f9d559`: alive-frame survival confidence
  guards, completion receipts and explicit disjoint smoke keys.
- Integrated smoke-02 used Python producer source
  `3c318aaea13cbee9848e6d94746f35ba2ed77734`. Subsequent changes clarified frontend
  wording and documented retirement of v1 efficacy; no Python behavior changed.
- Final smoke-03 used merged source
  `de13b916cab73a3303a919a60406e89f8dd17824`, including main's shared-kernel
  changes. It is the final combined-source entrypoint evidence.

## Reproducible checks

Run from the worktree root unless a command explicitly changes directories.
The absolute project venv is used. Numerical libraries use one thread per
process; the harness permits one parent plus one worker. No xdist or automatic
worker fan-out was used for pytest.

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q \
  tests/test_b8_release_monitoring.py tests/test_t34_pathb_serving.py \
  tests/test_t34_pathb_controller.py tests/test_t34_pathb_harness.py tests/test_t34_pathb_analysis.py

env BLACK_NUM_WORKERS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  make PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python lint

env BLACK_NUM_WORKERS=1 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m black --check --line-length 100 \
  research/t34_pathb research/t3_rollout_labels_20261015/engine.py tests/test_t34_pathb_*.py
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m isort --check-only --profile black \
  research/t34_pathb research/t3_rollout_labels_20261015/engine.py tests/test_t34_pathb_*.py
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m flake8 \
  research/t34_pathb research/t3_rollout_labels_20261015/engine.py tests/test_t34_pathb_*.py
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m flake8 --jobs 1 \
  src web/backend research/t34_pathb research/t3_rollout_labels_20261015/engine.py tests/test_t34_pathb_*.py

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -m 'not slow'

cd /Users/josenunez/Projects/ml/snake-dqn-pathb/web/frontend
env GOMAXPROCS=1 OMP_NUM_THREADS=1 npm test -- --pool=forks --poolOptions.forks.maxForks=1 --poolOptions.forks.minForks=1
env GOMAXPROCS=1 OMP_NUM_THREADS=1 npm run build
cd /Users/josenunez/Projects/ml/snake-dqn-pathb
git diff --check
```

The focused Python check passed **76 tests** after repairing compatibility with
monitoring fixtures that construct partial sessions via `__new__`. T34-specific
checks cover **42 tests**: real clone identity, default-off/rollback scope,
bounded outstanding work, stale/late/invalid results, post-validation deadline,
source/seed/roster/log bindings, complete receipts and analysis acceptance guards.
The full frontend check passed **185 tests in 22 files**. The production frontend
build passed. Lint and whitespace checks passed.

The first non-slow attempt exposed the partial-session regression above and was
interrupted; it is not a passing verification. The second attempt's terminal
result is recorded below when complete. An initial frontend `singleFork` run
produced DOM-cleanup failures; normal file isolation with exactly one fork passed
without test changes. Existing React `act` and Node localStorage warnings remain.
The read-only UI review found no functional blockers; its initial-decision copy
finding was corrected and covered before the final frontend run. Rendered layout
and browser accessibility-tree behavior were not verified because no live server
was started.

## Real entrypoint smoke

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-pathb
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.t34_pathb.run smoke \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-pathb-20261019/smoke-02 \
  --smoke-key implementation-v2
```

This command already completed; do not reuse its create-only output directory.
The key names a second **v1 plumbing check**, not design v2. The smoke contains
eight paired world records, 24 arm logs and four timing rows. Its analyzer reports
`complete: true`, `decision: SMOKE_ONLY`, `timing: PLUMBING_ONLY`,
`release_authorized: false`, and `client_latency: NOT_MEASURED`. No efficacy
values are interpreted or reported. The smoke forces a short H2 search at each
living decision; it cannot establish real long-body feasibility or acceptance.

Artifacts are outside Git at
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-pathb-20261019/`.

| Smoke-02 file | SHA-256 |
| --- | --- |
| `manifest.json` | `21450b1c8ad33aea568a692f4277b7f8d03c9428610dd4502005c125a759a6e7` |
| `analysis.json` | `df49ba778dfc4fe3d10b8b4a487612b512c811d32fcb27b900a9f3016a884949` |
| `complete.json` | `8b095f115d3bee1de5f6767382074bc5d7b6a2a7eae7db846f8ed68f5b4e859f` |
| `seed-audit.json` | `bddb9b7fe222648b4bc437538352402baf254fc8a9d35a0bc8152c4910bd74f3` |

Smoke-01 is preserved separately, with its original source identity. One smoke-02
preparation was refused before allocating a bank because an unrelated untracked
document was present. The producer guard was narrowed to untracked producer code
and config, then the explicit smoke command above ran once successfully.

An isolated in-process `GameSession` check also confirmed that flag-on installs
the eligible assisted hook and flag-off plus `SNAKE_SERVE_CHECKPOINT=b8-s3`
constructs bare B8. It started no server and changed no live process.

## Boundaries

- No real efficacy bank, strict worlds, v2 measurements, browser/network latency
  qualification, natural long-body timing qualification or release.
- No Colab, RunPod, paid allocation, live service or tmux operation.
- No direct modification of released checkpoints, serving pins, other worktrees
  or original hash-bound protocol bytes. Clarifications are sibling files;
  main's owner-ratified authority/contract updates were imported by merge.
- The two untracked spec addenda written by another actor are preserved and are
  not staged as this task's changes. The committed v2 design is the basis for
  retiring v1 efficacy; the earlier H300 addendum is not implemented here.

## Full non-slow terminal result

The required serial pre-merge run completed in **4,164.79 seconds**:
**8,060 passed, 14 failed, 26 skipped, 101 deselected**. It is not a green suite.
All 42 Path B tests passed. The retained log is
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-pathb-20261019/non-slow-02.log`.

Thirteen failures load compiled modules against missing `@rpath/libc++.1.dylib`.
The other failure is
`tests/test_colab_cli_supervisor.py::test_provider_calibration_timeout_is_a_real_elapsed_bound`:
the receipt reports `181.76300000000708` rather than the fixture's expected `135`.
Its cleanup succeeded and stop was verified. Supervisor, fixture, test and
root-conftest files are byte-identical to branch baseline `3ea25070`.

The supervisor case passed alone with the original environment (**1 passed in
17.70 seconds**). A separate venv-first-PATH attempt failed earlier in the fake
CLI version step. Both logs are retained; the original full-run failure is not
erased or described as fixed. The exact isolated commands were the serial pytest
command above with only this node ID, first with
`PATH="/Users/josenunez/Projects/ml/snake-dqn/venv/bin:$PATH"`, then without that
PATH override. Evidence files: `supervisor-isolated.log` and
`supervisor-original-env.log` in the artifact root.

## Incoming-main integration review

The read-only review against incoming main `105debee` found no overlapping changed
filenames (203 incoming paths and 28 Path B paths, merge base
`3ea25070cd63beca11cb8d021becafcd49157c1d`). Incoming raster/v8 feature kernels
are used by live served policies, so fresh post-merge clone/parity tests and a new
plumbing smoke remain necessary.

Another implementation lane independently documented these inherited failures:

- S1 admission rejects changed source hashes in bound historical timing evidence.
- Two T22 frozen-source tests reject the newer `membership` keyword.
- T4 snapshot graft rejects mismatched `_food_array` capacities.

Its clean-baseline reproduction log is
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-exit-fit-prep-20261019/acceptance.ZCCuSG/baseline-four-failures.log`.
Its `research/t34_exit_b8_fit/VERIFICATION.json` also records compiler-loader
failures separately from a passing system-runtime follow-up. These are context,
not a substitute for this branch's own test results.

Main subsequently advanced to `dc1daf79`: its owner-ratified authority/statistics
updates and implementation work were merged after the full run completed, with
no conflicts. The newer S1 test fixture fixes its historical source-hash issue;
the earlier four-failure description is not a claim about this newer main.
P7 retires the v1 bank-block statistical frame. No v1 analysis qualifies a
future fixed controller under P7. The harness's T15 constructor disables B8
release monitoring, and Watch runs do not submit leaderboard scores.

## Combined-source checks

The integration command below passed **347 tests, 2 deselected in 101.65 seconds**
after merging main at `fecf9c51945c7f50c3d6a8af13cd8f57b4ba95f2`. The later
README-only commit changes no tested Python behavior. The full long suite was
not repeated after that merge; these checks cover the changed shared feature
dependencies and all Path B tests.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-pathb
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -m 'not slow' \
  tests/test_b8_release_monitoring.py tests/test_t34_pathb_analysis.py \
  tests/test_t34_pathb_controller.py tests/test_t34_pathb_harness.py tests/test_t34_pathb_serving.py \
  tests/test_ego_live_identity.py tests/test_ego_raster.py tests/test_ego_raster_bv8.py \
  tests/test_t24_v8_bodymap.py tests/test_grid_sim_nb.py tests/test_grid_sim_parity.py \
  tests/test_web_watch_roster.py tests/test_web_served_checkpoint.py \
  tests/test_with_slots.py tests/test_s1_admission.py

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  TORCHINDUCTOR_COMPILE_THREADS=1 MAX_JOBS=1 CXX=/usr/bin/clang++ DYLD_LIBRARY_PATH=/usr/lib \
  TORCHINDUCTOR_CACHE_DIR=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-pathb-20261019/compile-cache-system \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -m 'not slow' tests/test_perf_compile.py
```

Evidence: `post-merge-tests.log` and `compiler-system-runtime.log` in the artifact
root. The compiler command supplies the system C++ runtime only to that process
and uses a fresh private cache; no installed library or global cache is changed.
Its complete non-slow module passed **26 tests, 2 deselected in 100.12 seconds**.
Style was checked again on combined source using Black's single worker, isort,
and `flake8 --jobs 1` over `src`, `web/backend`, the Path B package, T3 engine and
the four Path B test files.

## Final merged-source smoke and evidence hashes

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-pathb
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.t34_pathb.run smoke \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-pathb-20261019/smoke-03 \
  --smoke-key merged-infrastructure-v1
```

Exit code 0. The create-only smoke-03 directory contains all eight expected
records; analysis reports `complete: true`, `SMOKE_ONLY`, `PLUMBING_ONLY`,
`release_authorized: false`, and true client latency `NOT_MEASURED`. The command
already ran; use the fresh owner-smoke directory/key in README for another check.
This proves combined-source plumbing, not natural long-body feasibility or skill.

All files below are relative to the external artifact root named above.

| Evidence file | SHA-256 |
| --- | --- |
| `smoke-03/manifest.json` | `99fa1d32342a0dc54a4f82ffcad168f4d0435e017ce28eabe0e5446ce8c8b96c` |
| `smoke-03/analysis.json` | `df49ba778dfc4fe3d10b8b4a487612b512c811d32fcb27b900a9f3016a884949` |
| `smoke-03/complete.json` | `43bd6463ed2ff46038eadad77287ceccc411cdff01660d0893e057e68333efb7` |
| `smoke-03/seed-audit.json` | `bddb9b7fe222648b4bc437538352402baf254fc8a9d35a0bc8152c4910bd74f3` |
| `non-slow-02.log` | `985b4b28d610634c7db09fdb57ed16d735e57fd2124922a420375e6c1449a8d2` |
| `post-merge-tests.log` | `3bdc46f3319e3f107dae531e9834ab38cbd7288674e2e3f4848774e0fecd1972` |
| `compiler-system-runtime.log` | `a5b276c5288bf9d99c18b7ca0fe327abc6d7adf2c5858db47631d9412ff4c7cb` |
| `supervisor-original-env.log` | `cc5a96636285f9a4bc66fe70480ad1b26f35d3a7b16024883336caa5d1776441` |
| `frontend-final.log` | `ab942e69edc8ac7da765a5c92ad9cfbe4f0381150f0a2de5118c59fe27e729ff` |
| `frontend-build.log` | `bcd1b9a4e5299bfe8cf7e4b403aa4829509abc3d59b44ddd27faa062d4bc2801` |
