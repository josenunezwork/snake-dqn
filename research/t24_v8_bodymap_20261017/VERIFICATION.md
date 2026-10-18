# T24 CPU qualification

Production source is the retained-template implementation at `4b4723ff`, with
shared actor integration at `6aae4bf8`. Later commits extend tests, benchmark
artifacts and documentation. Source hashes in the paired performance receipts
identify the exact measured bytes.

Final result: **410 passed, zero skips** in two disjoint suites: 382 core tests
in 414.03 seconds and 28 compiler tests in 151.23 seconds. Both include slow
markers. Black, isort and flake8 pass for changed Python source/tests/harnesses.
The compiler used Apple clang 21.0.0 (`clang-2100.3.34.2`).

Use the project Python from the main checkout. The worktree has no separate
venv. Run with one host thread and CPU forced:

```bash
export PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu
export CXX=/usr/bin/clang++
export OMP_PREFIX=/Users/josenunez/Projects/ml/snake-dqn/venv/lib/python3.12/site-packages/torch
export TORCHINDUCTOR_CACHE_DIR=/tmp/t24-inductor-apple
"$PY" -m pytest -q --no-showlocals --tb=short tests/test_perf_compile.py
"$PY" -m pytest -q --no-showlocals --tb=short \
  tests/test_v8_nb.py tests/test_ego_raster.py tests/test_ego_raster_bv8.py \
  tests/test_ego_live_identity.py tests/test_ego_live_bv8.py \
  tests/test_grid_sim_parity.py tests/test_grid_sim_nb.py tests/test_perf_actor.py \
  tests/test_m3c_trainer.py tests/test_t16_colab_bundle.py \
  tests/test_t16_colab_compare.py tests/test_t16_colab_environment.py \
  tests/test_t16_colab_notebook.py tests/test_t16_colab_sampler.py \
  tests/test_t16_colab_smoke.py tests/test_t19_compare.py \
  tests/test_t19_league_pool.py tests/test_t19_mixed_collector.py \
  tests/test_t21_actor_control.py tests/test_t21_actor_snapshot.py \
  tests/test_t21_colab_long_run.py tests/test_t21_demo_snapshot.py \
  tests/test_t21_execution_snapshot.py tests/test_t21_full_state.py \
  tests/test_t21_multi_integration.py tests/test_t21_sync_integration.py \
  tests/test_t22_frozen_pool.py tests/test_t22_random_safe.py \
  tests/test_t24_v8_bodymap.py
```

No slow markers are deselected. Checkpoints and demonstration fixtures are
present on this Mac. The trainer cases are bounded 12k/14k-transition CPU
regressions, not research training or performance results. The Colab tests
inspect static/mocked contracts and do not operate the running Colab session.

The first qualification attempt exposed an existing default Inductor cache
library referencing `@rpath/libc++.1.dylib` without an LC_RPATH. A process-local
Apple compiler, bundled Torch OpenMP headers/library, and fresh private cache
resolved the compiler tests. Installed packages and the shared cache were not
modified as a workaround. A `DYLD_LIBRARY_PATH` probe was rejected after it
crashed during framework import; it is not part of this reproduction.

The new actor method also required updating the scripted-only static-pool test
double. The initial failure caused pytest's default local-variable rendering
to access a closed memory map; `--no-showlocals` prevents that diagnostic crash.
The mock now delegates its learner-only feature request normally. Production
code was unchanged during these qualification repairs.

The final text and JUnit receipts are `verification_compile.{txt,xml}` and
`verification_core.{txt,xml}`. The frozen pre-T24 kernel is checked by SHA-256,
and the active membership BFS's entire discovery queue (including the queued
frontier beyond a cap) is compared with the frozen BFS. The six-policy oracle
uses independent legacy feature imports, physical long bodies, and forced
all-policy death/respawn.
