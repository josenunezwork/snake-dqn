# Verification commands

Final synthetic selection: 108 passed, 0 failed in 1.23s.

```bash
nice -n 15 env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ./venv/bin/python -m pytest -q -o addopts= -p no:xdist tests/test_f5_status_extractor.py tests/test_f5_closure_runtime.py tests/test_f5_closure_collisions.py tests/test_f5_external_collision_metadata.py tests/test_bl_residual_f5_r10a_annex.py
```

Style checks (nice 15; same single-thread environment):

```bash
./venv/bin/python -m black --check --line-length 100 research/bl_residual/f5_r10a_closure_20261026 research/bl_residual/f5_status_extractor.py tests/test_f5_status_extractor.py tests/test_f5_closure_runtime.py tests/test_f5_closure_collisions.py tests/test_f5_external_collision_metadata.py
./venv/bin/python -m isort --check-only --profile black research/bl_residual/f5_r10a_closure_20261026 research/bl_residual/f5_status_extractor.py tests/test_f5_status_extractor.py tests/test_f5_closure_runtime.py tests/test_f5_closure_collisions.py tests/test_f5_external_collision_metadata.py
./venv/bin/python -m flake8 research/bl_residual/f5_r10a_closure_20261026 research/bl_residual/f5_status_extractor.py tests/test_f5_status_extractor.py tests/test_f5_closure_runtime.py tests/test_f5_closure_collisions.py tests/test_f5_external_collision_metadata.py
git diff --check
```

These are metadata/synthetic checks. No real status scan, post-mortem, replay, fit,
canary, Colab or dispatch was executed. The final annex references all metadata
recovery plans and outputs by exact SHA-256.
