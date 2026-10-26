# F5 engineering amendment verification — offline

Status: implementation verification only; HOLD remains. No real plan migration,
claim issuance/consumption, targeted timing, Colab/provider/dispatch operation,
scientific fit, outcome read, Mac companion workload, serving or release occurred.
The root coordinated bounded test lanes; every pytest command used nice 15 and
one numerical thread. The concurrency tests use two Python threads for claim-file
races, not numerical workers. macOS pytest's repository conftest selected one
OpenMP runtime; its waiting wrapper is not an additional numerical worker.

## Root integration results

Existing fit/data/job/protocol/timing/resume/accounting/qualification/package/
native-v6/ops admission/recovery suite: **245 passed, 1 skipped, 42.43s**. The
skip is the real-CUDA test, deliberately unavailable offline.

```sh
env OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q tests/test_bl_residual_f5_fit.py tests/test_bl_residual_f5_data.py tests/test_bl_residual_f5_job.py tests/test_bl_residual_f5_protocol.py tests/test_bl_residual_f5_timing.py tests/test_bl_residual_f5_timing_resume.py tests/test_bl_residual_f5_timing_accounting.py tests/test_bl_residual_f5_qualification.py tests/test_bl_residual_f5_package.py tests/test_f5_kernel_provenance_v6.py tests/test_ops_f5_timing_admission.py tests/test_ops_f5_plan_admission.py tests/test_bl_residual_f5_timing_recovery_package.py
```

Final new engineering integration suite: **63 passed, 38.45s** (21 reducer,
5 source closure, 9 plan successor, 23 supplement, 5 original protocol tests).
These test metadata-only plans of the real registered 2000x256 shape. They do
not open or migrate the original F5 plan, selected corpus or C/W outcomes.

```sh
env DYLD_LIBRARY_PATH=/Users/josenunez/Projects/ml/snake-dqn/venv/lib/python3.12/site-packages/torch/lib OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q tests/test_f5_amended_reduction.py tests/test_f5_engineering_seal.py tests/test_f5_plan_successor.py tests/test_f5_timing_supplement.py tests/test_bl_residual_f5_protocol.py
```

Verified boundaries include original published reduction content and annex
immutability; rehashed source/environment substitution; absent crossing vs role
coverage; terminal branches vs missing full-H300 path coverage; kernel strata;
all-actions bundle vs individual timing; no legal-count divisor; complete shard
reservations charged once; frozen recipe/loss/corpus/allocation; exact ordered
512,000 indices and typed RNG equality; independent predecessor file pin and
sampler-build proof; ordinary runtime BatchPlan export; consumed/retired/copied
claims; concurrent sole issuance and consumption.

Final Black check (100 columns), isort check (Black profile), flake8 and
`git diff --check` pass on the six changed/added engineering Python modules and
four new test modules. Lint removed an unused test import/assignment and shortened
closure classification text after the 63-test run; no behavior changed.

```sh
nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m black --line-length 100 --check research/bl_residual/f5_amended_reduction.py research/bl_residual/f5_engineering_seal.py research/bl_residual/f5_plan_successor.py research/bl_residual/f5_timing_supplement.py research/bl_residual/f5_protocol.py research/bl_residual/f5_package.py tests/test_f5_amended_reduction.py tests/test_f5_engineering_seal.py tests/test_f5_plan_successor.py tests/test_f5_timing_supplement.py
nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m isort --check-only --profile black research/bl_residual/f5_amended_reduction.py research/bl_residual/f5_engineering_seal.py research/bl_residual/f5_plan_successor.py research/bl_residual/f5_timing_supplement.py research/bl_residual/f5_protocol.py research/bl_residual/f5_package.py tests/test_f5_amended_reduction.py tests/test_f5_engineering_seal.py tests/test_f5_plan_successor.py tests/test_f5_timing_supplement.py
nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m flake8 research/bl_residual/f5_amended_reduction.py research/bl_residual/f5_engineering_seal.py research/bl_residual/f5_plan_successor.py research/bl_residual/f5_timing_supplement.py research/bl_residual/f5_protocol.py research/bl_residual/f5_package.py tests/test_f5_amended_reduction.py tests/test_f5_engineering_seal.py tests/test_f5_plan_successor.py tests/test_f5_timing_supplement.py
git diff --check
```

The create-only engineering seal reopens all original published evidence files,
compares them byte-for-byte against the historical Git publications, binds final
source/v6/loss/amendment bytes and checks the explicit executable seam bridge.
Its later successful `seal` and `verify` outputs are publication checks, not a
CUDA qualification, actual-plan compatibility verdict or ACCEPT_F5_FIT.
