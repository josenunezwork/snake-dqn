# F5 HOLD engineering V4 verification

Status: **ENGINEERING ONLY; HOLD REMAINS; NO DISPATCH OR FIT ACCEPTANCE.**

Source commit after rebase and publication to main:
`00af09a16913eb8ef3eab60003891657858dff69`.
The V4 package pins that commit. Later publication commits add metadata only.
All 233 executable inventory members are unchanged by the rebase from the
pre-rebase implementation commit `58a5a454`. Upstream changes were outside that
closure; this task did not edit or stage any sandbox file. The engineering seal
verifies the original 563 historical paths against their original Git objects,
the unchanged scientific files against both historical sources, and the V3 seal
against its original digest. Prior seals/packages/evidence remain unchanged.

Completed test commands: **345 passed in 149.55 seconds** and **61 passed in
19.25 seconds**. These are two overlapping runs, not 406 distinct tests.
The follow-up run includes the final-package dry-run assertion added after the
broader run. No source behavior changed between the two successful runs.

```bash
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
nice -n 15 env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 "$PY" \
  -m pytest -q -n 0 -p no:cacheprovider \
  tests/test_bl_residual_f5_package.py tests/test_f5_engineering_seal.py \
  tests/test_f5_amended_reduction.py tests/test_f5_timing_supplement.py \
  tests/test_bl_residual_f5_protocol.py tests/test_f5_plan_successor.py \
  tests/test_f5_step3_driver.py tests/test_f5_step3_package.py \
  tests/test_bl_residual_f5_fit.py tests/test_bl_residual_f5_timing_resume.py \
  tests/test_bl_residual_f5_raw_envelope.py tests/test_f5_kernel_provenance_v6.py \
  tests/test_colab_cli_f5_timing_transfer.py tests/test_colab_cli_canary.py

nice -n 15 env \
  DYLD_LIBRARY_PATH=/Users/josenunez/Projects/ml/snake-dqn/venv/lib/python3.12/site-packages/torch/lib \
  SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 "$PY" -m pytest -q -n 0 -p no:cacheprovider \
  tests/test_f5_step3_package.py tests/test_colab_cli_runtime.py \
  tests/test_colab_cli_admission.py tests/test_colab_cli_schema_failures.py \
  tests/test_colab_cli_supervisor.py::test_existing_output_directory_is_refused_as_nonfresh
```

The macOS conftest supplies the same Torch OpenMP library for the first run;
the second sets it before startup. Each command uses serial pytest and one
numeric thread. No timing ProcessPoolExecutor was launched: fixture execution
is replaced by bounded synthetic test workers. An additional broad supervisor
integration run was stopped in this task's own pytest process and replaced by
the focused completed follow-up above; that interrupted run is not claimed as
a passing result. It used only fake provider fixtures. No existing job or tmux
session was stopped or changed.

Black (`--check --workers 1 --line-length 100`), isort (`--profile black
--check-only`), flake8 (`--jobs 1`), `git diff --check`, and
`bash -n research/bl_residual/f5_step3.sh` pass. Python checks cover:

```text
research/bl_residual/f5_amended_reduction.py
research/bl_residual/f5_protocol.py
research/bl_residual/f5_engineering_seal.py
research/bl_residual/f5_step3_driver.py
research/bl_residual/f5_step3_package.py
tests/test_bl_residual_f5_protocol.py
tests/test_f5_amended_reduction.py
tests/test_f5_step3_driver.py
tests/test_f5_step3_package.py
```

A separate read-only reviewer read both full reviews and inspected the final
engineering paths, including final canary admission, resume acceptance, elapsed
accumulation, retained shard liability and direct successor lineage. No material
finding remained. This is static engineering review, not Astra/Fable acceptance.

Create-only V4 engineering seal:
`research/bl_residual/ledger/F5_ENGINEERING_SEAL_20261026_V4/ENGINEERING_SEAL.json`.
SHA-256 `e73b2d24144cd3e375aeffca1706dd5c677c3b23da95dce49db30d15ba3f9038`.
Executable inventory SHA-256
`5b79b55097beb5b04dc71dbc08e756f975e74737cb5d10526b6a6ebd16117e7f`.

Create-only package root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/sol-20261026/f5-hold-corrections-v4/step3-package`.
Its seal SHA-256 is
`72e739989e13a1adbee19cbf40e13f280092fe1d11bca43464a0cf0a4417ffa4`.
The sibling `OFFLINE_VERIFICATION.json` binds all manifests, envelope, inventory,
dry-runs and test counts. `offline_verify.py` contains the exact offline checks.
Repository `PUBLICATION.json` references those artifacts by absolute path/hash.

Executed actual module operations under `nice -n 15`, all numeric thread
controls (including OMP_THREAD_LIMIT) set to one, and bytecode disabled:

```text
python -m research.bl_residual.f5_engineering_seal seal ABS_V4_SEAL
python -m research.bl_residual.f5_engineering_seal verify ABS_V4_SEAL
python -m research.bl_residual.f5_step3_package seal ABS_V4_PACKAGE
python -m research.bl_residual.f5_step3_package dry-run ABS_V4_PACKAGE
python ABS_V4_PARENT/offline_verify.py
```

`python` above is the absolute project venv interpreter; `ABS_V4_SEAL` and
`ABS_V4_PACKAGE` are the paths above. The local script is run with
`PYTHONPATH=/Users/josenunez/Projects/ml/snake-dqn-f5rawenv`. The package parent
must exist; an initial absent-parent refusal created no package and no existing
artifact was overwritten when the parent was prepared.

The seal's real `verify` operation passes. Package and real-driver dry-runs pass
against all staged inputs and the dedicated empty ledger. The unchanged
supervisor validates the preliminary qualification manifest offline and refuses
the measurement template with `dispatch requires a SHA-bound family canary
receipt`. No provider is constructed or called by these validations.

Synthetic final target→acceptance→sealed pair→driver dry-run tests pass, including
the unchanged real `canary.evidence`/`canary.check` and runtime scanner with
synthetic broker transfer/CUDA observations. Input mutation and missing-export
refusals pass. Those observations are test fixtures, not actual CUDA proof or
operational receipts. No production acceptance/qualification/forecast/final pair
is manufactured. The initial package is a HOLD template; the real final pair
requires the future artifacts and independent reviews in `ENGINEERING_V4.md`.

No real timing supplement, fit, evaluation look, Colab/provider call, dispatch,
release or serving/checkpoint/pin change occurred. No `tools/sandbox/` or
`research/sandbox_*` file was edited or staged by this task.
