# F5 engineering successor verification

Final local test result: **231 passed in 121.95 seconds**. No CUDA test was
executed or claimed. All test execution used `nice -n 15`, CPU, `pytest -n 0`,
and OMP/OMP_THREAD_LIMIT/MKL/OpenBLAS/Numba/VecLib thread caps of one.

```bash
nice -n 15 env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  -m pytest -q -n 0 \
  tests/test_bl_residual_f5_package.py tests/test_f5_engineering_seal.py \
  tests/test_f5_amended_reduction.py tests/test_f5_timing_supplement.py \
  tests/test_bl_residual_f5_protocol.py tests/test_f5_plan_successor.py \
  tests/test_f5_step3_driver.py tests/test_f5_step3_package.py \
  tests/test_bl_residual_f5_fit.py tests/test_bl_residual_f5_timing_resume.py \
  tests/test_bl_residual_f5_raw_envelope.py tests/test_f5_kernel_provenance_v6.py \
  tests/test_colab_cli_f5_timing_transfer.py
```

Black (`--check --workers 1 --line-length 100`), isort (`--profile black
--check-only`), flake8, and `git diff --check` passed for the sixteen changed
Python files (eight implementation files and their eight focused test files).
`bash -n research/bl_residual/f5_step3.sh` passed. The actual staged driver and
package dry-runs are recorded separately by immutable digest in the publication
record; syntax checks alone are not application qualification.

An independent read-only reviewer inspected both original rejection artifacts,
the corrected source/tests and expanded bridge. The reviewer found no remaining
actionable defect in the original engineering corrections after follow-ups for
canonical lineage existence, shard evidence bindings, measurement forecast
admission, resumed wrapper allowlisting and nonadditive collection timing.
This is static engineering review, not Astra/Fable ratification, live L4 proof,
independent package acceptance, or operational dispatch authority.

The final implementation commit after rebase is
`f86f39b25114893f1523daa28b27f9a66032858d`. The package pins this exact commit.
Subsequent publication commits add only immutable seal/verification metadata;
the executable inventory must remain byte-identical. No source changes were
introduced by rebasing. Historical comparison is performed by the engineering
seal against both designated source revisions and the original evidence Git
objects. The original seal is preserved and linked as predecessor.

No timing supplement, original timing rerun, fit, evaluation look, dispatch,
Colab/provider call, tmux action, serving/pin change, or released-checkpoint write
was performed. No `tools/sandbox/` or `research/sandbox_*` file was edited or
staged. The final package remains pending genuine matched CUDA qualification,
complete measured envelope and independent acceptance.
