# Native compile VM dependency repair

Prepared only; independent Codex-V LIGHT PASS. No dispatch or CUDA compile was performed.

The failed `native-compile-l4-1912f403` reached the package's eager `DeviceSim`
import, then `src.core.config_loader`, and stopped on missing `yaml` before nvcc.
`native_compile_requirements.txt` now includes the shared spike requirements, as
the other spike jobs do. This supplies PyYAML and Pydantic as well as Ninja;
the supervisor's Torch 2.9.1 / NumPy 1.26.4 constraints remain in force.
The compile manifest also includes `src` in its source bundle for these imports.
Native Python/shell entrypoints, compiler flags, CUDA/C++ sources, and caps are unchanged.

Fresh runtime worktree:
`/Users/josenunez/Projects/ml/snake-dqn-gpu-compile-yamlfix-frozen-01`
on `t34-gpu-compile-yamlfix-20261023`, published pin
`14f76fef53eccc7fd23c37643a8d78d97a8d6691`.
Its transport tree is the existing qualified
`d0d92bdb572b770284a1fea6152417ffa5f06bea`. As in the preceding compile
package, this frozen pin is recorded in main's ancestry while main retains its
existing transport. The qualification registry was not changed.

Fresh artifact package:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-engine-sprint-20261023-03/sparse-cap160-exactness-yamlfix-01`.
It was built with `native_compile_pack`, `sparse_preflight`, and `sparse_pack`.
Previous failed packages, manifests, receipts, and logs remain intact.

## Verification

- New isolated Python 3.12 venv, no system or user site packages, installed
  `torch==2.9.1 numpy==1.26.4 -r research/gpu_sim_spike/native_compile_requirements.txt`.
  `pip check` passed. `import-preflight.py` and `import-checkout.json` retain the
  exact audit and installed versions in the fresh artifact package.
- The same isolated interpreter, with `-I`, passed from the actual extracted
  pinned `source.prepare` fallback bundle: 342 files, 5,071,404 unpacked bytes;
  archive SHA256 `035081c41568d4296be089e2c7369a6a2b311c5685e8cb5845e33c92efa545ff`.
  `source-audit/import-bundle.json` records 23 local and 946 external modules.
  The audit covers eager imports, deferred `torch.utils.cpp_extension.load` and
  `sparse_units` imports, real immutable publication, CLI `--help`, and all five
  extension inventory entries. It does not fabricate a native compile receipt.
- Real Mac sparse Bash preflight completed scripted/frozen/mixed E1 units and
  saved-state exactness replay in 11.665 seconds at the published runtime pin.
  The saved fixture hash remains
  `68e182b29e66a564743053dcc3c41b3fc0a8f55b2294a747c336a58a220028d2`.
- `SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q research/gpu_sim_spike/test_native_compile.py tests/test_colab_cli_canary.py tests/test_colab_cli_schema_failures.py tests/test_ops_qualification.py`:
  **107 passed** from the implementation worktree.
- With the same environment prefix/interpreter, from the frozen worktree:
  `-m pytest -q tests/test_colab_cli_source.py research/gpu_sim_spike/test_sparse_bench.py`:
  **29 passed**.
- Black, isort, flake8 on `native_compile_pack.py`, Bash syntax on native and
  generated dispatch scripts, and `git diff --check`: PASS.
- Codex-V LIGHT independently checked the two-file repair, isolated import
  reports, all package sidecars, all 16 sparse-manifest input hashes, source
  qualification at worktree and pin, compile manifest validation, absent fresh
  output/state directories, and compile-before-sparse admission: PASS.

Supplemental validation limitation: a broader run including
`tests/test_colab_cli_supervisor.py` showed failures in fake-CLI cold/staging
tests and was stopped before a complete failure report. Those failures were not
classified or independently reproduced, and the broad suite is not claimed green.
Only that run's explicitly identified owned pytest processes were terminated;
no live jobs or tmux sessions were stopped.

## Exact dispatch commands

Run the compile check first:

```bash
/Users/josenunez/Projects/ml/snake-dqn-gpu-compile-yamlfix-frozen-01/tools/ops/dispatch.sh native-compile-l4-14f76fef /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-engine-sprint-20261023-03/sparse-cap160-exactness-yamlfix-01/native-compile/native-compile.json --broker /Users/josenunez/Projects/ml/snake-dqn-gpu-compile-yamlfix-frozen-01 --cwd /Users/josenunez/Projects/ml/snake-dqn-gpu-compile-yamlfix-frozen-01 --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-engine-sprint-20261023-03/sparse-cap160-exactness-yamlfix-01/native-compile/state --dispatch-txt /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-engine-sprint-20261023-03/sparse-cap160-exactness-yamlfix-01/native-compile/DISPATCH.txt
```

After its same-pin PASS receipt with verified stop exists, run the sparse canary:

```bash
nice -n 15 bash /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-engine-sprint-20261023-03/sparse-cap160-exactness-yamlfix-01/dispatch-canary.sh
```

The helper verifies the compile receipt before creating/validating the final
dispatchable sparse manifest. Compile: L4, estimate 5 minutes, effective cap
20 minutes / 1.2 units including 0.2 reserve, rate ceiling 3 u/hour.
Sparse canary: A100, estimate 10 minutes, effective cap 24 minutes / 4 units
including 2/3 reserve, rate ceiling 10 u/hour. Existing sprint limits apply.
CUDA compilation, native exactness, and performance remain unexecuted.
Released checkpoints, serving pins, prior hash-bound evidence, and live jobs
were not modified.
