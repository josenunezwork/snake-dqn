# Native compile upload-cap repair

Prepared only. Codex-V LIGHT PASS; no allocation or dispatch.

The refused compile package staged a 1,264,488-byte source archive against
its unchanged 1,048,576-byte per-file transfer bound. The new package replaces
broad directories with the five production native source/loader triples,
compile controls, the audited eager import/publication closure, requirements,
and the supervisor's required bootstrap files. No runtime code, compiler flags,
preflight, scientific inputs, namespace, or budget was changed.

`tools/ops/repack_gpu_upload.py` makes this repair create-only at the existing
published runtime pin. This deliberately leaves the frozen runtime and original
packer bytes intact. It expands the sparse package's conservative dependency
directories into code/config files, excludes historical evidence/fixtures/logs,
and preserves all 131 files in its source-hash contract. The 17,876-byte saved
exactness NPZ remains a separate SHA-bound input. Checkpoints remain public
download inputs; neither source archive contains checkpoints or binary fixtures.

Runtime/broker (unchanged, qualified transport tree
`d0d92bdb572b770284a1fea6152417ffa5f06bea`):

```text
/Users/josenunez/Projects/ml/snake-dqn-gpu-compile-yamlfix-frozen-01
14f76fef53eccc7fd23c37643a8d78d97a8d6691
```

Final package:

```text
/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-engine-sprint-20261023-03/sparse-cap160-exactness-uploadfix-03
```

The original YAML-fix package and intermediate uploadfix-01/-02 preparation
artifacts remain intact. Only uploadfix-03 is the final operator package.
Its job IDs end in `-uploadfix03`; all output/state/ops names were unused at
preparation. The package includes an immutable copy of the repacker so dispatch
preparation does not depend on later authoring-checkout changes.

| Offline transport check | Compile | Sparse canary, pending compile |
| --- | ---: | ---: |
| Qualified `supervisor.validate` | PASS | PASS |
| Source files | 48 | 423 |
| Source archive bytes | 137,321 | 1,540,232 |
| Unpacked source bytes | 522,223 | 6,219,819 |
| Per-file transfer limit bytes | 1,048,576 | 67,108,864 |
| Fallback upload forecast seconds | 70.492 | 116.496 |
| Upload forecast ceiling seconds | 150 | 150 |
| Fallback plus 120-second fetch bound | 190.492 | 236.496 |
| Total transfer ceiling seconds | 300 | 300 |

The timing numbers use the existing upload qualification's measured rate and
fixed allowances. They are offline forecasts, not new provider observations.
Compile retains estimate 5 minutes, wall cap 20 minutes, unit cap 1.2 including
0.2 reserve, L4 at 3 u/hour. Sparse retains estimate 10 minutes, wall cap
24 minutes, unit cap 4 including 2/3 reserve, A100 at 10 u/hour.

## Verification evidence

`upload-audit/{compile,sparse-pending-compile}.json` bind the manifests, source
inventories, staged/local-input size checks, and upload-time qualification.
The repacker calls the frozen `supervisor.validate`, `source.prepare`, and
`research.bl_upload.qualify`; it checks every actual staging file and local
input against `max_export_bytes`. It does not change or bypass the supervisor.

`OFFLINE_VERIFICATION.json` records actual fallback installation with hash
verification for both archives, all 131 sparse hashes equal to the prior Mac
preflight, rejection of the original oversized bundle, refusal of absent
compile proof before creating a bound canary, explicit retry-receipt handling,
and absent final output/state paths.

`source-audit/compile-import/import.json` records import-only verification from
the actual extracted compile archive in the prior isolated Python 3.12 venv:
23 local modules, five native inventory entries, `torch.utils.cpp_extension.load`,
immutable publication, and CLI help. `source-audit/sparse-import.json` records
isolated sparse/exactness imports and both CLI help entrypoints from the actual
sparse fallback, with all 131 source hashes identical. No CUDA execution or
numerical workload was involved in these import probes. The existing 11.665-s
Mac E1/pre-recorded-state preflight is reused at the identical runtime pin.

Independent reviewer `/root/codex_v_light` returned PASS with no open findings;
its attributed create-only `CODEX_V_LIGHT.json` binds helper and manifest hashes.
It independently checked pinned/current qualification, supervisor validation,
all archive members, source-hash preservation, input bounds, fresh names,
generated shell syntax, missing-proof refusal, and accounting-retry support.

Additional authoring-checkout commands:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-gpufix
nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q tests/test_colab_cli_source.py tests/test_colab_cli_supervisor.py::test_existing_output_directory_is_refused_as_nonfresh research/gpu_sim_spike/test_native_compile.py
# 26 passed in 21.61s; macOS OpenMP isolation wrapper completed successfully.
nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m black --check --line-length 100 tools/ops/repack_gpu_upload.py
nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m isort --profile black --check-only tools/ops/repack_gpu_upload.py
nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m flake8 tools/ops/repack_gpu_upload.py
git diff --check
```

The full project suite was not run for this artifact-packaging-only repair.
The helper's real `prepare` command and saved `offline_verify.py` were executed
at nice 15; both completed successfully. Both generated dispatch scripts passed
`bash -n`. The initial uploadfix-02 probe correctly refused absent compile proof
but its test assertion expected the missing leaf rather than pathlib's missing
parent path; the final uploadfix-03 probe corrects that assertion and passes.

## Exact dispatch commands

Compile first:

```bash
nice -n 15 bash /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-engine-sprint-20261023-03/sparse-cap160-exactness-uploadfix-03/dispatch-compile.sh
```

After its same-pin PASS receipt with verified stop, sparse canary:

```bash
nice -n 15 bash /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-engine-sprint-20261023-03/sparse-cap160-exactness-uploadfix-03/dispatch-canary.sh
```

The sparse helper verifies the native compile receipt and its saved manifest,
terminal record, complete export hashes, all native sources/flags, and both
architectures before publishing `sparse-bench-canary.json`. It then reruns full
offline size/time validation before invoking the qualified ops wrapper. No
compile receipt has been fabricated. If the qualified dispatcher's permitted
accounting retry succeeds in `output-retry-1`, use this explicit override:

```bash
NATIVE_COMPILE_RECEIPT=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-engine-sprint-20261023-03/sparse-cap160-exactness-uploadfix-03/native-compile/output-retry-1/receipt.json nice -n 15 bash /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-engine-sprint-20261023-03/sparse-cap160-exactness-uploadfix-03/dispatch-canary.sh
```

An already-bound canary refuses a different receipt rather than replacing its
immutable binding. There is no main-benchmark dispatch in this repair package.
CUDA build success, GPU exactness, and performance remain unexecuted. No live
jobs/tmux sessions, released checkpoints, serving pins, qualified transport,
qualification registry, or prior hash-bound artifacts were modified.
