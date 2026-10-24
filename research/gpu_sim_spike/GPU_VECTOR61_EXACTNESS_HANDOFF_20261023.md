# GPU vector61 exactness canary handoff

Prepared and Codex-V LIGHT ACCEPT; native A100 compile/parity NOT_RUN.
Runtime pin: `3ab2ef04a003161b2c582d98689126c020eb3d57`, contained in origin/main.
Frozen detached checkout: `/Users/josenunez/Projects/ml/snake-dqn-gpu-exactness-frozen-01`.
It applies the repair on qualified `42c84584`; qualified CLI tree remains
`d0d92bdb572b770284a1fea6152417ffa5f06bea`. Current main's newer CLI tree is
unqualified by the existing receipt, so dispatch uses this frozen checkout.
The runtime pin was merged into main without reverting upstream changes.

Artifact root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-engine-sprint-20261023-03/sparse-cap160-exactness`.
Canary manifest SHA-256:
`ef981f50db21253a550c19c8f19b438106255daaf8a9df6b9c118030af08b5fd`.
`VERIFICATION.json`, `DISCREPANCY_REGISTER.json`, `CODEX_V_LIGHT.json` and their
create-only sidecars bind the evidence. Original failed exports remain unchanged.

The declared repair and complete operation/field inventory are in
[GPU_VECTOR61_EXACTNESS_REPAIR_20261023.md](GPU_VECTOR61_EXACTNESS_REPAIR_20261023.md).
Dense column 39 used float32 danger distance/normalization/bonus arithmetic;
the recorded input's ordered CPU double result casts to 1059027653, whereas
A100 returned 1059027652. Static column 8 used torch float32 density division:
11/18.75 should cast to 1058418633; CUDA produced 1058418634, reproduced by
float32 reciprocal multiplication. This identifies the causal arithmetic chains,
not a proven disassembly-level FMA diagnosis.

Both backends now use integer count/squared-distance work, explicit RN binary64
native arithmetic, `__double2float_rn` at the output boundary, CPU sector LUTs
and pixel-distance carry. Flood normalization indexes a finite table of CPU
final-cast bytes. No fast math; FMA contraction off; precise div/sqrt on; TF32
and reduced-precision reductions disabled. Both arms stop on any CPU byte failure;
the GPU report includes every column's mismatch count and first raw-bit pair.

Verification: real frozen Mac Bash preflight 11.825734709 s, all three tiny roster
units PASS and all four saved-state direct/indexed x dense/static 61-column,
updated-carry and tensor-preservation probes PASS. Final frozen repair plus
transport canary/schema/qualification tests 126 passed / 5 CUDA-skipped in 5.84 s;
independent focused review 31 passed / 5 CUDA-skipped. Earlier full GPU package
suite 206 passed / 23 CUDA-skipped. `make lint`, changed-Python Black 100/isort/flake8,
Bash syntax and whitespace checks passed. No GPU job was dispatched.

Supplemental broad transport testing is not green. The unchanged offline
`test_colab_cli_dispatch_warmup.py` fake CLI requires its guard hook before the
initial read-only account inventory, which intentionally precedes supervisor
hook setup. Root and Codex-V independently reproduced this fixture failure on
the qualified checkout. Two supplemental broad runs were stopped after fixture
failures; only their owned test processes/descendants were terminated. No full
transport qualification claim or transport edit follows from this work.

Reproduce the final bounded tests from the frozen checkout:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-gpu-exactness-frozen-01
SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q \
  tests/test_colab_cli_canary.py tests/test_colab_cli_schema_failures.py \
  tests/test_ops_qualification.py research/gpu_sim_spike/test_sparse_bench.py \
  research/gpu_sim_spike/test_static_capped.py research/gpu_sim_spike/test_vector_geometry.py \
  research/gpu_sim_spike/test_vector_device.py research/gpu_sim_spike/test_vector_exact_native.py
```

Exact operator canary dispatch (prepared only):

```bash
nice -n 15 bash /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-engine-sprint-20261023-03/sparse-cap160-exactness/dispatch-canary.sh
```

Fresh job `sparse-bench-a100-canary-3ab2ef04`; fresh `state-canary` and
`canary-output`. Empty pin-specific ledgers. A100 >=12 vCPU, estimate 10 min,
wall cap 24 min, 4 units including 2/3 reserve, 10 u/hour, 64 MiB export ceiling.
Saved failed-state replay must pass natively before the E1024 all-roster ruler.
Main remains a non-dispatchable template until its helper validates a genuine
same-pin canary PASS, exports and release. Existing series 216 min / 36 unit ceiling,
prior charges, global balance/deadlines and parity-INVALID stop rule still apply;
no new allowance, automatic retry, labels, policy qualification or release.

No live tmux/jobs, released checkpoints, serving pins, qualified transport or
immutable prior evidence was modified.
