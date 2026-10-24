# R32 CPU canary directory fix — prepared, no dispatch

This supersedes only the CPU canary dispatch in `HANDOFF.md`. The failed pack,
its inputs, receipts and log remain unchanged. No runtime was allocated or
dispatched, no factory lease was accessed, and no fit/world/timing run was started.

## Frozen executable pin

- Pin: `c1ab27508ef097108260b89061564f81aaae26fc`.
- Frozen worktree: `/Users/josenunez/Projects/ml/snake-dqn-r32-canary-fix-20261024`.
- Branch: `t34-r32-canary-fix-20261024`, published and contained in `origin/main`.
- Base: original failed CPU pin `7b114eaa6c672a95c67bdecb1c4fa01ba3ff6231`.
- Qualified transport tree: `d0d92bdb572b770284a1fea6152417ffa5f06bea`.

The pin adds only `lifecycle.py` and `tests/test_r32_package.py` to the original
CPU source. `start_attempt` accepts the supervisor's pre-created empty output
directory; files, symlinks, nonempty directories, payloads and sidecars refuse
before publication. Atomic create-only exports are unchanged.

The fix is also present on the current main line. Newer independent transport
changes on main have a different tree from the recorded qualification. Both
`--broker` and `--cwd` below must use the frozen sibling; do not substitute the
current main checkout. No concurrent main changes were reverted.

## Evidence

Artifact root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-32/implementation`.

- `pack-c1ab2750-01/pack.json`: standard `package build` succeeded, including
  source-bound inputs, empty ledgers and the original consumer trace matching
  this CPU pin. It remains preparation, not scientific or serving authority.
- `pack-c1ab2750-01/CANARY_REFRESH.json`: preflight, validation, exact prior
  failed-canary spend and remaining shared-series allowance.
- `pack-c1ab2750-01/CODEX_V_LIGHT.json`: independent final review.
- `pack-c1ab2750-01/SUPERVISOR_VALIDATE.json`: captured successful validation
  command, cwd, exit code and output; validation does not allocate.
- `preflight-c1ab2750-01/output/receipt.json`: actual Bash CLI canary entrypoint,
  pre-created `output/`, 3/3 units; unit times 0.802340 / 0.324418 / 0.430052 s,
  total worker time 1.557003 s excluding interpreter startup.
- `pack-c1ab2750-01/clean-import-preflight.json`: fresh managed Python 3.12.15,
  88 import statements / 34 local modules, no system/user site packages.
- Fresh canary manifest SHA256:
  `64b7f849641cfc1323687c78f8d35d8893379a56f2444c890d69fbd0d377f6bd`.

Root verification: **21 passed in 12.08 s** for both R32 test files. Independent
Codex-V execution on the frozen pin: **11 passed, 3 deselected in 2.04 s** for
the new regression tests. All numerical work was serial, one thread, with nice
floor 15 (the reviewer could inherit an even lower scheduling priority).
Black (100 columns, one worker), isort, flake8, Bash syntax and diff checks passed.
Supervisor `validate` passed for the fresh canary, without allocation.
The two targeted supervisor guards also passed on the frozen pin: **2 passed,
76 deselected in 6.21 s**, preserving refusal of a used local output directory
and refusing exhausted shared reservations before allocation.

The root test command was:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -p no:xdist \
 tests/test_r32_package.py tests/test_r32_rehearsal.py
```

The same environment and interpreter ran the supervisor guard selection:

```bash
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -p no:xdist \
 tests/test_colab_cli_supervisor.py \
 -k 'existing_output_directory_is_refused_as_nonfresh or shared_reserved_budget_refuses_before_new_allocation'
```

The tiny local CLI used fresh `preflight-c1ab2750-01/inputs` from
`package tiny-inputs`, explicitly created an empty sibling `output/`, and ran
`bash research/r32_rehearsal/colab.sh` from the frozen worktree with
`R32_MODE=canary`, `COLAB_CLI_ROOT` and `COLAB_INPUT_DIR` pointing to that preflight,
the project venv first in `PATH`, and the same one-thread environment.

## Exact canary dispatch command — operator only

Verify current account liabilities, positive debit, host capacity and a free
funded assignment under the existing priority/account rules before admission.
The command has not been run:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
 /Users/josenunez/Projects/ml/snake-dqn-r32-canary-fix-20261024/tools/ops/dispatch.sh \
 r32-cpu-canary-c1ab2750-01 \
 /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-32/implementation/pack-c1ab2750-01/canary-template.json \
 --broker /Users/josenunez/Projects/ml/snake-dqn-r32-canary-fix-20261024 \
 --cwd /Users/josenunez/Projects/ml/snake-dqn-r32-canary-fix-20261024 \
 --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-32/implementation/state-cpu-canary-c1ab2750-01 \
 --dispatch-txt /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-32/implementation/pack-c1ab2750-01/DISPATCH_CANARY.txt
```

Canary caps remain 20 minutes / 0.0266666667 u, reserve included 0.0044444444 u,
at planning 0.08 u/h. Prior `r32-cpu-canary-7b114eaa-01` has an INCOMPLETE receipt
with verified stop and zero exported units. Its 136.264540542 s / 0.0030281009 u
charge leaves 285.728924324 min / 0.3809718991 u before this canary, within the
unchanged shared 288 min / 0.384 u ceiling. No automatic retry was introduced.

Do not dispatch `main-template.json` directly. A later fresh main admission needs
same-pin successful canary proof, verified release and an envelope reduced to the
actual remaining series allowance, including all attempts. A full new canary cap
would leave at most 265.728924324 min / 0.3543052324 u, below the old main template.

The generated pack's `mac-consumer-plan.json` is not a Mac dispatch instruction.
Retain the external guarded `MAC_PLAN_880c8553.json` described in `HANDOFF.md`;
the live factory still blocks a current-guard fit. Consumer/strict/registry and
serving readiness remain blocked. No live tmux session, released checkpoint,
serving pin, factory process, prior pack or hash-bound evidence was modified.
