# Kernel creation identity correction — 2026-10-22

Status: **NOT QUALIFIED; Claude must requalify before merge or dispatch.**
This change modifies the qualified `research/colab_cli` Git tree and executable
inventory. The previous qualified tree is
`553af3f692515b2bf66173356fe4d099acc96942` (registry qualified commit
`f21a92edb86550df5263dbed4be7e2ff0bfa4c17`). Offline regressions do not renew it.

## Evidence and root cause

Incident: `p1-y3-06-canary-20261022`, job
`bl-p1-e8-04-canary-5d877123-pack04`. Immutable evidence:

- `/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/logs/p1-y3-06-canary-20261022.log`
- `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-latebody-06-e8-04-review-20261022-04/final/canary-output/receipt.json`
- The sibling `warmup-measurement.json` and ops state directory
  `ops/state/p1-y3-06-canary-20261022/bl-p1-e8-04-canary-5d877123-pack04/`.

The first warmup exec timed out after 30.0104 seconds. On warmup attempt two,
the guard's GET inventory had more than one entry and refused. The session
ended after 42.6124 seconds: `failure_class=SUPERVISOR`, `terminal=null`, no
uploads or exports, `stop_verified=true`, `cleanup_error=null`, and no account
assignments afterward. No workload ran. Cleanup removed the CLI session row;
the durable assignment record still identifies the endpoint and notebook UUID.

CLI 0.7.4 `ColabRuntime.kernel_client` calls `start()` and websocket setup
before its `on_kernel_started` callback saves the ID. The installed
`JupyterKernelClient.start` POSTs `/api/kernels` before starting channels.
A websocket/startup timeout can therefore leave a real kernel with no saved ID.
The runtime's internal ReadTimeout/ConnectTimeout retry reconstructs its client
using the still-empty runtime ID and can POST another kernel. Independently,
a pre-existing runtime kernel also makes unique-inventory recovery fail.

**Evidence limit:** the failed helper did not retain the inventory or startup
trace, so the exact origin of the additional entry (internal retry, provider
pre-existing kernel, or stale listing) cannot be established retrospectively.
The confirmed supervisor defect is relying on unique inventory after losing the
creation identity. There was only one supervisor exec before the refusal;
Contents readiness itself submits no kernel work.

## Minimal correction

The existing requests safety hook now captures the authenticated creation
response before websocket startup. It binds the exact URL and proxy token to
the supervisor's explicit config/session and durable endpoint/notebook record,
then saves that response's nonempty kernel ID using the CLI StateStore.
A saved ID forbids another creation POST. A process-local fence also forbids a
second POST after an uncertain response, preventing the upstream internal
creation retry. A subsequent supervisor exec still requires independent idle
proof and reconnects to the saved ID; a busy kernel does not permit resubmission.

No inventory sorting, arbitrary selection, deletion, extra allocation, raised
cap, or new retry authority is introduced. Missing identity, malformed creation
responses, and unowned multi-kernel inventories still refuse. If termination
occurs before identity can be saved, the existing strict discovery rules apply.

## Claude execution after review

Use this clean branch at the final handoff SHA; do not modify transport bytes
between plan, execution, review and merge. The full acceptance profile remains
`final-only`, all eleven required gates, with the existing 4.0-unit total and
0.8 reserve. Do not combine acceptance with a production retry.

```sh
REPO=/Users/josenunez/Projects/ml/snake-dqn-kernelamb
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
cd "$REPO"
git fetch origin
git status --short
git rev-parse HEAD origin/supervisor-kernel-ambiguity-20261022
# Light, single-process offline regression only (owner's timing constraint).
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" -m pytest -q -p no:xdist tests/test_colab_cli_guard.py \
  tests/test_colab_cli_warmup.py tests/test_colab_cli_dispatch_warmup.py \
  tests/test_colab_cli_latency_policy.py tests/test_ops_qualification.py
# Prepare a freshly reviewed endpoint-to-positive-rate account map.
# An empty object is valid only when the account is actually empty.
ACCOUNT_MAP=/absolute/path/to/reviewed-current-account-rates.json
"$REPO/tools/ops/requalify.sh" plan --broker "$REPO" \
  --other-assignments "$ACCOUNT_MAP" \
  --state-parent /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/kernel-ambiguity-requalification-20261022
# Inspect the printed plan, manifests, source inventory and budget first.
PLAN=/absolute/path/printed/by/plan/ops-review.json
"$REPO/tools/ops/requalify.sh" execute --plan "$PLAN" \
  --review-token COLAB_CLI_ACCEPTANCE_REVIEWED
```

Require final-only acceptance PASS, exact source identity, all eleven observed
gates, CPU and L4 cold-kernel recovery, verified exports and backend disappearance.
Review the creation path's saved kernel identity and first Contents upload too.
The wrapper updates the registry only on PASS. Failure consumes that plan;
prepare a new reviewed plan rather than re-executing it. After PASS, merge the
unchanged qualified tree and repin the retry science package to that published
transport tree before a fresh family canary. An old manifest pin does not become
qualified merely because the broker was requalified.

Codex-V's independent create-only review receipt is outside the Git tree:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/kernel-ambiguity-20261022/codex-v-review.json`.
