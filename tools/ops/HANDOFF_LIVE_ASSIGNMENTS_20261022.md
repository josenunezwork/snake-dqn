# Live assignment accounting and A2 main handoff

The dispatch wrapper now refreshes `budget.other_assignments` from live account
inventory under `ops/alloc.lock` before each launch, including the existing exact
allocation retry. Each endpoint must join to a broker-owned active record. L4
uses its registered 1.54 u/hour rate, CPU 0.08 u/hour, and A100 its broker-record
rate. Calibrated records must supply an observed price; a reserved ceiling is
insufficient. Foreign endpoints, ambiguous identities, malformed inventory and
invalid or conflicting registered rates refuse before submission.

The original manifest remains immutable. Dispatch creates `NAME.live.json`,
records its SHA256 and inventory/record provenance in
`ops/dispatch/NAME/ATTEMPT.accounting.json`, validates the effective manifest, and
checks its hash immediately before start. All supervisor admission checks and
the sole exact allocation retry remain in force. `research/colab_cli` is
unchanged; its qualification tree remains
`553af3f692515b2bf66173356fe4d099acc96942`. No requalification is required.

A2's fresh preparation is
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a2/pack-20261022-07`.
It reuses the genuine pack-06 A100 canary PASS at runtime pin
`cb67a4b90ec2577ed398d76b59c18427d6ae764a`, receipt SHA256
`0f1706436178c1d93f4cdb1e1a1f538f77e09bb22007da3de68edebca846bf77`.
That proof records 89.02549954 seconds of transfers and 223.68381837 seconds of
source-plus-transfer staging, verified exports, CUDA/A100 execution and verified
release. The unchanged runtime pin, five input hashes, source inventory, resume
ledger and scientific allocation remain covered by that canary.

The failed pack-06 main receipt is retained unchanged. It proves a preallocation
accounting refusal: only `version`, `sessions`, `usage` were called; endpoint,
accounting and terminal are null; assignment records are absent and the process
has `EXIT=1`. Its original `stop_verified=false` is not reinterpreted as release
proof. Its 1.738232208 seconds are conservatively charged at 10 u/hour, adding
0.004828423 units to the series ledger. This consumed no main fit/completion
attempt. All five ancestor receipts remain bound by hash.

Main retains 242 minutes, 40.34 units, 8 units reserve, 120 minutes estimate and
10 u/hour ceiling. Including prior attempts and the entire 18-minute/3-unit
canary allowance, the maximum series is 287.93321057 minutes / 47.99553510 units,
within the original 288-minute/48-unit envelope. No cap is raised or doubled
again. This sibling was prepared and validated only; Claude/operator dispatches.

Light verification uses `-p no:xdist` and one native thread. Tests cover departed,
new and foreign endpoints, malformed inventory, registered/calibrated rates,
immutable source/effective manifests and hashes, effective-manifest tampering,
inventory refresh on the sole exact retry, and all existing refusal paths.
Final commands, results and file hashes are recorded in pack-07's
`OFFLINE_VERIFICATION.json`. Independent Codex-V review is
`CODEX_V_LIVE_ASSIGNMENTS_REVIEW_20261022.json`, with SHA sidecar and completion
marker beside this handoff and in pack-07. `PREPARATION.json` retains its
historical build-time source hashes; the final verification/review receipts bind
the final code and tests.

```bash
/Users/josenunez/Projects/ml/snake-dqn-acctfix/tools/ops/dispatch.sh \
  a100-a2-main-20261022-05 \
  /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a2/pack-20261022-07/main.json \
  --broker /Users/josenunez/Projects/ml/snake-dqn-acctfix \
  --cwd /Users/josenunez/Projects/ml/snake-dqn-acctfix \
  --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/state/a100-a2-main-20261022-05 \
  --dispatch-txt /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a2/pack-20261022-07/DISPATCH.txt
```

No Colab allocation or stop, research execution, tmux mutation, released
checkpoint, serving pin, previous manifest or receipt was changed in this work.
Dispatch still refuses an unexplained account change after refresh; its existing
bounded retry policy has not been expanded.
