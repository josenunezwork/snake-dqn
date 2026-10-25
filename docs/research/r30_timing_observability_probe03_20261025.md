# R30 timing probe-03: offline observability handoff

Prepared and verified; **G5/main HOLD**, independent predispatch acceptance still
required. No Colab calls or live dispatch. Runtime source is published on main at
`e18cdbcae140f3b25d7439a5d78cd9fc85b404f0`. Worktree:
`/Users/josenunez/Projects/ml/snake-dqn-r30post`, branch
`sol-r30-post-f4-20261025`; original clean baseline `0f90a3c277de296e9ecd1031e1b432f56f76ff67`.

The full Astra ruling and its sidecar were read and verified:
`JUDGMENT_R30_PROBE02_INCOMPLETE_OBSERVABILITY_20261025.md`, SHA256
`bc5e471c28ab824b1ed06abc86ba72a5f7e5e2d018c3f687a167d41389a74888`.
The [prospective amendment](../../research/bl_round30/TIMING_PROBE03_OBSERVABILITY_AMENDMENT_20261025.md)
records the narrow operational correction and accounting change.

The producer publishes durable immutable operational snapshots at each phase
start/end, plus failure/completion records. Records carry timestamps, elapsed
time, completed-phase timings, active phase, an allowlisted exception type and a
fixed diagnostic code. Exception text, traceback, locals, numerical scientific
values, intermediate parameters and certificate outcomes remain suppressed.
Payload/SHA/completion publication is fsynced; delivery of deadline/termination
handlers is deferred over the metadata transaction. Failed immutable slots are
abandoned, and incomplete phases never supply completion durations. SIGKILL
retains previous records and the last phase start. All output, including
sidecars, is capped at 512 KiB; transport stays 64 MiB. The qualified transport,
shared exporter and scientific consumers are unchanged.

The synthetic CPU entrypoint passed both before and after the correction,
including feature cache, the fixed B8 masked decision, plan coefficients,
equality, Adam, closures and certificate. **No dtype/device workload defect was
reproduced. Probe-02's real failure cause remains UNKNOWN.** Boundary fault
injection found and closed a new observability interruption window; it also
verified transient I/O failures cannot erase the original type/code by reusing a
partial immutable slot. The independent implementation/metadata reviewer found
no remaining concrete blocker. This review does not grant judge acceptance or
operator dispatch authorization.

Verification used `./venv/bin/python`, nice at least 15, single-thread CPU
settings, serial pytest (`-n 0`) and at most two compute processes:

- Offline timing/packet/forecast/telemetry, dispatch, source, R30 packet and fake
  supervisor regressions: **258 passed in 598.17 seconds**.
- Focused rerun against the published source: **74 passed in 11.08 seconds**
  (overlaps the regression selection). Includes genuine SIGTERM/SIGKILL, worker
  alarms, certificate timeout, payload/SHA/marker alarm and I/O faults, redaction,
  incomplete forecast/ledger refusal, complete CPU entrypoint and output caps.
- Black `--workers 1 --check`, isort `--check-only`, flake8 `--jobs 1` over all
  eight changed Python files, and `git diff --check`: PASS.
- AST comparison: `workload`, `bookkeeping_probe`, `_b8_masked_action_decision`
  and `_worker_deadline_seconds` are identical to baseline. Scientific consumers,
  shared exporter, dispatcher and qualified transport are byte-identical.
- Actual offline upload preflight: **13 files PASS**, source archive 1,214,463
  bytes, opaque bundle 36,189,672 bytes; each file below 67,108,864 bytes.
- Actual dispatch dry-run: PASS, zero allocations; live accounting NOT_CHECKED.

Exact commands and verification evidence are in packet `VERIFICATION.json`,
`OFFLINE_TESTS.log`, `SCOPE_PRESERVATION.json` and `DISPATCH_DRY_RUN.json`.
No real scientific inputs were unpacked or inspected. No scientific units,
D4096 or `t34-round0-study-01/` were opened. No tmux sessions, released checkpoints,
serving pins or old hash-bound packets/receipts/reservations were modified.

Separate replacement reservation: base 15/effective 30 minutes; original
40-minute liability retained without reclaiming it; combined envelope
**472 + 30 = 502 effective minutes**. R30-1/2 allowances stay 144/288 minutes,
main estimate/timeout ceilings 53/106 minutes, worker/session/reserve deadlines
1,500/1,800/300 seconds and certificate cap 600 seconds. Full admitted input
hashes, 2,000-batch objective, continuous 30 updates, closures 0/10/30, L4,
precision, single-thread settings, 150/300-second transfer/staging bars,
account/lease/watchdog, verified stop and `no_automatic_retry=true` are retained.
Another incomplete calibration requires independent disposition.

Packet root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/timing-calibration-probe-03`.
Job ID: `r30-1-timing-calibration-probe-03-e18cdbca`. Packet-relative
`runtime-output` and `supervisor-state` are unused.

| Payload | SHA256 |
|---|---|
| G5_TIMING_OBSERVABILITY_PACKET.json | `cdbbb6e2e523af20fce8c830d01102a96f877400330d4f383a523ab40bfb59cf` |
| job.json | `3d314954235173057be8e40766a540ae77f6258469de19c7f969c1a9c446ea41` |
| inputs/timing-config.json | `0ab6c87f0875c94104e1962b955be80dafb5a72192cbee94f36d7956dbe44225` |
| accounting-reservation.json | `765347652319cac2f377f35691ec2192165a70fd67c6705ec1a1ad30c7b1223a` |
| VERIFICATION.json | `7aaa812dfcbc45d7f75ed8cdc6c403b0bf838a9b858b81b5009842bcab7b2348` |
| DISPATCH_DRY_RUN.json | `b33cc405252042650b9f391e9f1abfbf11c08df90126f90b9f08136c67fc73db` |

Actual dry-run output:

```text
PASS_DRY_RUN r30-timing-calibration-probe-03 allocations=0 live_accounting=NOT_CHECKED
```

Exact operator command after independent predispatch ACCEPT and live account
review; **not executed**:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-r30post
PATH="$PWD/venv/bin:$PATH" tools/ops/dispatch.sh r30-timing-calibration-probe-03 \
  /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/timing-calibration-probe-03/job.json \
  --broker /Users/josenunez/Projects/ml/snake-dqn-r30post \
  --cwd /Users/josenunez/Projects/ml/snake-dqn-r30post \
  --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/timing-calibration-probe-03/supervisor-state
```

Append `--dry-run` for the offline check. No live timing result or release proof
exists for probe-03. The packet's `PRE_DISPATCH_REVIEW.json` remains REQUIRED.
