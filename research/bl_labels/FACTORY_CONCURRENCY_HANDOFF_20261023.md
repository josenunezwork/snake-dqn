# F0 factory bounded-concurrency handoff

> **PREPARED_NOT_QUALIFIED — ACTIVATION HOLD.** The implementation is prepared
> for review, but it must not start the companion monitor, client/tool lanes, or
> a D256/D1024/D4096 continuation under this ruling. `require_activation()` is
> intentionally an unconditional HOLD until a real native descendant-ownership
> backend is implemented and independently qualified. No environment variable,
> manual PASS file, unit suite, or reviewer approval overrides this fence.

This handoff prepares parts of Astra's `ALLOW_BOUNDED_CONCURRENCY` operational
amendment for future D256, D1024, and D4096 production windows. It does not meet
all mandatory acceptance requirements. The factory remains first
priority.  The frozen collector closure stays at `61900af28932e47e22e332840fc4406559adb062`;
the operational wrapper pin below is separate and must be the reviewed, published
implementation commit.  No factory command is started by this handoff.

## Companion commands during a production window

Use the committed wrapper for every local Astra/Sol client and tool command. It binds the
judgment SHA `af117cd98d51f428cd83d6e33138fcd13def0b4d7c7cf63a8b2e2b83db99105f`,
passes the active factory attempt identity to the Mac-wide supervisor, and the
supervisor sets and verifies absolute nice 15 for light work and nice 19 for the
single test lane.  It also exports CPU-only and every required numerical-thread
cap before a child begins.

When a future native backend opens the qualification fence, start or reuse the
one monitor before the first 60-second light-only observation and before the
factory continuation. It is a persistent owned process;
the wrapper uses a nonblocking start lock, validates the process start identity
before reuse or stop, and the supervisor independently refuses a second monitor.
It begins as `pending`; the driver binds the live attempt identity when collection
starts.  Do not use `monitor --once` as a readiness claim.

```bash
# FUTURE PROTOCOL ONLY: this command currently refuses with ACTIVATION HOLD.
MONITOR=/Users/josenunez/Projects/ml/snake-dqn-f0/tools/ops/factory_companion_monitor.sh
"$MONITOR" --repo /Users/josenunez/Projects/ml/snake-dqn-f0 start
"$MONITOR" --repo /Users/josenunez/Projects/ml/snake-dqn-f0 status
```

Leave the monitor running throughout the production window.  Only after the
factory has terminally drained, released its lease, and all local clients are
quiescent may the operator stop its verified owned PID:

```bash
"$MONITOR" --repo /Users/josenunez/Projects/ml/snake-dqn-f0 stop
```

Claude must substitute the active attempt identity recorded by the driver for
`ATTEMPT_ID`.  It must use argv, not a shell command string.  Light work is
limited to `rg`, `cat`, `sed`, `jq`, `git status|diff|show|log`, and
`/bin/bash -n FILE`; each command remains subject to the supervisor's 30-second
light-command limit and aggregate resource budget.

```bash
COMPANION=/Users/josenunez/Projects/ml/snake-dqn-f0/tools/ops/factory_companion.sh
ATTEMPT_ID='exact-active-factory-attempt-identity'
CLIENT_MANIFEST=/absolute/path/to/reviewed-client-manifest.txt
# The client manifest uses the same v2 source_commit and fixture hash records.
CLIENT_MANIFEST_SHA256=$(shasum -a 256 "$CLIENT_MANIFEST" | awk '{print $1}')
"$COMPANION" --repo /Users/josenunez/Projects/ml/snake-dqn-f0 \
  --lease-identity "$ATTEMPT_ID" --class client \
  --approved-test-manifest "$CLIENT_MANIFEST" --approved-test-sha "$CLIENT_MANIFEST_SHA256" -- codex exec --help
"$COMPANION" --repo /Users/josenunez/Projects/ml/snake-dqn-f0 \
  --lease-identity "$ATTEMPT_ID" --class light -- rg -n 'needle' research/bl_labels
```

There is one Mac-wide test slot.  Use it only after the supervisor reports its
60-second light-only warmup and the factory has no protected baseline, rearming,
new-stream, or W12-sanity interval.  The command must use an explicit reviewed
node ID or a small named file list and this exact pytest shape:

```bash
TEST_MANIFEST=/absolute/path/to/reviewed-focused-test-manifest.txt
# v2 lines bind schema, this worktree's source_commit, fixture=PATH|SHA256, and node=NODEID.
TEST_MANIFEST_SHA256=$(shasum -a 256 "$TEST_MANIFEST" | awk '{print $1}')
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$COMPANION" \
  --repo /Users/josenunez/Projects/ml/snake-dqn-f0 \
  --lease-identity "$ATTEMPT_ID" --class test \
  --approved-test-manifest "$TEST_MANIFEST" --approved-test-sha "$TEST_MANIFEST_SHA256" -- \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q \
  -o addopts= -p no:xdist -p research.compute.factory_companion_pytest \
  research/compute/test_factory_companion_policy.py::test_exact_thresholds_are_admitted
```

Do not run whole suites, `-n`, coverage, benchmark/watch plugins, GPU/MPS,
training, rollouts, builds, or arbitrary shell commands during a factory window.
The supervisor reserves 90 CPU-s, starts cleanup by 116 seconds and 85 CPU-s,
and records an incomplete resource-limited command as
`INCOMPLETE_RESOURCE_LIMIT`; it is never automatically retried.  A companion
breach sheds owned companion work before it stops the factory.  Measurement,
ownership, cleanup, or monitor failure remains a conservative factory preemption.
Timing, strict, serving, F1, and calibration arms require companion quiescence
and the existing factory drain/release path.

## Future D256 continuation protocol

Do not run this command while the activation fence is HOLD. Once a future
qualified native backend is accepted, use a fresh dispatcher name and the reviewed operational pin, never the frozen
science pin, at the next lawful ledger continuation:

```bash
/Users/josenunez/Projects/ml/snake-dqn-f0/tools/ops/dispatch.sh --mac \
  --repo /Users/josenunez/Projects/ml/snake-dqn-f0 \
  bl-f0-d256-guard-w12-03 NEW_OPERATIONAL_PIN -- \
  /bin/bash research/bl_labels/RUN_D256_RESUME.sh --host-check \
  --calibration-receipt \
  /Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/w12-admission-01/receipt.json
```

The resume script and driver retain the registered ledger, width admission,
wall/worker accounting, cumulative charges, source identity, and immutable
receipt chain.  Record the ruling SHA, implementation pin, source hashes,
verification evidence, companion receipt, and any warning/trip incidence in
the campaign log before treating this operational allowance as active.

## Qualification boundary

Synthetic process/clock fixtures exercise the implemented admission, ownership,
accounting, and shedding transitions; they do not establish complete enforcement.
Native integration and label-identity work require a
separate maintenance window after draining the factory; they are not companion
tests. No implementation test, reviewer result, or this handoff starts a factory
job or establishes an achieved throughput benefit. The known gaps are: native
proof of escaped/double-forked descendant ownership, cross-process-group CPU
crediting, watchdog and monitor topology proof, and exclusive-launch verification.
The frozen-label identity proof excludes global escape behavior, so it does not
qualify the production concurrency allowance.
