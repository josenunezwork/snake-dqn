# Colab transport robustness audit

Prepared offline on 2026-10-18. No live allocation, execution, or stop was made
for this pass. Live acceptance remains required. The protected S1 and Q0/Q1
research directories are outside this change.

## Measured evidence and limits of the measurements

Receipts are under
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/colab-cli-acceptance/`.
Each path below ends in `outputs/cpu/receipt.json`, with observer evidence in
its sibling `acceptance-observer.json` and arm disposition in
`acceptance-result.json` two directories above.

| Run directory | Observed failure / timing | Release evidence |
| --- | --- | --- |
| `acceptance-1792377693` (run 3) | 174,172,160-byte (166.1 MiB) whole-source archive; write timed out; whole arm 64.368 s | `INCOMPLETE`, `stop_verified=true`, `cleanup_error=null` |
| `acceptance-1792378723` (run 4) | Flat 300 s staging ceiling during the 600 MiB re-upload; whole arm 853.155 s; 600 MiB local snapshot exists, but no complete re-upload / roundtrip timing | Same clean release |
| `acceptance-1792380711` (run 5, commit `b8b01bde`) | `exec --timeout 20` killed by its 30 s outer subprocess bound; whole arm 169.721 s | Same clean release |

Run 4's successful intentional 65 s silence occupied 66.213 s; run 5's took
68.015 s, each under its then-90 s RPC bound. Run 5 records verified small
uploads taking 3.021–5.144 s for 236–12,918 bytes. The calibration cycles
completed in 13.458, 14.124, and 13.090 s respectively, **including** two
five-second sleeps and three pairs of sessions/usage reads. They are not
individual request latencies. There is no defensible live p95 or measured
600 MiB roundtrip goodput in these artifacts. Failed operations give censored
lower bounds, not completed-call durations.

The 120 s normal allowance is four times the failed 30 s outer bound and
provides substantial headroom over observed small transfers. Source and
allocation budgets also cover setup. These are engineering allowances, not
claims that these samples establish a latency distribution. New receipts
record each CLI attempt's command, elapsed time, effective timeout, attempt
number, and outcome without recording arguments, code, or credentials.

## Complete call inventory and policy

`policy.py` owns the Mac defaults. Unless stated otherwise, every allowance is
clipped to the earliest existing wall, priced-unit, and narrower operation
budget. Three attempts means three total attempts. Backoff is 2 s then 5 s;
a retry does not restart the operation clock. Only transient connection,
timeout, and HTTP 500/502/503/504 failures qualify. Authentication, quota,
identity/hash, hardware, and acknowledged kernel errors fail immediately.

| Boundary / callers | Class | Timeout / total budget | Retry or reconciliation |
| --- | --- | --- | --- |
| `version`, `whoami`, `url`, `sessions`, `usage`: admission, accounting, status, acceptance, reconnect | Idempotent read | 120 s shared operation | Up to 3, backoff 2/5 s |
| Provider-rate calibration: sessions + usage samples | Idempotent reads | Each up to 120 s; **730 s derived calibration allowance (6 reads × 120 s + 2 × 5 s), clipped to first caps** | Read retry policy; insufficient stable samples remain INCOMPLETE |
| Guard HTTP: backend assignment lookup, account reads, runtime proxy discovery, kernel REST state, Contents, OAuth transport | Per-verb; no hidden mutation retry | Connect/read 30/120 s, always under owning subprocess budget | Retry at the owning CLI/helper layer only |
| Guard assignment recovery GET and kernel-idle REST helper | Idempotent read | 150 s helper total, or caller's smaller remaining budget | Up to 3 with backoff 2/5 s |
| `new`: GET assignment precondition + POST assignment, durable UUID and endpoint | Non-idempotent allocation | 180 s, one dispatch | **Never allocate again automatically.** On ambiguity recover by recorded notebook UUID / durable endpoint, then verify cleanup; unresolved identity retains pending record |
| Runtime launcher `exec`, ready/hardware admission, credential rotation, source installer, export split/assembly, acceptance control/corruption/resume RPCs | Non-idempotent / single-dispatch kernel mutations | Normal exec 120 s + 10 s client completion grace; source RPC 150 s + 10 s grace; all clipped to session caps | No blind replay. Missing nonce fences kernel work and enters cleanup. No second launcher or replacement allocation |
| Telemetry/read RPC, hash verification, UUID transfer mkdir/remove | Idempotent read or repeated same control state | 120 s shared budget including independent idle probes/backoff | At most 3; retry only after REST proves kernel idle. Unknown/busy remains fenced |
| Contents input/chunk/control PUT, immutable data and metadata GET | Idempotent same bytes / same path | 120–180 s per call (actual 2 MiB PUT / 8 MiB GET use 120 s); 3 attempts under per-file `max(300, 120 + bytes/524288)` total | 2/5 s backoff; SHA/size verification and temporary paths remain mandatory. Export sidecars use this same policy |
| Initial source/input staging | Idempotent transfer with single-dispatch assembly | **Existing 300 s staging cap** plus per-file/session caps | Never borrow staging time from a later transfer or extend a cap |
| Generation-bound lease Contents PUT | Idempotent control | Immediate, then every 30 s; each at most `min(120, lease/3, remaining caps)` | Separate from kernel; next heartbeat may recover a failed PUT. VM filesystem mtime defines age, never Mac wall time. Failed communication still expires the watchdog |
| Runtime GitHub shallow fetch of exact pinned commit | Idempotent fetch | 120 s total, including local setup/backoff | Up to 3 on reachability/5xx only; then bounded sparse fallback. Missing commit/tree mismatch refuses |
| Mac OAuth access-token mint / acceptance forced refresh | Idempotent credential read/refresh, no assignment | 150 s outer; HTTP connect/read 30/120 s; clipped to session/price | Up to 3 transient attempts with 2/5 s backoff inside the same 150 s budget; auth/scope/malformed failures refuse immediately, never start interactive auth |
| Owned client-loss `exec` fault subject | Non-idempotent exec | CLI 300 s; Mac observation/release handshake at most 300 s, clipped to session caps; owned client terminate/kill waits 5 s each | One execution. Started marker, confirmed SIGTERM of a live owned client, absent finished marker, explicit Contents release, then finished marker prove the same request continued |
| Acceptance completion file polls, negative finished-file check | Idempotent read | Up to 120 s per read, bounded observation interval | Transient retries; expected 404 polls; no kernel resubmission |
| Mac-loss checkpoint observation | Idempotent read | Up to 120 s, within final-renewal expiry interval + unchanged 15 s grace | Poll immediately after final renewal, before expiry; never sleep past the checkpoint window. No lease renewal or kernel work after suppression |
| Mac-loss disappearance sessions poll | Idempotent read | Up to 120 s within **unchanged 300 s after earliest possible lease expiry** | Transient retry/poll within that window; broker cleanup does not count as watchdog success |
| Broker `stop` and endpoint-unassign helper | Non-idempotent release | **Existing three 75 s stop attempts, 60 s sessions observations** | Before another stop/unassign, account sessions must positively show the same endpoint still present; unknown state never authorizes another mutation. Absence is the only release proof |
| Runtime watchdog unassign GET/XSRF POST | Endpoint-bound release | **Existing 20 s request bound, 3 attempts, 130 s total release envelope** | Fresh authenticated endpoint/XSRF GET before each POST. This precondition is not account disappearance proof; the Mac verifies disappearance independently |

Local Git inventory/archive/identity commands (5–20 s), local GPU discovery
(5 s), local toy resume subprocess (120 s), child process waits, filesystem
hashing, and optional local backup are not Colab HTTP calls. They retain
bounded local execution; they cannot allocate or launch a second remote job.
The runtime's job process, post-terminal wait, checkpoint grace, credential
scope/expiry checks, and lease fencing are unchanged.

Contents heartbeat bytes use a separate `lease-heartbeat.json`: a chunked PUT
cannot truncate the VM-owned atomic `lease.json`. The watchdog caches only a
stable, generation-validated VM mtime and retains it through a partial write;
a malformed or failed write cannot renew or erase the last good heartbeat.

The CLI hook is published atomically because foreground calls and the lease
thread can import it concurrently. The lease thread never executes kernel
code. It is fenced before cleanup, and quiesced before simulated Mac loss.
A final VM-side renewal durably disables Contents heartbeats (including a late
commit of a timed-out PUT) and establishes an expiry interval; using its earliest
possible expiry for the release deadline preserves the 300 s safety claim
without assuming zero acknowledgement latency.

## Cap feasibility

CPU remains **20 minutes**, L4 **30 minutes**. Unit caps, reserve, first-cap-wins
pricing, account cap, and stop verification are unchanged. Timeouts are
allowances within these limits, not additive reservations or permission to
consume them all. Slow transfers may safely finish INCOMPLETE at a real cap.

A 600 MiB upload's standalone policy allows **1,320 s**, already longer than
the **1,200 s CPU arm**, before bootstrap, silence, downloads, resume, and
cleanup. Therefore the full worst-case transport allowance cannot fit that
arm. No evidence from runs 3–5 proves a larger CPU wall cap sufficient, and
this change does not silently increase one. If qualification requires the
entire standalone upload allowance plus all other stages, a separately
priced CPU-cap amendment is necessary. The current cap still permits a fast
run; it cannot guarantee PASS under every allowed latency/fault sequence.
The L4 worst-case reservation also does not fit: a 1,320 s snapshot download
plus its 180 s lease and full 300 s disappearance window consumes all 1,800 s,
leaving no setup or control time. Honoring every standalone worst-case budget
would require separately priced amendments for both arms. No exact larger cap
is justified by the available successful timings.

Likewise, the required checkpoint exists on a VM that deliberately releases
after a 15 s checkpoint grace. Starting collection before expiry removes the
old artificial 3 s call ceiling and missed-window sleep, but a transport
outage spanning the whole availability window can still make that evidence
unobservable. It must remain INCOMPLETE; increasing HTTP timeouts cannot
manufacture evidence from a released VM or justify weakening release checks.

## Offline fault qualification

`tests/fixtures/fake_colab.py` accepts a seeded `FAKE_COLAB_FAULT_PLAN` with
per-class random latency, timeout, connection reset, and 5xx events. Set
`latency_mode: "real"` for actual subprocess waits; the many-seed suite uses
an elapsed-time model that advances the supervisor clock, enforces each passed
timeout, and services independent heartbeat events during waits. A separate
one-second subprocess check verifies the physical-latency mode. An
8-class by 4-fault deterministic matrix plus 24 random seeds and two clean /
90-second-launcher anchors exercise the actual acceptance observer, including
9 MiB chunk splitting/assembly and a real terminated fake client process.
Scheduled pre/post-mutation faults cover ambiguous commits. The fixture keeps durable
account state and serializes concurrent commands so heartbeat activity cannot
lose an assignment or launch counter. Property tests must inspect the actual
generated run directory and require non-vacuous release receipts for allocated
arms, no live fake assignments, unique allocation names, and at most one
runtime launch per arm. Fake safety evidence is test evidence, never live
qualification. See the test output reported with this change for counts and
coverage.

## Operator commands (not executed by this pass)

Run from the published worktree with the primary venv. A fresh output directory
is mandatory on each attempt. Preparation performs no Colab command:

```bash
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.colab_cli.acceptance \
  --repo /Users/josenunez/Projects/ml/snake-dqn-colab-safety \
  --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/colab-cli-acceptance
```

After reviewing the fresh plan and current accounted-assignment rate map, the
operator command uses the AC/open-lid and caffeinate wrapper:

```bash
/Users/josenunez/Projects/ml/snake-dqn-colab-safety/research/colab_cli/launch.sh acceptance \
  --repo /Users/josenunez/Projects/ml/snake-dqn-colab-safety \
  --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/colab-cli-acceptance \
  --colab /Users/josenunez/.local/bin/colab \
  --oauth-python /Users/josenunez/.venvs/colab-cli/bin/python \
  --execute-after-review --review-token COLAB_CLI_ACCEPTANCE_REVIEWED
```

If other account assignments exist, include `--other-assignments` pointing to
the reviewed endpoint-to-positive-rate JSON. Unknown assignments refuse rather
than borrowing another lane's budget. This document authorizes no live run.

## Verification evidence for this pass

The 58 full acceptance flows (32 deterministic class/fault combinations,
24 random seeds, and two healthy/slow-launch anchors) produced **22 simulated
PASS and 36 INCOMPLETE** outcomes. All **74 allocated arms** had clean,
verified release receipts. Independent artifact aggregation found **zero
orphans, duplicate allocations, or duplicate launches**. All 32 class/fault
combinations occurred. The 9 MiB fixture exercises real chunk splitting and
assembly; the existing 600 MiB fake-transport roundtrip also passed. These
are offline transport results, not live Colab qualification or trainer proof.

Evidence is outside Git at
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/colab-cli-robustness-20261018/`:
`fault-summary.json`, raw JUnit reports, and the local toy resume smoke.
The broad sweep selected 205 tests: 204 passed and one test-only assertion
incorrectly rejected the valid CPU+L4 pair in a successful latency case.
That assertion was corrected to apply only to ambiguous failed mutations;
the existing strict per-arm allocation and exact release-receipt checks remain
in force. The corrected case passed its focused rerun. An additional test
passed the physical one-second fake-latency mode, for 206 distinct checks
verified across the sweep and focused runs. The merged-tree focused report
records the final replay of these checks without replacing the raw evidence.

Commands (use the primary venv):

```bash
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
SNAKE_DQN_DEVICE=cpu "$PY" -m pytest -q -n 4 tests/test_colab_cli_*.py
"$PY" -m pytest -q tests/test_colab_cli_fault_properties.py -k 'fault_classes and new and latency'
"$PY" -m pytest -q tests/test_colab_cli_fault_properties.py -k real_wall_clock
"$PY" -m black --check --line-length 100 research/colab_cli tests/test_colab_cli_*.py tests/fixtures/fake_colab*.py
"$PY" -m isort --profile black --check-only research/colab_cli tests/test_colab_cli_*.py tests/fixtures/fake_colab*.py
"$PY" -m flake8 research/colab_cli tests/test_colab_cli_*.py tests/fixtures/fake_colab*.py
make PY="$PY" lint
git diff --check
```

The local real `acceptance_probe --resume-input` entrypoint matched the
nine-step uninterrupted toy hash
`6c54b0cb1ed845c960cea8fe28b80d3ef8d58a81785473de89ccd4509f96d485`.
The guarded `launch.sh acceptance --check` printed the expected command
without making a Colab call. No existing acceptance receipts were modified.
