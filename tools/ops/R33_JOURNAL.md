# R33 durable operator journal

`task_journal.py` is a small SQLite journal used by the existing `watchd.sh` and
`wait-event.sh`. It does not allocate, dispatch, retry, alter a transport, or
decide scientific admission. Claude/operator keeps those actions under the
existing allocation lock and `dispatch.sh`.

## State and identity

A task identity contains campaign, stage, attempt, source-closure SHA-256,
manifest SHA-256, and prerequisite-receipts SHA-256. It is serialized
canonically and hashed into the SQLite key. `PREPARED` is not evidence.
`READY` requires separately recorded static proof and authority receipt hashes.
`ELIGIBLE_READY` additionally records a current resource/admission check; it
does not cache the next live admission. Source revocation moves the task to
`HOLD`, deletes only pre-intent claims and undelivered action notices, and must be
reconciled before a new attempt exists.

Terminal evidence, allocation release, and retained allocation are independent:
a retained chain records an ongoing charged lease while a completed job remains
terminal. Terminal evidence without either verified release or a lawful retained
lease remains `ALLOCATION_STOP_UNVERIFIED/HOLD`. A retained chain must not invent a release between jobs or block its
lawful successor. Unknown submission/stop remains `HOLD`.

Only one `ELIGIBLE_READY` task may exist per resource lane. A lane head with a
typed blocker does not prohibit a separately eligible lane.

## Consumer protocol

Events wake the journal; `watchd.sh` reconciles it each normal 60-second loop.
Its PID/log observations are stored as `WATCH_WAKE` events only. They never
create a task or prove READY. Each consumer calls `deliver CONSUMER` and must
explicitly call `ack CONSUMER NOTICE_ID`; delivery is deliberately at-least-once.
A new consumer starts from unfinished notices, not EOF.

Claiming increments the task generation. A claim can expire and reoffer only if
static proof, authority, source validity and current eligibility still hold and
there was no submission intent. `POSSIBLE` submission or an expiry at the
submission boundary is `SUBMISSION_AMBIGUOUS/HOLD`; it never produces a
replacement launch. A confirmed submission becomes `RUNNING` only after the
single `POSSIBLE` intent for that generation.

The journal emits five-minute acknowledgment reminders and fifteen-minute
operator escalations. `metrics` retains all ready/eligible/ack/start/terminal
intervals, resource eligibility, typed blocker/owner, and reports p95/median
only once there are ten actual transitions; otherwise it reports
`INSUFFICIENT_LT10`.

## Proof reuse charter

`proof_dependencies.py` records the complete closure for source, input,
environment, platform, command, fixture, allocation, and contract. Unknown,
added, or removed dependencies fail closed. Reuse is only presented for an
opted-in `destination:` or `transport_location:` delta whose existing proof
binds the complete prior inventory and permits reuse. Changed source, runtime,
platform, allocation, contract, or a changed procedure produces
`FULL_REVIEW_REQUIRED`.

The generated file is only `PENDING_INDEPENDENT_ACCEPTANCE` input. It requires
fresh native-canary and live-admission obligations. Sol/builder identities are
refused as reviewers. Codex-V alone writes the separate create-only acceptance
record and must bind the input SHA-256 plus downloaded-byte, destination,
access/integrity, runtime-environment and consumer-behavior checks. A false
acceptance disables that dependency class; the target of 50% less review time
is unmeasured until a future report contains real observations.

## Offline commands

These author-only commands do not dispatch a job:

```bash
PY=./venv/bin/python
$PY -m pytest --noconftest -q tests/test_r33_journal.py
$PY -m pytest --noconftest -q tests/test_r33_proof_dependencies.py
bash -n tools/ops/watchd.sh tools/ops/wait-event.sh
```

Run them only under the shared serial host-test lock and the active factory
window rules. The fixtures are deterministic metadata/SQLite tests; the ten
campaign-log anchors are historical metadata transitions with unknown durations,
not a handoff-performance sample.
