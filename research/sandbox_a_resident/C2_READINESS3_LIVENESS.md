# TRAIN-01 c2 admission and liveness boundary (readiness 3)

This is an implementation handoff for Opus review, not science ratification,
transport qualification, a seal, or permission to dispatch. No r6 input or output
was read or changed. The producer remains the unchanged `research/colab_cli`
tree `5833e54e9c5396121cbbf99a5cd4e52006e564fd`.

## What the retained producer captures prove

`fixtures/supervisor_goldens/` and `fixtures/supervisor_goldens_v2/` are immutable.
Tests replay their complete retained output trees and logs. The first corpus's
runtime-timeout and runtime-lease-expiry cases held retention artificially; they
are now named `RETAINED_TERMINAL_NON_PRODUCER` and excluded from `ELIGIBLE`.
They remain useful terminal-schema fixtures, **not producer liveness evidence**.
The v2 captures model native runtime self-release faithfully.

| Captured stop | Exact receipt error | Ordinary admission |
| --- | --- | --- |
| wall-cap | `Refused('whole-session wall cap reached (including setup/export)')` | Admits verified released envelope and retained tree |
| helper-deadline | `Refused('whole-session deadline reached before helper')` | Admits verified released envelope and retained tree |
| cli-deadline | `Refused('whole-session deadline reached before CLI call')` | Admits verified released envelope and retained tree |
| runaway-guard | `AccountingError('priced session unit cap reached')` | Admits verified released envelope and retained tree |
| calibration-reserve | `Refused('study reserve reached during calibration')` | Admits verified released envelope and retained tree |
| lease-renewal-timeout | `Refused('deadline reached during telemetry recovery')` | Admits verified released envelope and retained tree |
| rpc-exec-poll-timeout | `Refused('deadline reached during telemetry recovery')` | Admits verified released envelope and retained tree |
| historical attempt-08/09 telemetry loss | `Refused('kernel remains busy/unknown after telemetry hiccup')` | Admits the immutable historical complete outputs and ops logs; not a current live observation |
| runtime-timeout (v2) | `AccountingError('supervised assignment disappeared from sessions')` | **Refuses**: native self-release leaves no broker-retained terminal |
| runtime-lease-expiry (v2) | Same assignment-disappearance error | **Refuses** for the same reason |
| provider-loss-no-receipt | No receipt | **Refuses**: no verified released incomplete receipt |
| provider-loss-after-download (fresh offline test) | No receipt, even with verified snapshot and native partial transfer | **Refuses**: inspection grants no stop/continuation authority |
| child-nonzero (v2) | `ClassifiedFailure('remote child exited 23: no traceback line published')` | **Refuses**: no mechanical terminal reason |
| clean-completion (v2) | No error; PASS | **Refuses**: already completed |

`test_real_producer_mechanical_stop_is_admitted_with_all_retained_bytes` proves
the seven positive envelopes against their actual retained files and supervisor
logs. `test_real_producer_liveness_and_exact_fault_twin` adds a root-watchdog twin
to each; the refusal is exactly `receipt records watchdog timeout exit code 124`.
The native producer emits **no receipt-root exit field** in these captures.
Additional tests permit only an integer mirror of a retained terminal integer;
this compatibility case is schema coverage, not a newly captured producer shape.
There is no exception for 124, conflicting exits, string exits, bools, or nulls.

These are offline transport/mechanical-class captures, with clocks and failure
boundaries injected through the subprocess fake. They are not real GPU TRAIN-01
executions. Native TRAIN-01 CPU orchestration/proxy/snapshot production is present
in the completion and runtime-stop captures, not every seven positive envelope.
Stop admission alone does not prove that a usable full-state snapshot exists.
Snapshot/ledger admission is separately required, and a setup/calibration stop
before a snapshot may have no resumable work. No r6 liveness claim is made.

## Ordinary producer stops without usable continuation proof

The runtime's `_supervise` loop in `runtime.py` emits `timeout`, `lease_expired`
(via `_lease_expired`), and `release_credential_expiring`, then calls
`_unassign_or_raise` immediately after publishing its terminal. The first two
have faithful v2 negative captures. Credential expiry has the same source path
but no faithful native golden here: its broker-retained terminal cannot be
assumed. An assignment-disappearance receipt always refuses with:

`ACCOUNTING receipt records assignment disappearance without a retained terminal; not admissible`

A raced broker download could retain a bound terminal before release; the
terminal schema accepts a mechanical reason only if all evidence agrees. The
old artificial-retention captures do not prove that race reliable. Do not design
c2 around runtime timeout, lease expiry, credential expiry, provider loss, or
receipt-less recovery under this producer.

`supervisor.py` additionally contains exact mechanical causes not independently
captured in this corpus: `read retry budget exhausted`, `study reserve reached`
outside calibration, `priced session unit cap reached` as a direct `Refused`,
`priced session unit cap reached during calibration`,
`priced session unit cap reached while awaiting account broker`,
`deadline reached while awaiting account broker`,
`telemetry recovery exhausted`, and `retention lease elapsed; no further job`.
The existing schema recognizes these strings, but **producer liveness is not
established for these paths**, and this revision grants no new reliance on them.
The busy/unknown-after-hiccup envelope is additionally covered by the separate
immutable real historical corpus (`fixtures/historical_mechanical/attempt-08`
and `attempt-09`); its full-tree/log positive and exact-message negative twins
are exercised by `test_historical_mechanical.py`.
Before a future seal relies on one, capture that exact native path with its
retained evidence and demonstrate usable snapshot/ledger admission. A chain
retention expiry is between jobs; it is not by itself a TRAIN-01 child stop.

Public-input fetch timeouts/failures, `popen_failed`, `runtime_error`, generation
fence loss, preparation/distribution/stall failures, unexplained child exits,
cleanup failure, stop-unverified, and unknown causes remain inadmissible. They
are not ordinary mechanical continuation triggers merely because release was
verified. Header/progress/metrics/tail/poll/log evidence with a non-finite number
or any diagnostic exception refuses even when a mechanical stop also exists.

## Diagnostic endings and the required future operating decision

Native `runtime.py` stores the last nonempty merged stdout/stderr line in
`terminal.last_traceback_line`; that name does not prove the line is an exception.
The positive retained-terminal schema fixtures end in an exact complete
`TRAIN01_PROGRESS { ... }` object. Those records are recursively scanned.
No qualified GPU stderr-ending corpus was provided or created in this offline
task. A final NCCL/torch warning, bare SIGTERM text, or any other non-progress
ending remains **inadmissible**, even if a warning without fault content is
allowed as an ordinary log/tail line. No general warning exemption was added.
Exception/warning repr, quoted exit facts, prefixed JSON failures, and a traceback
header all have exact-message negative twins.

For a future c2 seal, the operator/reviewer must explicitly choose an operating
path supported by evidence. The available captured positive path is a **broker
wall cap before runtime self-release**, retaining previously downloaded immutable
snapshot units and clean diagnostic history. The broker checks this cap at
`timeout_minutes * 60 - 45`, including setup/export. It needs an independently
checked timing envelope and a demonstrated usable snapshot; this document does
not assert that 45 seconds suffices for a large transfer. The native runtime's
post-terminal deadline is measured from job start (F4), so ordinary jobs do not
gain a fresh full export window. Alternatively, a separately reviewed and
qualified broker/runtime repair could preserve the terminal before release.
Neither option has been sealed, timed for c2, or dispatched here.

This resolves M5 by explicitly limiting the liveness claim and listing the
inadmissible/unproven paths. It does **not** establish universal continuation
liveness or clear the pre-seal operating decision/F4 gate.

## Other review boundaries

The readiness-3 fixes cover B1 and M1-M5. Related minors 1-4, 6, 8-10 have bounded
corrections/tests (generation formats, supervisor-log location, truncated list
exemption, aliases, native offset filename, native spec validator, diagnostic
size bounds, and test cwd). Minor 5's historical wrapper cross-product remains
unchanged and is not a producer-fidelity claim. Minor 7's receipt-less replay
still grants inspection only; no identity-free recovery authority was added.
Minor 11's internal `allow_completed` inspection parameter remains; ordinary
`verify_mechanical_stop` never exposes or enables it. F5/F11 publication
clearance and independent review remain unclaimed. No existing evidence was
rewritten, and no qualified transport source was changed.
