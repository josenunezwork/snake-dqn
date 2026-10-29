# Accounting and transient exec recovery, 2026-10-29

Owner-approved scope: fix the 041a01b6 Opus review and recover transient exec/poll
outages on the same owned assignment. Qualification is the offline fake-provider
supervisor/runtime suite; no allocation or live acceptance is authorized here.

Count-lag grace uses broker membership and assignment count alone, bounded at two
polls. Rates only warn. Calibration requires three positive stable rate readings
with matching identity and count; variation resets the stability window. If only
identity/count settles before the calibration window expires, publish a null
observed rate, `CALIBRATION_RATE_UNSTABLE`, and a clearly informational priced
estimate. Never publish a zero or unstable rate as a measured job rate. Identity
and count failures remain fatal, and uncalibrated active records remain locked
until successful publication or verified cleanup.

For an owned live assignment, `exec_poll_recovery_seconds` optionally narrows the
consecutive-failure window. Default/base = 600 seconds; effective default = 1200
seconds under the owner's standing 2x rule; the maximum is 1200. These are transport
recovery envelopes, not changes to the job's registered whole-session timeout.
Each exec remains <=120 seconds; backoff is 2, 5, 10, 20, then 30 seconds. The window
ends at the earliest explicit caller timeout, staging deadline or session deadline.
Staging retains its existing recovery and upload limits. Success ends the outage;
a later call receives its own bounded window.

Before replay, verify the allocation notebook UUID, endpoint, local session binding
and account-wide backend membership, then prove the tracked kernel is idle using
the independent helper. Missing assignments, changed identities, permanent errors,
unknown ownership, or exhausted wall time refuse. Busy/unknown kernels receive
further probes but no exec until idle. The independent generation-fenced Contents
heartbeat continues during recovery. Release credentials are minted with lease +
recovery-window + 300 seconds of validity and rotated before long polling; a broken heartbeat still expires remotely.
Durable mutation UUIDs still prohibit replay of an operation started without a
completion record. No new assignment or job launch is retried by this change.

The mutable registry is `/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json`.
It records `PENDING` until independent review. `qualification.check` enforces this
for local and pinned refs before dispatch; the passing review state is `ACCEPT`.
Existing sealed packets must be regenerated against the accepted transport tree,
qualification record, broker commit and all downstream hash closures. See the
external implementation SUMMARY for packet-specific repinning instructions.
