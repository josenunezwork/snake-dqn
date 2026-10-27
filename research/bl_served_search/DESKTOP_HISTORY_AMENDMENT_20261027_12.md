# Prospective desktop history amendment — attempt 12

Owner-directed 2026-10-27. Applies before any attempt-12 host observation or panel.
Supersedes the classifier/state/replay descriptions in the attempt-10 and
attempt-11 repair notes. The original ruling, amendment, sealed registry, prior
receipts, and invalid attempt dispositions retain their original bytes.

`classify_row(history_for_pid_birth, row, registry)` is the sole desktop-policy
classifier. It is a pure function: filesystem/process/signature observations are
explicit exported row facts; the function has no filesystem access or hidden
mutable state. It returns the final kind, a complete immutable history record,
and the reason. The live adapter and independent reducer invoke this same
function. Workload and system-executable precedence are part of this function.

A PID/birth identity is permanently revoked for the rest of the attempt on any
observed executable/argv change, UID change, executable replacement (including
before admission), verification failure, or explicit revocation. Restoring a
file or returning to an earlier identity cannot readmit that PID/birth. A fresh
birth after restart starts a new history and may qualify after fresh off-tick
verification. This terminal rule is deliberately stricter than Astra's ruling
and remains within it. Revoked processes receive ordinary unknown/workload
classification. An initial unadmitted verification failure retains the ordinary
>=20% per-process unknown/workload CPU rule. Observed executable/argv/UID
transitions, enrolled executable replacement, and failure of an admitted identity
retain the existing immediate identity-contamination guard. Unenrolled file
replacement remains terminal and follows ordinary unknown/workload CPU rules. Terminal revocation and that contamination flag are
separate explicit history fields. Unknown/workload CPU never receives a desktop
background exemption. Verification cannot reset histories midway through an
attempt; a new birth may use fresh digest verification against the sealed entry.

Each captured row carries its observation-time decision and an event reference.
The previous CPU row is classified at its own capture time, before any later
point observation or current-row transition. Interval attribution uses those two
frozen decisions with workload-first conservative precedence. It never checks
an old executable against a newer filesystem or registry state. Executable and
displayed argv are captured together by one macOS process-table observation.
The exact argv vector is read separately with birth, executable, UID and argv
checks on both sides of that read. Inaccessible or racing evidence fails closed;
displayed argv never substitutes for the vector. Argv is identity evidence,
not proof of the activity an application performs.

`host.json` exports the complete ordered history transition sequence, including
unadmitted and idle processes, all transition inputs, immutable states before/after,
classification and reason, and final per-PID/birth histories. The pure function
runs on every observation. A repeated unchanged identity/state/decision reuses
its existing immutable event reference; it does not duplicate history at each
CPU counter update. Every changed identity, revocation or decision retains a
new event, and references are valid only during that event's effective lifetime.
Every CPU and point classification refers to its observed event. Reducer replay starts from empty
histories and must reproduce every event's complete classification and state,
then every referenced row and interval classification exactly. Missing,
reordered, altered or inconsistent history evidence is rejected. No claimed
classification flag is an input to the replay decision.

This amendment changes no enrollment, scientific threshold, exposure, timing
bar, shared background ceiling, per-process foreign-CPU rule, coverage rule,
all-tick retention rule, release authority or serving behavior. The new amendment
is bound into the source closure and desktop intent. Tests, smoke and `--check`
are implementation evidence only. Host-only and panel execution remain deferred
for independent acceptance and the operator's natural suitable Mac window.
