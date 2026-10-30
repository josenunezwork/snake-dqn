# Pre-ready source-fetch timeouts, 2026-10-30

Diagnosis and offline fix only; no dispatch or qualification-registry update.
Evidence and the operator handoff are retained in
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/transport-hiccup-diag/SUMMARY.md`.

## Confirmed boundary

The DOOM A100 fit and Watch6 L4 transfer canary both completed input transfers,
then timed out in the synchronous `deliver_source("fetch")` RPC. Neither reached
source identity publication, `ready.json`, hardware admission, dependency setup,
or its scientific entrypoint. There are no scientific polls or exports. The
watchdog subprocess was already running, waiting for `ready.json`.

The failing receipts' supervisor SHA-256 is
`1cedfd0475ba63e080ccbfab55e26284f1c8bcb4d16a0010dd12eae8a41824fc`;
this matches the pre-fix source. Each source helper SHA-256 is
`76dd3918da14090d412cfa51e3d9e7f82bb46290aa3f80262e8d723ebdc5080f`.
The same helpers were used by two successful Watch replay canaries. Those
canaries also timed out at 30 seconds, proved idle, recovered the completed
durable operation record, and reported source-delivery times of 44.561 and
38.566 seconds. The DOOM canary at the exact failed fit commit completed its
source delivery in 28.731 seconds.

Staging applies a 30-second control limit even to source delivery, although
`source.fetch` allows 120 seconds for setup/fetch and another bounded 20 seconds
for identity/checkout. The non-staging 1,200-second recovery window explicitly
does not apply. Staging instead permits three idle probes (two 2-second sleeps),
so fast busy/starting responses can exhaust recovery only about seven seconds
after a timeout. This explains why a still-running source operation can be
stopped well before its own internal deadline. The earlier DOOM input-ledger
verification timeout did recover with idle proof; it was not the final failure.

The historical records do not save raw kernel states or remote Git logs.
Consequently, they cannot distinguish busy from starting/unknown or identify
whether Git was downloading, unpacking, checking out, or the exec was lost.
An in-progress source fetch is strongly supported by the successful controls;
a GPU/compiler/scientific-job stall is excluded by the execution order.
`RPC_EXEC_POLL_TIMEOUT` alone remains insufficient evidence of mechanical failure.

## Minimal correction

Only `deliver_source` enters a thread-local source-delivery scope with a
150-second RPC/operation allowance, matching the existing source-RPC allowance
documented in ROBUSTNESS.md. This covers both GitHub fetch and verified sparse
fallback installation. It does not extend the 300-second staging deadline,
whole-session/price deadline, explicit caller timeout, or any manifest cap.
Other staging controls retain 30 seconds and the existing recovery policy.
Normal post-ready RPCs retain 120 seconds and their existing recovery window.

The source operation retains its durable UUID, started/completed fence, nonce,
and required idle proof before any replay. It still runs synchronously, but
receives enough time for its existing bounded work. An async source protocol
would be a larger change and is not necessary to correct this mismatch.
The independent Contents heartbeat remains separate from the kernel.

Source RPCs now carry `source-fetch` / `source-install_fallback` timing tags.
`upload-measurement.json` includes failed source-delivery elapsed time rather
than leaving it at zero when the source RPC raises. No scientific package,
scientific retry policy, runtime child command, or study data is modified.

## Qualification and retry boundary

This changes the qualified `research/colab_cli` tree. The old mutable registry
must continue to reject the new tree until the operator records the new
qualification and independent Argon review. Do not bypass that check or reuse
sealed manifests against an unqualified tree. Qualification requires the
offline supervisor/runtime/fault suite and the source-control regressions:
45/90/140-second source calls succeed once; a 160-second call is fenced at
150 seconds, does not release ready, and verifies stop; staging/session/explicit
deadlines remain binding; ordinary controls and independent heartbeat allowances
remain unchanged. Retain exact commands, counts and tree/commit hashes.

The 2026-10-29 offline-only qualification authority covered that specific change;
it is not silently extended here. The operator must record the applicable
qualification authority for this successor. A live acceptance, if required by
that authority, is operator work; none was performed here. A synthetic delayed
source-fetch acceptance within the existing staging cap is the appropriate
live check, rather than a scientific fit used as a transport test.

Both historical assignments have `stop_verified=true`, no cleanup error and
no exported scientific units. They are eligible for an outcome-blind fresh
attempt under their registered completion rules, after remaining attempts and
the verified ledger are checked. Retry reliability is not established on the
old transport. After qualification, regenerate both packages with the accepted
source/transport pins, input and ledger hashes, fresh output directories and
receipts, and current-code canary proof before a main dispatch. An empty ledger
is appropriate only after verifying no units were previously collected across
all attempts. No job-package heartbeat can repair this pre-entrypoint failure.

## Retained offline results

Artifacts live beside the external SUMMARY. Focused recovery/staging tests:
44 passed. Final strengthened source-control file: 8 passed, including successful
Contents heartbeats during the delayed source calls. Admission/dispatch complement:
432 passed. The previous qualification's file selection plus the new source-control
file accounts for 732 passing cases across the retained XML reports (see
`prior-qualification-selection.json`); this is not a separate 732-test invocation.

The exploratory all-transport sweep reported 650 passed, 3 failed and 3 collection
errors. All six problems reproduced at untouched baseline `322aaea7` in the
separate review worktree: the run7 fault test still expects the superseded
300-second telemetry limit, two T34 admission fixtures omit explicit devices,
and bundle/retention tests import absent modules. They were not changed here.
See `offline-suite.log` and `baseline-failures.log` for the exact failures.

Independent Argon review was attempted with `gemini-4-argon-high`, but Antigravity
returned HTTP 429 / `RESOURCE_EXHAUSTED` (individual quota). No review verdict
was produced. Keep review and successor qualification pending; passing tests
do not replace the other implementer's review. The registry remains unchanged.
