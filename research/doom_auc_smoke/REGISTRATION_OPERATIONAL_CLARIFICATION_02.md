# Operational clarification 02 — A100-only admission — 2026-10-30

Status: owner-directed, pre-main-fit clarification for independent review fixes
F1–F3. The synthetic canary is already running; no registered fit or scoring
look is authorized by this document. Original registration and packets stay intact.

Supersede only the requirement for both hardware canaries when the saved,
manifest-bound H100 canary receipt records allocation failure with HTTP 503 /
Service Unavailable, no endpoint, no terminal worker and no exports. In that
case a genuine same-pin, same-input A100 family canary with PASS, verified stop,
and unchanged supervisor validation admits an **A100-only** main manifest.
Its `prefer` field is removed and allowed_gpus is exactly [A100]; no H100
variant is admitted without its own genuine qualification. The H100 failure
receipt is provenance, never successful canary proof or verified cleanup.
In particular `stop_verified: false` and `ALLOCATION_UNRESOLVED` remain recorded;
this clarification does not resolve pending allocation/accounting state or
bypass execute-time broker safety checks. Operator must resolve that separately.

Mac admission keeps the runtime commit, source closure, inputs, config identity,
resume declaration and A100 fallback output/job identity from packet-03. Only
Mac preparation/scoring code changes. It writes fresh create-only admission
artifacts outside the running packet and binds this sibling and the pre-look
amendment in its Mac record. No altered upload invalidates the canary proof.

The measured A100 useful forecast remains <= 2,700 seconds: for each arm,
first measured step + 2,999 times the slower of warm steps 2/3, plus 120 seconds
across arms for loading/inference. The 45-minute worker limit, 60/120-minute
base/effective whole-session timeout, A100 8/16-unit informational caps and
common 2/4-unit reserves are unchanged. Nothing changes data, N, labels,
models, 3,000 updates, seeds, bars, bootstrap, single look or release scope.

Implementation is conditional preparation only. No dispatch or automatic retry.
