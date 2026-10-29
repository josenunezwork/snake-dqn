# TRAIN-01 c2 admission and liveness boundary (readiness 4)

Implementation for independent Opus/Argon review, following the operator's
owner-pre-approved M-new-2 decision. No production seal, dispatch or transport
change is made here. `research/colab_cli` remains tree
`5833e54e9c5396121cbbf99a5cd4e52006e564fd`.

## Real evidence regression and diagnostic fixes

`fixtures/attempt15_liveness/evidence.tar.xz` contains byte-for-byte copies of
attempt-15's supervisor log (1,428 lines), every retained poll (1,279), and its
executed manifest. `inventory.json` binds each uncompressed file by size/SHA-256.
The fixture is compressed only for repository storage; tests unpack into private
scratch and verify the entire inventory before running admission. No synthetic
Torch stderr replaces this corpus. The existing exact terminal receipt-write
ENOSPC policy admits the full log including its `EXIT=1`; all 1,279 polls admit.
This is end-to-end diagnostic admission, not a new account-release claim.

The exact Python-3.12 Torch/Inductor `compile_fx.py:312` TensorFloat32 performance warning
admits, including its following `warnings.warn(` line. Only the full known text
and runtime path shape are exempt. The native byte-tail variant missing its first
slash is covered by the r6 observation and a twin. Appended, prefixed, replaced
and nested exception evidence still refuses. No generic warning exemption exists.

An unparseable **first physical line only** of `job_log_tail` may be a native
byte-window fragment. It still receives the raw fault/exit/non-finite-text scan.
Complete JSON there receives normal structured checks; later truncated execution
lines still refuse. A real attempt-15 learner-finite-check fragment and injected
fault twins exercise this boundary. The outer JSON raw pass masks diagnostic
strings only after recursive line validation, avoiding reinterpretation of escaped
nested text. Inventory `path`, `dest`, and `sha256` strings now receive the same
fault scan before any raw-pass masking. The exact native inventory path
`training/stall-watchdog/faulthandler-stacks.log` is permitted only with integer
size zero; a nonempty dump or a modified/fault-bearing path refuses. Plain `value_loss=nan`, `loss: inf` and
signed/Infinity variants refuse.

A truncated scientific staged publication no longer fails the early JSON-only
precheck. Parseable staging still receives all scientific gates; truncated staging
must pass the existing transport prefix-of-verified-canonical check. Raw retained
faults, divergent bytes, unsupported partials and non-finite JSON still refuse.

## New receipt-less compact-export-prefix path (c2 only)

`training_prefix_recovery.py` implements schema
`sandbox-a-train01-retained-export-prefix/v1`. The eligible operator-declared
causes are exactly `host_loss`, `lease_expired`, and `runtime_timeout`.
`operator_confirmed_stopped=true` is required. It records the operator's stop
assertion; it does not manufacture a provider receipt or prove VM release.
Credential-expiry and arbitrary child/scientific failure are not added causes.
Ordinary receipt admission remains unchanged and still refuses assignment loss
without a retained terminal; the new path is explicit, never an exception fallback.

Before preparing a c2, the operator calls
`build_prefix_recovery_certificate(output_root, source_control, supervisor_log,
certificate_output, stop_kind=..., operator_confirmed_stopped=True,
source_packet_id=..., source_packet_manifest_sha256=...)`.
The destination must be fresh and outside the prior output. The function reads
local evidence, validates a private candidate, and publishes a create-only
certificate plus SHA/completion sidecars. It does not query Colab/the account,
write the prior output, seal a packet, or dispatch. These instructions describe a
future post-stop operation and must not be applied to a still-running job.

The verifier requires:

1. Frozen executed-manifest digest, job, source commit, source packet digest,
   absolute output root and exact generation bindings.
2. The latest complete eligible full-state snapshot(s) under the frozen control,
   with compact derivative, compact manifest, metadata, actor, and all immutable
   completion-last SHA/marker bindings. No older/better snapshot may replace the
   latest locally complete snapshot. A final endpoint cannot be continued.
3. A contiguous retained poll prefix from `poll-000000.json` through the **first**
   poll whose export inventory contains every retained snapshot payload's exact
   path/size/hash. Every retained poll has matching job and generation and passes
   the complete recursive diagnostic scan. Missing polls, foreign identities,
   mismatched inventories and later cutoff selection refuse.
4. A supervisor-log byte prefix ending on a complete newline at that poll's exact
   native event: job, status, terminal=false, export count and 4,096-character
   tail agree. Every line from the beginning through that event is scanned. The
   prefix has its own byte length and SHA-256; the verifier never reads its suffix.
5. An exact hash-bound retained file set: executed manifest, contiguous polls,
   snapshot payloads and sidecars. Every retained text/JSON payload is scanned;
   model bytes remain opaque path/size/hash evidence. Any retained fault refuses
   even if an operator rebinds the affected hashes.

Later log text, later polls, receipt/terminal text, unfinished publications and
other post-cut diagnostics provide **no evidence** in this mode. They cannot
improve or poison the retained prefix. They remain on disk, untouched. A newly
complete later snapshot invalidates the old certificate's exact snapshot file
set; regenerate the certificate from the new latest verified snapshot. No trust
or success is inferred from an ignored suffix. This policy is deliberately
narrower than full-attempt scientific admission: it resumes the last verified
state on the operator's stated outcome-blind mechanical cause and cannot claim
that unretained execution after it was healthy.

The certificate is accepted through the existing `--recovery-evidence` builder
argument and `completion_packet` recovery API. The c2 manifest embeds the verified
admission summary under `completion_attempt.admission`. The future SEAL declares
`prior_stop_evidence_kind=receiptless_retained_export_prefix` and retains the same
summary under `prior_release_proof` with `receipt_less=true`,
`original_receipt_used=false`, `post_hoc_release_verified=false`,
`after_last_verified_export_trusted=false`, the explicit poll/log cutoff and
retained-file hashes. Receipt presence is `null` (uninspected), and its hash is
`null`; no receipt is bound. Preflight reopens the committed certificate, repeats
verification, requires completion index 2 and checks both packet labels exactly.
Independent provider reconciliation/stop confirmation remains an operator task;
this input-admission path grants no release, allocation or dispatch authority.

The compact decoder, tensor/full-state validation, migration allowlist, frozen
science/seed/cadence checks, storage/envelope guards, two-completion maximum and
qualified dispatch path remain separate mandatory gates. Receipt-less prefix
admission never decodes or selects numerical outcomes. No new live timeout/lease
trial or transport qualification is claimed by offline positive/fault twins.

## Existing producer captures and limits

The seven eligible broker-stop goldens continue to admit their full retained
outputs/logs: wall cap, helper deadline, CLI deadline, priced runaway guard,
calibration reserve, lease renewal timeout and RPC-exec-poll timeout. Historical
attempt-08/09 telemetry loss also remains covered. All exact receipt/exit/fault
precedence gates remain in force.

The faithful native v2 timeout/lease captures demonstrate why an ordinary receipt
can be unavailable or lack a terminal after self-release. They remain negative
ordinary-receipt fixtures. They do not, by themselves, prove the new compact-prefix
path because they did not retain the complete required compact snapshot evidence.
The old artificial-retention timeout/lease cases remain labelled non-producer
terminal-schema coverage. Unexplained child nonzero exits, clean completion,
public-fetch failures, generation fencing, scientific/preparation/distribution/
stall failures, unverified cleanup and unknown causes gain no admission exception.

This revision supersedes readiness-3's policy that receipt-less recovery has no
positive continuation path. It adds the specific operator-ratified path above;
it does not claim universal producer-stop liveness or a successful c2 execution.
