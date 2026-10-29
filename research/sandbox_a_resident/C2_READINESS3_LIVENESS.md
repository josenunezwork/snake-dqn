# TRAIN-01 c2 admission and liveness boundary (readiness 5)

Implementation for independent Opus/Argon review. This revision addresses the
readiness-4 REJECT and supersedes that revision's policy of ignoring suffixes and
accepting operator-only release assertions. No production seal, dispatch or
transport change is made. `research/colab_cli` stays at tree
`5833e54e9c5396121cbbf99a5cd4e52006e564fd`.

## Explicit receipt-less c2 recovery

`training_prefix_recovery.py` uses schema
`sandbox-a-train01-retained-export-prefix/v2`. Version 1 is refused. It remains
an explicit recovery mode, never an exception fallback, and is restricted to
completion index 2 by completion-packet admission and packet preflight.

The operator prepares a certificate only after a stop, using the existing
`build_prefix_recovery_certificate` API with a new required keyword:
`account_status_capture=Path(...)`. That file must be an existing immutable
`sandbox-a-colab-account-release-capture/v1` capture, with SHA/completion sidecars,
created by the qualified account-capture tool. The builder does not query an
account, contact Colab, stop a runtime or dispatch. Its destination is create-only
and outside the prior output. It validates a private candidate before publishing.
No capture or recovery certificate for the currently live r6 was created here.

Admission requires all of the following:

1. The frozen executed manifest, source commit, packet identity and digest,
   absolute output root, and exact generation agree. The latest eligible compact
   full-state snapshot and its original/derivative/header/actor bindings pass all
   existing completion-last and SHA checks. A final endpoint cannot continue.
2. A contiguous poll prefix ends at the first poll listing every required compact
   snapshot payload. The entire supervisor prefix is hash-bound and scanned;
   **every** native event matches its poll in order (job, status, terminal=false,
   export count and log tail), with finite monotonic elapsed time. Matching only
   the last event is insufficient. Suffix exports cannot bind any resume bytes.
3. The certificate records the complete recursive export-directory listing,
   including unfinished files and sidecars, by path/size/SHA. Any listing drift
   invalidates it. A newer snapshot advertised by any retained poll poisons an
   older cutoff, even if its local payloads were removed before rebuilding the
   certificate. Offline admission cannot reconstruct evidence removed before
   capture from every retained source; preserving original evidence is mandatory.
4. **All** retained output files and the complete supervisor-log suffix are scanned
   for scientific faults, failed stages, non-finite values, foreign identities,
   distribution material gaps, stack dumps, completed summaries and unknown
   execution failures. A later fault refuses even if it happened after the last
   verified export. EOF-truncated post-cutoff JSON is tolerated only after raw
   scans and checks of complete retained execution/fault fields. Incomplete files
   never support admission. Existing receipt/terminal files must be complete.
5. Any receipt must match the exact known ACCOUNTING assignment-disappearance
   envelope with verified stop, null cleanup error, frozen identities and a
   retained mechanical terminal. Every terminal must match the same job and
   generation, status `failed`, reason `timeout` or `lease_expired`, and a known
   producer stop returncode (0, -9, -10, -15 or -30). All other fields, including
   nested diagnostics, still receive the recursive fault scan. An arbitrary child
   failure plus a mechanical reason cannot enter this path. The only omitted
   receipt fields are the exact independently validated envelope values. The
   ordinary receipt path and its eligibility rules are unchanged.
6. The declared cause agrees with retained evidence. `lease_expired` needs exactly
   that mechanical terminal reason. `runtime_timeout` needs exactly `timeout` and
   same-job elapsed evidence within 60 seconds before through 600 seconds after
   the manifest timeout. Three hours of a 46-hour run cannot qualify. `host_loss`
   requires an interrupted healthy prefix and no contradictory terminal/receipt;
   its cause remains an operator assertion constrained by the retained evidence.
7. The shared ENOSPC-path account-capture verifier checks the immutable release
   capture's timestamp (after the supervisor log, no future timestamp, maximum
   age 24 hours), empty server sessions, zero assignments, zero rate and empty
   unresolved state. The helper reads the supplied capture only. It does not read
   live account state. Future dispatch still requires the qualified operator's
   normal admission process; an old capture does not assert perpetual emptiness.

The packet's explicit `receiptless_retained_export_prefix` label remains.
`receipt_less=true` and `original_receipt_used=false` mean that no ordinary receipt
provides positive resume authority; an existing known mechanical receipt is
nevertheless inspected, hashed and reported. `original_receipt_present` is now a
boolean, and its digest is bound when present. `post_hoc_release_verified=true`
records the checked capture, `suffix_scanned_for_poison=true` records the scan,
and `after_last_verified_export_trusted=false` denies suffix snapshot authority.
Preflight reopens the certificate, repeats all checks and compares the summary,
receipt presence/hash and both c2 labels. The full-state decoder, migration/source
pins, frozen science/cadence, storage, envelopes and retry limit remain mandatory.

## Diagnostic regressions and evidence

`test_c2_readiness5.py` commits each readiness-4 demonstrated attack as a refusal
twin across certificate building, verification and completion admission. It also
covers account/cause contradictions, arbitrary suffix exceptions, truncated fault
fields, binding every prefix event, export-list drift, deleted newer snapshots,
and suffix inventory attempts to support a prefix. The earlier test that admitted
a later arbitrary receipt/fault is replaced by a refusal assertion.

The first-line diagnostic fragment scan now catches arbitrary failed stage/phase
values, including `"stage": "update-failed"}`. The non-finite text boundary admits
hyphenated words such as `mode: nan-guard`, while actual `loss=nan`, `loss: inf` and
Infinity values refuse. Existing exact Torch TF32-warning and fragment policies
remain covered by the readiness-4 suite.

The original immutable attempt-15 corpus remains unchanged: 1,428 log lines and
1,279 polls, verified against its inventory before full replay. The fresh
`fixtures/r6_readiness5_liveness` corpus contains byte copies of 1,296 log lines,
1,088 polls and the executed manifest, captured 2026-10-29 06:35:48 UTC using only
local reads into private scratch. Every copied file is inventory-verified before
admission. These are diagnostic-liveness regressions; neither manufactures account
release, a healthy suffix, an experimental outcome or dispatch authorization.

The native supervisor goldens retain their prior ordinary-admission behavior.
Actual native timeout/lease captures without the required retained terminal and
compact-export evidence remain refused; this revision does not reconstruct or
invent absent evidence. No new live timeout/lease trial or transport qualification
is claimed by offline positive/fault twins.
