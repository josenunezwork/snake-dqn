# FIX-FIRST: F4 post-mortem exporter

Astra independent read-only review, 2026-10-25.
Implementation a31a1d540b1d00eaab01525e7fedcdb966ae6522; execution pin 7f3c0857dda8df0e59ad296aa1d51c859dffb68c.
Governing exception SHA256: 8c4480a5e80f63ec500abd7baa6062adc757866937ce560291cbbaab68b5eeec.

**FIX-FIRST. No operator run command is approved by this record. F4 remains STOP_FUNDING.**

## Blocking findings

1. **P1 — publish-before-inspection is incomplete.** In research/bl_residual/f4_postmortem_result.py:232-244, verify proves remote equality only for binding-listed files. At :246-251 it discards verify_seal's returned outcome/hash inventory; :253-289 then deserializes any locally present failure.json/diagnostic.json. A published diagnostic-only bundle with an additional locally sealed failure, or a published failure-only bundle with an additional locally sealed diagnostic, can therefore expose an unpublished outcome. Before any outcome parsing, require exact equality of actual sealed outcome/file/hash inventory and the remotely verified binding. Add synthetic refusal tests for both extra-outcome directions, proving refusal precedes deserialization. This finding is established by source tracing; no result bundle was opened or modified.

2. **P1 — the hard 60-minute ceiling is not independently enforced.** research/bl_residual/f4_postmortem_runtime.py:206-214,254-258,299-311 implements both soft and hard deadlines through a Python SIGALRM callback; even os._exit(124) requires that callback to execute. Long or stalled native inference/finalization can defer it. The tests invoke fake signal handlers, establishing control flow rather than a hard native-stall bound. A data-free local probe requested SIGALRM after 0.020 seconds during hashlib.pbkdf2_hmac and observed the callback only after 0.138940750 seconds. This demonstrates handler deferral, not an observed exporter overrun. Provide an independent kill mechanism that does not depend on exporter Python callbacks, remains active from before claim through all terminal finalization, preserves the spent claim, and never retries. Prove it with a short synthetic native-stall/finalization test before publishing the revised pin.

## Checks that passed this review

- Clean reviewed worktree; git ls-remote origin refs/heads/main returned exactly 7f3c0857dda8df0e59ad296aa1d51c859dffb68c. a31a1d54 is its ancestor. All 13 manifest implementation hashes and package sidecars checked; package bytes checked against the pin. Original exception bytes equal the published copy.
- Diff against f9c08082 shows optional observer instrumentation only in fit/heldout; original action/harm/margin computation remains. D256 then D1024 reuse rowwise heldout evaluation. Export retains exact H=C+U, disjoint unknown-side components, authorized denominators, four fixed bins and every parent including zeros; no new inferential statistic or funding invocation.
- The exclusion ledger equals all 29 ruling/census parents in order and permanently excludes descendants and unexposed siblings; unresolved ancestry fails closed.
- Publication gate precedes claim and payload verification. Fixed-directory atomic reservation and create-only sealed plan/claim spend partial attempts. Exclusive live Mac lease, CPU, nice 15, one math thread, AC/lid/thermal/wake checks are present. Timer scope includes terminal exports, subject to finding 2.
- Sol reports 130 combined and 44 published-pin tests passed. Synthetic evidence and test source were reviewed; these suites were not rerun here. No exporter was invoked.

## Mandatory sequencing

**Even after these fixes are accepted, the run must wait until the F5 registration is ADOPTED and published before diagnostic exposure.** The inspected F5_DRAFT_PREDATA_REGISTRATION_20261025.md explicitly remains DRAFT_PREDATA — NOT ADOPTED; its fixed selector consumes this export. The current exporter does not itself enforce F5 adoption. Operator instructions must carry this prerequisite; no drafting, implementation acceptance or this review substitutes for adoption. Result publication must still precede outcome inspection. No claim, retry, F5 execution or release is authorized here.

Review boundaries: source, named published metadata and Sol report only; no checkpoint/table/outcome payload, D4096 or t34-round0-study-01 access; no commits, dispatch or tmux. The only writes are this create-only record and its SHA256 sidecar. Repository remained clean. Two independent read-only reviewer lanes checked runtime and publication boundaries.
