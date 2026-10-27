# F5 attempt-03 verification and forecast/target handoff

**Preliminary qualification VERIFIED PASS. Complete measured forecast HOLD;
prepare-target REFUSED; no TARGET.json or eligible target package exists.**

Publication first: `bc22e072a70941fab45b37fc624276bcd7453e08` was pushed to
origin/main before verification/forecast preparation. `PUBLICATION.json` SHA256
`05cc31778183ce951f70e81ebb0f90a26a1ef12e47c0cce8490b010418e47122`.
Reconstructed 71 originals / 14,868,322 bytes, including all 60 output files /
14,795,858 bytes. Every published part matches that commit and the original.
The complete output includes staging source archive, original export companions,
receipt, terminal, uploads, polls, accounting and saved broker release evidence.

| Verification | Result | Evidence |
| --- | --- | --- |
| L4 terminal receipt | PASS | 753.1065790839493 s, status PASS, terminal succeeded/returncode 0, cleanup/error null |
| Genuine CUDA | PASS | One source-attributed fp32 B8 CUDA forward/masked argmax at cuda:0, shape [1,6], synchronization; saved native CUDA/CPU parity PASS |
| Native environment v6 vs capture-05 | PASS | Current validator returns no failures; frozen v5 capture-05 comparison adapter admits matching v6 canonical identity while retaining distinct raw GPU/kernel identities |
| Required exports | PASS | environment.json 13,420,032 bytes and qualification.json 421 bytes, SHA/size/sidecars/marker verified |
| Saved release | PASS | stop_verified true, sessions.json empty, post-release assignment count zero; no live account query made |
| Source/package identity | PASS | Original sealed 233-entry executable closure/inputs verified; documentary publication did not alter runtime source |
| Focused current-project tests | PASS | 33 passed in 36.01 s, project venv, ordinary pytest config, serial numeric threads one |
| Forecast | HOLD | Independently accepted long-body/worker/complete incremental costs absent |
| prepare-target | REFUSED | Real entrypoint, absolute nice15, exit 1, no TARGET.json |

Independent verifier report SHA256
`749e09780f3dacbb082b56f22f707433b1ba88fb70b09b0aee2b617e78d2501c`.
Its extra 39-test Anaconda run is supplementary. The source-of-truth project
venv passed the ordinary repository configuration; no project pytest blocker
is inferred from the verifier's unrelated interpreter.

## Forecast and accounting

The preliminary lifecycle is measured at 753.1065790839493 s. Its 177.01096445892472-s
staging includes 87.49885679106228-s source delivery and 88.10770795796998-s
input transfer; these nested measurements are not added again to the lifecycle.
Historical short-body diagnostics are 171.274947290 s scans and
299.2773495600022 s bundle-plus-six-individual expansion. Adding those diagnostics
to the preliminary lifecycle gives **1223.658875933952 s**,
a partial calculation, not a complete forecast or admissible bound.

No measured_total_seconds is asserted. Long-body crossing clone/digest/features/
legal setup, actual bundle and separately cloned individual H300 costs, scan
instrumentation, six-worker imports/model/JIT/cache and pool costs, incremental
unit/ledger/summary I/O, imbalance/idle and complete cleanup remain unresolved.
V4 ENGINEERING_V4.md lines 24-27 requires independently accepted operational
evidence and expressly forbids inferring long-body bounds from short-body timings.
ENGINEERING_CORRECTIONS.md preserves the same distinction. The source review
confirmed historical branch timing occurs before the natural long-body trace.

ACCOUNTING_APPEND.json preserves attempts 01/02/03 at 247.70002616604324 /
625.7038177910727 / 753.1065790839493 s, totaling **1626.5104230410652 s**.
Saved account balance deltas are observations with provider lag, not exact final
billed debit. Declared-rate inclusive conservative charges are recorded separately;
an absent first-attempt crosscheck stays null. Historical timing 10,902.09342033323 s
and historical canaries 3,849.601884124102 s keep their separate original attribution.
The prospective 480-minute effective envelope, measurement 60/120 min, archive
96/192 h, shard 6/12 h and two global completion rounds remain unchanged.

No Mac companion benchmark was collected. Read-only snapshot at 17:30:46 PT:
18 logical CPUs, load 6.54/6.97/8.15, CPU 73.23% idle, AC attached, no thermal
warning recorded. This does not certify an uncontended timing window. Later
step-5 Mac/production forecasts remain separate and unmeasured here.

## Concrete preparation result and next Astra action

Review dossier root: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/sol-20261026/f5-attempt03-target-preparation`.
Forecast SHA256: `839ebe3190215430302cc76b89e4403a6a7f06d56393094d2e0c16d085cdfab2`.
Absolute-nice15 refusal receipt SHA256:
`ab95904a4c72295c8c614a819877dafbe6fc95259d3df0c78d845a38e1c33b3f`.
Partial staged control SHA256: `196a02f9c3980a3713d2df5060e7ba09f2a65bd304a291d8ecabbeba98116cd8`.
Partial staging: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/sol-20261026/f5-attempt03-target-preparation/prepare-target-nice15`.

The real prepare-target invocation refuses exactly
`measure requires finite qualified 60/120-minute envelope`. It staged immutable
inputs before validation, so the partial directory is preserved; it contains
no TARGET.json and is not an independently acceptable measurement target.
The first local preparation at more conservative nice20 is also preserved; the
absolute-nice15 successor changes no input/science/source bytes. Neither is a
Colab attempt, timing-unit retry or scientific look.

Astra must first identify/accept a valid independently measured cost basis or
issue a narrow prospective disposition resolving the missing-cost and ordering
prerequisite. Current V4/V5 authority cannot close that requirement from a
qualification-only PASS. With a genuinely complete <=3600-s forecast, Sol can run
prepare-target into a fresh directory; Astra then accepts the resulting
acceptance_target_sha256, execution-envelope digest, qualification/source/fixture
identities and empty component ledger. Seal-final, exact-final-package/order
review, matching final-inventory canary and separately authorized measurement
remain later operations. No acceptance payload was invented.

No dispatch, Colab/provider call, seal-final, timing-fixture collection, scientific
look, fit, evaluation, serving/release change, sandbox access, tmux modification,
checkpoint overwrite, pin change or hash-bound historical edit occurred.
