# P1 round-14 late-body amendment 02 handoff

Status: **VALIDATED_NOT_DISPATCHED**, zero Colab calls. This sibling supersedes
the latebody-01 preparation instructions. Old prepared attempts have immutable
`QUARANTINED_ROUND14.json` siblings; their original bytes remain unchanged.

Implementation commit: `ea1f88dd5813ba8d1161927b1debe38149eb2e16`.
Published executable pin: `44929777e9e5a7f580be6a2b766fb8e15c54e487`, the
merge with concurrent density-only changes. Later publication metadata does not
change this executable pin. Original registration SHA:
`944aaf3798f9ad698698c5f7abe16e1f5325a034fb1b032ba3cc9387d9111a56`.
Amendment 02 SHA:
`e1ca389ebb5193cbe75b00cf633e5c8657376413bf1bebd6923e4978afb12ec8`.

The gate uses natural completed held-out L>=800 rows only. The registered
EV>=.1 and MSE<baseline bar is unchanged. Twelve fixtures are diagnostic in a
separate bank; they never fit PopArt/baseline or contribute to the gate.
Natural support floors are 160 fit lives/640 rows and 40 held-out lives/160
rows/8 lives per mix. One support-only continuation is bounded at <=4M fit
transitions including supplemental fit rows; continued shortage is NOT_READY.
Sufficient-support accuracy failure never extends the fit. Numeric refusals,
support tables, monitor-time histograms and natural feature/return sidecars
remain available. The registered offline three-head comparison cannot
substitute a head into P1.

## Prepared operator artifacts

Artifact root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/`.

| Phase | Directory under artifact root | Estimate/timeout | Cap/reserve |
|---|---|---|---|
| Y3 source | `y3/source-latebody-02/` | 75/150 min | 6.4/1.28u |
| Y3 replacement template | `y3/replacement-latebody-02/` | 20/40 min | 2/.4u, conditional on actual Y3 remainder |
| 1M qualification template | `qual-1m/prepare-latebody-02/` | 40/80 min | 4/.8u |

Each directory contains `manifest.json`, `DISPATCH.txt`, create-only preparation
and validation receipts, recomputed pinned-source Z-9 upload readiness, and
verified unpacked control/audit sidecars. Each uses
`budget.other_assignments={}` as a preparation assumption, never a live account
inventory. Local `colab-output` directories remain absent. The common index
and dispatch file are `registration/latebody-02/manifests.json` and
`registration/latebody-02/DISPATCH.txt`. The committed
`P1_LATE_BODY_INDEX_02.json` binds their digests and independent reviews.

Y3 remains inside its original **8u** phase envelope: the judgment accounts
1.6u for source-02, leaving at most 6.4u for the amended source and subsequent
replacement together. The replacement template is not a fresh 2u allowance;
Claude must reprice it from the actual residual invoice. Source uses the
predecessor's 49min L4 measurement plus ~25min support contingency as a forecast,
not a measured complete-job rate. First wall/unit/transition cap wins.

All manifests retain preparation-only HOLDs and operator blockers. They need
fresh usage/inventory and verified predecessor release before allocation.
Replacement additionally needs actual source receipts, a distinct assignment,
and a real transitive HTTPS snapshot/trace catalog. Qualification needs actual
Y3 PASS and the original qualified origin catalog. No assignment, URL, live
PASS or funding receipt was invented. Inventory or prerequisites changing
requires a new prepared attempt; immutable controls are not patched in place.
Claude owns actual dispatch and campaign-log entries. No release is authorized.

## Verification and review

Integrated targeted selection: **115 passed in 72.63s**. It includes support
floors and fixture isolation, numerical refusal, exact entrypoints, full Y3
state fingerprints, qualification, every tranche resume, required-sidecar
failure and nonfinite KILL retention. Changed Python files pass Black, isort
and flake8; shell syntax and diff checks pass. Work ran at nice 10 with at most
two processes/one math thread each. The full repo suite was not run.

`P1_LATE_BODY_REVIEW_02.md` records the independent read-only review and the
root's second review against the full judgment. Final source hashes,
test log and verification receipt are under
`verification-latebody02-01/`; the publication index binds the artifact review.
The positive tiny replay uses explicitly synthetic natural-labelled bank
support and controlled predictions inside pytest. It establishes no natural
support, critic accuracy, scientific qualification or actual L4 Y3 identity.

No Colab CLI command, tmux operation, serving change, protected-package edit
or sealed-directory access was performed. Source-02 remains the original
accuracy refusal with unrecoverable old metrics; it is not relabelled or
rescored. The offline diagnostic is registered, awaiting new exported rows;
run it outside K-1/X5 windows at 0u before proposing any recipe change.
