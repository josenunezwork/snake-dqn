# E8_LATE_BODY_AMENDMENT_02 — natural late-body qualification

Status: JUDGE_AUTHORIZED_PROSPECTIVE_PRE_DATA, 2026-10-22. Authority: round-14
`JUDGMENT.md` ruling (c), all seven clauses, and the owner's implementation
request. Judgment SHA-256:
`95359ebf0e835ea3fc14ab138e94bdfa2f34a627a08b2381c23229ebac0a0314`.
No affected source-latebody-02 data exist at registration; no Colab calls are
part of this work. Freeze this sibling and executable source before new data.

This supersedes only the support construction, fixture eligibility, support
continuation, and evidence exports of `E8_LATE_BODY_AMENDMENT.md` for P1.
The original amendment and source-02 evidence remain immutable. Earlier
`source-latebody-01` preparations are quarantined: do not dispatch them.
`REGISTRATION_P1.md` remains byte-identical, SHA-256
`944aaf3798f9ad698698c5f7abe16e1f5325a034fb1b032ba3cc9387d9111a56`.

## Premises and estimand

Source-02 refused accuracy after selecting nonempty late rows; the exception
lost its metrics. The natural stream reached 778 late lives, 650 completed
(207 survivors, 443 deaths), and 540,912 L>=800 decisions in 2.52M transitions.
These counters establish exposure; exact retained held-out support and old
EV/MSE cannot be recovered from that status. The judgment estimates about 130
completed late held-out lives; source-latebody-02's exported retained-support
table will test the premise. Source-02 stays NOT_READY; it is never rescored.

The registered head is linear on `[z,h]`, z=256 frozen-core features and h the
normalized remaining horizon. Its ability to beat the fit-only `(1,h,L)`
baseline is NOT_READY until measured. The complete-job <=24h nominal invoice
is also NOT_READY until the 1M qualification. No new CUDA kernel, precision,
network path, reward or actor curriculum is introduced by this amendment.

## Frozen qualification and support rules

1. Qualification computes **EV >=0.1 AND MSE < baseline-MSE** on natural,
   completed, held-out L>=800 rows only. Whole worlds stay split by
   `world_id % 5 == 0`. Complete-life suffix returns use gamma one and real
   death/H5000; a rollout cut never closes a target. Fit the baseline on fit
   worlds only. No fixture enters this gate, baseline regression or PopArt fit.
2. Retained natural held-out support requires >=40 distinct completed lives
   with at least one L>=800 row, >=160 retained L>=800 rows and >=8 such lives
   in each scripted/frozen/mixed mix. Retained natural fit support requires
   >=160 distinct completed late lives and >=640 L>=800 rows. Counts refer to
   sampled completed rows/lives, not raw decisions, open lives or duplicates.
3. At >=2M natural fit-world transitions inspect support, before examining
   accuracy. Below any support minimum record
   `INSUFFICIENT_LATE_BODY_SUPPORT`; allow exactly one natural critic-only
   continuation, capped at <=4,000,000 fit-world transitions including any
   supplemental fit rows. Make the final natural rollout shorter to respect
   the cap; charge every collected row. Check
   support once more. Still below => `NOT_READY(P1 support)`, INCOMPLETE critic
   evidence, zero actor updates. Never PASS below minimum or fit until accuracy
   passes. If support is sufficient at 2M, use the original refit duration.
4. The predeclared bounded 25%-root critic-only supplement is permitted only
   when natural **fit support is below its minimum**, never merely for a poor
   accuracy result. Keep its provenance distinct and its fit-only restriction;
   it cannot count as natural support or supply held-out rows. It remains
   bounded at H5000 steps/world, charged inside Y3, with natural actor resets.
5. Collect the twelve fixed first lives separately as `fixture_late_body`,
   using the existing disjoint namespace 2026102111, L900 hero and opponent
   lengths 600/400/300/150/150. Keep their frozen seed exclusions and isolated
   actor RNG. Score them diagnostically with the natural-fit baseline; never
   pool them into support, fitting, qualification or ordinary-bank diagnostics.
   Their targets, counters and metadata remain part of durable replay state.
6. Export finite numeric EV/MSE/baseline-MSE/bias on refusal, with counts and
   explicit validity/support labels (empty support has unavailable metrics and
   cannot pass). Preserve the registered EV computation. Export support by
   provenance x mix x length bucket
   (800–999/1000–1199/>=1200, plus <800) x remaining-horizon bucket
   (<=1500/>1500 frames), including distinct world/life identities. Every
   monitor tick exports hero-length histograms by 500-frame bucket per mix in
   an immutable `monitor-NNNN/training-status.json` with sidecars; the root
   `training-status.json` remains the terminal receipt. The
   length bins are 100 wide with >=12000 overflow. Retain cumulative state
   through pause/resume. Export completed held-out 257d feature/return rows in
   bounded sidecar chunks of <=10MB each with `.sha256` and `.complete`, all
   referenced by a hash-bound row index; also export natural fit rows for the
   registered offline diagnostic. All refusal paths retain
   available evidence; nonfinite data still follow the existing tripwire.
7. Y3 reproduces the support construction, natural/fixture separation,
   qualification verdict and all algorithmic state byte-for-byte; wall time
   alone is excluded. CPU tests do not establish live Y3 PASS.

## Registered offline diagnostic — zero units, no substitution

After source-latebody-02 exports, use natural completed FIT rows to fit to
convergence (a) the registered `[z,h]` affine PopArt head, (b) the predeclared
`[z,h,h*z]` affine PopArt head, and (c) `(1,h,L)`. Evaluate only on the exported
natural held-out rows; no fitting, normalization selection or model selection
uses held-out targets. For these linear heads, deterministic double-precision
least squares with fit-target normalization gives the converged quadratic
PopArt regression solution. Report EV/MSE/bias by mix and length bucket,
sample counts and input digests. This diagnostic is descriptive, costs 0u,
and runs on the Mac outside K-1/X5 windows. It cannot substitute a head or
alter a P1 qualification; any recipe change needs its own prospective
amendment with this diagnostic receipt as premise.

## Caps, transport and unchanged scope

Seed 20261021, fixed 100M endpoint, gamma1, lambda.995, opponents, actor natural
resets, one look, <=24h nominal invoice, tripwires, no checkpoint selection,
banks and release authority are unchanged. Keep the original Y3 <=8u envelope:
source-02 used 1.6u, so source-latebody-02 may consume **at most 6.4u remaining**;
replacement must be repriced from the actual residual shared envelope, never
add a new allowance. Source forecast 75min (49min predecessor plus ~25min
support continuation), timeout150min, L4 ceiling2u/h, reserve1.28u; first wall,
unit or support-transition cap wins. The 1M qualification stays <=4u,
40min estimate/80min timeout/.8u reserve and requires actual Y3 PASS. A time
cap is INCOMPLETE, never permission for another support check.

Prepare new attempt paths `y3/source-latebody-02` and
`qual-1m/prepare-latebody-02`, optionally a blocked replacement template.
Use the established Z-9 small-catalog plus bundled control/sidecars upload;
checkpoint bytes are fetched by verified HTTPS catalog on the runtime. Run
pure local supervisor validation, unpack/verify all sidecars and recompute
normal/fallback pinned-source upload forecasts. `other_assignments` is exactly
`{}` in prepared manifests, an assumption requiring fresh operator inventory
before allocation. No invented download URLs, assignment identities or PASS
receipts. Independent review and new pinned-source preparation precede any
dispatch. Claude owns live receipts/release and campaign-log dispatch entries.

The predecessor source-02 ran ~49min on L4 with this transport; the amended
source is the family's first L4 canary, as allowed by round-14 section5 because
changes affect CPU refit-bank handling and exports, with no new CUDA path.
D1′ Fri12:00 holds only if Y3 passes Fri AM; otherwise the operator records a
create-only slip. No Mac timing work, protected research package, K-1 job,
served B8/checkpoint/pin or sealed study is touched.
