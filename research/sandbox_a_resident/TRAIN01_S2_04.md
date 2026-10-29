# TRAIN-01 S2-04 — authorized fresh seed-2 restart

Prepared only; operator dispatches. Owner-delegated operator decision (2026-10-29)
authorizes `sandbox-a-train01-20261028-s2-04` as a **fresh start with no reused
state**, the same recipe and seed allocation. All S2-03 evidence remains intact.

## Contract accounting

TRAIN01.md:191-196 says: “There is one original dispatch plus at most two
**manual**, outcome-blind completion attempts after a verified mechanical stop
and missing work.” It further says: “Calibration-only or failed-preparation state
is not a valid full-PPO resume.” TRAIN01.md:184-189 requires complete hash-bound
full-trajectory units; TRAIN01_S2_01.md:53-59 requires the latest eligible verified
snapshot and preserved RNG, counters, preparation and banks.

S2-03 executed and remains **INCOMPLETE** after its released mechanical RPC poll
stop. It exported no eligible trajectory snapshot; its calibration boundary is
not a resume unit. No completion attempt was dispatched for S2-03. The offline
S2d checks consumed no attempt. S2-04 is **attempt 1 (original dispatch) of a new,
explicitly operator-authorized fresh run**, not completion attempt 2 of S2-03.
The contract's completion path resumes a trajectory; it does not turn an empty
ledger restart into a completion. The fresh-run authority is the explicit new
operator decision, not an inferred restart entitlement in TRAIN01.md. Preserve
S2-03 as incomplete and charge its actual compute to its retained receipt; do not
count S2-04 as an additional independent efficacy replicate or replacement seed.
This new run has at most two manual, outcome-blind completions from its latest
eligible verified full-state snapshot, after mechanical failure, verified release
and missing-unit checks. No automatic completion retry is authorized.

## Science and finite envelope

Same B8 start, allocation index 2, training offset 2,000,000,000, 384 PPO updates,
1024 worlds, rollout 256, horizon 5000, 100,663,296 actor rows; unchanged learning,
calibration, critic preparation, numerical runtime, opponents and proxy banks.
Existing seed/union audits are reused byte-for-byte. The initial ledger is empty.
This deliberately reuses the authorized seed domain to restart the incomplete
replicate; it is not a newly disjoint bank. Snapshots remain at iteration 0,
every 16 PPO updates, and the final boundary. Proxy cadence is unchanged.

Base/effective attempt timeout: 690/1,380 minutes (23 hours effective). Estimate
690 minutes is an attempt window, not a full-endpoint forecast (A100 matched
full-workload forecast remains 2,009.04 minutes). H100 first, A100 fallback;
whole-timeout base/effective caps 138/276 units at 12 u/h H100 provisional ceiling,
92/184 units at 8 u/h A100; reserves 0/0. The owner's 2x rule does not double N,
rates, cadence, measured estimates or the two-completion maximum.

## Operational changes without a science amendment

TRAIN01.md:72 freezes the calibrated/pre-actor snapshot boundary. S2-04 opts into
`SANDBOX_A_EARLY_PRE_ACTOR_SNAPSHOT=1`: publish that existing full iteration-0000
unit after successful preparation and the zero-update greedy check, before the
initial proxy. It includes full optimizer/task/RNG/preparation state and an actor
export, using compact completion-last publication. A stop in the initial proxy
can resume it and rebuild the absent baseline on the unchanged bank. Calibration
alone and failed preparation still cannot resume. Historical default behavior is
unchanged. A bounded production-path regression compares the resumed early unit
with the default uninterrupted run's final metrics, task state and RNG streams.
This does not recover a stop during critic preparation.

The retained-evidence scanner is derived from accepted da85a426. Its exact eager
fallback, TF32 warning and mid-line tail rules are retained. Only scanner machinery
is imported; this packet does not introduce the newer receipt-less prefix-recovery
route or change the handed branch's host-ENOSPC recovery contract. Fixes admit the
canonical calibrated-boundary.pt through existing streamed hash/sidecar proof and
recognize the native OAuth CLI exec TimeoutExpired only at a validated, released
RPC_EXEC_POLL_TIMEOUT receipt root with matching session and finite bounded timeout.
Nested/compound faults still refuse. Full-root replay also exposed native v2
screen statuses; only the exact distribution-screen publication/schema and healthy
producer status vocabulary receive a scan view, with all other fields recursively
checked. Fault twins preserve material-gap, nonfinite, identity and nested-error
refusals. None of these changes creates a trajectory header or makes S2-03 a c1.

Runtime/payload pins, independent Argon review, test counts, same-pin sandbox
admission 3x per GPU, real dispatch wrapper dry-run and exact dispatch command:
`VERIFICATION_TRAIN01_S2_04.json` and
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/astra-impl-train01-s2e/SUMMARY.md`.
No live GPU execution, CUDA throughput, new canary, serving change, release or F5
crossing is claimed. Publication is branch-only; main and all prior packets remain
unchanged. Operator dispatch remains the live accounting/host/hardware boundary.
