# Sibling correction: A4 omitted allocation lifetimes

Decision: **A4's admission is insufficient; A5 admission is REFUSED under the
conservative allocation bounds below. No Q1 dispatch, no further cap raise.**
This corrects operational accounting, not a scientific result. The A4 admission
at `3f231b2e`, its 120 unqueued manifests, old accounting, attempt receipts,
registration prefix and owner admissions remain unchanged.

The original registration charges "All startup, qualification, copies,
evaluation, failed attempts, and cleanup are charged". The old
`q0_accounting.py` summed claim-to-terminal intervals. Those intervals include
some queue setup but omit allocation before the first claim, between claims,
and after the final claim. A4's 29.468238339864364 combined GPU-hour forecast
therefore did not establish compliance with its 29.6 usable limit.

## Evidence and bounds

The sibling `allocation_accounting.py` retains all 12 original Q0/Q0b attempts
as provenance and charges each runtime lifetime once instead of adding claim
durations again. It also includes the overnight host-contract claim inside its
A100 lifetime. The generator and runtime admission revalidate worker/claim
coverage, historical hardware, original attempt duration, full lifetime totals,
and the independently frozen A3 credit envelope.

Evidence root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/q1-a5-20261018`.
`allocation-bounds-v2.json` records each source path and SHA-256; its evidence
includes immutable snapshots of the four retained heartbeats, queue claims,
terminal receipts retained in the original accounting, queue control records,
the overnight operational note, attempt worker logs, owner-supplied facts and
read-only account-session observations. Attempt logs establish environment
bootstrap, not assignment time. No aggregate worker log or first-heartbeat
history was found: queue heartbeat JSON is overwritten, so a retained heartbeat
is a latest snapshot, never an allocation-start or release receipt.

**These are conservative policy upper bounds, not measured consumption.**
Assignment starts are missing. The disclosed start bound is **2026-10-17
18:00 PDT (2026-10-18 01:00 UTC)**, before the first preregistration commit at
20:52:39 PDT and within the owner-described evening scope. Each unidentified
start is charged from that boundary. This can substantially overcharge
overlapping uncertainty. The unidentified earlier L4 is reserved separately
because its identity cannot be reconciled to the current L4; that can double
charge one allocation. It is not asserted that all these lifetimes overlapped
in reality. Additional undisclosed allocations invalidate the inventory;
they may not be assigned zero cost.

The prospective Q1 dispatch assumption is **2026-10-18 20:00 PDT**. The retained
L4 is charged through **20:30 PDT**, exactly dispatch plus 30 minutes, including
all idle time after its r6 packet ended around 17:57. Its actual release was
verified earlier; this reconciliation retains the requested prospective bound
and does not seek admission by rebating the unused interval.

| Runtime / hardware | Relevant observations (PDT) | Charged end bound (PDT) | Lifetime upper bound (hours) | Unit upper bound |
| --- | --- | --- | ---: | ---: |
| `3854efbc34ec…` / A100 | Worker started Oct 18 00:36; first Q0 claim 00:36:44; retained heartbeat 02:03:20; explicit verified-disconnect control 07:26:28 | Oct 18 07:26:28 | 13.44111111111111 | 71.23788888888889 |
| `291b7018260c…` / A100 | First claim Oct 18 08:03:58; final attempt ended 15:21:43; owner reports idle 15:22–18:38 and unassign at 18:38; retained heartbeat 18:38:04 rounds release upward | Oct 18 18:38:04 | 24.634444444444444 | 130.56255555555555 |
| `d946df279cf1…` / CPU | First claim Oct 18 00:33:32; idle heartbeat 00:37:19; no individual release receipt; do not assume the ten-minute idle timer succeeded | Oct 18 18:58:38, account has no active sessions | 24.977222222222224 | 12.488611111111112 |
| `notebook-bb4112838e9b…` / current L4 | Worker started Oct 18 about 17:12; claim 17:12:28; terminal 17:56:00; owner packet-end estimate 17:57; still idle at initial observation | Oct 18 20:30, prospective assumption | 26.5 | 53.0 |
| Earlier L4 / unresolved identity | Owner reports an earlier L4; no assignment, worker or release history located. Another worker starting does not prove this allocation ended | Oct 18 18:58:38, account has no active sessions | 24.977222222222224 | 49.95444444444445 |

Rates remain the historical receipt's conservative ceilings: A100 5.3,
L4 2.0 and CPU 0.5 units/hour. CPU and L4 rates are planning inputs, not fresh
billing measurements. Summed Q0 allocation bounds are **89.55277777777778 GPU-h,
24.977222222222224 CPU-h and 317.24350000000004 units**. The old claim totals
were 13.394975767532985 GPU-h, 0.06305555555555556 CPU-h and
68.63035524611341 units. The difference is bounded unclaimed/unknown allocation
and identity uncertainty; it must not be described as an observed debit.

For scale, a *tighter diagnostic scenario* that starts at the observed workers
rather than the missing allocation starts would charge 6.841111111111111 hours
for the first A100, 10.568333333333333 for the second, and 3.3 for the current
L4 through 20:30. With Q1's 16.07326257233138 hours this is
36.782707016775824 GPU-hours. This scenario already exceeds 36 but is neither a
certified upper bound nor a measured minimum: the first A100 disconnect control
is itself an end upper bound, and omitted setup/earlier L4 remain unresolved.
It is not substituted for the conservative admission calculation.

## Final prospective limits

A5 quotes the owner's exact decision, freezes scientific choices and states no
further raises. GPU is **45 nominal / 36 usable**. Unit and CPU limits stay fixed:
203.63035524611342 nominal / 168.63035524611342 usable units; 40 nominal / 32
usable CPU-hours. The frozen A3 Q0 unit cap remains 68.63035524611341 and Q0 CPU
cap remains 8 hours. Historical breaches are disclosed, never retroactively
validated by this amendment. A2/A3 remain explicitly post-diagnostic.

The remaining Q1 costs reuse unchanged A4 evidence: 12.243442220561551 training
GPU-hours plus 3.8298203517698304 overhead hours; 26.32554603290217 CPU-hours;
32.14652514466276 training units plus 13.162773016451087 evaluation units.
The complete upper-bound forecast is therefore:

| Resource | Q0 allocation bound + Q1 | Usable limit | Nominal cap | Decision |
| --- | ---: | ---: | ---: | --- |
| GPU-hours | 105.62604035010915 | 36 | 45 | REFUSE |
| CPU-hours | 51.30276825512439 | 32 | 40 | REFUSE |
| Units | 362.5527981611139 | 168.63035524611342 | 203.63035524611342 | REFUSE |

Q0's historical unit and CPU caps also fail under these bounds. The large
unknown intervals are an evidence limitation, not a reason to increase a cap
or assume free allocations. Q1 does not run under A5. There are no eligible
enqueue commands and no new runtime plan to start. The conditional L4 + CPU +
CPU plan would require full admission and at most three account-wide
assignments, including any retained notebook workers; it is not authorized by
this refusal.

The current L4 was terminated through Colab's existing-session UI after refusal.
Read-only `colab --auth oauth2 sessions` reported **No active sessions found on
server** at 2026-10-19 01:57:53 UTC and again at 01:58:38 UTC. No notebook cell
was executed, no new runtime allocated, no Q1 job enqueued, and no old queue
evidence altered.

The actual clean-commit generator invocation, terminal refusal, exact commands,
test results and artifact hashes are recorded in the sibling A5 admission
handoff after publication. A refusal writes reviewable `study-plan.json` and
`admission.json` but materializes zero queue manifests.
