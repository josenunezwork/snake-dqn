# Prospective remote-supervision evidence amendment

Status: **IMPLEMENTATION INCOMPLETE — LIVE ADMISSION UNAVAILABLE.** This is a
prospective engineering policy and offline evidence implementation, not an
implementation acceptance or permission to run a classified panel alongside an
unobserved supervisor. Policy ID: `classified-supervised-v2`. `classified-v1`
and its original amendment retain their historical interpretation.

Authority: Astra's complete ruling at
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/astra-guard-supervisor/RULING.md`,
SHA256 `83e47ff2b3cfddba0d9ee4a6ec2bab48efb645259857d7e81ea5fd23459f7426`.
The ruling is conditional; this amendment does not ratify its own implementation.
Attempts 04/05 and other historical aborts remain INVALID and are never rescored.

## Identity and permitted activity

Seal each declared root's UID, PID, microsecond-or-better birth identity,
canonical interpreter, exact module/argv structure, canonical actual source
repository/cwd and state directory, source hashes including generated startup
hooks, manifest digest, and non-secret assignment/session/generation. Revalidate
throughout the attempt. Disk hashes alone do not prove which bytes an already
running interpreter loaded. Restart/reuse/assignment changes require new admission.

A child needs a witnessed birth-aware launch chain and a verified operation
template. Neither a module name, state-dir substring, tmux name, ancestry alone,
reparenting, nor a caller's tag establishes permission. Retain intermediate
identities after exit and track exec changes. Only compact state/terminal reads,
usage/sessions reads, and exact generation-bound lease renewals qualify. Remote
exec must match the pinned compact read_state/renew_lease code; upload must match
the exact small heartbeat content and destination. Request plus response
application payload is at most 65,536 bytes, excluding protocol overhead. Retain
sizes and digests, never credentials. Generic exec/upload/download, allocation,
staging, bulk export, decompression, substantive hashing, fitting, rollout and
pytest are ordinary work. Credential/cleanup authority continues but does not
grant a timing exemption. Leaving the envelope invalidates the panel; it must
never disable remote safety work.

## Exact engineering ceilings

These are prospective one-core CPU ceilings, not measured harmlessness and not
Colab envelopes; do not apply the owner's envelope doubling rule to them.

| Check | Invalid boundary |
| --- | --- |
| Ordinary workload/unknown, each process | >=20% in historical point and interval screens |
| All admitted roots and permitted descendants, all lanes | >50% in any monitored 0.5–5 s CPU interval |
| Any rolling 10 s | >2.0 CPU-s |
| Any rolling 60 s | >3.0 CPU-s |
| Individual operation, entire helper subtree | >1.0 CPU-s |
| Aggregate service/renderer background | strictly >150% for strictly >60 s, original continuity/reset rules |

Equality is allowed for the four supervision ceilings, and forbidden at the
ordinary 20% boundary. Nominal sampling is 1 s. Intervals outside 0.5–5 s cannot
certify supervision compliance. Require 60 s of complete actual-workload
host-only evidence before warm-up or measured exposure. Continue one accounting
stream through warm-up, all cells/gaps and final capture; no debt reset at cell
boundaries, root/child churn or exit. Include already-active operations or wait
until a complete window is available. A missing/gapped stream >5 s invalidates,
never becomes zero CPU or a cleared window. Keep historical point CPU as a
covariate for *verified* supervision; its interval/operation rules replace the
ordinary point abort only when all identity/accounting evidence is available.

Use witnessed launch/exec/exit records with per-child final user+system CPU,
independent ongoing process sampling and root deltas. Include between-sample
children. Reconcile descendant-inclusive wait4 usage without double counting.
Enforce while operations run, then reconcile final counters. Negative counters,
rounding ambiguity, missing exits, unobserved helpers, dropped journals or
unresolved discrepancies fail closed. Unplaced CPU is charged in full to every
intersecting interval and rolling window, including trailing windows after exit;
retain conservative bounds separately from measured totals. Before sealing,
obtain finals for all helpers overlapping scored exposure within qualified call
deadlines. Short final tails must receive bounded complete coverage or INVALID.

## Immutable evidence and reporting

The registry, ruling and implementation source hashes belong in the intent.
Create-only payload/SHA/completion triplets bind the registry and exported
evidence. Retain raw point screens, root identities, operation IDs/templates,
launch/exec/exit identities, monotonic brackets, sampled/final CPU, observed and
conservative maxima, payload sizes, exclusions, gaps and abort reasons.
The reducer independently replays evidence and refuses summary or binding
mismatches. This is contamination evidence for accidental local work, not a
security guarantee against malicious same-UID processes.

Retain every measured tick and client receipt in authoritative distributions
and the existing verdict. No CPU subtraction, favorable cell repetition, tick
deletion, latency-based selection or poll shifting. Report each cell and overall
root/aggregate CPU, burst/10 s/60 s maxima, operation counts/durations, overlap
counts, completeness and ordinary/background/unknown covariates. A valid panel
may FAIL. Its scope is the declared co-running environment, with no causal claim.

The sensitivity subset keeps the old complete-coverage/churn rules and further
requires background plus supervision <=20% in *every* overlapping interval.
Exclude operations' full spans expanded 1 s on either side. Apply the exclusion
and complete coverage to entire original consecutive-client receipt gaps;
never bridge omitted receipts. Empty subsets have count zero and null statistics.
The subset cannot change FAIL to PASS or establish absence of contention.

## Current-root blocker and future boundary

Read-only inspection on 2026-10-27 identified root PID 7425, UID 501,
birth `1793116688948465000` ns, module `tools.sandbox.supervisor`, actual cwd
`/Users/josenunez/Projects/ml/snake-dqn-train01-07-broker.noindex`, state-dir
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/sandbox-a-resident/train01-attempt-07/state`.
Bind that broker's source, including its generated sitecustomize, rather than
this served worktree. The candidate sealed registry is a declaration only.

Darwin's installed sys/event.h states NOTE_FORK does not deliver the child PID
through kevent, and NOTE_TRACK/NOTE_CHILD are unsupported. NOTE_EXIT does not
provide final CPU. proc_pid_rusage is available for live processes/zombies;
the existing parent immediately reaps its children via subprocess.run, so a
panel-side query races reap and misses between-sample children. Parent cumulative
child CPU cannot authenticate those children or separate concurrent operations.
The supervisor keeps operation/call records in memory until its final receipt,
without the required process/final-CPU/payload linkage. A longer observation
cannot repair these missing records.

No qualified-source edit is inherently required for **future** sandbox roots.
A future adapter can own process-local subprocess proxies for both qualified
supervisor and cli_guard helper paths, a pinned CLI entry wrapper covering its
descendants, wait4-aware poll AND wait paths, exact operation/template and actual
application-payload measurement, ongoing sampler reconciliation and durable
live journals. The new sandbox accounting primitives are explicitly
NOT_ADMISSIBLE: they are not integrated into future dispatch and do not cover
all those paths. Do not claim otherwise or activate them through an environment
variable. Complete, pin and independently test that future adapter before
declaring capability. No code under research/colab_cli was changed.

This work cannot make **the existing** attempt-07 root admissible without an
already-qualified observer or prohibited live attach/instrumentation/restart.
The implementation therefore refuses live supervision before output creation,
lease acquisition or game imports, even if a caller supplies a capability flag
or a qualification filename. No supervisor/job/cadence/heartbeat is modified.
The requested live qualification and full panel remain blocked, not accepted.

## Offline proof implementation and remaining boundary

The versioned offline verifier now separates semantic identity proof,
exact CPU accounting, ordinary load predicates and timing reports:

- `supervision_identity.py` freezes boot/root/source/assignment identities,
  revalidates complete observation and root-phase coverage, verifies witnessed
  launch brackets and exec/exit chains, regenerates the qualified compact
  request grammar, and verifies independently captured payload lengths/digests.
  Positive tests capture the actual unchanged qualified CLI methods with their
  launch boundary intercepted, without any provider request.
- `supervision_cpu.py` reconciles exclusive samples with final self/subtree
  counters. Inclusive finals require witnessed descendant reaping/inclusion
  records. Integer or rational nanosecond numerators preserve exact Mach tick
  conversion; rounded/ambiguous counters are refused. CPU amounts are conserved
  once, with read brackets charged in full to every intersecting interval/window.
  Online prefixes enforce current ceilings; final replay requires exits and the
  full 60-second pre-exposure stream. Root bookkeeping and concurrent lanes share
  the same ledger, with no debt reset.
- `remote_supervision.py` cross-binds exact process and operation inventories,
  root aliases to OS creation keys, parent/ownership/launch/exit spans, final
  counters and inclusion witnesses. It replays both ordinary >=20% paths and
  the original background ceiling. An independent supervision interval row must
  carry exact delta numerators matching the ledger; rounded ps deltas cannot
  silently receive an exception. Raw point percentages remain covariates.
- `supervision_report.py` retains every timing row, reports complete per-root
  and aggregate covariates and poll overlap, and applies the conservative
  operation-expanded sensitivity subset, including entire client-gap spans.
- `supervision_io.py` verifies immutable triplets and intent/source/ruling/
  registry bindings, independently recomputes the result, and rejects changed
  summaries. The source-bound panel reducer remains closed because no complete
  live observer exists.

A matching synthetic transcript can produce **OFFLINE_VALID**. Its
`live_capture_capability` remains **NOT_ADMISSIBLE**. Passing offline cases does
not establish that a collector captured every real phase, hidden response,
transient helper or final counter. The actual collector, including complete
future sandbox dispatch integration, is **not implemented or qualified** here.
The standalone sandbox wait4 primitive truthfully reports descendant-inclusive
usage and a launch bracket, not a complete OS birth/exec witness. A future
waitid/WNOWAIT plus zombie proc_pid_rusage bridge must preserve native Mach
counter units; a floating microsecond wait4 value cannot be relabelled an exact
nanosecond counter. No fallback sampling or metadata flag opens admission.
The collector must bind its native counter API, raw counter kind, captured Mach
timebase numerator/denominator and exact conversion provenance into the immutable
intent/evidence. Declaring a denominator or `provenance_complete` in a synthetic
ledger does not establish that provenance for a real process.

The source-bound verification receipt must retain this distinction. The ruling's
live §6 item 5 remains unexecuted and unavailable for the current root. No
qualification receipt or full panel PASS can be created by these offline tests.

## Prepared commands — do not run either observation here

The 10-minute command and full panel command below deliberately fail closed for
the currently declared root. They are reviewable target commands, **not ready
dispatches**. Replace the registry only after a future fully instrumented root
and pinned observer have been independently qualified; implementation is still
needed to activate the live capture path and verify its host-only receipt.

```bash
/Users/josenunez/Projects/ml/snake-dqn-served/research/bl_served_search/run_tick_probe.sh --classified-host-load --remote-supervision-registry /Users/josenunez/Projects/ml/snake-dqn-artifacts/served-tick-20261026/supervision-registry-06.json --host-only-seconds 600 --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/served-tick-20261026/host-supervised-06

/Users/josenunez/Projects/ml/snake-dqn-served/research/bl_served_search/run_tick_probe.sh --classified-host-load --remote-supervision-registry /Users/josenunez/Projects/ml/snake-dqn-artifacts/served-tick-20261026/supervision-registry-06.json --qualification-receipt /Users/josenunez/Projects/ml/snake-dqn-artifacts/served-tick-20261026/host-supervised-06/report.json --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/served-tick-20261026/panel-supervised-06
```

The existing launcher provides nice 15, CPU, five thread caps of 1 and caffeinate.
Preserve fixed manifest/order/seeds/exposure/parity, power/thermal/lease/heartbeat
and timing bars. Host-only evidence must precede any fresh full panel; the source
bound offline receipt alone never admits it. Strict gates/serving qualification
must explicitly adopt applicable changes; release still requires the owner.
