# T34 Path B v2 paced retention — prospective development registration

DRAFT_PENDING_OPERATOR_RATIFICATION

The pre-data FIX-FIRST rulings, including the second review, the unpaced
Colab ruling, the final contention and admission review, and the fourth
timing-cap review, are incorporated
from [the first operator specification](../../docs/research/specs/2026-10-20_t34_retention_prereg_fixes.md)
and [the second operator specification](../../docs/research/specs/2026-10-20_t34_retention_fixes2.md),
with the [unpaced-arm ruling](../../docs/research/specs/2026-10-20_t34_unpaced_arm_colab.md)
and [FIX-FIRST #4](../../docs/research/specs/2026-10-20_t34_retention_fixes4.md).
They supersede the earlier draft precedence and execution ordering below.

Written 2026-10-20 before any retention efficacy row. The owner delegated the
ratification decision to Claude; only the operator ratifies after review. This
draft, its committed banks and its OC simulation authorize no pre-check,
efficacy, strict-gate run, training, cloud spending or serving change.

## Amendment 5 addendum — FIX-FIRST #4 timing cap and contention precedence

The [FIX-FIRST #4 ruling](../../docs/research/specs/2026-10-20_t34_retention_fixes4.md)
amends this unauthorized draft before data. It replaces only Amendment 4's
timing-cap, timing-contention, and idle-baseline rules; its arm scope,
admission requirements, and pre-check/study rules remain in force.

The registered phase map is `timing: 3600` seconds, `precheck: 28800` seconds,
and `study: 144000` seconds. With the frozen 2% allowance, the corresponding
phase contention budgets are **72 seconds**, **576 seconds**, and **2880
seconds**. The timing gate has a hard **one-hour elapsed** cap. Its measured
elapsed and CPU time also debit the enclosing 40-hour elapsed / 60-CPU-hour
study caps; the pre-check remains separately capped at 8 elapsed hours and
does not debit either study cap.

For timing, zero contended **scored probe ticks** is an independent prerequisite
that precedes all timing bars. Any contended scored probe tick yields the exact
verdict **INCONCLUSIVE-retry**, produces no efficacy row, and admits no
pre-check or study work. The operator may manually rerun that timing gate only
on a quiet host and in a fresh output directory. Each invocation preserves its
own create-only `timing_attempt.json`, including admission-refusal diagnostics;
there is no automatic retry. Warmup contention is recorded separately from
scored-tick contention and is assessed only through the timing phase's existing
72-second, 2% contention budget.

At admission, the idle baseline is
`min(first measured load_1m, (os.cpu_count() or 1) / 4)`. The record retains
the CPU count, that absolute ceiling, the capped baseline, and the resulting
load threshold. This prevents steady background load from raising the baseline.
Unconditional controller faults in the Mac `bare` or `paced` arms remain KILL
conditions; this contention retry rule does not weaken them.

## Amendment 4 addendum — pre-data contention accounting and arm-scoped faults

The [final FIX-FIRST #3 ruling](../../docs/research/specs/2026-10-20_t34_retention_fixes3.md)
amends this draft before data. Its purpose is to make host-contention evidence
prospective and phase-local, while preserving the descriptive status of the
separately transported `unpaced` arm.

For each Mac phase, the contention allowance is exactly 2% of that phase's
**registered elapsed-time cap**, checked only when that phase ends. The former
statement here that timing used the 40-hour study cap is superseded by Amendment
5: timing uses its 1-hour cap and 72-second phase budget. The reused-T3
pre-check uses its separate 8-hour cap and 576-second budget, while the
retention study uses its 40-hour cap and 2880-second budget. Apart from the
scored-tick timing rule in Amendment 5, a contended sample or an interim
fraction above the allowance does not stop a row or phase.

The HostGuard samples every 5 seconds; every sample classified as contended
adds 5 seconds to that phase's measured contention total. Load-average excess
and a process-pattern CPU condition count only after the applicable excess has
persisted continuously for more than 60 seconds. The phase evidence records the
configured 3.0 load allowance, the measured idle baseline, and the CPU count
used to interpret that baseline. The baseline is measured at admission rather
than assumed from an earlier host observation.

Before each timing, pre-check, or study phase, the host must remain quiet for a
registered 120-second admission window, including raw load and other-process
CPU checks. If it does not, the runner refuses that phase with a nonzero exit;
it writes no `decision.json` and no terminal phase record, so the operator may
retry admission later. This is an admission refusal, not an INCONCLUSIVE or a
scientific result.

In this registration, “any unconditional fault -> KILL” applies only to the
Mac `bare` and `paced` arms. The `unpaced` Colab arm is a descriptive,
separately transported reference with no decision authority: its missing or
failed rows remain missing/failed descriptive rows and never change a Mac GO,
KILL, or INCONCLUSIVE verdict. This boundary prevents an export or transport
failure in a non-decision arm from deciding Claim B.

## Amendment 2 addendum — pre-data resize and cap separation

[The operator resize ruling](../../docs/research/specs/2026-10-20_t34_retention_resize.md)
supersedes the prior draft sample sizes and resource caps before any retention
or reused-T3 pre-check data exist: terminal worlds **32 -> 48 per mix** (144
paired), complete mixed Watch6 product sessions **16 -> 32**, and study caps
**32 -> 40 h elapsed / 48 -> 60 CPU-h**, $0, stopping at the first study cap as
INCONCLUSIVE. The reused T3 pre-check is development-only, has its own **8 h
elapsed** cap, and is never debited against the study's 40 h elapsed or 60 CPU-h
caps. The timing gate is part of the registered study and remains debited to its
study caps.
The preceding N32/16 development OC found T3-like (+92 mass, +.25 scripted
no-death) joint GO probability only .26630/.31715 at 20% Watch death reduction
(T15/T13b), and 16-session product power .46160/.54845. This prospective resize
responds to that power review; it does not use retention or pre-check outcomes.

Keep all 113 original seeds, rows, rosters and the timing seed. Append 48
terminal seeds (16 per mix) and 16 product seeds using the frozen allocator's
next namespace indices; audit all 64 additions together against all visible
prior banks, including the retained retention bank and Round 0 draft bank.
The refreshed audit retains the original audit evidence and hashes and records
the additional zero-hit scan; original draft bytes remain in Git history.
Re-run development-only OC at N48/32. Protected false-pass must remain <=.06.
If T3-like joint GO power at 20% Watch death reduction remains <.5, report it;
**do not change N again**. Thresholds, arm identities, one-look inference,
ratification requirements and scope are unchanged. This package remains an
unauthorized draft; no efficacy, timing or pre-check gameplay is run here.

## Amendment 3 addendum — pre-data descriptive unpaced Colab arm

Before any retention efficacy row, the operator moved only the `unpaced` arm
to the qualified `research/colab_cli` transport. It remains a **descriptive
deadline-loss reference**, not a Claim B test: it has no mass hypothesis, no
decision authority, and no role in fault precedence, multiplicity, GO, KILL or
INCONCLUSIVE. Missing, failed, malformed, or incomplete unpaced exports are
retained and reported as descriptive missing/invalid rows; they never change
the Mac verdict. A valid export with a cross-platform bare-prefix divergence is
retained as a valid descriptive row with its divergence report, not relabelled
missing or invalid.

This prospective addendum authorizes only the local Mac preflight of exactly two
disjoint development worlds at no more than four frames each. It persists no
efficacy metric and cannot dispatch Colab, spend credits, ratify the draft, or
admit a timing/pre-check/efficacy phase. All production collection remains
operator-only after ratification.

The Colab arm consumes the same already registered bank: 144 terminal H5000
worlds (48 each scripted/frozen/mixed) and 32 complete mixed-Watch6 H5000
product sessions. It does not allocate, substitute, retry, or extend a world.
Each create-only world export carries its identity and SHA-256, and the final
descriptive report compares the physical-state hash stream of its shared bare
prefix against the Mac bare record for that world. Prefix tracing is a
world-record diagnostic only; it is not collected as timing-probe overhead.
Cross-platform fp32 argmax ties are expected in roughly one episode per 50; a
reported prefix divergence is descriptive evidence, not an automatic Mac
identity failure or a reason to rerun a world.

The Mac study now runs only `bare` and 24-fps `paced` efficacy records, after
the Mac-only timing gate and the separately capped pre-check. Its registered
40 h elapsed / 60 CPU-h study caps and separate 8 h pre-check cap are unchanged.
The original two-hypothesis Holm family is preserved conservatively without
observing an unpaced result: the paced one-sided pooled mass p-value must be
`<= .05` (equivalently its fixed worst-case Holm tail is `.05`) as well as
meeting the raw LB90 and all other paced bars. No unpaced mass value can relax,
reorder, or otherwise affect that threshold.

The separately priced Colab envelope is **42.533128 units**, including
**7.088855 units** reserve: `362.498247 s/world * 176 / 3600 * 2.0 L4
units/h * 1.20`. Its unreserved projection is 35.444273 units. Cold setup,
exports, idle time, and the manifest timeout must fit the aggregate unit cap
and reserve; the manifest builder enforces that budget. L4 is the default at
2.0 units/h. A CPU runtime is admissible only after its positive measured rate
is recorded and priced; it is never treated as zero-cost. At most three
account-wide Colab assignments may exist, including unrelated notebook or CLI
work. The qualified transport is used as-is; this amendment authorizes no
`research/colab_cli` change.

## Authority and question

Binding in order: [SUCCESS_CONTRACT](../../docs/research/final_run_20261019/SUCCESS_CONTRACT.md),
[DESIGN_V2_SHADOW Amendment 2 §3](../../docs/research/final_run_20261019/DESIGN_V2_SHADOW.md),
[P7_STATISTICS](../../docs/research/final_run_20261019/P7_STATISTICS.md),
[decision advice §4](../../docs/research/final_run_20261019/speed_advice/m2m3_fable_on_astra.md),
and [PROTOCOL_V2](PROTOCOL_V2.md) with [build details](PROTOCOL_V2_BUILD_RULES.md).
Historical M2 and M3 NO-GO results stand. This is a Tier 1 fixed-controller
screen of Claim B, not a training-recipe claim or release qualification.

Question: does the frozen D500 assisted B8 s3 controller retain a material mass
and persistence benefit under actual 24 fps deadlines, with the contract's
survival, Watch6 product and processing bars, against released **bare B8 s3**?
Comparator and hero initializer SHA-256:
`3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`.
No intermediate checkpoint selection, training replicate or replacement seed.
Every gating comparison is against B8 s3; historical champion/FRP3 comparisons
carry no gating authority.

## Fixed arms, information and action rule

1. `bare`: released B8 s3, rowwise fp32 PyTorch CPU, no decision-time veto or
   assistance; paced at 24 fps in terminal and product episodes.
2. `paced`: the same B8 s3 plus the frozen v2 controller, real 24 fps pacing.
3. `unpaced`: the identical assisted controller with the simulation clock held
   between ticks until its sole worker finishes maintenance/search. No rewind;
   no extra request or backend. It runs only as the separately priced Colab
   descriptive arm. It is never part of the Mac study's resource caps or a
   Claim B decision.

D=500 exact greedy shadow, one confirmed advance per consumed real transition;
initial/refork catch-up is background work. Stop a doomed future at hero death.
On doom search earliest-first from first detection, at most eight chronological
branch points per native batch, independent alternative/opponent/v8 carry,
persistent bounded scratch. Alternatives descend in B8 Q, ties lowest index.
Continuation horizon 500, censored at the remaining terminal H5000 horizon.
T3 ranking: keep bare if it survives; otherwise prefer horizon survivors; if all
die prefer longest alive; ties Q, then bare, then action index. Only a fully
evaluated improvement at a still-future frame may be scheduled. Horizon rescues
and all-die delays stay separate; delays are not decisive teacher examples.

Use deployment-available state only. Bind ordered bodies/food, every simulator
RNG stream, opponent carry including v8, frame, session, hero life, checkpoint
and generation. Any mismatch/intervention discards the old shadow and results,
then reforks. Late, stale, illegal, unavailable or unconfirmed results use bare;
no blocking tick or hindsight intervention. One assistance worker, at most one
request, bounded controller history and snapshots as frozen in source inventory.
Isolated RNG; native kernels release the GIL. ONNX and serial LAD/v1 fallback
are excluded. Count every miss/fallback in its assigned arm's efficacy.

The host guard is asynchronous: it samples through a separate lightweight path
or non-blocking between-tick reads and never sleeps in `_collect` or between
ticks. It can pause only between complete rows (worlds or sessions). Before a
pause it drops every pending assistance request; each dropped request is counted
as a fallback in the assigned arm. A row is neither excluded nor retried for
contention.

## Banks, opponents and order

`RETENTION_BANK_V2.json` freezes the exact row order, roster slots and seeds:
48 paired terminal H5000 worlds **per mix**, scripted/frozen/mixed; 32 complete
paired continuous-respawn mixed-Watch6 H5000 sessions; one disjoint timing seed.
Each group has its own namespace. All 177 seeds are unique and disjoint from
prior visible banks, including strict banks. The Mac `bare` and `paced` arms,
and the Colab `unpaced` descriptive arm, share each world's seed, hero slot 0
and ordered opponents. Execute the Mac terminal rows in committed order, then
the Mac product rows. The separately dispatched Colab arm uses that same order
within its immutable shards; it is not a third Mac pass through the bank.

Terminal rosters are materialized by the frozen historical
`bank_engine.planned_rosters(gid._context(1), seeds, mixes)` authority. They are
not replaced by product opponents. The bank records every slot's spec and member
hash. Terminal hero ends at first death; dead remainder contributes zero, with
no respawn credit or additional pacing required after death.

Product and timing use the actual offline T15 GameSession plus exact Grid mirror:
hero B8 s3 (assisted only in active arms), opponents slots 1–5 each fixed
FRP3 s12 SHA-256 `eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723`,
each with independent v8 carry, method
`free-space-veto/v8-space-and-head(lambda=8.0)`. Continuous respawn includes
all hero lives; no life/seat/tick becomes an independent sample. Full-tick
timing includes bridge, snapshot and digest overhead. Profile is pinned in
`RETENTION_PROFILE_V2.json`: `promotion-v2-watch-rect`, deployment YAML
SHA-256 `4146baa3a06102b8afd627b1fba8384e9a2f47aaac4a9bc96292c3eb71715aa5`.
The registry's released checkpoint and every terminal opponent are separately
bound by `{path,sha256}` in the registration.

The allocator is frozen before allocation. Derive the complete candidate union
once, audit research/, docs/, scripts/, src/, all visible artifact text and
concrete historical FRP exclusions. Reject the **whole allocation** on any hit,
including internal duplication. Never silently replace a seed or retry. A
future allocation requires a prospective sibling erratum. Preserve rejected
allocation evidence. `SEED_AUDIT_RETENTION_V2.json` records scan coverage,
limits and hashes; opaque binaries/unrecorded external runs remain a stated
coverage limit. This task commits allocations only and consumes no world.

## Endpoints, denominators and inference

One look at complete fixed N, no outcome peeking, early efficacy stop, automatic
extension, replay of a failed scientific run, replacement bank or source swap.
Paired **worlds** are independent units within disjoint mix strata; product
units are the 32 **complete paired sessions**. No five-training-seed inference.

For each terminal metric, form candidate-minus-bare paired differences within
mix. Pooled effect is the equal mean of the three mix means. Its variance is
`sum(s_m^2/n_m)/9`, with independent-stratum Welch–Satterthwaite degrees of
freedom. Zero spread gives a bound equal to the mean. Per-mix and session bounds
use Student t with n−1 df. Do not pair arbitrary indices across disjoint mixes.

| Bar | Metric and denominator | Frozen rule |
|---|---|---|
| Skill mass | Mean post-transition hero logical length over all 5000 frames, dead frames zero; equal-weight mix means of paired deltas | pooled point >= +40 and one-sided LB90 > 0 |
| Mass multiplicity | Original two-hypothesis family retained prospectively; only paced is decision-bearing | fixed worst-case Holm: paced one-sided p <= .05, plus raw LB90 > 0 |
| Survival guard A pooled | Alive frames / 5000 per terminal world; equal-weight pooled paired differences | one-sided 95% lower bound strictly > −.05 |
| Survival guard A each mix | Same alive fraction, 48 paired differences per mix | every one-sided 95% lower bound strictly > −.075 |
| Survival absolute floor | Mean candidate alive-frame fraction in each mix | >= .30 in every mix |
| Survival guard B | Scripted indicator of zero hero deaths through H5000; paired mean over 48 worlds | point gain >= .03; .03 is not rounded to zero |
| Delivered boost | Total executed hero boost decisions / total executed hero decisions across all 144 terminal worlds | inclusive [.005, .10]; dead frames excluded; fallbacks included |
| Watch6 persistence | Each session's hero deaths * 1000 / executed hero decisions; candidate-minus-bare rate per paired session | mean rate lower, two-sided paired 95% interval upper < 0; 20% fewer is a target, not an extra bar |
| Watch6 growth | Per-session (candidate full-H5000 mass − bare full-H5000 mass) / bare full-H5000 mass | two-sided paired 95% interval lower >= −.05; zero bare denominator is invalid evidence |
| Processing | Real Watch6 ON/OFF at both 12/24 fps, fixed 5000 scored ticks after 200 warmup ticks per mode/rate | zero paced overruns (full tick >= period is overrun); paced update p95 <=11.688 ms at 24 fps; report p95 at 12 fps |
| Delivered cadence | Scored ticks / measured paced elapsed seconds, ON compared with OFF at matching rate | ON >= .95 * OFF at each rate |

The original two-hypothesis Holm family is frozen before data. Because unpaced
is now descriptive and may be absent, apply the conservative worst-case paced
threshold directly: one-sided pooled mass p <= .05 (and report its adjusted
p-value). Do not compute, rank, or use an unpaced p-value in a decision.
The survival intersection-union components use alpha .05 without dividing
across mixes: every component must pass. Survival band v2 margins and floor
are unchanged. The profile binds the original amendment document by SHA-256
and its exact ratification tuple (Decision: ratified; Option: 1; Changes: none). The single final look uses its one-look OBF level .05; no
sequential continuation is introduced. A fixed-controller independent-stratum
Welch bound replaces arbitrary shared-index covariance, per P7.

Report median/p95 full-horizon hero length, deaths at pre-action length >=800,
late deaths at frame >=2000 and their overlap by session; arena mass, kills,
label accuracy and rescue/delay counts are diagnostic. Report update and full-tick
p95/p99, delivered cadence, first-result latency and shadow core share. First-result
p95 <=2 seconds is an engineering target. Record observed long-body tails; an
absence of long bodies is not qualification of that condition. Quiet timings
cannot establish contention tails; historical contended smoke is disclosed only.

Client latency deterioration <=10%, Watch8/12 at 12 fps, Play, visible assisted
hero labelling and production serving qualification remain **separate required
release checks**, unmeasured here. The offline timing adapter qualifies neither
a browser nor a production hook. Development GO claims no release readiness.
Rollback at a separately authorized release disables assistance and pins
`SNAKE_SERVE_CHECKPOINT=b8-s3`; this task changes no serving selector.

## Decision rules (verbatim advice §4)

- **GO (Claim B proceeds to the strict gate):** paced arm meets every contract bar — pooled mass ≥ +40 with
  LB90 > 0, scripted no-death ≥ +0.03, survival guards A/B, boost share [0.5%, 10%], Watch6 deaths/1k lower
  with paired 95% CI excluding 0, zero paced overruns, update p95 ≤ 11.688 ms — with misses/fallbacks counted.
- **KILL (drop Claim B):** paced arm fails any skill bar, or an unconditional
  fault in the Mac `bare` or `paced` arm occurs; teacher labels then wait on
  Round 0. The Colab descriptive arm cannot cause or avoid KILL.
- **INCONCLUSIVE:** missing rows, parity/identity failure, cap reached, LB90 within ±10 of 0 with point estimate
  ≥ +40, or product bars pass but one skill bar misses by less than its interval — report, no extension, owner
  decides whether to fund a second fixed block.

Executable precedence: unconditional faults in the Mac `bare` or `paced` arms
KILL. Before the timing gate, the reused T3 pre-check, and the retention study,
the host must pass the 120-second quiet admission window; refusal to admit exits
nonzero without a terminal record or `decision.json` and may be retried later.
The quiet-host timing gate runs first, before the reused T3 pre-check or any
efficacy world. At both rates require zero overruns and the cadence bar; require
update p95 <=11.688 ms only at 24 fps. Before those timing bars are judged,
**any contended scored probe tick** is exactly **INCONCLUSIVE-retry**: it
produces no efficacy row and permits neither pre-check nor efficacy work. The
operator may rerun manually only on a quiet host in a fresh output directory;
each attempt, including an admission refusal, preserves `timing_attempt.json`.
There is no automatic retry. If no scored probe tick is contended, a failed
registered timing probe stops with KILL-timing and no pre-check/efficacy work.
Warmup contention is separately retained and remains subject to the timing
phase's 72-second 2% budget. At phase end only, a phase's contention above its
registered allowance (72 seconds timing, 576 seconds pre-check, 2880 seconds
study), or an AC/lid/thermal interruption beyond the registered 120-second
pause limit, is INCONCLUSIVE. Missing or invalid Mac evidence is INCONCLUSIVE.
At the sole complete valid look, failed timing is KILL;
failed paced skill decides KILL, subject only to the uncertainty exceptions
below; all paced skill/product bars pass -> GO; product bars not established ->
INCONCLUSIVE. Unpaced Colab evidence has no branch in this precedence.

The near-zero exception (`abs(raw_mass_LB90) <= 10`, point >=40) applies only
when mass is the sole failing category and product passes. Raw/Holm
significance and the +40 point threshold are one mass category. It cannot
mask survival, boost, absolute floor, product or timing failures, and cannot
turn a complete pass into INCONCLUSIVE. The separate one-skill-bar interval
overlap exception applies only with product passing. Every efficacy tick >=
41.7 ms is diagnostic: its bare fallback remains counted in assigned-arm
efficacy. The zero-overrun bar applies only to registered timing probes.

P7 unconditional faults in the Mac `bare` or `paced` arm are non-finite neural
Q (and loss/gradient if applicable), illegal action, or >90% of one delivered
action in any rolling 100,000 delivered hero decisions; kill immediately. This
study has no fit, so fit-drift tripwires are inapplicable. Boost outside its
delivered range fails its skill bar; it is not the obsolete >2% training
tripwire. Any cap expiry without a trustworthy unconditional Mac fault is
INCONCLUSIVE, never efficacy failure or success. Unpaced Colab faults are
reported descriptively and are outside this precedence.

## OC, power and the fixed-N choice

`retention_oc_v2.py` produces `RETENTION_OC_V2.json` before efficacy. Bind the
simulator, extracted development pool, original development files and complete
final executable source inventory by SHA-256. Use T3 dense paired deltas and
historical B8 Watch baselines only, never retention/pre-check outcomes. T3's 64
trajectories share 32 physical seeds across two checkpoints and are scripted;
Watch is mixed continuous respawn, not a terminal frozen/mixed paired pool.
These limitations must be visible; proxy resampling and normal stress scenarios
are planning evidence, not observed retention effects or a power guarantee.

Report no-change pass probability and joint pass with protected pooled −.05
or each one-mix −.075 regression, at mass effects 20,26.8,40,60,80. Every
protected-regression joint rate must be <=1.2*.05=.06, or registration stays
blocked; no margin/alpha/floor retuning. Report +40 mass superiority power at
48/mix, separately from probability of meeting the point >=40 bar and the
conservative near-zero decision exception. Full joint skill/product power is
not inferred from mass power alone. Extend development-only simulations with
T3-shaped Path B scenarios: scripted no-death gains +.10/+.25 crossed with
mass gains +40/+60/+92; report joint GO probability, conditional on timing and
boost compliance and the explicit unchanged-product-growth planning assumption, and disclose proxy assumptions. Report 32-session Watch6
paired deaths/1k power at 20% and 40% reduction from T15/T13b B8 Watch baselines.
Keep N=48/mix even if joint GO power at T3-like effects is <.5: flag it for the
operator; implementation may not change N.

P7 §5's cost comparison is prospective: fixed N consumes 144 paired terminal
worlds and 32 paired product sessions for the Mac `bare` and `paced` records.
The same rows are also exported by the separately capped Colab descriptive arm.
Report illustrative
sequential expected cost at +40 using development variance and explicit spending
and stop assumptions; this does not authorize a sequential design. The pre-data resize amendment
fixes this study at N=48/mix and one look; do not convert after commencement.
Minimum pacing exposure if all heroes survive: 176 rows * 2 Mac arms *
5000/24 = 20.370 h, plus four timing probes (0.347 h + 0.014 h warmup) and Mac
setup; 40 h is a hard cap, not a runtime promise. Terminal early deaths shorten
pacing only; their dead remainder remains zero-scored. Descriptive unpaced
search cost is priced only against the separate Colab unit envelope.

## Admission, resources and evidence lifecycle

Mac CPU only, $0. Timing has a hard **1 h elapsed** cap and its elapsed/CPU
debits also count toward the enclosing study's first-of **40 h elapsed / 60
CPU-h** caps. The pre-check has its own first-of **8 h elapsed** cap and does
not debit the study caps. The Mac runs only the `bare` and `paced` arms. Main
simulation plus one assistance worker uses at most two CPU threads; every
numerical pool is capped at one
(OMP/MKL/OpenBLAS/VECLIB/NumExpr/Numba/Torch). One process, no parallel worlds.
Charge Mac setup, source checks, waiting, failed work and exports to the phase
that incurs them. OC has its own bounded resumable simulation budget and is not
retention gameplay. Use
AC, open lid, available and acceptable thermal telemetry. No service or tmux
action. `RETENTION_PROFILE_V2.json` repeats this as
`timing_max_elapsed_hours: 1`, `precheck_max_elapsed_hours: 8`, and the study's
`max_elapsed_hours: 40` / `max_cpu_hours: 60`.

Admission requires acceptable AC, lid, and thermal telemetry plus 120 continuous
quiet seconds under the registered raw load and other-process CPU checks. From
the first measured load, the runner sets the idle baseline to
`min(load_1m, (os.cpu_count() or 1) / 4)` and records the CPU count, ceiling,
capped baseline, and configured load threshold. If the window cannot complete,
the phase is refused before it begins: the command exits nonzero and creates
neither a terminal phase record nor `decision.json`; the operator can retry
later. Unavailable telemetry cannot pass admission. No hard-coded
quiet/Phase-0-completed/zero-job attestation establishes it.

After admission, HostGuard samples every 5 seconds without blocking a tick. A
load excess above the measured idle baseline plus the registered 3.0 allowance,
and a process-pattern match (`research/`, a runner, or another listed heavy
job) above the registered CPU threshold, are contention only after persisting
continuously for more than 60 seconds. Each contended sample contributes exactly
5 seconds, and the record
retains raw load, PID/command evidence, the threshold, idle baseline, and CPU
count. A row is neither excluded nor retried for a contended sample.

At the end of each phase only, compare its cumulative contended seconds with 2%
of its registered elapsed cap: timing's 3600 seconds permits 72 contended
seconds, the T3 pre-check's 28800 seconds permits 576, and the study's 144000
seconds permits 2880. Exceeding that allowance makes only the affected phase
INCONCLUSIVE; an early contended sample or an interim excess cannot end a phase.
For timing, this phase-end rule includes separately recorded warmup contention;
it does not replace the earlier exact INCONCLUSIVE-retry rule for any contended
scored probe tick. Power, lid, or thermal changes pause only between rows; a
continuous non-acceptable condition for 120 seconds makes the affected phase
INCONCLUSIVE. Paused time counts against the applicable phase cap. The exact
policy keys and thresholds are frozen in `RETENTION_PROFILE_V2.json` and copied
unchanged to `registration.load_profile`; its `max_contended_fraction` is
`0.02`. A policy change requires a prospective sibling registration before
affected data.

Dependency: the separately authorized reused T3 pre-check must cover original
b8_s0/b8_s3 x env0..31 at 24 fps, 64 trajectories / 32 physical seeds, original
checkpoint/roster/ceiling-receipt identities, and reach >=50/64 paced survivors.
Its source_commit and full source hashes must match the frozen retention source.
It decides no Claim B result. Its independent 8 h elapsed cap applies before
the fresh study; it does not consume the timing-plus-efficacy study budget. This
implementation task does not run it.

Sequence: commit and publish the implementation; freeze `source_commit` and its
deterministic actual import closure, including every transitive Python module and
data file read by the registered entry points (`retention_v2`, `analyze_v2`,
`timing_v2`, `host_guard_v2`, `registration_retention_v2`, and
`retention_oc_v2`). Retain the committed audited banks; run the bounded
development-only OC; then build, validate and commit the draft with every
evidence binding except the final trace receipt. The operator produces/reviews
the final-source trace receipt, writes the create-only ratified JSON in the
authoring worktree, commits and publishes it before timing or the T3 pre-check.
The Markdown remains a draft; only the operator JSON ratifies it.

Execution occurs from a fresh, owned worktree detached exactly at the ratified
`source_commit`. The published ratified JSON is passed by absolute path from the
authoring worktree; its generated bytes are not part of the frozen executable
closure. The detached worktree must have a clean index and worktree before
timing, and its `HEAD`, index and worktree must remain unchanged through
analysis. The runner refuses drift or dirtiness. Timing and pre-check receipts are
admissible only with a matching create-only `source_validated.json`, written after
the source monitor closes successfully. A sibling `source_failure.json` invalidates
the receipt even if the tree is later restored. Run the quiet-host ON/OFF timing
gate at 24 and 12 fps first (~.36 h). Only its PASS receipt admits the separately
authorized T3 pre-check; only matching PASS timing and >=50/64 pre-check evidence
admits efficacy in a fresh artifact root. Follow
[README operator commands](README_V2.md#retention-registration-package).

The create-only decision.json written by an admitted runner at the end of the
single look is the only verdict. A contended scored timing tick writes the exact
INCONCLUSIVE-retry decision before timing bars or efficacy; its attempt record
is preserved, and any later manual timing attempt requires a quiet host and a
fresh directory. Other timing failure writes the terminal decision before
efficacy. A failed 120-second quiet admission writes neither a terminal record
nor `decision.json`, but its `timing_attempt.json` diagnostics are preserved.
Interrupted work retains an INCONCLUSIVE decision and fault evidence.
Re-analysis is explicitly descriptive, preserving the saved decision and never
publishing a replacement verdict. No automatic retry or resumption.

Trace receipt and OC receipt must say PASS, bind exact source hashes, and carry
real supporting evidence. No opaque success flag, synthetic pre-check or hash
placeholder authorizes a run. Any executable closure change invalidates bindings
and requires prospective source freeze/reverification before data. Preserve all
create-only receipts, source/checkpoint hashes, paired raw records and terminal
fault/cap evidence; analyze only at the registered look. No automatic retry or
resume of efficacy. GO admits fresh Mac strict-gate preparation only; strict
gate, serving qualification, release, teacher labels and further funding remain
separate operator decisions. A failure means this controller failed its
registered bar, never that B8 is the bare ceiling.
