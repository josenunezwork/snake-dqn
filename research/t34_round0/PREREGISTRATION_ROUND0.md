# T34 Round 0 — deterministic teacher premise test

**DRAFT_PENDING_OPERATOR_RATIFICATION — 2026-10-20, before Round 0 data.**
This committed draft authorizes preparation and the excluded A6 Mac preflight only.
An operator must create the separate, immutable ratification receipt after A5 and
review the computed manifest forecast before dispatch. No scientific arm is run by
this implementation task. Do not edit this document after ratification; use a
prospective sibling amendment before affected data.

## Authority and question

Binding, in order: [success contract](../../docs/research/final_run_20261019/SUCCESS_CONTRACT.md),
[design v2 including Amendment 2](../../docs/research/final_run_20261019/DESIGN_V2_SHADOW.md),
[P7 statistics](../../docs/research/final_run_20261019/P7_STATISTICS.md), and
unsuperseded [FINAL_PLAN §8](../../docs/research/final_run_20261019/FINAL_PLAN.md).
Implementation request: [Sol spec](../../docs/research/specs/2026-10-20_t34_round0_prereg.md).

Does the deterministic offline search teacher, acting over released bare B8 s3,
clear its premise bar sufficiently to justify Claim A label rounds? This is Tier 1
development evidence, conditional on this fixed B8 checkpoint; there is no training
replicate population. GO permits planning label rounds, never fitting, dispatching
another study, a strict pass, serving qualification, or release by itself.

Comparator: released **bare B8 s3**, SHA-256
`3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`.
All bars and guards compare to this checkpoint. Phase 0 runs independently under
its own +80/LB90>+40 rule; it is a separate cross-check, never substituted, pooled,
relabelled, or interrupted here. A failed Round 0 is reported as **this teacher
failed its registered bar**.

## Fixed controllers and world bank

Arms: `bare` (greedy B8 s3 with its normal advisory-mask/fallback semantics and no
serving veto) and `teacher` (the same checkpoint plus the existing deterministic
`research.t34_exit_b8.teacher_v2` kernel, beta=1, H=300, shadow depth D=300,
`bisection-radius3-prepared-cache`). On shadow doom it searches bisection with
radius-three refinement and schedules the latest discovered horizon rescue. A
scheduled action executes only while the shadow is confirmed; intervention
invalidates and reforks it. All-die delay choices never execute. No rewinds,
length-trigger substitution, stride-10 rule, stochastic rollouts, tower, training,
or student selection. Offline teacher: no delivery deadlines or deadline fallbacks.
The job's operational wall cap does not truncate successful controller searches;
expiry makes the study incomplete. Search may extend beyond H5000 on private
clones; only delivered H5000 decisions are scored.

Exactly **240 paired terminal-hero H5000 worlds**, slot 0, six snakes, mechanics v1:
96 scripted, 72 frozen, 72 mixed. Each pair uses identical seed, roster, seats,
profile, and RNG stream identifiers; both arms execute in the **same runtime and
process**. A pair cannot be assembled from different hosts or attempts.

`BANK_ROUND0.json` is the materialized authority. Namespace root
`t34-round0-premise-20261020/v1`; each mix has its own suffix and disjoint seeds.
Derivation is the first four SHA-256 bytes of `{namespace}/world/{index}`, big
endian. `bank.py` uses the historical strict roster materializer for scripted and
mixed patterns; checkpoint slots rotate FRP3 s12 and champion_a5, each with v8,
matching the FINAL_PLAN teacher pool. Explicit ordered opponent specifications,
veto lambda 8, and profiles travel with every row; all checkpoint identities come
from the existing public release `s1-inputs-20261019` and its hash-bound catalog.

`SEED_AUDIT_ROUND0.json` freezes the Mac audit of all 240 selection seeds plus two
excluded fixtures. Coverage: research/, docs/, src/, scripts/ where present,
sibling artifact text history, concrete FRP training exclusions, and specifically
the Path B retention bank (whose file hash is recorded). Hidden and ignored text
files are included. Opaque binaries and unrecorded external runs remain coverage
limits. Any hit rejects the **whole allocation**; no world replacement, silent
reroll, post-data namespace change, or strict-bank reuse.

The two excluded fixtures have separate namespaces
`t34-round0-preflight-20261020/v1/{scripted,mixed}` and H500. They never enter any
estimate, fit, or controller selection. The real paired entrypoint runs these two
fixtures only, on Mac CPU with one native math thread. No campaign arm runs on Mac
as part of this task.

## Endpoints, inference, and one look

Primary efficacy: paired **full-horizon logical-mass integral**, dead frames zero;
each arm's mass sum divided by 5000. Alive-frame fraction and no-death survival are
distinct endpoints. Terminal death ends credit; the hero never earns respawn
credit. Boost share is boosted action decisions / all delivered hero decisions,
summed over all mixes, rather than the mean of per-world ratios.

Independent unit = paired world. Means are equal-weight across mixes even though
N differs. For paired deltas d, pooled G=(mean(d_scripted)+mean(d_frozen)+
mean(d_mixed))/3; variance=sum(s_mix²/n_mix)/9. Welch-Satterthwaite df uses the
three independent strata. LB90/UB90 are one-sided 90% t bounds (two-sided 80%
interval endpoints). Per-mix bounds use paired t statistics with n_mix−1 df.
Survival guard A uses one-sided 95% lower bounds, per the survival-band amendment.
Guard B is the scripted no-death point gain. Exactly one active controller and one
look: Holm over this singleton changes nothing, family alpha .10 for efficacy;
the guard intersection has its own .05 bounds. No interim outcome peeks, optional
stopping, extension, extra arm, replacement, or checkpoint search.

The pre-data [operating-characteristic note](OPERATING_CHARACTERISTICS.md) and
`OC_ROUND0.json` compare fixed N with an explicitly costed sequential alternative
at +40 using historical development variance and sensitivity models. Neither is
Round 0 efficacy data. This registration retains the spec's complete 240-world
premise measurement; the expected-cost comparison and any limitation are reported
before ratification. Do not convert this fixed design after data start.

## Ordered decisions

`INCONCLUSIVE` for absent, duplicate, unexpected, incomplete, non-finite,
unreconciled, unbound, cross-runtime-paired, cap-expired, or divergent evidence.
No partial scientific outcome is inferred from unfinished chunks. Preserve all
evidence; no automatic job retry or seed replacement.

An unconditional controller fault (non-finite Q, illegal action, or >90% one-action
share in any rolling 100,000 delivered-decision window) stops that arm and yields
`STOP`, with the exact fault retained; this fault takes precedence over missing
remaining rows. Fit-only tripwires do not apply because there is no fit.

On complete valid evidence, `GO` requires **all** of:

| Clause | Required value |
|---|---|
| Teacher scripted no-death survival | ≥ 0.66 |
| Paired scripted no-death gain vs bare B8 s3 | ≥ +0.10 |
| Equal-mix pooled mass gain | ≥ +40 |
| Efficacy LB90 | > 0 |
| Delivered teacher boost share | [0.005, 0.10], inclusive |
| Survival guard A pooled alive-frame delta LB95 | > −0.05 |
| Survival guard A each mix alive-frame delta LB95 | > −0.075 |
| Survival guard A absolute alive-frame fraction each mix | ≥ 0.30 |
| Survival guard B scripted no-death point gain | ≥ +0.03 (subsumed by +0.10) |

`STOP` if a complete-data point bar fails (scripted absolute/gain, pooled mass,
boost interval, absolute alive fraction, guard B), or a survival guard's upper
95% bound is at/below its NI boundary. Otherwise a failed mass or survival lower
bound with compatible point bars is `INCONCLUSIVE`. Exact floating-point thresholds
are applied without display rounding. No second amendment to rescue a failed
survival bar. STOP closes this teacher as tested; INCONCLUSIVE records insufficient
evidence and authorizes no extension; GO buys only the next planning decision.

## Platforms, identity, exports, and replay

Production = Colab L4 only, fp32 inference; no automatic GPU substitution. Both
arms of each pair stay on one runtime. `research/colab_cli` is the only allocator;
its qualification record is verified against current and pinned commit bytes.
No Drive, direct Colab job commands, legacy queue, auth, or live operations in this
implementation. Mac strict gate and serving qualification remain separate.

Ratification freezes commit, conservative tracked executable source inventory,
these rules, bank, audit, and A5 bytes. The runtime refuses drift before building a
world. Input checkpoints/public archives are SHA-256 checked; output directories
and receipts are create-only. Per pair export seed/row/roster, same runtime identity,
all six snakes' delivered action traces, RNG stream identifiers, physical-state
hashes at initial/every 250/final frames, metrics, and trace hashes. Search labels
are stripped: **no Round 0 output is admitted to fitting**. Telemetry is operational
progress only, about every two minutes; do not inspect outcome files until all N.

Before the single analysis, A2-style Mac replay regenerates a predetermined
stratified 10% sample: first 10 scripted, first 7 frozen, first 7 mixed bank rows,
both arms (24 worlds, 48 traces). The choice is frozen without outcomes. Report
every hash/outcome divergence; any unexplained divergence is INCONCLUSIVE, never
drop or replace the world. This trace replay cross-check is distinct from proving
cross-platform greedy-action identity or a Mac strict gate.

## Chunk sizing and spend cap

No invented A5 rate: operator supplies a hash-bound `t34-round0-a5/v1` receipt,
measured **paired** worlds/hour for the real entrypoint, active row-steps/s ≥3000,
measured exported bytes/pair, and successful two-concurrent-lane rehearsal evidence.
Lane 3 stays closed until that rehearsal; all other account assignments (including
legacy notebook workers) count against the cap of three and need reviewed positive
rates. No allocation retry, automatic job retry, credit purchase, or new authority
is inferred from a CLI exit code.

Let r be A5 paired worlds/hour, s measured setup/export/cleanup minutes. Chunk size
starts at floor(2.5r), limited to remaining N and reduced if needed so s+60n/r≤180.
At most three lanes; serial chunks on each lane, fresh chunk/attempt paths. Each
chunk has timeout 360 minutes; first wall or unit cap wins. At **2.0 units/hour**,
360 minutes costs **12.0 units/chunk**, plus **2.4 units/chunk (20%) reserve**:
manifest unit cap **14.4**, reserve_units **2.4**. For K=ceil(240/n) chunks,
worst reserved envelope **14.4K units**, expected priced amount
2*(240/r+K*s/60), with the reserve and worst case shown separately in PLAN.json.
No supplied measurement means no dispatchable manifests; the historical ~16-unit
estimate is not a new fixed budget. Operator reviews the computed whole-plan
forecast and account balance before dispatch. Setup, idle, failed attempts and
exports are charged. Reserve is held, never an automatic retry budget.

Use bounded immutable exports per chunk; measured export-size projection must fit
the qualified limit. Require supervisor terminal receipts and verified disappearance
of each assignment before admitting completed chunk evidence. Read-only validation
does not prove allocation, execution, completed artifacts, cleanup, or release.

## Implementation restrictions

All preparation/tests/preflight here run on the Mac, with ≤2 CPU threads total
while Phase 0 runs. No live Colab/tmux operations; no campaign arm runs. Preserve
all active workers, released checkpoints, serving pins, release-monitoring logs,
and existing hash-bound documents. Commit/push coherent steps and merge to main.
Exact ratification, validation, dispatch, replay, and one-look analysis commands
are in [README.md](README.md).
