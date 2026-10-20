# T34 Path B v2 paced retention — prospective development registration

DRAFT_PENDING_OPERATOR_RATIFICATION

Pre-data FIX-FIRST rulings (2026-10-20) are incorporated from
[the operator specification](../../docs/research/specs/2026-10-20_t34_retention_prereg_fixes.md).
They supersede the earlier draft precedence and execution ordering below.

Written 2026-10-20 before any retention efficacy row. The owner delegated the
ratification decision to Claude; only the operator ratifies after review. This
draft, its committed banks and its OC simulation authorize no pre-check,
efficacy, strict-gate run, training, cloud spending or serving change.

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
   no extra request or backend. Wait time counts toward both resource caps.
   Descriptive deadline-loss reference only, never independently advances Claim B.

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

## Banks, opponents and order

`RETENTION_BANK_V2.json` freezes the exact row order, roster slots and seeds:
32 paired terminal H5000 worlds **per mix**, scripted/frozen/mixed; 16 complete
paired continuous-respawn mixed-Watch6 H5000 sessions; one disjoint timing seed.
Each group has its own namespace. All 113 seeds are unique and disjoint from
prior visible banks, including strict banks. All three arms share each world's
seed, hero slot 0 and ordered opponents. Execute terminal rows in committed
order, then product rows; within every row execute bare, paced, unpaced.

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
units are the 16 **complete paired sessions**. No five-training-seed inference.

For each terminal metric, form candidate-minus-bare paired differences within
mix. Pooled effect is the equal mean of the three mix means. Its variance is
`sum(s_m^2/n_m)/9`, with independent-stratum Welch–Satterthwaite degrees of
freedom. Zero spread gives a bound equal to the mean. Per-mix and session bounds
use Student t with n−1 df. Do not pair arbitrary indices across disjoint mixes.

| Bar | Metric and denominator | Frozen rule |
|---|---|---|
| Skill mass | Mean post-transition hero logical length over all 5000 frames, dead frames zero; equal-weight mix means of paired deltas | pooled point >= +40 and one-sided LB90 > 0 |
| Mass multiplicity | Two active-arm superiority hypotheses vs the shared bare comparator, one look | Holm family alpha .10; paced must reject in addition to raw LB90 > 0 |
| Survival guard A pooled | Alive frames / 5000 per terminal world; equal-weight pooled paired differences | one-sided 95% lower bound strictly > −.05 |
| Survival guard A each mix | Same alive fraction, 32 paired differences per mix | every one-sided 95% lower bound strictly > −.075 |
| Survival absolute floor | Mean candidate alive-frame fraction in each mix | >= .30 in every mix |
| Survival guard B | Scripted indicator of zero hero deaths through H5000; paired mean over 32 worlds | point gain >= .03; .03 is not rounded to zero |
| Delivered boost | Total executed hero boost decisions / total executed hero decisions across all 96 terminal worlds | inclusive [.005, .10]; dead frames excluded; fallbacks included |
| Watch6 persistence | Each session's hero deaths * 1000 / executed hero decisions; candidate-minus-bare rate per paired session | mean rate lower, two-sided paired 95% interval upper < 0; 20% fewer is a target, not an extra bar |
| Watch6 growth | Per-session (candidate full-H5000 mass − bare full-H5000 mass) / bare full-H5000 mass | two-sided paired 95% interval lower >= −.05; zero bare denominator is invalid evidence |
| Processing | Real Watch6 ON/OFF at both 12/24 fps, fixed 5000 scored ticks after 200 warmup ticks per mode/rate | zero paced overruns (full tick >= period is overrun); paced update p95 <=11.688 ms at 24 fps; report p95 at 12 fps |
| Delivered cadence | Scored ticks / measured paced elapsed seconds, ON compared with OFF at matching rate | ON >= .95 * OFF at each rate |

Holm orders the two one-sided pooled mass p-values (paced wins stable equal-p
ordering), compares to .05 then .10; report raw and adjusted p-values. Unpaced
is in this family despite being descriptive, as required by Amendment 2.
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
- **KILL (drop Claim B):** paced arm fails any skill bar while the unpaced arm passes them (deadline loss), or
  both arms fail the skill bars, or any unconditional fault/timing overrun; teacher labels then wait on Round 0.
- **INCONCLUSIVE:** missing rows, parity/identity failure, cap reached, LB90 within ±10 of 0 with point estimate
  ≥ +40, or product bars pass but one skill bar misses by less than its interval — report, no extension, owner
  decides whether to fund a second fixed block.

Executable precedence: unconditional faults KILL. The quiet-host timing gate
runs first, before the reused T3 pre-check or any efficacy world. At both rates
require zero overruns and the cadence bar; require update p95 <=11.688 ms only
at 24 fps. A failed registered timing probe stops with KILL-timing and no
pre-check/efficacy work. Sustained contention or AC/lid/thermal interruption
beyond the registered pause limit is INCONCLUSIVE. Missing or invalid evidence
is INCONCLUSIVE. At the sole complete valid look, failed timing is KILL;
failed paced skill decides KILL (deadline loss if unpaced passes, otherwise
both fail), subject only to the uncertainty exceptions below; all paced
skill/product bars pass -> GO; product bars not established -> INCONCLUSIVE.

The near-zero exception (`abs(raw_mass_LB90) <= 10`, point >=40) applies only
when mass is the sole failing category and product passes. Raw/Holm
significance and the +40 point threshold are one mass category. It cannot
mask survival, boost, absolute floor, product or timing failures, and cannot
turn a complete pass into INCONCLUSIVE. The separate one-skill-bar interval
overlap exception applies only with product passing. Every efficacy tick >=
41.7 ms is diagnostic: its bare fallback remains counted in assigned-arm
efficacy. The zero-overrun bar applies only to registered timing probes.

P7 unconditional faults: non-finite neural Q (and loss/gradient if applicable),
illegal action, or >90% of one delivered action in any rolling 100,000 delivered
hero decisions; kill immediately. This study has no fit, so fit-drift tripwires
are inapplicable. Boost outside its delivered range fails its skill bar; it is
not the obsolete >2% training tripwire. Any cap expiry without a trustworthy
unconditional fault is INCONCLUSIVE, never efficacy failure or success.

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
32/mix, separately from probability of meeting the point >=40 bar and the
conservative near-zero decision exception. Full joint skill/product power is
not inferred from mass power alone. Extend development-only simulations with
T3-shaped Path B scenarios: scripted no-death gains +.10/+.25 crossed with
mass gains +40/+60/+92; report joint GO probability, conditional on timing and
boost compliance and the explicit unchanged-product-growth planning assumption, and disclose proxy assumptions. Report 16-session Watch6
paired deaths/1k power at 20% and 40% reduction from T15/T13b B8 Watch baselines.
Keep N=32/mix even if joint GO power at T3-like effects is <.5: flag it for the
operator; implementation may not change N.

P7 §5's cost comparison is prospective: fixed N consumes 96 paired terminal
worlds and 16 paired product sessions, every arm observed. Report illustrative
sequential expected cost at +40 using development variance and explicit spending
and stop assumptions; this does not authorize a sequential design. Amendment 2
fixes this study at N=32/mix and one look; do not convert after commencement.
Minimum pacing exposure if all heroes survive: 112 rows * 2 paced arms *
5000/24 = 12.963 h, plus four timing probes (0.347 h + 0.014 h warmup), setup and
unpaced work; 32 h is a hard cap, not a runtime promise. Terminal early deaths
shorten pacing only; their dead remainder remains zero-scored. Descriptive
unpaced search cost can dominate and is included in both caps.

## Admission, resources and evidence lifecycle

Mac CPU only, $0, first of **32 h elapsed / 48 CPU-h** wins. Main simulation plus
one assistance worker <=2 CPU threads; every numerical pool capped at one
(OMP/MKL/OpenBLAS/VECLIB/NumExpr/Numba/Torch). One process, no parallel worlds.
Charge setup, source checks, waiting, failed work and exports; OC has its own
bounded resumable simulation budget and is not retention gameplay. Use AC,
open lid, available and acceptable thermal telemetry. No service or tmux action.

The runner measures quiet-host admission and continuously rechecks AC, lid and
thermal telemetry during timing, pre-check and efficacy at most every 30 seconds.
It measures one-minute load excess above its own two-worker allowance; excess
>3.0 indicates contention. Any other process above 50% CPU for more than 60
seconds (PID and command listed) or a known heavy Phase 0/research runner also
indicates contention. The exact process patterns and policy keys are frozen in
RETENTION_PROFILE_V2.json and copied unchanged to registration.load_profile.
On contention or a power/lid/thermal change pause; continuous nonquiet conditions
for 120 seconds stop INCONCLUSIVE. Paused time counts against resource caps.
Unavailable telemetry cannot pass. No hard-coded quiet/Phase-0-completed/zero-job
attestation establishes admission. Policy changes require a prospective sibling
registration before affected data.

Dependency: the separately authorized reused T3 pre-check must cover original
b8_s0/b8_s3 x env0..31 at 24 fps, 64 trajectories / 32 physical seeds, original
checkpoint/roster/ceiling-receipt identities, and reach >=50/64 paced survivors.
Its source_commit and full source hashes must match the frozen retention source.
It decides no Claim B result. This implementation task does not run it.

Sequence: commit and publish code; freeze source_commit and inventory; retain
the committed audited banks; run bounded development-only OC; build/validate
and commit the draft with every evidence binding except the final trace receipt.
Operator produces/reviews the final-source trace receipt, then writes the
create-only ratified JSON and commits/pushes it BEFORE timing or the T3 pre-check.
The Markdown remains a draft; only the operator JSON ratifies it. The runner
requires exact ratified JSON bytes in HEAD and current HEAD published to origin.
Run the quiet-host ON/OFF timing gate at 24 and 12 fps first (~.36 h). Only its
PASS receipt admits the separately authorized T3 pre-check; only matching PASS
timing and >=50/64 pre-check evidence admits efficacy in a fresh artifact root.
Follow [README operator commands](README_V2.md#retention-registration-package).

The create-only decision.json written by the runner at the end of the single
look is the only verdict. Timing failure writes the terminal decision before
efficacy. Interrupted work retains an INCONCLUSIVE decision and fault evidence.
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
