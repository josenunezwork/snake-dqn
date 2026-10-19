# DRAFT_PENDING_OWNER_REVIEW — S1 mature collection screen (2026-10-18)

**Status: APPROVED_S1_MATURE_20261018.** Approved by Claude (planner) 2026-10-18 ~21:00 PT under the owner authorization quoted below, before any S1 training or evaluation data. Dispatch still requires the independently passed CLI live acceptance (final-artifact throughput), fresh account evidence and a PASS admission plan.

This is a prospective **new** study. It is DRAFT until Claude inserts
`**Status: APPROVED_S1_MATURE_20261018.**` within the first twelve lines while
preserving this DRAFT header; no package artifact, including
`OWNER_AUTHORIZATION.json`, authorizes dispatch before then. It does not amend
Q1. Q1 remains **INCOMPLETE for budget**, with its ledger, A2/A3
post-diagnostic amendments, A5 refusal, and all original artifacts retained.

Owner authorization in chat at approximately 2026-10-18 7:35 PM PT selected:
“2 variants, 60 credits (Recommended)” — “League-collection vs task-aligned
collection, both with the mature schedule; 5 seeds x 8M each on one L4; Mac
eval. Caps 60 credits / 25 GPU-h (48 / 20 usable). Expected ~15-20 credits.”
This registration implements that bounded authorization only.

## Question, endpoint, comparator, and design

Can either mature fine-tuning collection package improve the released bare B8
initializer, and is terminal-hero N1 collection better than static-pool N3
under the same mature schedule? The endpoint is terminal-hero **H5000
full-horizon mass integral**, with dead frames contributing zero, versus bare
B8 SHA-256
`3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`.
There is no serving veto and no serving action in this study.

Both arms start from identical B8 online and target bytes, with fresh Adam,
empty replay and label rings. They are fine-tunes, never exact resumes. Both
fix gamma `.998`, epsilon `.01`, and anchor `.1` from decision zero. They retain
all Q1 invariants: fp32; batch-scale 2; LR `1e-4`; five-step returns; 768 TD +
256 anchor rows/update; one update per 256 ingested transitions; unchanged
masks, reward/mass settings and zero kill scale; qualified actor counts/rings,
counter reconciliation and overshoot bounds; and B8 initializer/collector/
target oracle checks. Changed sources require fresh validation before admission.

The frozen qualified settings are N3=6 actors and N1=3 actors; 128 worlds;
raw rings with gather threads `g3` and queue depth 2; fp32; batch-scale 2; and
`torch.compile` reduce-overhead. Export only the terminal 8M online weights
(`final.pth`), terminal target weights (`target.pth`), counters/provenance/parity
receipts, telemetry and logs. No mid-run weights, full-state or replay snapshots
are created or exported. The adapter sets `--full-state-every 0` and puts the
weights checkpoint interval beyond the endpoint plus overshoot; the trainer's
correct full-state barrier implementation remains available to other studies.
The immutable final archive and its index are bounded to **64 MiB total per
attempt**; an oversized artifact is incomplete, never silently truncated.
Neither terminal weights file supplies an exact resume. The counter overshoot limit
is `actors * worlds * learner seats - 1`; emitted decisions reconcile as
`256 * updates + debt`; clean actor exits and ring overruns both equal zero.
Admission pins B8, the demo/league inputs, and their SHA-256 hashes as immutable
supervisor-uploaded inputs, and requires passing B8 initializer and collector
oracle receipts.

`N3` is Q1’s N=3 static-pool collection package. `N1` is Q1 arm D’s
task-aligned terminal-hero collection package, including its historical gate
rosters and joint seat/roster/termination semantics. Collection is a package;
this screen cannot attribute an effect to one component. Each arm has five
fresh collision-audited seed blocks (`223213833, 3055723932, 1783343308,
1125234883, 1809856951`),
one L4 run at a time, and exactly 8,000,000 learner decisions. No intermediate
checkpoint exists for selection or substitution.

## Immutable banks, one look, and statistics

`WORLD_BANK.json` registers 64 scripted, 32 frozen, and 32 mixed H5000 worlds
per seed: 640 decision worlds. Each row is used by the N3 final, N1 final and
shared B8 final, yielding **1,920 policy episodes**, not 1,920 worlds. The
separate 10-world `timing_worlds` fixture is disjoint and never enters a
decision. Evaluation worlds, training seeds, and train/eval namespaces are
fresh and audited against bound strict inventories plus all visible repository
and artifact JSON/JSONL, including Q1. An earlier entire provisional block was
rejected before S1 data because its collision audit found a collision; none of
its seeds/worlds are retained or substituted. Opaque binaries are a declared
audit limit; a later collision makes the study INVALID.

The only efficacy look is after every final and every required row is present,
identity-bound and merged. For each arm and seed, calculate the equal-mix mean
of its B8-paired mass effects. The primary interval is a one-sided 90% Student-t
lower/upper interval across the five seed effects (`df=4`). Report the paired
N1 minus N3 equal-mix contrast with the same interval. A hierarchical bootstrap
over seeds and within-seed worlds uses 2,000 replicates, RNG seed `20261018`,
and shared paired indices; it is sensitivity only and cannot alter labels.
Intervals are development screens, not multiplicity-controlled discoveries.

This is an explicit fixed-N exception to sequential Phase R: five paired seeds,
one complete final look, no interim efficacy inspection, no early stopping,
extension, replacement, resampling, checkpoint selection, or automatic retry.
The sole operational exception is the prospective from-scratch rule below.

## Ordered decision rule

Apply these labels in order per arm.

1. **INVALID / INCOMPLETE**: any missing/hash-mismatched input, receipt, final,
   row, world, sample count, counter reconciliation, parity/merge proof, clean
   exit, cap or timing/admission requirement; a namespace collision; or failed
   transport identity. Preserve evidence. No substitute final or replacement seed
   is permitted. Only an eligible operational loss may use the once-per-arm
   from-scratch rule below; all other failures remain INVALID / INCOMPLETE.
2. **KILL**: Q1 tripwire (non-finite loss/Q/grad or illegal action; >90% one
   action share in a complete 100k window; greedy boost >2% overall or >1%
   below length 700); pooled alive-fraction effect below -.03 on any mix; any
   seed scripted alive-fraction effect below -.15; pooled no-death fraction
   below .50 on any mix (survival comparisons use tolerance `1e-9`); or primary
   `UB90(G) <= 0`.
3. **GO**: all KILL guards pass, `LB90(G) > 0`, mean `G >= 20`, scripted mass
   point effect `>= 0`, and at least four of five seed effects are positive.
4. **INCONCLUSIVE**: every other valid complete outcome.

KILL precedes GO. Rank qualifying GO arms by `LB90(G)`, then mean `G`, then N1.
A GO funds only preparation of a separately priced, fresh Mac strict gate; it
does not constitute strict pass, release, serving change, a longer run, league,
teacher study, or cap raise.

A verified tripwire halt is a KILL-triggered arm outcome: retain its evidence,
do not replace its seed/final, and skip that arm's future jobs while completing
the healthy arm. Contrast and bootstrap sensitivity are unavailable if either
arm lacks five valid finals. Any integrity uncertainty remains INVALID rather
than being relabeled as a KILL.

## Operational failure rule (pre-data)

A run lost to a disconnect, transfer failure, or runtime reclaim **before its
final export is verified on the Mac** may be rerun **once from scratch per
recipe arm**: at most one N3 rerun and one N1 rerun across their five seed
blocks (two additional attempts total). Use the identical failed seed block,
namespace, B8 initializer, uploaded inputs, source commit and training recipe;
start with fresh Adam and empty replay/label rings. Use a fresh attempt identity
and output directory. There is no resume, seed replacement or training extension.

The operator must record the infrastructure cause, absence of any efficacy
look, and the failed attempt's immutable receipts before preparing a replacement.
A verified final cannot be discarded. A scientific tripwire, parity/hash/counter
failure, or ambiguous cause is ineligible. Reconcile request-to-disappearance
elapsed time and debit, and prove runtime release before any replacement can
launch. Unknown intervals or release evidence block further allocation. Do not
retry automatically. The same budget includes both failed and replacement
attempts. Admission reserves the complete worst case for the ten primary
attempts only; optional operational reruns are conditional, never pre-funded.
At the moment an operational rerun would start, the ledger must atomically
reserve its full unchanged worst-case liability, including cleanup, together
with spent amounts, all outstanding liabilities and the complete remaining
primary/evaluation liabilities within **every** usable cap.
A second operational failure in that arm, or insufficient remaining budget,
leaves that arm INCOMPLETE; no second exception or efficacy look is allowed.
Other integrity failures retain INVALID precedence. Preserve all failed bytes.

## Governance, accounting, and admission

The nominal envelope is **60 compute units / 25 GPU-allocation hours / 24 Mac
worker-hours / 48 elapsed hours**. Twenty percent remains withheld, so usable
caps are **48 units / 20 GPU-hours / 19.2 Mac worker-hours / 38.4 elapsed
hours**. The live CLI acceptance is included and limited to 4 units. Account
context at registration is “Current balance: 1174.37 compute units; Usage rate:
0.00/hr; Active assignments: 0”; it is a snapshot, not a substitute for fresh
admission evidence.

The study-wide ledger reserves each job’s full worst-case liability *before*
launch and refuses if spent + outstanding liabilities + complete remaining
upper bound would exceed any usable cap. Allocation is charged from request
through supervisor-confirmed disappearance, including setup, idle, failure,
export and cleanup. Unknown intervals, missing supervisor receipts, unreconciled
assignments, missing pricing or changed hashes block launch. First cap wins. No
borrowing reserves, raises, substitutions, automatic retries/recovery, or
parallel L4 study run exists. The ten primary runs and at most two expressly
prepared operational replacements are serialized.

The Mac evaluation uses at most two deterministic single-thread workers,
resumable per-world partials and a fixed-order merge proven byte-identical to
serial. A disjoint fixture of at least 10 episodes prices the complete upper
bound. A tiny real M3-C entrypoint run and real eval entrypoint timing are
required before admission; admission refuses if the complete forecast does not
fit all usable caps. Transport liability is the measured final artifact size
(or its enforced conservative bound) divided by the measured throughput lower
bound from the independently qualified CLI acceptance receipt, plus final
preparation and cleanup. There is no full-replay transfer liability. A local CPU
fixture does not measure L4/home-connection throughput. Acceptance must bind the
current transport/adapter sources and final-size receipts before admission.

Conditional training planning is `6 * (30 + 68) / 60 = 9.8 GPU-hours` including
one possible rerun per arm, before acceptance, setup, final export and cleanup.
This optional scenario is planning only, never a pre-funded entitlement.
Hard manifest estimates/timeouts remain N3 30/60 minutes and N1 68/136 minutes.
Admission reserves the ten primary timeouts plus five-minute cleanup tails and
two acceptance GPU-hours: `5 * (65 + 141) / 60 + 2 = 19.1666666667 GPU-hours`,
or `38.3333333333 units` at 2 units/hour. Mac worker time is 15.8656728896 hours;
serialized elapsed time is 27.0995031115 hours. These fit all usable caps, but
final-only L4 throughput qualification remains required for admission. The prior
22.6 GPU-hour / 45.2-unit reservation included both optional reruns and is
superseded by this primary-only rule. Caps, timeouts, the withheld reserve and
the worst-case-reservation principle are unchanged.

New Colab work uses only `research/colab_cli` after its independent live
acceptance. No Drive queue, direct `colab` operation, runtime retention,
purchase, training, evaluation dispatch, tmux action, serving change or release
is authorized by this document.

## History and interpretation limits

T23 was KILL: equal-mix mass was -80.8 versus B8 and all five seeds were
negative. Q0’s mass audit located food/alive-frame loss by 2M but did not
identify its cause. Q1’s attempted factorial remains INCOMPLETE because A5
refused its conservative budget forecast; it is neither reset nor silently
continued here. A2/A3 were diagnostic/execution amendments in Q1, not evidence
that S1 has already passed an invariant. N3/N1 are mature-schedule candidate
recipes; this study cannot identify schedule, collection, optimizer-reset or
interaction causality. Failure closes these recipes within this package, not
all RL.


## Pre-data change log

- 2026-10-18 (planner decision, before S1 training/evaluation data): replace
  four full-replay snapshots per run and 500k diagnostic weights with terminal
  online/target weights and decision evidence only. The measured capacity record
  `CHECKPOINT_SIZE_MEASUREMENT.md` at commit `66c9135e` showed about 868 GB of
  snapshot exports across ten runs, before finals; this transport is infeasible
  on the home connection during paid allocations and cannot support a registered
  exact resume. Keep the trainer barrier correction; remove it only from S1's
  export path. Add the once-per-recipe operational rerun rule, charged to the
  same caps and settled before any efficacy look. Replace full-replay transport
  qualification with final-artifact size/CLI throughput qualification. Preserve
  DRAFT status, seeds, recipes, endpoint, statistical thresholds, fixed look,
  overshoot bounds, serving boundary and all historical evidence. Record the
  resulting 22.6-hour hard reservation refusal explicitly.

- 2026-10-18 (Claude planner decision, S1 **DRAFT, pre-data**): make
  operational reruns conditional and never pre-funded. Supersede only the prior
  requirement to reserve both optional reruns before first launch, recorded in
  `ADMISSION_STATUS_FINAL_ONLY.json` (commit `e5dd9bb3`). Reserve all ten primary
  attempts at admission; reserve a rerun's complete unchanged worst case only at
  its explicit launch if spent plus all outstanding and remaining liabilities
  fit every usable cap, otherwise record the affected arm INCOMPLETE. This is a
  prospective edit to the unratified DRAFT, before S1 training/evaluation data;
  it is not ratification or dispatch authorization. Preserve caps, timeouts,
  20% reserve, worst-case accounting, seeds, recipes, endpoint, thresholds,
  fixed look, identities, counter bounds and serving boundary. The earlier
  change-log entry and historical refusal receipts remain as history.
