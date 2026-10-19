# DRAFT_PENDING_OWNER_REVIEW — S1 mature collection screen (2026-10-18)

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
`torch.compile` reduce-overhead. Full state (online/target/optimizer/replay,
rings, counters and RNG) checkpoints every 2M learner decisions in bounded
transport chunks no larger than 2 GiB; weights-only snapshots occur every 500k
for diagnostics and cannot be resumed or selected. The counter overshoot limit
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
one L4 run at a time, and exactly 8,000,000 learner decisions. No 2M/4M output,
if retained for transport diagnostics, can decide or replace a final.

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

## Ordered decision rule

Apply these labels in order per arm.

1. **INVALID / INCOMPLETE**: any missing/hash-mismatched input, receipt, final,
   row, world, sample count, counter reconciliation, parity/merge proof, clean
   exit, cap or timing/admission requirement; a namespace collision; or failed
   transport identity. Preserve evidence. Do not rerun, substitute a final or
   replace a seed.
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
borrowing reserves, raises, substitutions, retries, automatic recovery, or
parallel L4 study run exists. The 10 runs are serialized.

The Mac evaluation uses at most two deterministic single-thread workers,
resumable per-world partials and a fixed-order merge proven byte-identical to
serial. A disjoint fixture of at least 10 episodes prices the complete upper
bound. A tiny real M3-C entrypoint run and real eval entrypoint timing are
required before admission; admission refuses if the complete forecast does not
fit all usable caps. The full-size transport/export upper bound is unpriced at
registration and therefore blocks admission until it is measured and fits every
usable cap; local preflight results are pending and are not claimed here.

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
