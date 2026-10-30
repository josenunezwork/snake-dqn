# T34 Watch6 exact-stop efficacy ceiling — 2026-10-30

Status: RATIFIED_BY_ASTRA_PRE_DATA_2026_10_30, subject to independent source review,
full Mac mirror qualification and passing same-pin measured L4 canary admission.
This sibling prospectively supersedes REGISTRATION.md; original bound bytes remain
intact. Owner appointed Astra as judge + implementer for this amendment and chose
Colab-precomputed exact-search replays for Watch on 2026-10-30. RULING.md records
the ruling. No scientific data exist. This assignment authorizes preparation,
tests, branch commit/push only; no dispatch, merge, or serving change.

Question: Does the unchanged L2-prime exact-stop action rule have worthwhile
product efficacy when delivered synchronously at every eligible root? This is a
Tier 1 go/no-go screen for the owner-selected Colab-precomputed Watch replay
serving path, on Colab CPU. It is not a strict gate, implemented replay player,
timing qualification, or release. C/T's action rule is not used.

## Controller and product

Hero: bare B8 s3, SHA256
3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8.
Five FRP3 s12 opponents, each with its own v8(lambda=8) carry, checkpoint SHA256
eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723.
Actual default GameSession Watch6 mechanics v2 configuration, continuous respawn
for all six snakes; rectangular arena, no long-body injection, no terminal hero.
The constructor reads configs/mechanics_v2.yaml directly. It never substitutes
research/apex_safety_20260926/deployment.yaml or a gate evaluation profile.
A Grid mirror checks physical state/RNG, vector61 carry and v8 counters against
the actual live game every frame; mismatch is INVALID. B8 uses rowwise fp32 CPU
inference and advisory masking with normal-action fallback.

Trigger: DenseTrigger(min_length=600, interval=5), clock starts on first eligible
hero decision, resets each respawn. No Q-risk cutoff. At each eligible root with
nonempty advisory mask, complete H300 continuations: bare first, then descending
raw Q / ascending action index; stop at first survivor. If all die, use the
unchanged ceiling._choose death-delay ranking and bare/Q/index tie breaks.
Legal set and fallback are exactly ceiling._legal (advisory, normal fallback).
No receding-horizon search inside branches; all later branch actions are bare B8.
Opponent policies retain v8 in every branch. Root RNG and carry are cloned, never
rekeyed; no future observed events, hindsight death roots, or C/T reranker.
Branch hero death is terminal to branch scoring; main product hero respawns.
Search always gets its chosen action applied at its root. No deadline or queue.
This measures ideal synchronous efficacy, not a mathematical upper bound on every
possible delayed or less frequent controller (interventions can harm).

## Allocation, looks and statistics

Exactly 96 complete paired sessions, H14400 (10 min at 24 game frames/s), hero slot
0. Namespace t34-watch6-exact-stop-efficacy-20261030-v2; SHA256(namespace/index)
first four bytes big endian gives each seed. Allocation index0..95 is immutable.
The pair, including both full sessions, is the export/resume unit. Freshness audit
rejects an entire colliding allocation before data. Visible committed banks,
historical artifact text and training inventories plus positive controls are
covered; opaque/unrecorded history is unauditable and disclosed.

One fixed-N look on the Mac after all96 paired units are complete and verified.
No interim outcomes, extension, additional seeds, controller selection or
outcome-dependent retry. Excluded fixtures34890123..34890125 never enter the bank.

Power is planned on deaths, not the pooled +83 mass result. The relevant prior is
MIXED: +27.6 mass, 5 productive rescues /11 baseline deaths in64 worlds. This is
historical v1 with a different roster, not evidence of v2 Watch efficacy;5/11 is
not treated as a known reduction probability. No exact-product paired-session
variance estimate exists. The advisors' ~170 baseline deaths for96 sessions uses
1.8/session from a6000-frame example. At the retained H14400, the older mixed
monitoring rate0.296/1k projects4.2624/session or409.19 baseline deaths. Both are
sensitivity scenarios, not assumed truth. With uncorrelated Poisson arms and
20%/30% reduction, N96 gives signal1.96/3.02 SE at1.8/session, and3.02/4.65 SE at
4.2624/session. Doubling the count variance divides these signals by sqrt(2).
`python -m research.watch6_search_efficacy.power` records normal planning power
with the df95 t critical and dispersion1/2. Pairing may help; overdispersion and
a smaller actual effect may hurt. No powered mass claim is made from the
historical mean alone. N96 is a fixed decision panel, with a declared no-go on
uncertainty; there is no variance peek or automatic N adjustment.

Co-primary bars from SUCCESS_CONTRACT section 3 and P7_STATISTICS:
1. Candidate minus bare deaths/1k has two-sided paired Student-t 95% CI wholly
   below zero. Target 20% reduction is reported, not an additional pass bar.
2. Ratio of pooled full-horizon means (candidate-bare)/bare has Fieller paired
   two-sided 95% lower limit >= -0.05. Full-horizon includes dead frames as zero
   and all respawned lives. Unbounded/disconnected Fieller set cannot pass.

Unit of inference is complete paired session, df95; lives/ticks never raise n.
One arm and one look; conjunction of both co-primary tests is an intersection-
union claim, so no Bonferroni penalty is needed. No separate primary claim for
either endpoint. Report median/p95 length, deaths at pre-action length >=800,
boost decisions, interventions, branch counts. Zero baseline deaths or zero
variance cannot manufacture evidence of superiority; degenerate equality is
INCONCLUSIVE. Full audit/integrity faults precede science. Missing/duplicate/
foreign/hash-corrupt/fixture rows => INVALID_OR_INCOMPLETE, no scientific result.
Finite-Q and legal-action faults kill the arm; >90% one action in any rolling
100k delivered decisions is unconditional fault. This bank has <=14400 decisions
per session, so that window cannot complete within a session. No cross-session
concatenation is treated as a continuous trajectory. H5000 boost/skill bars stay
with the later Mac full-contract study; diagnostic product boosts are reported.

GO_EFFICACY_ONLY = both product bars pass. FAIL_EFFICACY = death-delta lower limit
>=0 with nondegenerate CI, or mass-relative upper limit < -0.05. All other valid
results INCONCLUSIVE. Integrity faults are INVALID_OR_INCOMPLETE, before science.

Replay-serving decision, frozen before data: GO_EFFICACY_ONLY admits preparation
and qualification of a labelled search-assisted Watch replay path using this
byte-identical evaluated controller. FAIL_EFFICACY and INCONCLUSIVE both mean
NO_GO_FOR_REPLAY_SERVING_UNDER_THIS_REGISTRATION: retain current bare B8 serving,
do not deploy/promote this replay path, do not add sessions or rerun this bank.
INCONCLUSIVE is insufficient evidence, not proof of no effect. A different
controller or later study requires a genuinely prospective fresh registration;
it cannot rescue this one. GO does not itself authorize serving: replay integrity,
mutating-control buffer invalidation, labelled assistance, rollback, Mac strict /
applicable full-contract and serving qualification, independent audit, and owner
release approval remain required. This is the efficacy go/no-go for that path.

## Placement, completion and guards

CPU science: fp32 singleton B8 and vector61 inference with native v8 simulator
branches. L4 host, >=12vCPUs, GPU unused; no hardware/precision substitution.
Workers4/6/8, one numerical thread/process, are allowed operational choices.
Serial==parallel pair bytes remain required. Workers and partitions do not alter
seed/controller/N or the statistical unit. The same-pin excluded L4 canary
measures all three worker counts, both actual Watch+5v8+mirror main loops for
H300/arm/worker, and three long-body H300 branch batches per worker. Full-H300
coverage and native-kernel/no-fallback counters are retained. Profile workers
warm up then synchronize at barriers to measure actual contention. It exports
operational costs only; no main-bank sessions or outcome look.

The mechanical forecast uses the largest observed whole long-body branch-batch
wall as a conservative singleton cost (no division by branches), each arm's
largest measured main-loop seconds/frame, ceil(H/5)=2880 roots/session and the
physical maximum6 branches/root, not the old1.2-branch assumption. Pair cost =
H14400*(bare main sec/frame + exact_stop main sec/frame) +2880*6*branch sec.
For each W4/W6/W8 and shard count3/6/12/24/48/96, the shard base estimate is
ceil(20 setup/export minutes +1.25*ceil(ceil(96/shards)/W)*pair seconds/60).
Choose the fewest feasible shards, then smallest measured aggregate runtime,
then smaller W. Fewer/larger shards are preferred only if this forecast fits.
Refuse if no choice fits base<=660min (effective2x wall<=1320min, below24h
provider lifetime with2h reserve). The25% forecast margin is distinct from the
owner's2x runaway wall/unit/reserve rule. Rates, N, measurements and retries are
never doubled. Costs are measured fixtures, not a mathematical upper bound on
all future long-body states; per-pair immutable exports bound incomplete loss.

`prepare` without --main creates ONLY the <=30min effective canary. `prepare
--main` refuses absent/failed/stale/foreign measured same-pin canary evidence,
transport/input/export mismatches or any forecast that does not fit with margin.
Main admission and real execute-time admission rederive the selected W, shard
count and base estimate from the exact recovered measured report. CLI worker /
shard overrides refuse. Main manifests carry those measurements in their bound
environment; uploaded input bytes remain identical to the qualifying canary.
No synthetic report can qualify main outside unit tests. No live measurement is
authorized in this implementation assignment; the operator runs the canary.

Before canary or main preparation, retain one excluded bare H14400 Mac session
through >=2 hero respawns and >=3 total respawns,14400 checked frames, zero mirror
mismatches. Receipt binds the actual gameplay/mirror source closure; it reports
no efficacy look. Three tiny real entrypoint pairs and Mac branch fixtures remain
additional transport/runtime checks, not substitutes for the full mirror proof.

Only qualified research/colab_cli via tools/ops/dispatch.sh. Astra ruling is valid;
there is no mandatory Fable persona. Approval binds judge, science identity,
REGISTRATION_V2.md and RULING.md hashes. Original plus at most two outcome-blind
completion attempts per missing shard after verified mechanical failure and
stop_verified. Empty source/allocation-bound ledger on attempt1. Mac recovery
reads path/index/size/hash completion metadata only. Complete pair payload and
SHA sidecar publish atomically, completion marker last. Fresh manifest/output /
state/receipt each attempt; never recompute verified IDs. No automatic retry.
Only the Mac aggregator takes the one look at the complete96-unit union; remote
outcomes stay sealed. Partial pairs are not resumable. Account cap3, refresh live
positive pricing and other_assignments before admission; units are not rationed.
Progress <=60s and stall watchdog900s. Mac mirror, preflight, transport checks,
execute-time admission, tests, independent Argon source review, published branch
pin, judge ruling and measured L4 canary are distinct evidence requirements.

Completion canaries must fit the original frozen W/shard/wall layout. Faster
measurements retain the original guards; slower measurements that exceed them
refuse completion. No mid-series increase or repartition is permitted.
