# Prospective amendment 04 — DENSITY-2, before replication data

Authority: independent Fable's 2026-10-22 spare-slot ruling, preserved at
`docs/research/advice/2026-10-21_big_lever/JUDGMENT_SPARE_SLOT_DENSITY2.md`
and `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/JUDGMENT_SPARE_SLOT.md`.
The latter was read fully. This sibling implements its DENSITY-2 registration.
METRICS.md and amendments 01–03 remain immutable. No replication data exist.

## Fixed allocation, independent of observed outcomes

| Arm | Frozen | Mixed | Scripted | Total |
|---|---:|---:|---:|---:|
| A | 96 | 72 | 72 | 240 |
| B70m | 24 | 24 | 24 | 72 |
| B70p | 36 | 36 | 36 | 108 |
| C-P1 | 24 | 24 | 24 | 72 |
| Total | 180 | 156 | 156 | 492 |

Namespace `bl-density-round14b-20261022`. A separate
`bl-density-round14b-canary-20261022` namespace contains twelve excluded canary
seeds. The allocator rejects duplicate seeds within/between these banks and all
90 seeds derived from pilot namespace `bl-density-round14-20261022`. All arms
and mixes remain; no post-data substitution, seed selection, extension or pooling.
Pilot episodes are used ONLY for sizing and excluded from every replication
statistic, classifier, reference and decision. One terminal look; no interim
outcome reads of exports. No third enlargement. Local engineering preflights use the additional excluded
namespace `bl-density-round14b-preflight-20261022`, also collision-checked.

Sizing disclosures from the ruling (historical pilot, never replication evidence):
A doom-bearing episodes frozen/mixed/scripted 1/2/2 of 12 each; witnessed-avoidable
roots 0/1/1. The pilot totaled 90 episodes and 4,532 CPU-s. A CPU seconds/episode
67.7/48.6/62.5; B70m 42.6/44.2/33.8; B70p 43.1/56.9/20.8; C-P1 50.0/57.6/48.9.
B70m and B70p contributed 5/18 and 3/18 doom-bearing episodes, C-P1 11/18.
Sizing uses these event rates, not a selected yield or classifier result. Frozen
A at N=96 gives about .96 probability of >=4 events at the pilot rate, about .57
at half that rate. No classifier AUC existed in the pilot. Pilot prefilter recall
was .6 (6/10 true dooms); natural rho95 .122/.105/.122, with 85/51/94 roots.
Pilot doom roots were 27/29 DEATH_SELF, L801–2094, median 1186.

## Frozen scientific behavior and kill rules

Round-14 section 2b implementation is retained byte-identically in analysis.py
and statistics.py. B8 checkpoint bytes, CPU float32 rowwise inference, mechanics,
rosters, arm constructions, features, classifier/split, prefilter, H5000 horizon,
stride 200, H300 continuations, four-witness-root cap, minimum-map rule, CPU
attribution, decision precedence and numerical thresholds remain unchanged.
REJECTED on yield <2*A, avoidable fraction <.5*A, AUC >=.80, or map violation;
unsupported realism never admits a cell. Existing classifier insufficiency reasons
identify INSUFFICIENT_REALISM without changing the frozen decision implementation.
Admission permits only registration of a separate A-transfer screen with round-14
09's falsifier. It never admits training, a P1 change or serving.

After the FULL allocation, a mix with new A <4 doom-bearing episodes is
INSUFFICIENT and densification rows 9/9a are CLOSED for that mix this campaign.
INCOMPLETE retains all per-episode work/cost and is not extended. Mechanical
transport/CUDA/export/hash failure permits at most one fresh prepared attempt,
never resume or automatic retry, and only within remaining registered caps.
Pilot and replication are never pooled. Natural A avoidable rate and rho95 are
secondary descriptives, not additional decisions. Prefilter recall <.8 disqualifies
it as a critic/served trigger; this measurement authorizes neither use.

## Bounded execution and preparation

Production: ten single-thread CPU workers on an L4 host with >=12 vCPUs;
`--max-seconds 10800`; session <=180 minutes; manifest unit_cap <=5.0,
reserve_units 1.0 (20% of the ceiling), first wall/unit cap wins. Pilot-sized
forecast is about 25,500 CPU-s (7.1 CPU-h); ten-worker scaling is UNMEASURED,
70–80 min is a planning estimate. The 1.54 u/hour historical L4 rate is a ceiling
to be checked at real admission, never a new measured rate. Reserve is included
in the envelope, not added to it; work planning stays <=4 u. Setup, copies,
canary, failed attempts and cleanup all count against the envelope. A finalized
manifest subtracts verified prior attempt charges and reserve before dispatch.
No new attempt is admitted when remaining capacity cannot cover the forecast.

Canary: twelve tiny eight-decision episodes, one per arm/mix, full H300 checks
where screened; separate excluded namespace; <=.5 u including .1 u reserve,
<=20 min (priced whole-session wall is tighter where necessary). Real supervisor,
identical entrypoint/source/input inventory and public hash-checked downloads;
PASS requires recovered/hash-verified exports, stage CPU counters, scanner and
verified backend release. Family-canary receipt is additionally required for
production. Z-9 readiness qualifies both actual inventories and fallback archives.
`budget.other_assignments` remains `{}`; live assignment inventory must be
reconciled immediately before dispatch. All exports and preparations create-only.

Preparation performs no Colab operations. The current qualified family gate also
requires a real CUDA action decision for an L4 canary. This conflicts with the
registered CPU-only density family. No synthetic CUDA observation, GPU change or
transport exemption is introduced here. Current transport qualification is verified
against the newer source-bound `colab-cli-main-acceptance-20261022` receipt, rather
than the older acceptance inventory in QUALIFICATION.md. Offline validation records this unresolved
live qualification prerequisite honestly; it is not a family PASS. The production
template must acquire genuine family and package recovery evidence before it can
pass supervisor admission. Release remains subject to the owner's approval.
