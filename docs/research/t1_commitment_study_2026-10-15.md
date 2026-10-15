development diagnostic, not gate evidence

# T1 Part A: exact-replay commitment and contrast study (2026-10-15)

## Routing rules (fixed before the commitment numbers)

- R1: >= 60% of the 49 late deaths (L >= 800) have a surviving alternative within 150 decisions AND
  median H_LAD <= 100 -> "T3 eligible". R1': < 30% have any alternative within 300, or median
  H_LAD > 300 -> "T3 off".
- R2: Q-gap at LAD < 2 Q units in a majority -> "value learning is the limit" (Part C's probe also
  feeds R2). R2': Q-gap > 5 favouring the fatal action in a majority -> "perception is the limit"
  (decided jointly with Part C).
- R3: the surviving action differs from the chosen one on tail-in-region or on boundary TTL in >= 70%
  of LADs -> tail-topology inputs GO; < 50% -> NO_GO.
- R4: a boost at LAD survives in >= 20% of late deaths -> include boost labels in T3.

## Method and results

All 75 B8 census deaths were completed: 49 late (L >= 800) and 26 early.
Each arm uses its original final checkpoint and census world seed/roster.
The executor captures the actual census evaluator's initial world and policies:
grid engine, terminal hero, respawning scripted opponents, fast anchors,
rowwise ego2s inference, advisory mask, eps 0, no B8 veto.

Recorded actions fast-forward the original trajectory. Every pre-action hero
length is checked against the census series. All advisory-legal actions, boosts
included, are followed by the same hero's own greedy continuation for 300
transitions or to decision 5000. The recorded action is a separate control
when advisory-unsafe; every control must reproduce the original death decision.

Indices are zero-based `fatal_t`/`t0`; census `frame` is one-based after preparation.
At `d = fatal_t - 300`, the control continues one extra transition beyond the
300-transition endpoint to verify death. Controls never count as alternatives.
The stride-10 scan includes f-10; the latest positive window is refined at stride 1.
The separately measured t0 can also establish a later positive inside the scan window.
All outcomes, controls, Q vectors and chosen/survivor geometry are retained in rows.

| Quantity | Late (49) | Early (26) |
|---|---|---|
| LAD observed | 46 | 26 |
| None within 300 | 3 | 0 |
| H_LAD quartiles (all deaths, censored ranks) | [21, 100, 254] | [2, 10, 25] |
| H_LAD quartiles (observed) | [16.5, 88.5, 248.5] | [2.25, 11.5, 24.5] |
| Q-gap quartiles at LAD | [0.21879065036773682, 0.44695305824279785, 1.27960205078125] | [0.12269997596740723, 0.255281925201416, 0.541587233543396] |
| Head-side region at t0 quartiles | [33.0, 160.0, 160.0] | [5.0, 17.0, 160.0] |
| Chosen next-cell region at t0 quartiles | [32.0, 130.0, 159.0] | [4.0, 16.0, 124.0] |
| Survivor within 30 decisions | 16/49 (32.7%) | 20/26 (76.9%) |
| Survivor within 100 decisions | 25/49 (51.0%) | 20/26 (76.9%) |
| Survivor within 150 decisions | 25/49 (51.0%) | 20/26 (76.9%) |
| Survivor within 300 decisions | 46/49 (93.9%) | 26/26 (100.0%) |
| Q-gap <2 at LAD | 40/46 (87.0%) | 24/26 (92.3%) |
| Q-gap >5 at LAD | 3/46 (6.5%) | 0/26 (0.0%) |
| Tail/boundary-TTL contrast | 5/46 (10.9%) | 2/26 (7.7%) |
| Surviving boost at LAD / deaths | 32/49 (65.3%) | 21/26 (80.8%) |
| Opponent head literal inside at t0 | 0/49 (0.0%) | 0/25 (0.0%) |
| Opponent head in/on component boundary at t0 | 16/49 (32.7%) | 9/25 (36.0%) |

| Rule | Verdict |
|---|---|
| R1 | does not fire |
| R1_prime | does not fire |
| R2 | value learning is the limit |
| R2_prime | does not fire |
| R3 | NO_GO |
| R4 | include boost labels in T3 |

R1 uses the full late-death median (rank 25 of 49) and the full 49-death denominator
for its survivor fraction. Censored quartiles use inverse empirical-CDF ranks
ceil(p*N), with missing horizons retained above 300 under the prescribed search.
Observed-only quartiles use NumPy's linear interpolation and are shown separately.
Missing LADs' exact horizons are unknown. R2/R2' use all observed LADs and retain the
Part C qualification. R3 compares the highest-Q surviving alternative with the
chosen action; empty boundary TTL is represented by null, not zero.

Sealer owners, late: `{'opponent': 13, 'hero': 13, 'unresolved': 7, 'none': 16}`; early: `{'unresolved': 6, 'opponent': 6, 'none': 9, 'hero': 4, 'no_t0': 1}`.
Resolved opponent kinds, late: `{'scripted:greedy_food': 13}`; early: `{'scripted:greedy_food': 6}`.

## Geometry conventions

Components are exact four-connected free regions, uncapped. The explicit spec
rule is used: empty cells and own-body TTL <=1 cells are free; opponent segments
block. This differs from ego_raster_b's all-owner tail-release rule. TTL is
logical length minus head-first segment index. Normal/boost variants use the
same first next cell. Tail membership includes the component and its boundary.
For the head-side region, only the current hero head is temporarily opened as
the flood seed. The region series covers fatal_t-59 through fatal_t inclusive.
A live opponent head is ordinarily blocked: literal component membership and
four-adjacent boundary membership are reported separately to avoid conflating them.
Sealers require a newly occupied cell with a demonstrated split; ambiguous
multi-cell/multi-owner cases remain unresolved. No proximity guess is used.

## Parity, timing and verification

The unmodified census clone passed two B8_s0 death worlds (source envs 3 and 8):
2,500 recorded warmup decisions followed by 1,000 paired clone decisions each.
All 9,000 checked length values and 2,002 full-state comparisons matched,
including raw body storage, grids/pads, food order/sets, RNG, policy state, masks,
timers and event/scratch arrays. Clone storage independence and restored pad
views passed. Prepared callback clones normalize only lifecycle guards to the
census pre_step phase, then apply their first action with step(), without preparation.

Early-state benchmark at decision 100 (warm paths, one CPU thread per worker):

| E | Clone (ms) | Step without hero (ms) | Step with hero (ms) | 300 decisions (s) |
|---|---|---|---|---|
| 1 | 1.060 | 0.542 | 2.942 | 0.802 |
| 32 | 30.858 | 1.946 | 27.882 | 9.349 |

Step 1 wall time: 3222.52 s with 2 worker processes.
Separately timed sealer review: 34.38 s, 230,944 recorded replay env-frames; 10 owner classifications changed.
Combined study and sealer-review wall time: 3256.90 s.
Counterfactual/control rollout env-frames: 2,671,306.
All Step 1 env-frames, including recorded/refinement replay: 2,906,615.
Combined study and sealer-review env-frames: 3,137,559.
Parity/timing and the preliminary pilot are separate from those totals.
Rollout workers used the frozen launch commit in launch-head.txt. A reviewed
joint-cell seal correction was applied in a separate recorded-action replay only:
no counterfactual outcome, LAD, Q-gap or action geometry was changed. Original
rows remain in rows-before-sealer-review; sealer-review.json pins source and costs.
Raw verification output is in `verification.log` under the artifact directory.
Final verification: the required test file passed 21 tests; the topology regression
file passed 6 tests. The required Flake8 command and Black checks were clean.
The independent artifact audit passed all 75 census identities and 2,929 recorded
death controls, checked every rollout total, and confirmed the reviewed rows retain
every original intervention outcome. Input bytes and protected dependencies matched
the launch hashes. No final verification failures remain.

## Caveats and artifacts

Small counts limit interpretation. 'No survivor' means 'no survivor under B8's own
greedy continuation' at the sampled decisions, not impossibility under other
policies or multi-action interventions. The stride-10 search can miss narrow
positive windows outside its final positive refinement window; LAD is measured
under that prescribed search, not exhaustive over the entire 300 decisions.
R2/R2' are descriptive Part A routing signals; Part C's representation probe is
not part of this run. Optional 11 incumbent and 10 champion reference deaths were
not run, to keep this pass within the mandatory B8 scope and two-process budget.
No training, gate evaluation, serving change, remote upload or push occurred.

Artifacts: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t1-commitment-20261015`. `rows.jsonl` (75 deaths), `rows/*.jsonl`, `summary.json`,
`parity.json`, `timing.json`, `run.json`, `manifest.json`, README, detached-run logs
and verification receipts retain the evidence. The manifest pins input/checkpoint
SHA256s and implementation/dependency source hashes.

## Reviewer notes (Claude, 2026-10-15)

- Of the 46 late deaths (L >= 800) with a last avoidable decision, 17 could be escaped only by a
  boost and 29 also had a non-boost survivor. B8 chose a boost at only 2 of the 46.
- Q(chosen) − Q(best surviving) at LAD, late deaths: p10 0.13, p25 0.22, median 0.45, p75 1.28,
  p90 2.55. B8 prefers the fatal action, but by a fraction of one Q unit.
- Combined with Part C (fuse probe AUC 0.84 at H100; −max Q AUC 0.85, not a length proxy):
  B8 knows the state is dangerous but ranks the escape marginally below the fatal action. That is
  a value-ranking failure, not missing information, and a third of the escapes need a boost, which
  B8 almost never uses (0.1-0.2% of frames).
- R1 (51% within 150 decisions vs 60%) and R1' both miss, a zone the panel left undefined. The
  survivors exist (43 of 46 within 300 decisions), so the T3 ceiling test is justified on
  evidence; whether training-time rollout labels count as "no veto" is still the owner's ruling.
