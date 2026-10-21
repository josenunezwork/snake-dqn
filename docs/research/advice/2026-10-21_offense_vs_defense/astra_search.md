**Bottom line — keep rescue deliverability first; test offensive search in one small parallel lane.**
Yes: T34 can evaluate offensive deviations against the actual reacting scripted/frozen policies.
It currently searches predicted deaths and ranks survival, so changing only a score will miss offense.
Start with profitable corpse interception; deliberate encirclement is a much larger planning problem.
The “mechanics v2 never implemented / current B8 abuses boost” premises are false in this checkout.
T3's +91.951 is a reused scripted-bank result, not a universal defensive ceiling; B8 s3 alone gained +122.812.
I would reverse priorities only after fresh, paced offense adds material mass over rescue without losing survival.

2026-10-21 · Independent Astra SEARCH/RL advice · source snapshot `61b8f751506dafff5fdee69d529309bd147d5232`.

This answers the open question in `QUESTION.md:32-52`. It is advice, not a registration, result,
launch instruction, or release authorization. No new gameplay was measured. Numbers below are
**recorded measurements**, **arithmetic derived from them**, or explicitly **proposed assumptions**.
I read repository sources and historical Git blobs, not the sealed Round 0 artifact directory.
The adopted direction reset is controlling: deliverability smoke first; registered worker count
N=1 versus N=6, with all timing bars preserved; fresh paired evidence and owner release approval
remain mandatory (`docs/research/final_run_20261019/CAMPAIGN_LOG.md:84-91`).

**What the facts actually support.** The target is bare B8 s3, particularly the mixed Watch6 hero
against five FRP3+v8 opponents. Terminal H5000 skill and continuous-respawn product behavior are
separate tests. The contract requires pooled skill mass +40/LB90>0, scripted no-death improvement,
and a product death-rate improvement, not simply more kills
(`docs/research/final_run_20261019/SUCCESS_CONTRACT.md:10-40`).

| Question premise | Checked evidence | Consequence for search |
|---|---|---|
| Mechanics v2 was never implemented | Live death drops and boost trails exist in `src/game/game_state.py:369-379,601-627`; Grid implements both in `src/simd_env/grid_sim.py:509-517,875-896`. B8's ego2s serving path selects the 61-D mechanics-v2 world, default-on with an explicit opt-out: `web/backend/session.py:323-344,755-774`. | Do not start by rebuilding mechanics or silently changing B8's benchmark. This is a source/default check, not inspection of a running server's environment. |
| A large kill deposits its entire usable mass | V2 attempts full-body drops, but corpse-class food has FIFO cap `max(0.5*max_food,10)`; Watch config has `max_food=300`, hence **150 pellets board-wide**, shared with boost trails. See `src/core/mechanics_constants.py:24-64,93-95`, `configs/mechanics_v2.yaml:31-40`, `src/game/food_manager.py:148-159`. | An 800-cell victim does not leave 800 collectible pellets. Deduplication, eviction, location, competitors and collection time matter. |
| Current B8 has ~0 kills and ~89% boost | T13b mixed Watch6 recorded **19 hero kills / 25 H5000 sessions = 0.76/session**, **0.17% executed boost**, mass 316.76: `docs/research/t13b_six_snake_roster_2026-10-17_results.md:25-36`. The 89% characterization refers to old Apex: `research/apex_safety_20260926/protocol.md:56-60`; the blueprint's ~0-kills argument diagnoses old economics: `docs/ml_redesign_blueprint_2026-07.md:33-36`. | B8 does kill, but not frequently in this product sample. Low boost does not establish that extra boost would be profitable. |
| Defense has a +92 maximum | Dense T3 pooled s0/s3 scripted mass was 700.005 versus 608.054 (**+91.951**); s3 was 686.662 versus 563.850 (**+122.812**). Survival was 54/64 versus 38/64, with 75 changes in 301,589 decisions. `docs/research/t3_ceiling_test_2026-10-15.md:113-129`. | This is a large observed defensive effect on reused development worlds, not an upper bound, three-mix effect, or paced serving result. Do not substitute the unrelated +92.4 training benchmark. |

**The reusable machinery is substantial, but the current algorithm excludes most offense.**
The teacher admits a search when its confirmed greedy shadow dies
(`research/t34_exit_b8/teacher_v2.py:756-788`). Its ranker keeps bare whenever bare survives;
among survivors it uses B8 Q, and among dying actions it uses alive steps. Mass is absent from
that ordering (`research/t34_exit_b8/teacher_v2.py:253-287`). Path B likewise requires a doom
and keeps a surviving baseline (`research/t34_pathb/engine_v2.py:617-625,941-966`).
The teacher defaults to H300; Path B v2 defaults to D500/H500, not the older D300 design
(`research/t34_exit_b8/engine.py:96-112`; `research/t34_pathb/engine_v2.py:408-415`).

The existing branch evaluator already sums post-step logical hero mass, with dead frames zero,
and applies one alternative first action followed by the B8 policy
(`research/t34_exit_b8/engine.py:320-355`). It clones opponent carry independently into branches
(`research/t34_exit_b8/engine.py:391-417`). Thus **no kill bonus or new RL trainer is needed to
reward successfully eating a corpse**: the resulting length is already in the mass sum.
Kill credit and corpse provenance are explanatory telemetry, not additional mass to add again.
Counting victim size as hero mass would reward uncollected, evicted or competitor-eaten food.

My first offensive controller would preserve the existing rescue rule when baseline is doomed.
When baseline survives, it would search at predicted opponent deaths, nearby head encounters,
and a small uniformly sampled set of safe states. Compare every legal first action with bare
under the **same H500 continuation**, require horizon survival, rank by
`delta_A500 = (sum(hero_length_alt) - sum(hero_length_bare))/500`, and retain bare on ties.
A prospective minimum `delta_A500 >= 5` is a proposed intervention threshold, not an estimated
effect. Re-run the actual closed-loop controller to H5000: many local +5 gains may disappear,
compound, or cause a later death. They cannot be summed into a claimed H5000 improvement.

This yields a narrow **mass-seeking policy-improvement step**, not a general hunting planner.
One turn followed by B8 may collect a corpse or alter an intercept; it may also immediately
abandon a multi-turn attack. If useful events require commitment, a later, separately bounded
variant can compare a fixed short macro library (e.g. 2/4/8 decisions) then B8, with exact legality
and carry throughout. Do not interpret failure of first-action search as an offense impossibility.
Do not start with unrestricted beam search: the separate T35 two-deviation rescue experiment
already found no novel rescues in its registered extension, a warning about indiscriminate
branch expansion, not evidence about offensive macros
(`docs/research/final_run_20261019/CAMPAIGN_LOG.md:73-76`).

**“Predicts exactly” means reacting policies, not recorded future actions.**
Historical M1 compared 15,000 frames across three mixes with no unexplained physical, carry,
v8, action or shadow differences. It compared independently initialized live/GameSession and
Grid worlds, including RNG and ordered food; it did not provide an arbitrary live-state loader
(`research/t34_shadow/RESULTS.md:29-62`). The newer Watch adapter builds the real five-FRP3+v8
roster and checks physical/carry/v8 identity at every step
(`research/t34_pathb/adapter_v2.py:245-280,346-376`). These are strong engineering grounds for
search against **our fixed Watch bots**, with source/checkpoint/config/platform identity held fixed.
They do not prove every new offensive branch, roster or counterfactual is exact.

Each changed hero branch must recompute each opponent's policy on that branch's state, including
its veto, trend memory, boost counters, food and respawn RNG. A frozen network still reacts.
No human player or external slither.io server is covered. Even here, FP32 action ties can differ
across Mac/x86; same-platform paired tests and Mac serving qualification remain necessary
(`docs/STATUS.md:90-96`). For a human/unknown policy, “exact simulator” does not imply an exact
opponent model; that would require a new uncertainty model and a different transfer claim.

**How often is there something worth attacking? There is useful evidence, but no measured B8
counterfactual opportunity rate.** I recovered the older census from Git, without opening its
artifacts. Below, `KILL_CENSUS` means
`10ca76af:docs/research/kill_opportunity_census_2026-10-03.md` (historical blob, not a file at HEAD).
The current index points to it at `docs/research/INDEX.md:249`.

| Recorded quantity | Scripted / frozen / mixed, per H5000 episode | What it does and does not estimate |
|---|---:|---|
| Champion+v8 forced cut-off windows within four frames | 1.18 / 0.45 / 0.25 | Restricted safe action-then-straight lines; 54/75 windows already taken. `KILL_CENSUS:9-10,28-47,78-82`. |
| Optimistic mass-integral benefit of those untaken forced windows | +7.6 / +5.4 / +1.1; pooled about +4.7 | Assumes instantaneous complete harvest retained to horizon; only the enumerated windows, not a global offense upper bound. `KILL_CENSUS:10,81-82,96-104`. |
| Open-loop replay windows | 14.8 / 15.0 / 14.2 | Opponents repeat old moves after hero deviation; the +202/+229/+170 optimistic bounds are not achievable-policy evidence. `KILL_CENSUS:45,84-94`. |
| Current B8 mixed-Watch6 supply of opponent deaths | **8.96/session**, derived | Arena causes sum to 261 deaths; hero causes sum to 37; `(261-37)/25=8.96`. This includes passive/self deaths and repeated small lives, not nine profitable corpse races. `docs/research/t13b_six_snake_roster_2026-10-17_results.md:25,35-36`. |

The old census used champion+v8, restricted lines and simplified geometry (frozen other heads,
ignored growth/head-swap); it explicitly calls its forced count a lower bound on deeper search
(`KILL_CENSUS:3,98-104`). Its “policy fitting would not transfer to serving” warning was appropriate
to that proposal. **T34 changes that premise:** the fixed served bots are available for reactive
simulation. Neither the old forced +4.7 nor the invalid replay +170–229 bounds the value of that
new policy-aware search. Also, the reported 736/88/73 corpse segments are generated corpse mass,
not verified mass consumed by the hero (`KILL_CENSUS:15,47,75`).

My planning prior, explicitly a belief: expect **0–1 additional profitable deliberate-kill
opportunities per H5000 world** from single-action+B8 search; price a three-event sensitivity,
not a claim of three events. Corpse interception has a larger encounter supply (roughly nine
opponent deaths/session in the product sample), but its incremental yield is unknown. I would
allocate effort to harvesting before encirclement because it can exploit existing deaths without
creating one. A census should distinguish total kills, controllable new kills, pre-existing corpses
captured earlier, and mass diverted from opponents; only the paired hero trajectory gives net gain.

Useful arithmetic for comparing with +92: if an intervention adds net retained mass `q` at
frame `t`, and that advantage persists without changing survival, its H5000 contribution is
approximately `q*(5000-t)/5000`. At the midpoint, +92 requires **184 extra retained cells**:
about 7.4 successful 25-cell events, 3.7 50-cell events, or 1.8 100-cell events per world.
One instantaneously harvested 150-pellet corpse at the midpoint contributes at most +75 under
those assumptions; gradual collection contributes less. This is a scenario calculation, not a
global bound: earlier kills, later growth, multiple corpses and prevented deaths alter it.
It shows why “one big kill doubles the snake” is a poor expectation for this capped economy.

**Horizon and compute estimates.** Proposed horizons, not measured distributions:

| Behavior | Lookahead worth testing | Rationale / limitation |
|---|---:|---|
| Already-adjacent cut-off or head-on | 4–16 decisions for contact; longer for payoff | Four is the historical census's scope, not a measured sufficient horizon. A kill certificate says nothing about harvesting. |
| Intercept an existing corpse | 128–256 initially | Travel plus collection; Grid eats at most one pellet and adds one length per snake per decision, including boosted traversal (`src/simd_env/grid_sim.py:725-753`). Collecting 100 pellets therefore takes at least 100 decisions after access. |
| Create a kill and harvest it | **300–500**, with H500 the first fixed test | This reuses T34's horizon machinery while allowing approach, collision and eating. Record time-to-kill and time-to-actual-gain rather than assume both fit. |
| Encirclement / sustained pursuit | 500–1,000+ and multiple deviations | My engineering expectation; a longer first-action+B8 rollout is not a multi-turn planner. The blueprint's ~330 is a discount-horizon design rationale, not an observed kill-time distribution (`docs/ml_redesign_blueprint_2026-07.md:192`). |

At most six first-action branches give `6*H` branch-world steps, before clone/shadow/refill costs.
The **measured** Mac fanout profile is 3.5945 ms/active row-step, three 120-frame worlds at H300;
it was contended and is not a quiet-host Path B timing result
(`research/t34_exit_b8/ENGINE_FANOUT.md:46-60`). Applying that rate gives these **projections**:

| Horizon | Six-branch work | Serial Mac projection | Ideal six-way division, not a forecast |
|---|---:|---:|---:|
| 128 | 768 rows | 2.76 s | 0.46 s + overhead |
| 300 | 1,800 rows | 6.47 s | 1.08 s + overhead |
| 500 | 3,000 rows | 10.78 s | 1.80 s + overhead |
| 1,000 | 6,000 rows | 21.57 s | 3.59 s + overhead |

The older scripted T3 continuation core achieved 1,126.1 rows/s, including clones but excluding
ordinary stepping/startup; at that rate H500 would be 2.66 s. It is an optimistic **different
workload**, not grounds to promise frozen-Watch latency (`docs/research/t3_ceiling_test_2026-10-15.md:129`).
Actual older serial Watch searches took 29.291–62.688 s across five dooms and missed all available
D300 deadlines (`research/t34_shadow/RESULTS.md:174-196`). The v2 ≤2 s target remained unmeasured
in the cited tiny implementation smoke (`research/t34_pathb/VERIFICATION_V2.md:130-144`).
Neither document measures offensive search or the newly adopted N=6 configuration.

At 24 fps one tick is 41.67 ms, four decisions are 0.167 s, and a 500-decision shadow provides
at most 20.83 s of lead. Even the ideal six-way H500 projection cannot react to an already-imminent
four-frame cut-off. It must discover a **future** action while the baseline shadow is still confirmed,
finish before that action, then invalidate/refork after intervention. H500 evaluation horizon is
not automatically 500 decisions of action lead. Deadline misses must fall back to B8 and remain
in the measured effect (`research/t34_pathb/controller_v2.py:137-172`). Dense offensive triggers
also consume the worker that rescues need; the hybrid should give rescue priority, cancel stale
offense, and log this competition. N=6 only helps after independent branch/carry/RNG isolation and
whole-tick contention measurements; division by six is not evidence of a sixfold speedup.

**The smallest decision-bearing queue I would fund next.** This is a proposed new development
lane alongside the already adopted rescue smoke; it does not amend or inspect any running study.
All numbers and rules below must be frozen before new data. Keep mechanics, checkpoint, precision,
opponent rosters, hero slot, terminal semantics and seeds paired. No new trainer or mechanics arm.

| Order and dependency | Work / placement | Proposed hard envelope, including reserve | Decision enabled |
|---|---|---|---|
| 0, independent of rescue outcomes | Provenance/identity fixtures plus actual H500 branch timing; 12 disjoint timing worlds, 4/mix, 2 roots/world; one L4 lane via qualified CLI. Mac identity reference only when capacity permits. | 2 u / 1 runtime-hour; reserve 0.4 u. Mac reference ≤0.5 wall-hour, ≤2 threads, nice 10. | Can the unchanged mechanics and reactive branches be measured honestly? |
| 1, only after 0 valid | Census on **48 fresh B8 s3 H5000 worlds, 16/mix**; at most 8 roots/world, six H500 branches/root; one L4 lane. | 8 u / 4 runtime-hours, reserve 1.6 u; 2M total row-step cap. | Is there enough realized, incremental mass to justify a closed-loop candidate? |
| 2, only after census GO | **96 fresh paired H5000 worlds, 32/mix**, arms B8, frozen rescue controller, rescue+offense; plus **16 fresh paired mixed-Watch6 H5000 sessions** for product diagnostics. L4 unpaced first; a second, disjoint Mac bank of the same sizes only after the candidate earns it. On Mac, pace the two assistants; bare efficacy stays unpaced, with separate matched bare timing probes. | 24 u / 12 L4 runtime-hours, reserve 4.8 u; 10M L4 row-step cap. Mac cap 24 wall-hours / 168 CPU-hours (tick + up to six workers); failure to fit is INCOMPLETE, not fewer worlds. | Does offense add value over rescue, and does that value survive deadlines? |

First-wave allocation is **10 u**, total conditional proposal **34 u**, about 3% of the owner's
~1,145-u input (`QUESTION.md:32-34`), with **6.8 u reserved across envelopes**. Reserve is not
routine working budget: stop at the ordinary 80% allowance unless an operational contingency is
explicitly admitted; never spend it to extend N after an outcome. Stop at the first wall/unit/work
cap, no automatic retry. No fitting, strict bank or release work is included in these ceilings.
Mac envelopes also reserve 20%: stage 2's ordinary allowance is 19.2 wall-hours / 134.4 CPU-hours.
Any future N=6 Mac test needs its own quiet-host admission after the current shared-job constraint
is lifted; it is not permission to exceed this task's two-thread limit. No such test ran here.

Cost basis is the recorded A5 L4 aggregate **377/432 row-steps/s** and 54.4–57.2 worlds/hour,
not training transitions/s (`docs/research/final_run_20261019/CAMPAIGN_LOG.md:59`). Those rates
belong to sparse rescue, so stage 0 must replace them for offense. At 377 rows/s, stage 1's
maximum branches cost `48*8*6*500 = 1,152,000` rows. Budget roughly 0.5M additional source/shadow/
fill rows: 1.65M/377/3600 ≈ **1.22 runtime-hours**, or 2.44 u at the **planning** 2 u/hour rate;
another 0.5 runtime-hour for setup/export adds 1 u. The 8-u envelope tolerates considerable
derating while retaining reserve. Use the actual billed rate at admission. Stage 2's 10M-row
cap costs 7.37 runtime-hours / 14.74 u at the same assumed rate, plus overhead; reject admission
if stage-0 measurements cannot fit the 19.2-u ordinary allowance. Two paced assistants across
112 H5000 worlds/sessions require up to `2*112*5000/24/3600 = 12.96` hours of nominal game time;
the 19.2-hour ordinary Mac allowance also covers bare unpaced efficacy, setup and matched bare
timing probes. Reprice before admission if those measured costs do not fit. These are budgets,
not promised runtimes.

For perspective, six H500 branches at **every** H5000 decision cost 15M rows/world: ~11.1 L4
runtime-hours / 22.1 u per world at 377 rows/s, before overhead. Sparse event detection is essential.
The aggregate K12 L4 rate cannot be used as single-request latency. An A100 does not solve this:
the campaign's measured B8 forward share was only ~8–13%, and A100 was rejected for host-bound
teacher work (`docs/research/final_run_20261019/CAMPAIGN_LOG.md:66`). The tiny CPU-only runtime
delivered 37 rows/s on two cores (`CAMPAIGN_LOG.md:59`); it has no demonstrated cost advantage
for this lane. No RunPod proposal is needed.

Stage 0 acceptance: exact source-state preservation and serial-versus-batched per-step actions,
physical state, carried policy/RNG state and mass; fixtures must include kill, corpse FIFO eviction,
boost burn, contested food, death/respawn and a changed hero action that changes an opponent's
reaction. Compare live/Grid on the actual Watch roster. Fail on any unexplained mismatch.
The simulator already exposes food-class, kills and victim lengths (`src/simd_env/batch_sim.py:251-268`),
but **corpse-class includes boost trails**. Add observer-only provenance to separate dead-body
food from trail food, track deduplication/eviction, and prove identical actions, RNG and world
hashes with the observer off/on. If inspecting legacy v1, death food is classified ambient
(`src/simd_env/grid_sim.py:881-896`); zero corpse-class consumption would then be misleading.

Stage 1 roots: four uniformly time-stratified roots plus up to four distinct event windows per
world, selected prospectively by a fixed reservoir, with 64-decision window deduplication.
Log every candidate event, sampling probability and count; do not call a capped sample the
complete opportunity census. Every selected root evaluates the same six actions to H500
(illegal actions marked, not executed), while H128/H300 prefixes are diagnostics. Report events
per world and per 1,000 alive decisions, length/time strata, new kills, actual food provenance,
boost loss, delta_A500 and any later hero death. Cluster uncertainty by world, not roots.
Selection is conditional on B8 states; the subsequent hybrid will visit different states.

Proposed stage-1 order: INVALID/INCOMPLETE for parity, accounting, missing worlds or caps;
KILL this candidate family if no valid positive survivor branch exists; GO to stage 2 if at
least 8/48 worlds, including at least 4 frozen/mixed worlds, contain a surviving `delta_A500>=5`
branch and an independently timed workload fits the next cap; otherwise INCONCLUSIVE, no extension.
These are cheap mechanism-screen thresholds, not population efficacy claims. They do not
establish the frequency of all possible multi-action opportunities or predict a +92 gain.

Stage 2 freezes the hybrid's trigger schedule, eight-offense-searches/world cap, priority/fallback
behavior, H500 ranker and worker count before data. Score equal-mix paired-world H5000 mass,
and session-clustered product metrics; calculate incremental hybrid-minus-rescue as well as both
versus B8. Predeclare the contrast family and Holm adjustment; use one fixed-N look. Proposed
order: INVALID/INCOMPLETE first; KILL on unconditional faults, survival-guard failure, or incremental
mass UB90≤0; GO only if incremental mass mean≥20 and multiplicity-adjusted LB90>0, and the hybrid
meets all current skill/product/timing bars versus B8; otherwise INCONCLUSIVE with no automatic
extension. With 16 sessions the product interval may be wide: that is a valid inconclusive result.
The +20 incremental requirement is my proposed resource threshold, not an existing contract bar.
The relevant units, survival guards and multiplicity rules are in
`docs/research/final_run_20261019/P7_STATISTICS.md:10-39`; the current boost floor is half the paired
bare share, ceiling 10%, per `docs/research/final_run_20261019/CAMPAIGN_LOG.md:75-81`.
For unpaced admission, apply the stated mass/survival criteria and report product diagnostics;
timing and paced product efficacy remain untested. Full GO requires the second, disjoint Mac
development bank and all bars under pacing. Only fresh Mac strict evidence, serving qualification
and owner approval can change the default. Preserve receipts and publish every failed arm.

**What changes my mind.** Fresh gains dominated by actual captured corpse mass, surviving the
paired rescue comparison and Mac deadlines, would move offense onto the critical path. A hybrid
increment of +40 or more across mixes would be especially persuasive; directly demonstrating
>+92 on a comparable fresh bank would settle the historical comparison, but the old +92 must
not become a new post-hoc gate. Conversely, fewer than one useful missed event/world, tiny victim
corpses, displaced ambient intake, frequent eviction, or gains that vanish when opponents react
would leave rescue first. Zero single-action positives would stop this variant, not all offense.

I would defer encirclement beams, learned terminal values, reward/kill-scale changes, richer
corpse caps and distillation until a directly acting controller shows the mechanism. Raising the
corpse cap may make a more interesting game, but it creates a new product/benchmark requiring a
fresh B8 comparison. Distillation is a separate generalization problem: T3 learned its label bank
yet lost gameplay mass (`docs/research/t3_diagnostics_2026-10-16.md:17-34`;
`docs/research/t3_screen_result_2026-10-15.md:29-37`). The original
[Expert Iteration paper](https://arxiv.org/abs/1705.08439) also separates planning from learning
to generalize those plans; it supplies no Snake-specific efficacy or cost evidence.

My judgment is therefore **defense for the fastest credible served improvement, offense for a
bounded attempt at additional upside**, with corpse interception before deliberate hunting.
The strongest new reason to try it is reactive access to the actual served bots. The strongest
reason not to promise a larger jump is the combination of limited incremental harvest, finite
corpse supply and deadline cost. The question is experimentally open; neither the old hunting
census nor the defensive +92 result closes it.
