# Advice: ideas beyond the exhausted levers (2026-10-17)

Advisory note (cloud advisor), requested by the local session. Not gate evidence.
Inputs: `t5b_risk_and_fuse_2026-10-16.md`, `t6_screen_result_2026-10-17.md`,
`t3_ceiling_test_2026-10-15.md`, `t1_commitment_study_2026-10-15.md`, `ego_raster_b.py`,
`eval_engine.py`, `src/scripts/ensemble_eval.py`.

## The constraint every failed arm shared

The dense ceiling changed 75 of 301,589 decisions and lifted survival 38/64 to 54/64. Every
training lever (reward, Huber, margin labels, two-sided, return, restarts) asks the network to
localise those 0.02% of decisions from the raster, and every probe (static 0.73-0.75, fuse
0.77) says the current representation cannot. Ideas that re-use the same inputs and the same
single network are therefore low value whatever the loss. The different levers are: (1) settle
whether the raster or the network is the limit, (2) change what the policy *is* at serving
time without a hand rule, (3) add the one competitive-snake input nobody here has tried.

## Ranked by expected value per Mac-hour

**1. Raw-raster probe (about 1 hour, decisive).** Train a small CNN from scratch on the T5b bank
(19,976 states, 103,264 branches, death-within-300 labels) with the packed ego2s-b
observation plus a one-hot action as input, grouped world folds as in T5. This is the missing
rung between "scalars" and "frozen fuse layer". AUC >= 0.85: the raster carries the signal and
B8's trunk/learning drops it, which reopens an auxiliary death head (your d) and ensembles.
AUC about 0.77: the raster lacks it; only new inputs (3, 4) or search (5) can help, and no
further label or loss arm should run. Either answer kills half the remaining list.

**2. Five-final ensemble, mean-Q and min-Q (zero training, 1-2 CPU-hours).** The fatal action
wins by a median Q-gap of 0.45 in states that are rare for every seed, so seeds will disagree
there and agree on the safe majority. Min-Q over members is pessimism exactly where
training data is thin, which is what every caution lever tried to impose globally. The eval
script exists. Test: both variants on the round-52 96 scripted worlds against the single
finals; look at scripted survival and mass. An ensemble is a policy, not a veto; it would need
its own pre-registration as a candidate, but the screen is nearly free.

**3. Arrival-race ("Voronoi") inputs (half a day including a probe).** Tron and Battlesnake
winners all use territory: cells the hero reaches before any opponent. The raster has a
time-aware `reach_time` flood for the hero only, and region size counts cells opponents will
take first. Add per-action scalars: Voronoi area after the action (log, uncapped) and the
fraction of the action's region contested. Opponent sealers were 13 of 46 late deaths and
opponent heads sat on the component boundary in a third, which region size cannot express.
Cheapest test: compute them on the restored T5 states and rerun the T5 GBM; a jump above the
0.75 feature baseline earns one training arm, no jump kills it in minutes.

**4. Deterministic-opponent planes (your b).** Legitimate as an observation if computed only
from state the hero sees (positions, headings, food) with the hero held fixed; that is
opponent modelling, and ego2s-b already does it for k=1 (`enemy_next`). It is not a veto,
because it changes the input, not the action. It is, however, bot-specific: in frozen and
mixed mixes the greedy_food forecast is wrong and Tier 1 tests those mixes. Prefer idea 3, which
is policy-agnostic; add an exact k<=8 greedy_food arrival plane only as a second probe column.
Higher scripted density in training is a weaker version of the same bet; its proxy (deaths per
1k at L>=800, 0.20 fresh) is cheap but the restart result shows that proxy moving without the
gate moving.

**5. Search as policy, gated by near-ties (your e).** The dense procedure cost 0.45 s per batch
and triggered on 20% of eligible decisions; most changed actions had a near-tie (87% of
LAD gaps < 2). Offline test first: rerun the dense replay gating on top-2 Q-gap < 0.5 and
count triggers and survival against 54/64. If near-ties are a few percent of decisions, a
self-rollout of the hero's own policy is an expert-iteration step, not a hand rule: no
flood-fill, no threshold on geometry, the policy consults itself. For Watch mode with
deterministic bots the owner may accept it; cost is the trigger rate times 0.45 s, and because
H_LAD median is about 88 decisions, a result that arrives a few ticks late still re-steers.
Distilling its choices back (DAgger on on-policy states, not death-window states) only works if
idea 1 says the raster can carry it.

**6. Strategy changes (your c): do not spend hours here.** Controlled shrinking costs the mass
efficacy clause directly and Huber already showed survival does not come back for mass.
Tail-following is only guaranteed in single-player snake; with respawning opponents and corpse
pellets it is not, and T1's R3 found tail or boundary-TTL contrast in 11% of LADs. Wall-hugging
shrinks Voronoi area. The only cheap check worth doing is already in T1's rows.

**7. Auxiliary death head (your d).** Bounded by its inputs; with current inputs it reproduces
the 0.77 probe inside the network. Run it only after 1 or 3 returns a positive.

**8. Literature (your f).** Battlesnake: minimax or MCTS over 2-4 ply plus flood fill and
Voronoi, i.e. ideas 3 and 5 combined. Tron: Voronoi plus alpha-beta. Shielding literature is
the veto. Hamiltonian-cycle solvers do not transfer. Nothing there is new beyond 3 and 5.

## Order

Run 1 and 3's probe in parallel (two CPU processes, under the six-process cap, do not touch T7),
then 2 as the next free evaluation slot. Decide 4, 5, 7 from those three readings. Phase R for
B8 decides the serving question regardless; none of this should delay or amend it.
