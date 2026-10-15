# Round 1, mechanics and game-theory memo (role: mechanics)

Read-only design discussion, 2026-10-14. Sources: the Mac-session report, digest blocks 0/1/2/7, and the
repository code at `/home/user/snake-dqn` (Oct 8 state). Every rule below is cited from code or the
parity spec, not from memory. Where I infer, I say so. This revision corrects one claim of the first
draft (section 2a: own-body closure CAN produce the 31-decision signature) and the composition of the
mixed mix (3 frozen + 2 random_safe).

## 0. Summary

B8's failing clause is a late-game, scripted-world *bottleneck* problem, not a generic "self-enclosure"
problem. A 1,000-cell body is a wall long enough to partition the 145x83 arena on its own; by frame
~2,900 B8 is always living on one side of a partition it made. The census's "trap formed 31 decisions
before death" is the signature of a bottleneck being sealed: the one-step tail-aware region count sees
the union of both sides (>= L) right up to the frame the neck closes, then the head's side is 30-60
cells and the snake fills it in ~31 frames. The sealer is the hero's own body (classic neck crossing),
a live greedy snake plugging the gap, or both; the fatal frame reads "self" either way because the last
free cell is bounded by own body. Which one dominates is measurable and not yet measured.

The scripted mix makes bottlenecks far more common than the other mixes because five `greedy_food`
snakes (i) steer by straight-line distance to the nearest pellet with no body avoidance, so they press
against and slide along the hero's body, (ii) die against the hero's head path ~20 times per episode
(v8 census: 22.2 kills scripted vs 3.5 frozen) and drop their whole body as a corpse line starting at
the hero's head, (iii) enter any region of >= min(L_greedy,160) cells, so they walk into the hero's
pockets to eat those corpses, and (iv) respawn within two frames anywhere, including inside a pocket.
Corpse lines parallel to the hero's body bait a mass-maximiser into 1-wide lanes between its own
folds; that is where necks are.

Three consequences. First, the champion's "safe plateau at 800-870" is not a strategy: it grows at
~0.17 cells/frame net of boost burn and crosses 800 only around frame 4,000+; its ~7,000 decisions
there predict ~1.2 deaths at B8's hazard, so zero observed is uninformative. Second, by the H5000
mass integral B8's growth is roughly break-even against its own late hazard (the measured hazard
slope is below the 2/(L*tau) threshold at which growing stops paying), so the objective is indifferent
to a length cap; only the guard clause cares. Third, the guard compares the candidate's alive-frame
survival with a snake that never lives in the long-snake regime, on seed 0's 32-world scripted bank
with a 1e-9 tolerance; it measures "did you stay small", not "are you safe for your size".

Levers that fit the mechanism, in order: exact-replay commitment analysis on the 49 deaths (scripted
worlds are deterministic given the hero's actions, so counterfactuals are exact and cheap), then
late-game scripted snapshot training with rollout labels; tail-topology inputs (is my tail in this
action's region, min boundary TTL, corridor length ahead); boost competence; corpse-bait discipline;
a growth governor only if the hazard rises with L rather than with frame.

## 1. Mechanics that matter (from code)

Arena: `configs/promotion_mechanics_v2.yaml` 1450 x 830 px, 6 snakes, mechanics v2, frame_rate 1,
max_food 300, initial_food 250; segment 10 px, so **145 x 83 = 12,035 cells**. Food and respawn
positions come from `find_empty_position` with margin = wall_thickness = 10 px (spec 5.2), so the
border ring never holds food or a respawn. A lap of the perimeter is 2*(145+83)-4 = **452 cells**.

Step order per frame (`BatchSim.step`, mirroring `GameState.update`): maintain ambient food ->
respawn dead -> all snakes move (boost second step) -> trail pellets -> food consumption -> collisions
(wall > self > head-on > body) -> corpse drop -> reward.

Movement (`_move_all`): 1 cell per frame; boost adds a second cell in the same frame if `length >= 5`
(`min_boost_length`). `boost_frames` increments once per boosted frame; every 3 boosted frames
(`boost_length_cost_frames`) `length -= 1` and the vacated tail cell becomes a corpse pellet
(`_settle_tails`, `_drop_trail_pellets`). Growth: eating adds 1 to `length`; the tail simply does not
pop that frame. So a snake eating k pellets holds its tail still for k frames.

Collision (`_resolve_env`): self = traversed head cell(s) in own `segments[3:]`. Head-on under v2:
the snake with `length >= 1.15 x other` survives and gets the kill (`HEADON_SIZE_RATIO`); otherwise
both die. Body hit kills the mover and credits the owner. A dead snake drops its **whole body** as
corpse pellets (`CORPSE_DROP_FRACTION_V2 = 1.0`).

Food economy: ambient topped up to 300 every frame; corpse-class pellets (kill corpses, boost trails)
are cap-exempt but bounded at `corpse_food_cap(300) = 150`, oldest evicted FIFO
(`mechanics_constants.py`). Total pellets <= 450 (~3.7% of cells). v8 census: nearest pellet mean 3.6
cells, ~420 pellets on board. Food is never scarce anywhere.

Respawn: `die()` sets `respawn_timer = frame_rate = 1`; `_respawn_dead` decrements then respawns at a
uniformly random non-snake cell, length 1, heading right (`CARDINAL[1]`). Opponents are back within
two frames; the hero is terminal (`hero_terminal=True` in the Watch profile).

Scripted opponents (`scripted_snake.py`): `greedy_food` takes the non-boost actions that are not
immediately fatal (AISnake's one-step simulation; **other snakes are static obstacles within the
frame**, so it does not anticipate where the hero's head will be), drops those whose reachable free
space (capped flood, cap `min(160, max(32, 2L))`, need `min(L, cap)`) is below its length *when an
alternative exists*, then minimises Euclidean distance from its next head cell to the nearest pellet,
ties straight > left > right. It never boosts. `random_safe` picks uniformly among non-fatal moves,
seeded. Both are deterministic given the world and the hero's actions.

Mixes (`tournament_eval.py`): scripted = 5 x greedy_food; frozen = round-robin over the 61-D pool
(champion_a5, best_apex_fs, best_apex_stage1_fs, best_apex_pre_fs); mixed = slots 0,2,4 from the frozen
pool and slots 1,3 random_safe, i.e. **3 frozen + 2 random walkers**.

Metric (`metrics.py`): `mass_integral` = sum over alive frames of logical length / 5000;
`survival_fraction` = alive frames / 5000. Phase R's `surv5` is this **alive-frame** fraction. The
census's 0.531 / 0.656 / 0.688 are exact k/32 fractions, i.e. **episode-level** survival. The
report's "target 0.64" subtracts the margin from the episode number; the clause is scored on the
alive-frame number, which is always >= the episode number.

Guard clause (`example_rules._common_clauses`): `c6_guard_scripted_survival`: hero minus
champion_a5+v8 `surv5` on the scripted mix `>= -0.05`, `SURVIVAL_TOLERANCE = 1e-9`, guard cell on
**seed 0's bank only** (`"seeds": [seeds[0]]`), protective.

What the student sees (`ego_raster.py`, `ego_raster_b.py`): local 31 x 31 at 1 cell/px with the head at
row 23, col 15 (**23 ahead, 7 behind, 15 to each side**): own TTL `clip(L-k, 0, 255)`, enemy TTL,
enemy head with size ratio, food (128 ambient / 255 corpse), wall, time-aware reach plane with K=24,
enemy next-cell plane. Global 37 x 37 at **8 x 8 cells per px** (head-centred; the arena is ~19 x 11
blocks of it): own segment count, enemy segment count, pellet count, outside. Scalars: log L and
L/1000 (uncapped), pending growth, boost availability and phase, hunger, four wall distances, alive
fraction, nearest and second-nearest enemy geometry, and **uncapped tail-aware free-region size for
each of the three next cells** (R/L capped at 4, and log R; a cell is free if its TTL <= 1). B8 adds 30
v8 per-action safety features whose list lives only on the Mac. The network flattens both towers into
linear layers, so its receptive field is the whole window; the limits are the window, the TTL clamp,
and the 8x8 blocks.

What the teacher saw (`vector61_featurizer.py`): length `min(L/150, 1)`, free space capped at 160.
The champion and frp3-s12 cannot tell L=300 from L=1,200 and cannot see a region bigger than 160. The
v8 veto's hard rule is `need = min(L, 160)` reachable cells (`free_space_threshold`); v7's re-rank area
cap is `min(max(32, 2L), 4096, open cells)`.

Reward in the repo (`_compute_rewards`): `gamma*Phi(L') - Phi(L)` with `Phi = L/10`, `Phi(death)=0`,
-3.0 on death, +0.3 x victim length per kill. B8's trainer uses `--reward mass --mass-kappa 1.0
--kill-scale 0` (Mac only); the literature track's note "death at L=1000 costs ~100 Q units vs
per-step 0.2" is consistent with a per-frame reward of ~L/5000 under gamma 0.998 (value 0.2/(1-0.998)
= 100). Exact form is a context question (section 5).

## 2. Answers

### 2a. Why bottleneck deaths at L >= 800 concentrate in the scripted mix

First, a correction to my earlier draft. The census flags a trap when the best one-step tail-aware
region is below L; at death lengths that is a region under ~1,000 cells. The hero consumes one cell per
frame, so it cannot fill 1,000 cells in 31 frames. I previously concluded the closure had to be an
opponent. That is wrong: the one-step region count is the size of the connected component containing
the next cell. If the head is crossing a neck between two sub-regions A and B with |A|+|B| >= L but
|B| < L, the count reads |A|+|B| until the body seals the neck, then reads |B|. Crossing a 1-cell neck
takes one frame. So **a 31-decision trap after a non-tight state is exactly what own-body neck sealing
looks like**, and it is equally what a greedy snake plugging the neck looks like. The census does not
record who sealed it.

What the scripted mix does to make necks common:

1. **Body hugging.** `greedy_food` steers by straight-line distance to the nearest pellet with no path
   planning and no body avoidance. A greedy snake on the far side of the hero's body from a pellet
   presses toward the body; its one-step veto strips the crash move and the survivors slide it along
   the body at distance 1. A greedy body lying parallel to the hero's body at distance 1 is a 1-wide
   lane, and its two ends are necks. Frozen vector61 opponents carry danger and free-space features and
   were trained with body-collision penalties; they keep distance.
2. **The kill pump.** Greedy snakes treat the hero as static within the frame, so they step into the
   cell the hero's head also enters; v2 head-on gives the kill to the snake >= 1.15x longer. Result:
   22.2 kills per episode for the champion in scripted vs 3.5 frozen (v8 census). Each kill drops the
   victim's whole body as a corpse line that **starts at the hero's head** and trails back along the
   victim's path, which was shaped by its attraction to pellets near the hero. A mass-rewarded hero
   follows the line; the line runs beside the hero's own body as often as not; following it lays a new
   body lane parallel to the old one at distance 1. Positive feedback: corpses near the body attract
   the next greedy to the body, which dies there.
3. **Pocket intrusion.** The greedy free-space veto only bites when an alternative exists and only up
   to `min(L_greedy, 160)`; a freshly respawned length-1 greedy has `need = 1`. They enter any pocket
   the hero has made if a pellet is in it, and the corpses from (2) are. A live greedy inside a 500-cell
   pocket is a moving wall that can seal the neck from the inside or cut the pocket in two.
4. **Respawn pollution.** Uniform over non-snake cells within two frames of death. With 20-40 opponent
   deaths per episode and the hero's region being 5-10% of free space, a greedy materialising inside
   the hero's region is routine.
5. **Fastest feeding.** The scripted mix feeds the hero most (champion survivors' final mass 882
   scripted vs 787 frozen). B8 reaches 800 earliest there, so its exposure to the long-snake regime is
   longest exactly where the necks are.

Why mixed (0.87) beats frozen (0.65) for B8: mixed has 3 frozen + 2 random_safe. Random walkers die
often (5.6 kills/episode in the v8 census) but wander away from the hero rather than hugging it; frozen
opponents are long, competent and take space, and they are the ones that kill the hero by body (7 of
75 B8 deaths were enemy body). Neither pumps corpses beside the hero's body the way greedy does.

### 2b. Geometry of a safe 1,000-cell snake

- A 1,000-cell body is 8.3% of the arena. The longest simple loop (perimeter) is 452 cells, so the
  body must fold at least twice. Any own-body path from wall to wall (>= 83 cells) partitions the arena;
  a 1,000-cell body partitions it essentially always. **The long-snake game is about which side of
  your own partition you are on**, and the census's "tight" flag (region < L) fires whenever the hero
  is on a side smaller than ~32 x 32.
- **Wall-hugging**: 2.2 perimeter laps, three concentric lanes (452 + 444 + 436 = 1,332 cells for
  three full lanes). The head ends on the innermost lane with the ~10,700-cell interior as its space.
  Inner lanes are walled by older outer lanes that open from the tail at 1 - g cells/frame (g = intake
  per frame). Safe as long as the head never turns back into its own lanes.
- **Serpentine**: lanes of 100-140 cells, 7-10 adjacent, a band 140 x 8. The hazard is the turn-back
  at a lane end: the new lane and the band form a 1-wide corridor whose far end must be open when the
  head gets there. A 30-cell corridor is a 30-frame commitment with the far end outside the 23-cell
  forward view.
- **Coil**: an inward spiral is bounded by the most recent body (TTL 700-1,000 frames) and is a death
  sentence at any pocket below L. An outward spiral (tail inside) is safe. The direction of the coil,
  not its tightness, decides.
- **Tail-chasing**: a Hamiltonian cycle on a 40 x 25 block holds 1,000 cells with zero waste and is
  perfectly safe at zero growth. Every pellet eaten needs one cell of slack (the tail does not move that
  frame), so at B8's ~0.28 pellets/frame a closed cycle needs a corridor into fresh space every few
  frames. Realistic long-snake play is "forage in the open, fall back on the tail when crowded".
- **The topological invariant.** A region bounded by own body is sustainable only if the region grows
  as fast as the hero fills it, i.e. if the **own tail lies on (or inside) the region's boundary**. Then
  the boundary vacates at ~1 cell/frame and a pocket of any size is safe. A pocket bounded by mid-body
  (TTL 300-900) does not grow for hundreds of frames and must be bigger than the time the hero needs
  to leave it. "Is my tail in this action's region" and "min TTL on this region's boundary" are the two
  numbers that define safety for a long snake, and neither is in the student's inputs (section 2f).
- **TTL arithmetic.** Segment k behind the head vacates in (L - k)/(1 - g) frames. A pocket of R free
  cells bounded by own body with minimum boundary TTL tau survives only if the hero can fill ~R cells
  (fill efficiency well under 1 in practice) for tau frames, or the pocket contains near-tail body.
- **"Tight" in cells.** Region < L at L = 1,000 means under ~32 x 32, bigger than the 23-ahead window,
  so a "tight" state is not locally visible as a wall. The pockets that kill in 31 decisions are
  30-60 free cells, fully inside the window. The gap between those two sizes is the neck crossing.
- **Closure mechanism.** Mixed. Own body supplies most of every boundary; the last cut is own body
  sealing a neck, a greedy body plugging it, or a wall. Corpses never block; they bait.

### 2c. Why champion_a5 + v8 "plateaus" at 800-870

Growth rate times horizon, not strategy:

- v8 census: pooled mean mass at frames 1,000 / 2,000 / 3,000 / 4,000 / 5,000 = 178 / 334 / 426 /
  485 / 509 (dead = 0); survivors end at ~825 (scripted 882) and still grow at a median 0.15/frame in
  frames 2,500-5,000. The champion crosses 800 around frame 4,000-4,500 and the clock stops at 5,000.
  "Median max mass 824" is where the episode ends.
- Intake ~0.19-0.2 pellets/frame, boost burn ~74 cells per episode (7.2-7.7% boost frames), net
  ~0.17/frame; B8 grows at ~0.28/frame with almost no burn.
- The champion cannot be length-aware: `min(L/150, 1)` saturates and free space caps at 160. Whatever
  it does at 800 it also does at 300. The v8 rule `need = min(L, 160)` is a constant 160-cell test for
  any snake longer than 160.
- "0 deaths at 800+": the champion reaches 800 in 20 of 32 episodes but spends 5.6% of ~128k
  decisions there, ~7,000 decisions. At B8's 0.177/1,000 the expectation is ~1.2 deaths; P(0) ~ 0.3.
  The incumbent (25% of decisions, 2 deaths, ~0.06/1,000) is weak evidence of being somewhat safer per
  decision up there.
- Cost: 369 vs 571 mass. B8's late deaths cost it roughly 65-70 mass (2d), so the champion's
  "safety" is bought almost entirely with size, and it is not demonstrably safer for its size.

### 2d. What the H5000 mass integral rewards

- Being alive at length L at frame t is worth L (T - t)/T over the rest of the episode (T = 5,000).
  A death at (t ~ 3,900, L ~ 1,000) costs ~220 mass; 49 late deaths over 160 episodes cost ~67 mass;
  the 26 early deaths ~40-50. Eliminating the late ones lifts B8 from ~571 to ~640.
- Growth 800 -> ~1,200 over frames 2,900-5,000 adds ~80 mass if the snake lives. B8's hazard above
  800 (0.177/1,000 decisions over ~2,100 frames) is ~31% cumulative risk at ~220 each, ~68 expected
  loss. **Growth is roughly break-even against B8's own late hazard**, so a plateau at 800-900 is worth
  little under the mass objective unless it removes most of the hazard.
- Marginal rule: growing one more cell at frame t pays iff 1 > h'(L) * L * tau / 2 (tau = T - t), i.e.
  h'(L) < 2/(L tau). At L = 1,000, tau = 2,000: h' < 1e-6 per frame per cell. B8's measured hazard rises
  from 0.068 to 0.177 per 1,000 decisions over ~400 cells, h' ~ 2.7e-7. By the objective, B8 should
  keep growing; what it should not do is die in necks. If the hazard is a function of frame (late-game
  crowding) rather than of L, a length cap buys nothing at all.
- Per-decision hazards (census): B8 below 800 0.068/1,000; champion overall ~0.08-0.09/1,000; B8 above
  800 0.177/1,000. B8 is safer than the champion when both are short.
- Answer: B8 is not over-growing relative to the objective; the neck handling is wrong. The objective
  already prices those deaths (~67 mass), so a learner that could see and sample them has the
  incentive. Note for my partner: gamma 0.998 compresses the 2,100 remaining frames to an effective
  ~500, so training under-weights a late death by ~4x relative to the eval integral, but growth is
  compressed the same way; the ratio that drives the policy is roughly preserved.

### 2e. Boost

Buys for a long snake: (i) 2 cells/frame against greedy snakes that never boost, so it wins every race
to a neck it can see; (ii) the whole body flows at 2 cells/frame, halving the time until a near-tail
stretch opens; (iii) burn releases one cell every 3 frames at the tail as a pellet, which helps a
pocket only if the tail bounds it. Costs: 1 cell per 3 boosted frames (0.1% of a 1,000-cell body), a
two-cell commitment with both cells collision-checked (the veto-era trap census found 57% of self
deaths were boosts into 1-24 cell pockets), and a pellet trail at the tail that attracts greedy snakes
behind the hero (possibly protective).

Economics: a 30-frame burst covers 60 cells for 10 cells of body, ~4 mass at frame 3,000. The champion
pays ~74 cells/episode. B8 at 0.1-0.24% has learned that boost is a pure mass cost under a mass
reward, which it is in every state except the few where it saves the episode. M2b's diagnosis
(`redesign_scope` section on `boost_diagnosis.py`): teacher boost labels 1.42% of decisions, student
boost agreement 17%; the deficit started in distillation and RL had no reason to fix it. Under-boosting
plausibly contributes on two counts: losing the race to a neck against a 1-cell/frame greedy, and
leaving the body at full length in a pocket where 31 frames of burn would buy ~10 cells. But only 8% of
B8's deaths involved a boost and no death census has tested "boost at the commitment decision"; that
counterfactual is exact under replay and should be run before investing.

### 2f. Information the student cannot see

- Local window: 23 ahead, 7 behind, 15 to the sides at full resolution. The far end of a 30-cell
  corridor is outside it on entry. Own TTL is clamped at 255, so at L = 1,000 the first ~745 segments
  all read 255: the plane cannot tell "opens in 300 frames" from "opens in 900".
- Global stack: 8 x 8 blocks with segment counts. Two own lanes in one block read "16 cells" whether
  they touch or leave a 1-wide gap; connectivity and gap width are not recoverable from counts. No
  enemy-head or heading channel globally; only the two nearest enemies appear in the scalars. With five
  churning greedy snakes the one about to plug a neck 30 cells away may be the third nearest.
- Region scalars are static one-step snapshots: component size now, not after the body seals the neck
  the head is entering, not after the greedy 5 cells from the gap arrives, and not whether the own tail
  is in the component or when its boundary opens. **Tail-in-region and min boundary TTL**, the two
  numbers that define long-snake safety (2b), are absent, and the tail itself is invisible once >15
  cells away.
- A 1-cell corridor (own body one side, greedy body the other, 20 cells long) is visible once inside
  the window, with enemy TTL and the K=24 reach plane. What is not visible is whether the far end is
  open and whether the corpse line it is following ends against its own body.
- The decisive unknown is the commitment horizon, not the onset horizon. The literature track's "41 of
  64 trap onsets unavoidable" says the fatal choice was earlier than the tight flag. Because scripted
  worlds are deterministic given the hero's actions, the earliest decision with a surviving alternative
  can be found by exact backward replay; only then can we say whether the distinguishing information
  was in the inputs. The M7 AUC study (death-within-H over all L >= 800 states) is weak evidence
  against hand features; a policy input earns its keep at the commitment decision, not on average.

### 2g. Levers, ranked (type in brackets)

1. **Exact-replay commitment study, then late-game scripted training data with rollout labels**
   [learned behaviour; data and labels]. Deterministic opponents make 300-frame counterfactual
   rollouts exact and cheap in GridBatchSim. For each of the 49 deaths find the commitment decision
   (earliest with an action that survives >= 300 frames), label it, and build the training
   distribution from late scripted snapshots (frame >= 2,500, L >= 700, corpse pool at cap) where the
   hero actually dies. Targets the actual mechanism (necks in crowded late worlds) and fixes the
   starvation of that regime that the literature track found.
2. **Tail-topology inputs** [input change]: per action, whether the own tail lies in the next cell's
   region; min and mean TTL of own cells on that region's boundary (scaled /1024); straight-ahead free
   run length (cells until blocked, capped 64); and a global own-TTL plane (min TTL per 8x8 block).
   Cheapest as 6-12 scalars appended to ego2s-b (a 20M run per seed at ~1.4 h on the Mac). Take it if
   the commitment study shows the fatal and surviving actions differ on these numbers.
3. **Boost competence** [learned behaviour]. Rollout labels will contain boost escapes whenever one
   exists; add them to the anchor stream. Go test: fraction of the 49 deaths where a boost at the
   commitment decision survives under exact replay. If < 20%, drop it.
4. **Corpse-bait discipline** [objective change]: lower mass credit for corpse pellets eaten within 2
   cells of own body when L > 600, or a penalty for entering a region whose boundary is >50% own
   mid-body. Conditional on the data showing corpse-following at trap formation.
5. **Growth governor** [objective change if a reward cap on L; veto in spirit if a boost rule]. Only
   if hazard rises with L in the bucket table; costs ~45 mass per surviving episode for a cap at 900.
6. **Hand-coded region or boost rules** [veto in spirit]: excluded. The shadow v8 agreed at all 75
   fatal decisions; anything deeper is a planner.

## 3. The guard clause

Survival in this game is "frames alive out of 5,000". For a long snake it is a different skill from
survival for a short one: every cell of body is a cell of wall that lives L frames and partitions the
arena, greedy snakes hug it, and the kills it causes feed the intruders. champion_a5 + v8 never lives
in that regime; it reaches 800 with a few hundred frames left. The clause therefore asks the candidate
to match the not-dying rate of a snake that stays small, on the one mix where the hero's size is most
dangerous. It cannot be satisfied by a snake that grows to 1,200 unless it also solves a problem the
champion never faced, so it functions as a growth cap, not a safety floor.

The mass integral already prices death by when it happens and how big the snake was; the alive-frame
survival clause is the same integral with L = 1 and a short-snake yardstick, so it double-counts
survival with the wrong reference. A floor that measures safety-for-size would be length-matched
hazard per decision (B8 below 800 is 0.068/1,000 vs the champion's ~0.085: it passes), or survival
against the served agent (passed, -0.003) plus an absolute catastrophic floor.

Two caveats. In Watch mode a user sees one episode, and a hero dead at frame 3,500 is a visible
failure whatever its mass; if the product goal is "rarely dies", the clause encodes that and the owner
should say so explicitly, in which case the right fix is a learned tail-topology skill, not a smaller
snake. And the clause as run is seed 0's 32 scripted worlds with a 1e-9 tolerance; paired survival SD
~0.41 gives SE ~0.07, a wide instrument for a 0.05 margin, and the report's 0.64 target mixes the
census's episode-level survival with Phase R's alive-frame metric.

## 4. Disagreements with the current plan

- "Champion plateau = safe strategy to learn from": it is slow growth plus the horizon; zero deaths in
  ~7,000 decisions is uninformative. Imitating it means staying small.
- "Self-enclosure, fix the hero's geometry with a tightness cost or death penalty": the mechanism is
  neck sealing in a crowded late world; the hero's one-step region inputs cannot see a neck until it is
  sealed, so penalising tight states or deaths harder gives the learner no new way to tell the fatal
  action from the safe one (M6 confirmed: null).
- "Fix training at L >= 800": agreed, but the regime is the late scripted world (corpse pool at cap,
  churned greedy snakes, intruders in pockets), not length alone; stratifying by L in mixed worlds
  under-samples it.
- "Hand features add nothing (M7)": AUC for death-within-H is the wrong test for a policy input. The
  right test is whether a feature separates the fatal action from the surviving alternative at the
  commitment decision, which exact replay makes possible. Also, the M7 families did not include
  tail-in-region as a per-action bit as far as the digests show.
- Growth governor as a boost rule is a veto in spirit; as a reward cap it is legitimate but only pays
  if hazard rises with L, which nobody has measured.
- No serving-time planner: agreed. But against the scripted mix, training-time rollout labels are
  exact, so the "planner as teacher" is unusually cheap and faithful here.

## 5. Context needed before committing (where it lives)

1. B8 and incumbent hazard per 1,000 decisions by length bucket (600-800, 800-900, 900-1,000,
   1,000-1,100, 1,100+) crossed with frame bucket (2,000-3,000, 3,000-4,000, 4,000-5,000), scripted
   mix. Decides growth governor vs late-game lever. [mac artifacts: b8-death-census per-decision
   captures]
2. For the 49 L >= 800 deaths: region-size time series over the last 150 decisions; the owner of the
   cell that sealed the neck (own body / opponent body / wall); whether a live opponent head was inside
   the hero's region at trap formation; whether the own tail was in the region. [new measurement on
   census worlds]
3. Commitment horizon by exact backward replay: earliest decision with an action that survives >= 300
   frames, per death; distance from head to the eventual closure point at that decision; whether the
   surviving action was a boost. [new measurement; GridBatchSim with scripted opponents is
   deterministic]
4. Body shape at death: bounding box, fraction of body cells adjacent to own body (parallel lanes),
   head-to-own-body distance histogram in the last 100 decisions, coil direction, min boundary TTL of
   the final pocket. [new measurement]
5. Corpse share of B8's intake at L >= 800 and head-to-own-body distance while eating corpse; kills per
   episode B8 vs champion in scripted. Decides the corpse-bait lever. [new measurement via
   `_last_corpse_food_ate`]
6. Boost usage by length bucket and tight flag for champion, incumbent, B8. [new measurement]
7. Frame-of-death distribution for the 49 late deaths; alive-frame vs episode survival on the Phase R
   guard bank for B8_s0 and the champion. Settles which metric 0.64 refers to. [mac artifacts: Phase R
   look-2 records]
8. The exact "mass" reward in train_m3c.py: per-frame L/5000 vs Delta-L vs potential form, death
   value, whether boost burn is charged, whether corpse and ambient intake score alike. [mac:
   train_m3c.py on branch m6]
9. The 30 v8 per-action features of obs version bv8 (names, caps). [mac: train_m3c.py / bv8 code]
10. Opponent length distribution and number of live greedy snakes inside the hero's region at frames
    3,000-5,000; greedy deaths per episode; corpse pellets within 5 cells of the hero's body over time.
    [new measurement]
11. Whether frozen-pool opponents in Phase R carry a veto, and the round-52 bank seeds. [repo
    `tournament_eval.DEFAULT_OPPONENTS` and mac probe configs]
12. Training telemetry: fraction of transitions in scripted worlds at frame >= 2,500 and L >= 700, to
    size the data lever; the unverified death_len_mean ~200-235. [mac: run logs / telemetry.jsonl]
