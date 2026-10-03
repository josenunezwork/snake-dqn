# Serving-time lookahead search on the v8 Watch hero: feasibility (2026-10-03)

This is a Tier-0 design study. It contains micro-benchmarks only. It played no gate episodes and no
rollouts at scale. It changes no default, champion, served veto or profile. It follows up the
portfolio review's "later lever" (`docs/research/experiment_portfolio_review_2026-09-26.md`).
It builds on the v3/v4 screens, the trap-horizon study, the v5 death census and the
v6/v7/v8 results.

Branch `search-design` (from `main` `a0cd04b`). New files:

- `research/serving_time_search_20261003/bench.py`: the micro-benchmarks below.
- `research/serving_time_search_20261003/rollout_search.py`: an opt-in prototype. Nothing
  imports it. It lives under `research/`, so the `src/` source closures that the strict and
  serving receipts bind are unchanged.
- `tests/test_serving_time_search_prototype.py`: 12 tests, about 1.2 s.

## 1. Recommendation (short)

1. **Build "veto v9" first: v8's choice plus a sparse, hero-only rollout check.** This is
   design (c). For the action v8 picks, simulate 40 frames of the hero alone. Use v5's exact
   move model and a deterministic space-greedy continuation. Treat opponents as static walls,
   and on the first frame also as v6's head-reach cells. Override v8 only when its action
   fails (dies, or ends with fewer than `need` reachable cells) **and** another masked-legal
   action passes. In that case take the highest-Q passing action, same speed mode first.
2. **Before writing gate code, run a Stage-0 ceiling and fidelity study** on the death
   census that is running now (section 6). It answers two questions. First, how many deaths
   could any action change still prevent? Second, does the static-opponent model see those
   deaths 20–60 frames ahead?
3. **Do not start with MCTS or Q-guided beam search.** Both cost 1–3 orders of magnitude more
   per decision. Both change most decisions, which breaks the paired-world ties that let v8
   pass. Both rely on Q as a value at depth, and Q was trained on reward v2, not on the
   H5000 mass integral.
4. **Do not distill v8 into the network now.** v8 already costs only 0.65 ms per decision.
   Its inputs are not recoverable from the 61-D observation, so the shield must stay. Any
   weight change also makes paired worlds diverge everywhere. Revisit distillation as
   search distillation only after a search wins.

Why (c) is the right first design, and what it can realistically gain:

- **Where the remaining mass loss is.** Under v8, survival is 0.83–0.92 of frames. In the v8
  screen, self deaths were 58 of 180 candidate episodes and head-on deaths were 3.
- **Why one-step checks miss these deaths.** Per the v5 census, the decisive event is
  entering a region smaller than `need`. That happens before the one-step point of no
  return. The count-only horizon has a median of 21 frames (q25 9, q75 34), and some
  unknown cases run up to about 60.
- **Why v4 failed and why (c) differs.** v4 used a depth-8 escape search triggered on low
  one-step counts. When it fired, the trap was already committed (694 no-escape fallbacks).
  v9 runs on every decision and looks 40 frames ahead, which covers the measured horizon.
- **Sparsity keeps the gate powerful.** v8 changed 2.2e-4 of decisions and still crossed
  the strict boundary at look 2, because most paired worlds stay exact ties. A v9 that
  fires only on failing choices keeps that structure.
- **Expected gain is my estimate, not data.** A dying hero loses about 12% of frames at a
  mass of roughly 260–800, so deaths cost an estimated 50–60 mass per world on average.
  The census found that most late escapes only delay death ("realistic gain is mostly
  delay"). Converting 20–40% of that loss gives **+10 to +25 mass per world**, against a
  v8–v7 delta of +30 to +43. On the low end this is not detectable at the current strict
  N_max (section 7, risk 1). Stage 0 and the Tier-1 screen exist to find out which end is
  true before strict compute is spent.

## 2. How the agent acts now

| Piece | What it does | Where |
|---|---|---|
| Network | `ApexNetwork`, dueling Q: 61 → 512 → 256 → (V: 256→512→1, A: 256→512→6), about 0.43 M MACs per row. Six actions: {left, straight, right} × {normal, boost}. | `src/model/apex_network.py`; champion `champion_a5_freespace_20260621.pth` (`43d4e2c5…`) |
| Observation | `vector61`: the 58-D hand-crafted vector plus 3 per-direction flood-fill free-space features (cap 160). The live `AISnake` reuses last frame's post-step state and mask (carry-forward, frame-guarded) and keeps nearest-enemy trend memory. | `src/game/snake_state.py`, `src/game/ai_snake.py:384-405` |
| Selection | masked argmax (`INVALID_Q_VALUE` on illegal actions), then `snake.safety_veto.apply(...)` if a veto is installed | `ai_snake.py:476-491` |
| v2 | one-step spacious test (`round(f·cap) ≥ need`, need ≤ 160); speed-preserving highest-Q replacement | `safety_veto.py` |
| v5 | v2 plus a boost landing check: flood from the two-cell landing with the first cell blocked | `safety_veto_v5.py` |
| v7 | among v5-eligible moves in the chosen speed mode, argmax of `Qn + λ·min(area, 2·len)/(2·len)`, where area is the tail-aware post-move area (λ=4 released, then superseded) | `safety_veto_v7.py` |
| v8 (served) | v7 at λ=8, then v6's opponent-head layer. If v7's choice can be reached by an opponent head the hero would not beat (1.15× rule, +1 growth margin), it re-ranks the non-risky eligible alternatives. Deterministic; timing is recorded only as diagnostics. | `safety_veto_v8.py`; STRICT_PASS + SERVING_PASS, default since `a0cd04b` |
| Serving | `GameSession.step` runs in `asyncio.to_thread`. The engine loop steps the game, broadcasts, then sleeps `1/speed` (default 12 fps, range 1–120). It is not a fixed-rate scheduler, so step latency adds directly to the frame period. The v8 wrapper is on the Watch hero only and bound fail-closed to the strict receipt (checkpoint sha, 7 source shas, method). Mean wrapper cost in the serving run was 0.65 ms per decision. | `web/backend/app.py:114-142`, `session.py:697-711`, `safety_veto_serving.py` |
| SIMD engine | `BatchSim`: E×S NumPy arena, bit-exact with live (CI parity). `Vector61Runtime`/`Vector61SimdPolicy` reproduce the live carry, mask and argmax. v7/v8 run via a rowwise live-veto adapter; H5000 parity was 24/24 identical and 1.6–3.1× faster. There is **no** live-`GameState` → `BatchSim` loader. Cloning is `copy.deepcopy(sim)` (`observation_probe.clone_sim`). | `src/simd_env/` |
| Strict gate | live `tournament_eval.rollout`, `promotion-v2-watch-rect`, H5000, 6 snakes, 3 mixes. OBF sequential, 4 looks at 63/125/187/249 worlds per mix. v8 stopped at look 2: 561 paired worlds, about 33–37 s per episode at 2 torch threads. | `research/apex_veto_v8_strict_20261003` |

## 3. Micro-benchmarks

Setup: Apple M5 Pro, torch 2.9.1, NumPy 1.26.4, `OMP_NUM_THREADS=1`, torch threads 1/1. The
config is the strict gate's `research/apex_safety_20260926/deployment.yaml` (1450×830,
6 snakes, mechanics v2, max_food 300). The machine was shared with the running death census
(load average about 8–9 on 18 cores). Treat every number as a median wall-clock design input,
not as evidence. All sections together used about 100 CPU-seconds.

```bash
cd /Users/josenunez/.codex/worktrees/search-design/snake-dqn
B="env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
   /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python research/serving_time_search_20261003/bench.py"
$B --section q                               # 2 s
$B --section live                            # 9 s  (warm 600 frames, hero length ~95)
$B --section live --live-warm-frames 2500    # 29 s (hero length ~388)
$B --section simd                            # 8 s
$B --section proto                           # 4 s
$B --section proto --live-warm-frames 2500   # 24 s
```

Live worlds use seed 20261003, with the champion in every slot and v8 on the hero.

### 3.1 Q forward pass (champion `ApexNetwork`, `torch.no_grad`, 1 thread)

| Batch | 1 | 6 | 18 | 64 | 256 | 1024 |
|---|---:|---:|---:|---:|---:|---:|
| torch, ms | 0.030 | 0.063 | 0.068 | 0.105 | 0.42 | 1.74 |
| NumPy float32, ms | 0.036 | 0.135 | — | 0.73 | — | — |

- **The Q forward pass is never the bottleneck.** Featurization and world stepping are 10–400× more expensive.
- **Batch shape changes the floats.** Row 0 of a batch-64 forward differs from the same row
  at batch 1 by up to 4.2e-5. NumPy differs from torch by up to 2e-4. A search that batches
  Q calls must use fixed batch shapes (or the rowwise call shape) to stay bit-reproducible.
  Otherwise near-ties in argmax can flip between the gate and serving.

### 3.2 Live engine (`GameState`)

| Quantity (median) | Hero len ≈ 95 (frame 600) | Hero len ≈ 388 (frame 2500) |
|---|---:|---:|
| One live frame (`gs.update`, 6 snakes, v8 on hero) | 8.8–9.5 ms | 12.8 ms |
| Hero `get_state` (fresh featurization) | 0.77 ms | 1.08 ms |
| Hero `_get_safe_actions` | 0.36 ms | 0.63 ms |
| Hero v8 `apply` | 0.37 ms | 0.65 ms (matches the serving run's 0.65 ms mean) |
| v5 `simulate_action` × 6 | 0.008 ms | 0.012 ms |
| v7 `landing_area` × 6 (tail-aware flood, cap 190 / 776) | 0.46 ms | 1.85 ms |
| **World fork**: `copy.deepcopy(gs)`, policies shared via memo | 0.94 ms | 1.23 ms |
| Fork + 8 frames, **naive** deepcopy | 131 ms (16.4 ms/frame) | 170 ms |
| Fork + 8 / 32 frames, **closure rebound** | 70 / 392 ms | 97 / 480 ms |
| Global RNG save + restore | 0.007 ms | 0.007 ms |

**A naive deepcopy is not a faithful fork.** I found this here and it is checked in
`bench.py`. `GameState` gives each AISnake `get_frame=lambda: self.frame`. `deepcopy` keeps
that closure pointing at the **original** world. Clone snakes then fail the carry-forward
frame guard: they re-featurize every frame (about 2× slower) and take a different selection
path from live.

| Fork check (16 frames, with the global RNG saved and restored around the clone) | Result |
|---|---|
| Naive fork: bodies equal live after 16 frames | **False** |
| Fork with `snake._get_frame` rebound to the clone: bodies equal live | **True** |
| Rebound fork: food equals live | **True** |
| Live frame counter untouched by forking | **True** |

Any search that forks the live world must:

- rebind that closure;
- save and restore `random`'s state, because food spawns draw from the global Mersenne
  Twister;
- share the frozen policies through the deepcopy memo.

### 3.3 SIMD engine (`BatchSim`, greedy_food scripted on every slot, lengths 12–125)

| Quantity (median) | Value |
|---|---:|
| E=1 step + greedy action pick | 0.63 ms/frame |
| `deepcopy(sim)` E=1 | 0.49 ms |
| E=1 clone + 8 / 32 steps (greedy) | 5.7 / 24 ms |
| E=16 step | 2.29 ms (0.143 ms per env) |
| E=64 step | 8.55 ms (0.134 ms per env) |
| vector61 featurize, 6 / 96 / 381 rows | 0.79 / 3.9 / 15.4 ms (about 0.04 ms per row when batched) |

A policy-driven SIMD world-frame (6 snakes, batched) costs about **0.4 ms per env-frame**
(step 0.13 + featurize 0.25 + Q 0.01). The live world costs 9–13 ms per frame, so SIMD is
about 25–30× cheaper. However, it has no live → `BatchSim` loader. Writing one means
transferring ring buffers, food order, boost counters, respawn timers, carry caches, trend
memory and the per-env `random.Random` state, and then proving parity. At serving, a
6-candidate × 40-frame search over full worlds would still cost about 90 ms. SIMD pays off
for throughput (many worlds per call), not for one decision's latency.

### 3.4 Prototype: hero-only rollout (`rollout_search.score_actions`, all 6 actions)

| Horizon | Hero len 95 | Hero len 388 | Nodes (6 actions) |
|---|---:|---:|---:|
| 8 | 3.5 ms | 7.1 ms | 111 |
| 20 | 8.6 ms | 19.2 ms | 291 |
| 40 | 17.3 ms | 37.1 ms | 591 |
| 60 | 25.7 ms | 54.5 ms | 891 |
| 40, policy flood cap 16 / 160 | 9.3 / 31.4 ms | 28.1 / 54.5 ms | 591 |

The global RNG was untouched (checked). This is unoptimized Python. Each node copies the
whole body (`simulate_move`) and rebuilds a tail set, so cost is O(length) per node.

**Projection, not measured:** keeping the body as an incremental deque plus an occupancy
counter, with integer cell ids, should cut cost 3–5× at length ≥ 400.

**Decision rule cost.** v9 rolls out only v8's choice first, and rolls out the others only
when that choice fails. The mean cost is about one candidate: **about 3 ms at length 95 and
about 6 ms at length 388 for H40**. The worst case is all 6 actions: 17–37 ms at H40 and
26–55 ms at H60.

## 4. Candidate designs

The common constraints for strict evaluation are:

- bit-reproducible: seeded and isolated RNG, no wall-clock budgets (node budgets only),
  fixed Q batch shapes;
- the same platform for the gate and the serving parity probes;
- the "check off == v8" invariant must be testable;
- no oracle access to opponent identity.

### (a) Depth-limited Q-guided rollouts / beam over the hero's actions

- **What is simulated:** at each node, the hero's next action is the argmax of Q on that
  node's `vector61` state, so every node needs a full world. Opponents' features enter
  `get_state`, food density included.
- **Opponent model, honestly:** there are three options.
  - Self-model: the champion plays every opponent. This is identity-blind and legitimate,
    and it costs a 6-row Q forward per frame.
  - Constant-velocity heads.
  - Static walls.

  Using the gate's real opponent checkpoints, or `greedy_food` for the scripted mix, would
  be an oracle tuned to the gate's pool. That is forbidden.
- **Cost:** a beam of width 3 and depth 8 is about 72 world-frames.
  - Live rebound fork, about 11 ms per frame: about **0.8 s per decision (about 1.2
    decisions/s)**.
  - SIMD, after a loader is built, about 0.4 ms per env-frame: about 30 ms, or about 33
    decisions/s (projection).
- **Fit:** depth 8 is the horizon v4 already showed is too short. Going deeper multiplies
  the cost. Q at depth is not a mass-integral value. Not first.

### (b) MCTS with Q as prior and value

- **What is simulated:** stochastic opponents and stochastic food. It needs determinization
  (sampled opponent models and private-RNG food), and every sample must be seeded from a
  hash of the world state, never from the global RNG.
- **Cost:** 200 simulations × depth 20 is about 4,000 nodes.
  - Hero-only model (0.15–0.3 ms per node): about 0.6–1.2 s.
  - Full live worlds: about 45 s.
  - SIMD: about 1.6 s.

  That is **about 1 decision/s at best**, against 4,350 decisions per H5000 episode. Neither
  a gate (about 70 slot-hours per arm at 1 decision/s) nor a web tick budget allows it.
  Q priors also change most decisions, so paired worlds diverge and the gate's SD rises.
  Not first; possibly later as an offline teacher.

### (c) "Veto v9": short simulation of k candidate actions, scoring survival and space (recommended)

- **What is simulated:** the hero alone, using v5's exact `simulate_move` (boost, burn and
  tail-pop order are reproduced; tested equal to live `Snake.move`).
  - Opponents are static walls (the v2–v8 convention), plus v6's head-reach cells on the
    first frame, with the 1.15× rule.
  - No food spawns and no growth. Growth only delays tail release, and v3's slack of 1
    absorbs one pellet.
  - Continuation: deterministic space-greedy over normal-speed moves (capped tail-aware
    flood; ties straight > left > right).
  - Score: (survives H frames, leaf tail-aware count ≥ need).
- **Opponent honesty:** identity-blind by construction. The model never reads an opponent's
  checkpoint or kind.
- **Weakness:** static walls are wrong at 40–60 frames, because opponents move 40–120 cells
  in that time.
  - The model will miss traps that opponents close and flag pockets they would open.
  - The sparse rule bounds the damage, since it only acts when v8's choice fails and an
    alternative passes.
  - Stage 0 measures how often the model is wrong.
  - Optional upgrades: dilate opponent heads by 1 cell per frame (pessimistic), or release
    opponent tails at 1 cell per frame (optimistic). Both stay identity-blind.
- **Cost:** about 3–6 ms mean per decision at H40 (prototype), 26–55 ms worst case. Per
  episode that adds about +13–26 s, against about 35 s now. With the 3–5× optimization
  (projection) it is about +5 s.
- **Determinism:**
  - integer cells only;
  - BFS neighbour order is fixed;
  - sets are used only for membership;
  - no RNG and no clocks;
  - node count is bounded by `horizon × 3`.

  Q floats are used only to order passing candidates, exactly as v8 uses them.
- **Web budget:** at 12 fps (83 ms period, 9–13 ms frame) the mean adds under 10%. A
  worst-case decision is a one-frame hitch of about 40–55 ms, which is invisible at 12 fps
  and noticeable at 60+ fps. A deterministic node cap bounds it.

### Decisions per second (single thread, derived from section 3)

| Design | Per decision | Decisions/s |
|---|---|---:|
| v8 today | 0.37–0.65 ms | 1,500–2,700 |
| (c) v9 H40, v8's choice first (typical) | 3–6 ms | 160–350 |
| (c) v9 H40, all 6 candidates (worst) | 17–37 ms | 27–58 |
| (c) v9 H60, all 6 | 26–55 ms | 18–39 |
| (c′) v9 with a live rebound fork and self-model opponents, k=3 × 40 frames | 1.3–1.8 s | ~0.6 |
| (a) beam w3 d8, live fork / SIMD (projection) | 0.8 s / ~30 ms | 1.2 / ~33 |
| (b) MCTS 200 × 20, hero-only / live | 0.6–1.2 s / ~45 s | ~1 / 0.02 |

## 5. Distillation (imitate v8's vetoed actions)

- **Data.** v8 overrides the base argmax on 0.22–0.29% of decisions: 169 of 770,590 changes
  versus v7 in the screen, and 360 of 125,000 replacements in serving. About 10^6
  on-policy decisions (about 230 H5000 episodes, about 2.2 slot-hours locally) give about
  2–3k override labels. Rows would be the 61-D state, the mask and v8's action, 244 MB as
  float32.
- **Training cost (derived from 3.1).** A 256-row forward is 0.42 ms, so forward plus
  backward is about 1.3 ms per step. 10^6 rows × 10 epochs is about 39k steps, or **about
  1 minute of CPU**. Use a DQfD-style large-margin loss on override states plus an L2 anchor
  to the champion's Q everywhere else. Training cost is negligible.
- **Why not now:**
  1. **Aliasing.** v8 uses information that is not in the observation:
     - tail-aware areas up to 2·length (the feature is capped at 160);
     - two-cell boost landings;
     - every opponent's head reach and length (the observation has only the nearest two
       enemies).

     Identical inputs will carry conflicting labels. The review already rejected two corpora
     for exact-input alias conflicts. The shield must stay, so the stack does not get
     cheaper, and v8's 0.65 ms is not a bottleneck anyway.
  2. **The best case is roughly zero gain.** Imitating v8 can at most match champion+v8.
  3. **Gating cost.** The candidate is a new checkpoint, so it needs a champion-change
     strict gate: (champion′ + v8) vs (champion + v8). That needs new serving bindings.
     Any weight change perturbs base decisions in every world, so ties disappear and paired
     SD grows. The one learned challenger in the review (6102) lost.
- **When it would make sense:** as *search* distillation (expert iteration), after v9 (or a
  heavier search) passes strict. Distill the search's choices into a network with an
  extended observation, a new `obs_spec`, so the network sees what the search used. Gate it
  as a checkpoint change with the search kept as the shield.

## 6. Staged plan

**Stage 0 — ceiling and model fidelity (DEV, no gate).**

Inputs: the running death census's captures, or deterministic re-plays from world seeds
(replaying to frame t costs at most t × 7 ms).

For each self or enclosure death, take every 5th decision in the last 60 frames. At each:

- **Oracle:** fork the live world with the closure rebound and the RNG isolated. Play each
  masked-legal action, then champion+v8, for 100 frames against the true opponents. Record
  whether any action avoids the death.
- **Model:** run the prototype's pass/fail at H20/40/60 on the same decisions.

Outputs:

- the addressable share of deaths;
- recall and precision of the model against the oracle;
- the earliest decision at which an alternative still saves the hero.

Cost: about 12 decisions × 6 actions × 100 frames × about 12 ms, which is about 90 s per
death. For about 50 deaths that is about 1.3 slot-hours.

**Go** if at least 25% of self deaths are addressable within the window **and**, at H40
(or H60), model recall is at least 50% with an estimated false-override rate of at most
1e-3 of decisions. Otherwise stop, or test opponent dilation first.

**Stage 1 — module.**

Write `src/evaluation/safety_veto_v9.py`:

- v8 runs unchanged; v9 is "v8, then the rollout check".
- With the check off, v9 equals v8 decision for decision, and tests assert this.
- Frozen-source tests cover v2–v8.
- A descriptor records horizon, caps and model flags; the probe uses the seven-counter
  shape; diagnostics give pass/fail counts and node totals; time is recorded only as
  diagnostics.

Optimize to about 2 ms mean at length 400. Run it on SIMD through the existing rowwise
live-veto adapter (as v7/v8). H5000 parity versus live needs 24/24 identical before SIMD is
used for any screen.

**Stage 2 — DEV sweep, then a Tier-1 screen versus released v8.**

1. DEV sweep: H ∈ {20, 40, 60}, 8 worlds per mix, burned namespace. It checks activity
   (overrides per episode), cost, and a 3-slot calibration.
2. Screen: 60 worlds per mix in a fresh disjoint namespace, with C=A/D=B determinism
   replays. The pre-registered rule is v8's: ADVANCE iff the pooled mean is above 0, the
   pooled one-sided 90% lower bound is above 0, and no mix has a one-sided 90% upper bound
   below 0.
3. Also pre-register a **power check**: proceed to strict only if the pooled screen effect
   is at least about +20 (section 7, risk 1).

**Stage 3 — sequential strict gate** (`research/sequential_strict_template`).

- OBF, 4 looks (63/125/187/249 per mix), family alpha 0.05, efficacy in at least 2 of 3
  mixes, scripted noninferiority, survival bands.
- Fresh namespace, standing authority.
- All episodes on **one platform**.

**Stage 4 — serving qualification** (copy of the v8 lane).

- Opt-in `SNAKE_SERVE_VETO_VARIANT=v9`, bound fail-closed to the receipt (checkpoint, all
  veto source shas, method/descriptor).
- 25 Watch + 25 Play episodes plus 2 parity probes against the strict rollout, on the gate's
  platform.
- A latency report: per-decision p50/p99/max, and the frame-period effect at 12/30/60 fps.
- Release is a separate user-approved step.

### Compute per stage

Assumptions (derived):

- An incumbent episode takes about 35 s per slot (2 torch threads).
- v9 adds about 5 ms × about 4,350 decisions, about 22 s, so a candidate episode takes
  about 57 s (about 40 s after optimization).
- One slot is about 2 vCPU. RunPod CPU costs about $0.03 per vCPU-hour; assume x86 vCPUs
  are 1–2× slower than the M5 cores.

| Stage | Episodes | Slot-hours | Local wall, 2 / 3 slots | RunPod |
|---|---:|---:|---:|---:|
| 0 ceiling (about 50 deaths) | n/a | about 1.3 | about 0.7 h / 0.5 h | about $0.08–0.16 |
| 2 DEV sweep (8/mix, 3 arms) | about 100 | about 1.3 | about 0.7 h / 0.5 h | about $0.08–0.16 |
| 2 Tier-1 screen (60/mix + 12 replays) | 372 | about 4.8 | about 2.4 h / 1.6 h | about $0.3–0.6 |
| 3 strict, stop at look 2 (187/mix) | 1,122 | about 14.3 | about 7.2 h / 4.8 h | about $0.9–1.7 |
| 3 strict, N_max (249/mix) | 1,494 | about 19.1 | about 9.6 h / 6.4 h | about $1.2–2.3 |
| 4 serving (50 + 2 at H5000, 1 slot) | 52 | about 1.5–2 | about 2 h | not recommended (see risk 4) |

Local is sufficient. RunPod mainly buys wall time, for example a 32-vCPU pod finishing the
whole strict run in about 1.5–2.5 h. That holds only if the strict run **and** its serving
parity probes run on the same x86 platform. Dollar cost is never the constraint.

## 7. Risks

1. **Power.** Strict paired SDs were 124–165 per world. At +15 mass the needed n is about
   ((2.54+0.84)·130/15)², which is about 860 worlds per mix, far above N_max 249. Only
   effects of about +25 or more are likely to pass. Sparse overrides help, because more
   worlds stay exact ties and the SD falls, but that is not guaranteed. Mitigation: the
   Stage-2 power check, before any strict compute.
2. **Model error at long horizons.**
   - Static opponents are wrong at H40–60.
   - The space-greedy continuation can fail where a smarter path survives, so "fails" is
     pessimistic.
   - Ignoring growth is mildly optimistic.

   False overrides cost mass: each one takes a lower-Q move. Mitigations: Stage-0
   precision measurement, override only when an alternative passes, and DEV activity
   bounds.
3. **Determinism traps.**
   - A naive `deepcopy` fork diverges from live (closure bug, section 3.2).
   - The global RNG must be isolated.
   - Q batch shape changes floats by about 4e-5.
   - Thread count and CPU architecture change GEMM floats. The strict run, its replays and
     the serving parity probes must share the platform and thread settings.
   - Budgets must be node counts, never milliseconds.
4. **Cross-platform serving parity.** Serving runs on the Mac. A strict run on RunPod x86
   would need its parity probes re-run against Mac-served rollouts, and a mismatch on Q
   near-ties is plausible. Prefer the local strict run for the receipt the server binds.
5. **Latency spikes in the web tick.** Worst case is 26–55 ms unoptimized, against 83 ms at
   12 fps. Bound it with a deterministic node cap. If the server is ever run at 60+ fps,
   decouple decision computation, or accept the hitches.
6. **Governance surface.** v9 adds a module whose sha the receipts must bind. Seven sources
   become eight. It also needs a new serving variant, fail-closed bindings and frozen-source
   tests for v2–v8. These are routine but must be done.
7. **Goodhart on the gate.** Any opponent model that reads the opponent's identity (a
   checkpoint, `greedy_food`) is an oracle for the gate's pool and invalidates the result.
   The prototype is identity-blind. Keep it that way, and if a self-model is ever used,
   pre-register it.
8. **Prior negative result.** v3 and v4 (short look-ahead) were flat. v9's case rests on
   two differences: a horizon of 40–60 instead of 8, and running on every decision instead
   of on low one-step counts. If Stage 0 shows the model cannot see the trap at 40–60
   frames either, the lever is weak and should be dropped.

## 8. Prototype status

- `rollout_search.py` provides `step_hero`, `rollout`, `score_actions` (read-only on live
  snakes) and `rollout_choice` (the sketched v9 rule).
- Tests, in `tests/test_serving_time_search_prototype.py`:
  - `step_hero` equals live `Snake.move` cells, length and boost counter on 60 random
    worlds × 6 actions;
  - one-frame death equals live wall, self and static-body collision on 60 worlds × 6 actions;
  - the trap-horizon pocket: the straight boost fails and normal straight passes;
  - a first-frame head threat is fatal;
  - the choice rule keeps, switches, falls back and never picks a masked action;
  - results are repeatable, the global RNG is untouched, and the snakes are not mutated.

  Command:
  `/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q tests/test_serving_time_search_prototype.py`.
  Result: 12 passed, exit 0.
- Not done: no installer, no serving hook and no SIMD port. No default behaviour changed.
