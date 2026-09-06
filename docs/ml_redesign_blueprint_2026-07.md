# ML Redesign Research & Blueprint

**Date:** 2026-07-05
**Scope:** July 2026 assessment snapshot of the ML stack (state representation, algorithm, training system, environment, evaluation) plus a verified redesign blueprint targeting: best-possible state representation and algorithm for the slither-style multi-agent sim, training in hours on one rented GPU, real-time serving on a MacBook inside the web backend.
**Provenance:** Produced by a multi-agent pipeline — 4 code audits (file:line evidence), 6 literature/web research sweeps (primary sources), 3 competing redesign proposals (clean-slate / evolve-in-place / pragmatic-modernization), a 3-lens judge panel (science, engineering, economics), one synthesis, and 10 adversarial verification agents that attacked every load-bearing claim (including git archaeology and an independently written-and-run throughput benchmark on this machine). Verification verdicts are in Appendix B.

---

## Part I — July 2026 audit snapshot

This part records the implementation and gate observed for the July audit. The
gate defects described below have since been repaired; see the status note and
[Project history and durable findings](project_history_and_findings.md) before
using this snapshot to describe current behavior.

### What is verified correct (keep)

- **TD math is sound.** Double DQN (online argmax, target eval) shared between both training paths ([td_targets.py](../src/training/td_targets.py)), per-stream n-step returns with correct truncation bootstrapping, PER with α applied exactly once and IS-weighted Huber loss, exact Ape-X epsilon ladder. The observed pathologies are **not** TD-math bugs.
- **Contract discipline.** Named state indices, runtime size checks, checkpoint contract validation, replay-consistent enemy-trend memory (deliberately not updated on next-state capture), exact simulator action masks stored per transition with 2-step boost semantics.
- **The champion artifact.** `champion_a5_freespace_20260621.pth` is real and stays the incumbent/anchor.

### Critical defects (state representation)

1. **World-frame observation driving relative actions.** All sector maps, food/enemy vectors, and wall distances are absolute-frame ([snake_state.py](../src/game/snake_state.py) `_angle_to_sector` uses raw atan2); actions are turn-left/straight/right. The network must learn every spatial→action association 4×, once per heading. Only per-action danger and free-space (6 of 61 dims) are ego-indexed.
2. **Train/deploy feature distribution shift.** Apex actors run on a 0.2-scale board (290×166) while the danger radius stays absolute at 300px (`DANGER_MAX_DISTANCE * segment_size`, [snake_state.py:218](../src/game/snake_state.py)) — danger/wall features are near-saturated everywhere in training data and sparse on the 1450×830 eval/production board. Board-relative normalizations shift too. The learner trains on one feature distribution and is evaluated/deployed on another.
3. **Enemy observability misses what killing requires.** Only the 2 nearest enemy *heads* encoded; enemies #3+ and all enemy bodies beyond 300px invisible; enemy boost state unobserved; `kill_opportunity` projects enemies 1 step ahead, wrong under boost.
4. **Non-Markov reward.** Starvation ramp and boost segment cost depend on counters (`frames_since_food`, `boost_frames`) absent from the state — observationally identical states get rewards differing by up to 3.0 (a food pellet), injecting irreducible TD noise that PER preferentially resamples.
5. **Own-body shape nearly invisible.** Free-space is a 3-scalar saturating BFS band-aid over a real POMDP gap (cap pins at 160 cells for length ≥ 80: "pocket" and "open half the board" both read 1.0).
6. **Feature extraction is the throughput ceiling.** ~1.5 ms/snake/frame in Python, and both `get_state` and `_get_safe_actions` are computed **twice** per snake per frame.

### Critical defects (reward/economics — these explain the known pathologies)

- **~0 kills is the rational policy.** Max kill reward 5.0 (`kill_max`, re-clamped by `reward_max=5.0`) ≈ 1.7 food pellets; typical victim ≈ 0.5–0.7 pellets — in an arena with 250–300 pellets worth +3.0 each. Kill credit lands on the victim's death frame (beyond n_step=3 credit reach of the causal maneuver) and is **dropped entirely if the killer dies the same frame** ([snake_reward.py:54-63](../src/game/snake_reward.py)). Head-ons are size-blind mutual death. Corpse drop is 50% and counts against the ambient food cap, suppressing baseline spawns (the kill's spoils cannibalize food that would have spawned anyway).
- **Boost-abuse was patched, not fixed.** The −3.0/segment penalty skips frames where the snake ate and fires only on the 1-in-3 boost frames that burn a segment; boosting through dense food remains net-positive. Structural fix: make boost drop trail pellets (mass transfer, like real slither.io) instead of destroying mass.
- Reward reads feature indices out of the observation tensor ([snake_reward.py:77](../src/game/snake_reward.py)), coupling reward semantics to observation layout.

### Critical defects (training system)

- **The H100 was never the problem; the data engine is.** Each actor: one Python process, one 6-snake env, ~1 ms/frame ≈ 2,100 transitions/s, ~94% pure-Python feature/collision code. Learner gated on **3 synchronous IPC round trips per gradient step** through a single-threaded buffer process → 100–250 steps/s regardless of GPU. H100 utilization < 0.1%.
- **Pure mirror self-play.** Every snake in every env runs the current policy; no opponent pool, no past checkpoints, no frozen anchors. Late-episode data is lone-survivor solo farming with zeroed enemy features (dead snakes never respawn in training), biasing replay toward "enemies do not exist."
- The deleted historical Colab fork is recorded in [Project history and durable findings](project_history_and_findings.md). It diverged from the production contract (58-D, no free-space, own reward code) and used a **grid** environment, a caution against treating it as a continuous-dynamics precedent.

### Critical defects observed in the July evaluation gate

- **The promotion gate could promote regressions.** In the July snapshot,
  `tournament_eval.py` ranked mean mass over **alive frames only**: a candidate
  that boosted to mass 60 and died at frame 200 could outrank one that held 35
  for all 3000 frames. Opponents were 5 clones of one checkpoint
  (single-opponent Goodharting), and paired-seed data was analyzed **unpaired**.
  Both evaluation scripts then defaulted to a deleted YAML. Phase 0 repaired
  these defects; this paragraph explains why the repair was prioritized.
- Architecture decisions made under that gate, the 0.2-scale actor distribution,
  and ~10⁷ data-limited frames had unknown validity (see Appendix B, claims 1
  and 10).

### Corrected historical record

**The claim in CLAUDE.md/README/docs that "CNN variants were evaluated and lost every paired benchmark" is false** — verified by git archaeology: the CNN code (written 2026-06-21, deleted 2026-06-28) never entered git, zero CNN checkpoints ever existed, and all 10 overnight benchmark arms were feedforward fine-tunes. Only GRU/DRQN was actually trained and lost. The docs should be corrected; the CNN question is genuinely open and the blueprint re-tests it cleanly (P4 control arm).

---

## Part II — Research landscape (what the literature supports)

**Prior art for this exact game class:**
- **Battlesnake** (closest competitive analogue): the meta is dominated by tree search on bitboards, but the strongest pure-RL agent (Binnersley / Asymptotic Labs) briefly topped the global arena with PPO on a **17-channel ego-centric 23×23 raster**, sparse win/loss reward, ~524M turns in 4 days on one desktop GPU (208 parallel envs, ~1,500 turns/s). The practical champion recipe: RL policy + shallow alpha-beta fallback for forced wins/losses.
- **generals.io** (arXiv 2507.06825): top 0.003% of the human 1v1 leaderboard in **36 hours on a single H100** once the env was vectorized to thousands of FPS.
- **Published slither.io RL attempts** (Stanford CS229, Cal Poly, OpenAI-Universe-era): all pixel-based against the live browser game, all throughput-starved, none got past "better than random." This repo's custom simulator is already ahead of the published slither.io literature; the win condition is investing in it.

**State representation:** hand-crafted vectors have a demonstrated ceiling in this class (a 3-channel grid CNN doubled the best published Snake score vs an 11-feature vector). The evidence-backed target is an **ego-centric, heading-rotated multi-channel occupancy raster + scalar hybrid** into a small CNN. Entity-attention encoders (OpenAI Five style) are wrong-sized here (homogeneous segments, purely spatial threat); ray-casting is informationally close to the existing 16-sector features and shares their ceiling.

**Algorithm:** at cheap-frame throughput, replay-based sample efficiency stops mattering; the field's answer is synchronous on-policy or buffer-free value learning on massively vectorized envs. **PQN** (Gallici et al., ICLR 2025): deletes replay buffer + target network, stabilizes with LayerNorm + L2, matches Rainbow's median Atari at up to 50× DQN wall-clock — and keeps Q-values (the web UI dependency). **PPO** (PufferLib school: 0.3–1.2M steps/s trainers on one consumer GPU) is the highest-precedent alternative. Rejected with evidence: full Rainbow, R2D2, BBF (expensive-frame regime, backwards for this env), MuZero/Dreamer-class (unjustified complexity).

**Self-play:** OpenAI-Five-minimal — ~80% latest / 20% past-checkpoint pool — prevents cycling/forgetting; league/exploiter machinery is overkill for symmetric FFA. **OpenSkill (Plackett-Luce)** turns each FFA episode's death order into one rating update over all participants; keep 1–2 permanent frozen anchors so ratings stay comparable over time.

**Reward:** boost-abuse is textbook non-potential-based shaping pathology. Fix structurally: potential-based shaping `r = γΦ(s′) − Φ(s)` (Ng et al. 1999) for dense terms, true objectives as sparse events, and in-world economics (trail pellets, corpse mass) instead of scalar penalties. Per-term reward accounting in telemetry is the early-warning instrument.

**Apple Silicon serving (measured on this machine):** current MLP ≈ 20 µs/snake on CPU; a generous 8ch 32×32 4-conv CNN ≈ 1.9 ms for 10 snakes **batched** (~6% of a 30 Hz frame). MPS is 3–21× slower (dispatch-bound) — CPU stays correct. `torch.set_num_threads(1)` matters (3× oversubscription penalty measured). torch.compile/ONNX/CoreML: zero-to-negative benefit at these sizes.

---

## Part III — The redesign blueprint (final synthesis)

# Snake-DQN Redesign Blueprint
## Dual-Scale Ego-Raster PQN on a Vectorized Cell-Exact Sim, with Champion-Gated Migration

**Status:** Final synthesis, dated July 2026. Chassis = "Pragmatic Modernization" (highest aggregate judge score, 248/300; second place in both lenses it did not win; no fatal objection in any lens), upgraded with grafts from COIL (dual-scale perception, GPU-raster escape hatch, encircle detector, single-sim endgame as the v2 vehicle) and from Ape-X Refit (cheap existing-stack actor fixes, the 3-arm kill diagnosis, distribution-histogram CI, max-Q telemetry). Both rejected chassis options had verified fatal flaws: the historical Colab fork was a **grid** environment, not a batched continuous-dynamics proof, and Refit's 800-1200 learner-steps/s claim collapsed to ~210 without an unbudgeted SumTree rewrite while conceding the representation ceiling. See [Project history and durable findings](project_history_and_findings.md) for post-blueprint status and provenance.

**Goal restated:** best-possible state representation and algorithm for a slither-style multi-agent snake sim; trains in hours on one rented GPU; serves 6-12 snakes in real time on a MacBook CPU inside the FastAPI web backend; the web UI's "watch it think" surfaces (Q-values, dueling V+A, decision traces, activations) remain first-class.

---

## 0. The ten open questions — definitive positions

1. **Movement model: staged, not dodged.** v1 keeps 4-cardinal grid movement, canonicalized to exact integer cell semantics. Rationale: it is what makes bit-exact parity, existing masks, the Battlesnake prior art, the 45-file test suite, and the shipped Play mode all survive; and cut-off/boxing aggression IS achievable on the grid (env-eval audit's own framing). The honest cap — no smooth coiling — is stated and priced. Every other layer (raster obs, 6-action relative space, trainer, self-play, gate, serving) is built **movement-agnostic**, so the continuous-heading upgrade is a scoped v2 (Phase 6, ~15-20 pd touching only sim dynamics core + rasterizer geometry + Play input, with parity degrading from bit-exact to tolerance+event-exact), decided by a written go/no-go memo using v1 champion behavior, the encircle-detector data, and human-play feedback. If the product hard-requires true coiling, v1 is the de-risked stepping stone that builds ~80% of the destination.
2. **Why the historical vector/GRU evidence was limited, and why now differs:**
   the feedforward and GRU/DRQN arms were judged at ~10^7 data-limited frames,
   on 0.2-scale boards with saturated danger features, without ego-rotation,
   with float observations throttling an already IPC-bound pipeline, and under
   the flawed July gate. No CNN arm was trained or judged: its code was deleted
   before training and produced no checkpoints. The proposed control changes
   all relevant conditions: ~3.5-4B frames, full-scale single-geometry env,
   heading rotation, uint8 transport, and repaired gate. **We re-test it:**
   Phase 4 carries an MLP-on-the-new-featurizer control arm in the identical
   loop, with replicate seeds; the gate decides. Either winner is valid.
3. **Trust in the "one winner" baseline:** trust the **artifact**
   (champion_a5 is real; it stays incumbent and permanent anchor) and the
   **engineering** verdicts (contract discipline, exact masks, InferenceAgent,
   tournament concept). Treat historical architecture outcomes as no ceiling
   evidence: CNN had no trained outcome, and GRU/DRQN was judged under the
   conditions above. Nothing is re-deleted on their authority; nothing new is
   promoted without beating the artifact under the repaired gate. GRU stays
   excluded for serving-complexity reasons only.
4. **Algorithm fork: PQN**, with dueling head retained, so every Q-value visualization survives byte-for-byte. Keep-Ape-X rejected: the 100-250 steps/s learner cap is architectural (3 synchronous IPC round-trips/step), and the corrected arithmetic kills the replay-ratio argument for keeping it. PPO rejected as primary but held as a **pre-specified fallback** (~2-3 days on shared infra) with its UI adapter designed *now* (critic V(s) as the value stream; τ·(log π − mean log π) as soft advantages; probability bars with masked-action hatching) and explicit tripwires that trigger the switch (§4.6) — so the fallback does not strand the product, and the UI does not pick the algorithm.
5. **Masking under vectorization:** masks move inside the batched step as cell gathers on the incremental occupancy grid — 3 target cells per agent plus the boost midpoint and landing cell (exact 2-step boost semantics preserved), computed in the same pass as collisions. ONE shared fatality function feeds the behavior mask, the stored per-transition 6-bit uint8 mask, and the Q(λ) masked-max targets — collapsing the audit's three divergent definitions. Trapped states bootstrap to the death value, not 0. Mask cost stops being ~half the actor bottleneck and becomes a handful of vectorized lookups.
6. **Throughput, honestly:** plan at 150k agent-steps/s aggregate; **no number is trusted until gated.** Gate 1 (before renting): ≥40k/s single process on the Mac including obs+masks+scalars. Gate 2 (before campaign spend): ≥100k/s end-to-end on the rental box **including** the K+1 frozen-pool forwards, learner, and coordination bubbles, after a 30-minute cloud-CPU derate bench (rented x86 cores are typically 2-3x slower than Apple Silicon per core — modeled, not discovered mid-rental). Escape hatches in order: more env workers → GPU rasterization (COIL's per-arena paint + crop; also the P6 vehicle) → Cython/numba on the two hot kernels only. This is 10-70x current throughput, deliberately not the 1M+/s C-grid-snake class, for which no existence proof exists for this obs pipeline.
7. **Gate repair is Phase 0** — before any candidate exists to judge — plus a calibration check (the repaired gate must reproduce the known ordering champion_a5 > scripted anchor > random-safe with paired significance) before it is trusted. After Phase 2 parity, evals run in the vectorized sim (`--engine simd`), making gating ~100x cheaper and closing the unbudgeted-eval-compute blind spot.
8. **Mixed-policy execution:** per-slot `policy_id` in the synchronous loop; obs grouped by policy each step; K+1 batched forwards (hero + ≤10 frozen pool nets resident on GPU); only hero-slot transitions train. Realized opponent-exposure distribution (fraction of hero transitions by opponent id, episode length by mix) is telemetered against the nominal 80/20 — covering the pool-x-population-floor interaction blind spot.
9. **Zero-kills bet:** economics primary (~55%), opponent quality secondary (~30%), data volume tertiary (~15%). Discriminator: **Refit's 3-arm experiment run cheap and early** — Phase 1, local CPU, existing Apex stack accelerated 2-4x by the Phase-0 actor fixes: (A) economics-only, (B) economics+pool, (C) baseline, judged on kills/ep + kill-opportunity telemetry under the repaired gate. A>C: economics confirmed. B>A: opponent quality adds. A≈C≈0 with rising opportunity counts: volume binds, and P4 budget shifts from reward sweeps to longer generations. All three remedies ship regardless; the experiment sets emphasis.
10. **Parity:** one integer transition function, two implementations, three shared artifacts — (a) cell-exact semantics + one constants module, (b) ONE pure event-based reward function (`ate/died/killed/boost_burn`) imported by both sims, (c) ONE batched featurizer used by trainer and web session. Enforced by golden-replay CI (seeded action logs, 10k frames × 20 seeds, bit-exact positions/deaths/kill-attribution/food-sets/rewards/obs/masks — achievable because the math is integer; the RNG-alignment cost is scoped by a single owned RNG stream with documented draw order, budgeted inside P2) **plus** a train-vs-eval-vs-serve observation-histogram KS check as a standing CI instrument (golden replay catches logic drift; the KS check catches distributional drift — the 0.2-scale bug class). Maintenance: ~0.5 pd per future mechanics change, paid as red CI instead of silent deployment drift. The deleted Colab fork remains a documented historical caution.

---

## 1. Game mechanics v2

All changes land in the live Python game AND the vectorized sim from the same shared constants + event-reward modules. Product sign-off on feel is a P1 exit criterion.

1. **Cell-exact canonicalization.** Movement is already fixed 10px cardinal steps on a 1450×830 board (= 145×83 cells); redefine collision/food-pickup as integer cell-occupancy equality instead of circle overlap. Near-zero behavioral change; the foundation of bit-exact parity and vectorization.
2. **Boost drops a trail pellet** at the popped tail cell (position already in hand at `snake.py:~164`). Boost becomes a mass *transfer* funding chase dynamics; the −3.0 scalar penalty and its `ate_food` loophole (`snake_reward.py:107-114`) are **deleted** — economics are structural now.
3. **Kills pay in mass:** 100% corpse drop (vs 50% at `game_state.py:473-475`); corpse food is exempt from the ambient `maintain_count` cap via separate ambient/corpse accounting (`food_manager.py:145-159`) so a kill's spoils stop suppressing baseline spawns.
4. **Size-resolved head-ons:** ≥1.15× length survives and receives kill credit; near-equal remains mutual death. Replaces size-blind mutual loss (`game_logic.py:193-204`); also makes big snakes feel powerful in Play mode.
5. **Same-frame killer credit:** kill reward paid before the `collided` early return (`snake_reward.py:54-63` vs the interaction call at line 101).
6. **Starvation reward ramp deleted** (non-Markov); starvation stays as a death mechanic, observable via the hunger scalar.
7. **Training-env-only:** episodes reset when alive < 3 (fixes lone-survivor replay skew; roughly doubles useful transitions/frame). The product game keeps its behavior.
8. **NOT changed:** 4-cardinal movement, 6-action relative space (turn L/S/R × normal/boost), 2-step boost semantics, rectangular + circular arenas, per-tick decisions.

**Kill/trap attribution under v2 (blind-spot fix — defined now, before it corrupts metrics):**
- *Contact kill:* owner of the collided body segment (existing semantics), credit paid even if the killer dies the same frame.
- *Head-on kill:* the surviving larger snake.
- *Entrapment event (probe-only in v1, never reward):* victim dies within K=60 frames of its reachable-free-space (BFS on the occupancy grid) dropping below threshold T, AND snake X's body forms ≥60% of the confining boundary during that window → entrapment attributed to X. This powers the encircle/cut-off detector and kills-adjacent probes. It is deliberately kept out of the reward: the *reward* payoff for traps flows through corpse mass, which the entrapping snake is positioned to eat — in-world credit needs no heuristic attribution.
- *Trail-lure / starvation-in-pocket:* no special attribution; corpse mass is the payoff mechanism.

---

## 2. Observation spec (obs_spec = `raster31v2`)

Everything spatial is rendered in the snake's heading frame ("always facing up"). With 4 cardinal headings this is an **exact** `rot90` — no interpolation. This kills the audited 4× world-frame/relative-action redundancy and makes horizontal-flip augmentation valid (flip columns, swap actions 0↔2 and 3↔5, negate lateral scalars).

### 2.1 Tactical raster — 31×31, cell = 1 segment (10px)
Head fixed forward-biased at (row 23, col 15): ~230px lookahead, ~70px behind, ~150px lateral. **Stored/transported as 2 uint8 planes** (1,922 B/agent): plane A = type code, plane B = value byte. Expanded to 9 float channels on GPU (÷255). Type codes with overlap priority (heads > bodies > predicted > corpse food > ambient food > empty):

| ch | content | value byte |
|----|---------|-----------|
| 0 | own body | TTL: time-to-vacate (tail low → head high; encodes where gaps open — replaces the 3 saturating free-space BFS scalars) |
| 1 | enemy bodies (ALL enemies in window) | same TTL encoding |
| 2 | enemy heads | clamp(their_len/own_len, 0, 2)/2 — killable vs dangerous |
| 3 | enemy predicted-next cells | 1-step from heading; **2 cells when that enemy is boosting** (fixes the boost-blind kill_opportunity at `snake_state.py:396-399`) |
| 4 | ambient food | pellet mass |
| 5 | **corpse/boost-trail food (separate channel from day one)** | pellet mass — the kill-economy signal is first-class (COIL graft) |
| 6 | wall / out-of-arena mask | handles rectangular and circular arenas uniformly |
| 7 | own head | boost-engaged bit in value |
| 8 | reserved (ablation slot) | — |

### 2.2 Strategic raster — 25×25, cell = 5 segments (50px) → ~1250px span
**Resolves the science judge's first-order objection** (single-scale = the same discredited ~300px horizon; enemies beyond ~200px invisible). Three uint8 density channels: enemy mass density (bodies+heads, mass-weighted), food mass density (ambient+corpse), own body density. Implementation keeps CPU cost near zero: each env maintains a 29×17 **coarse accumulator grid** incrementally (each cell add/remove updates one coarse counter); the *whole per-env coarse grid* (1,479 B) is uploaded once per step (256 envs → ~0.4 MB/step total), and the per-agent 25×25 ego crop + rot90 happens **on GPU** via precomputed index maps (4 headings × positions). CPU crop cost therefore stays at the tactical-only budget the economics judge verified.

### 2.3 Scalars — 26 float32, all ego-frame, Markov-completing
`length/max_length`, `log1p(length)/log1p(max)`, `boost_available`, `boost_cost_phase = boost_frames/3`, `hunger = frames_since_food/starvation_max`, `episode_progress`, `alive_count/max_snakes`, `mass_rank_percentile`, wall distance ahead/right/behind/left (4, cells/64 capped), nearest-food-beyond-raster ego (dx, dy, dist), nearest-2-enemy-head summaries (ego dx, dy, size ratio, is_boosting × 2 = 8 — cheap redundancy for beyond-tactical threats), normalized world x, y, arena-type flag. `boost_cost_phase` and `hunger` remove the audited non-Markov reward aliasing; `max_length` and the full obs spec become checkpoint-contract fields.

Transport per obs: 1,922 B tactical + 104 B scalars + amortized ~1.5 B coarse ≈ **2.0 KB** — ~15× smaller than naive float32.

### 2.4 Network (~1.3M params, ~22 MFLOP/forward)
- Tactical trunk: Conv3×3 9→32 s1 → Conv3×3 32→64 s2 → Conv3×3 64→64 s2 → flatten (8×8×64 = 4096) → FC 256 + LayerNorm.
- Strategic trunk: Conv3×3 3→16 s2 → Conv3×3 16→32 s2 → flatten (7×7×32 = 1568) → FC 64 + LayerNorm.
- Fusion: concat (256+64+26) → FC 256 + LayerNorm → **dueling V(1) + A(6)** (UI contract preserved).
- LayerNorm everywhere per the PQN recipe. Horizontal-flip augmentation in the learner from day one.
- Mac cost: measured scaling from the mac-inference sweep (1.9 ms for a 360k-param CNN, 10 snakes batched) → gate at **≤8 ms for 12 snakes batched, one core**; well inside a 33 ms frame.

---

## 3. Algorithm

### 3.1 PQN core
Synchronous Q(λ) on vectorized envs (ICLR 2025 recipe): no replay buffer, no target network, no PER, no actor/buffer IPC — deleting exactly the systems audit's three critical bottlenecks rather than optimizing them.

- Loop: 256 envs × 8 snakes = 2,048 agent slots per env-worker process; rollout T=16 → 32,768 transitions/update/worker; 4 minibatches × 2 epochs.
- **λ-returns with per-agent stream handling (the subtlety, spelled out):** returns computed backward over each agent's rollout stream independently; at *death* steps the return is the reward alone (death value is in r; no bootstrap); at *truncation* (rollout edge, episode cap, alive<3 reset for surviving snakes) bootstrap from masked-max Q(s′); **trapped non-terminal states bootstrap to the death value, not 0** (carries the audit fix at `td_targets.py:36`).
- **Masked max** over valid next actions using the stored per-transition 6-bit mask; −inf on invalid logits at selection.
- Exploration: ε-greedy 1.0→0.02 over the first ~50M agent-steps, per-agent ε ladder retained for diversity.
- Stability: LayerNorm everywhere, Huber loss, grad-norm clip 10, **no TD-target clipping** (removes the ±50/±100 divergence class) — replaced by a standing max|Q| alarm (§7).
- Hyperparameter starting points: γ 0.997 (kill-setup horizon ~330 frames), λ 0.65, Adam lr 5e-4 linearly annealed, adam_eps 1.5e-4 (single config source), wd 0. Sweep axes: lr {2.5e-4, 5e-4, 1e-3}, λ {0.5, 0.65, 0.8}, kill scale {0.15, 0.3, 0.6}.

### 3.2 Reward (v2)
- Shaped: `r_pot = γ·Φ(s′) − Φ(s)`, Φ = length/10, **Φ(death) = 0** — dying forfeits accumulated potential, so dying rich is intrinsically penalized. (Acknowledged deviation from strict PBRS — the forfeiture is deliberate shaping equivalent to a mass-at-death penalty.)
- Sparse: kill **+0.3 × victim_length, unclamped**, paid on same-frame death; death −3.
- Nothing else: no food term (mass flows through Φ), no boost term (mechanics), no hunger term (observed scalar). Per-term reward accounting in telemetry is mandatory.

### 3.3 Self-play
80/20 latest/pool; FIFO pool of 10; snapshot offered every ~50M agent-steps, admitted by the repaired gate; PFSP weighting `(1−winrate)^p` only if the hero beats the pool uniformly; no league/exploiters. **Kill-opportunity seeded resets:** 10-20% of episode resets use opportunity presets (hero spawned near a smaller scripted-prey snake, or mid-chase configurations), competence-gated OFF once median kills/ep clears threshold — the targeted-exploration fix for rare kill events that uniform self-play may never stumble into.

### 3.4 Opponent modeling (shared blind spot, addressed at bounded cost)
- **Auxiliary enemy-prediction head (P4 sweep arm):** predict the nearest enemy's next cell (ego frame) as a small cross-entropy loss (weight ~0.1) on channel-3 ground truth. Tron-lineage precedent shows direct win-rate gains; costs one small head.
- **Shallow lookahead (post-v1 optional module, not load-bearing):** 1-3-ply paranoid masked rollouts using the fast vectorized sim at serve/eval time to veto provably fatal actions and convert forced kills (Binnersley's alpha-beta-fallback pattern). Six actions × the fast sim makes this cheap; needs no retraining; explicitly scoped so its absence blocks nothing.

### 3.5 PPO fallback (pre-specified, tripwire-triggered)
Same env loop, rollout storage, masking, mixed-policy execution. Switch cost ~2-3 days. UI adapter designed now: critic V(s) → value stream; τ·(log π − mean log π) → advantage stream; probability bars with masked hatching (~1-2 days frontend, the same class of contract change already shipped once for the dueling split). Triggers: NaN/divergence unrecoverable by lr/λ backoff; max|Q| alarm persisting 3 consecutive checks; failure to beat the scripted anchor after 200M steps in P3 shakedown while telemetry shows healthy data.

---

## 4. Training system

### 4.1 Vectorized sim (`src/simd_env/`)
NumPy-batched, cell-exact: fixed-capacity body ring buffers (E, S, 150, 2) int16; per-env int8 occupancy grids (145×83) updated **incrementally** (head-add/tail-remove, ~4k writes/step); per-env 29×17 coarse density accumulators; collisions = gathers of head target cells (plus boost midpoint/landing); **masks = the same gathers** (one shared fatality function); tactical raster = per-agent 31×31 crop of a padded coded grid (contiguous row copies + group-wise rot90 by heading), emitted as 2 uint8 planes into a pinned buffer; ONE H2D copy per rollout for obs (per-step H2D/D2H for action selection is budgeted — ~4 MB and ~1-2 ms/step, an economics-judge correction we accept into the ms budget).

### 4.2 Mixed-policy execution
Per-slot `policy_id`; group obs by policy; K+1 batched forwards (hero + ≤10 frozen pool nets on GPU); only hero transitions to the learner. Trivial because we own the synchronous loop — the reason third-party C/GPU envs were rejected.

### 4.3 Throughput budget (with every judge correction internalized)
Per 2,048-agent step, single Mac core: game-logic array ops ~1 ms; tactical crops dominate at 2,048 × 1,922 B ≈ 3.9 MB of structured uint8 copies → 10-40 ms at 100-400 MB/s; coarse-grid upload ~0.4 MB; action-selection round trip ~1-2 ms. → 50-190k agent-steps/s/process on the Mac.
- **Derates applied to planning:** cloud x86 cores 2-3× slower per core (→ 8-10 env workers on a ≥32-vCPU box instead of 4-6); self-play K+1 forward tax 15-25%; coordination/Amdahl bubbles 10-20%.
- **Plan: 150k/s sustained aggregate. Gate 1 (Mac, pre-rental): ≥40k/s single process incl. obs+masks. Gate 2 (rental, pre-campaign): ≥100k/s end-to-end incl. pool forwards + learner, after a 30-min cloud-CPU smoke bench.** At 100k/s, 1B steps = 2.8h — even a 3× shortfall keeps the campaign hours-scale.
- Learner load at 150k/s ≈ 20 TFLOP/s — a few % of a 4090; **the env is the binding constraint by design.**
- Escape hatches, in order: more workers → **GPU rasterization** (upload compact segment/food arrays; per-arena paint + per-agent crop on GPU — COIL's core trick, which also keeps the single-torch-sim P6 endgame reachable) → Cython/numba on the two hot kernels only. No C rewrite, no third codebase.

### 4.4 Hardware and rental economics
Default rental: **32+ vCPU, 4090/5090-class GPU, spot/preemptible** (typically 2-3× cheaper; every run here is a 1-12h ideal spot workload). Preemption-safe: checkpoint every 10 min; save/restore time measured in shakedown; resume under contract v2. The H100 is justified **only** if the GPU-rasterization hatch activates (that path becomes HBM-bandwidth-bound). Select rentals by vCPU count first.

### 4.5 Campaign plan (~16-24 GPU-hours, ~$40-120 spot)
0.5h shakedown + 1h soak (incl. tripwire dry-run and checkpoint save/restore timing) → sweep: 9 configs × 100M steps (~3h) **plus replicate seeds (2-3) on the two decisive comparisons — CNN-vs-MLP control and kill-scale** (~1h; the statistical-power blind spot: single-run arms cannot separate seed noise from signal on the decisions the redesign hinges on) → 3 self-play generations × ~1B agent-steps (~8.5h at plan rate) with gate checks between (run in the vectorized sim, minutes not hours) → margin for one re-run. Total frames ~3.5-4B — past the 0.5-1B band where prior art reached strong play (Battlesnake 524M; generals.io ~36 GPU-h).

---

## 5. Evaluation

### 5.1 Gate repair (Phase 0 — before anything is trained)
1. Headline metric: **mass integral over TOTAL frames** (dead frames = 0), replacing alive-conditioned mean_mass (`tournament_eval.py:143-166`) — dying rich now loses. Survival, deaths, kills reported alongside.
2. **Paired per-seed deltas** vs a designated baseline with 95% CI and wins/N — porting the already-correct math from `ensemble_eval.py:177-196`.
3. **Opponent diversity:** round-robin over ≥3 checkpoints + a scripted greedy-food/flood-fill-safety anchor + random-safe (replacing 5 clones of one checkpoint at `tournament_eval.py:124-127`). A scripted-aggressor bot exists for probes/curriculum.
4. Fix the crashing default `--config` (line ~193 → deleted YAML).
5. **Calibration before trust:** the repaired gate must reproduce champion_a5 > anchor > random-safe with paired significance.
6. **Power analysis:** pilot variance from the P0 re-baselining sets seed count for a minimum detectable effect (~3% mass-integral); default ≥40 paired seeds, raised if the pilot says so.

### 5.2 Promotion rule
Candidate promotes iff paired per-seed mass-integral delta vs incumbent > 0 at 95% CI across ≥2 opponent mixes, AND no regression vs scripted anchors, AND behavioral probes within bands, AND a 50-episode serving-path spot check (batch=1 CPU in the live Python game) matches training-sim scores within CI.

### 5.3 Ratings and probes
- **OpenSkill (Plackett-Luce)** over {pool, champions, anchors} from FFA death order (tie-break final mass), nightly, with 2 permanent frozen anchors for cross-month comparability. Ratings inform; the paired gate decides.
- **Probes (co-equal gate metrics and training telemetry):** kills/episode, deaths-by-cause, boost-frame fraction + boost mass ROI (trail-adjusted), self-collision rate, survival, **encircle/cut-off detector** (entrapment events per §1), per-term reward accounting, kill-opportunity frequency, realized opponent-exposure distribution.
- **Observation-histogram KS check** (train vs eval vs serve) as standing CI.
- **Eval engine:** after P2 parity, `tournament_eval --engine simd` runs gates ~100x faster; the Python game remains the serving-path spot-check venue.

### 5.4 Human play as an evaluation channel (blind-spot fix)
Every Play episode logs {mechanics_version, checkpoint_id, human score, AI outcomes, probe telemetry} to a versioned ScoreStore (new `mechanics_version` column; v1 leaderboard preserved read-only; v2 starts fresh, announced). Human games are the only out-of-distribution adversarial opponents available: a weekly review of human-vs-AI outcomes is the exploit-detection channel, and pool checkpoints of graded OpenSkill rating provide difficulty tiers for the product.

---

## 6. Deployment and web UI

- **Serving unchanged in shape:** eager fp32 CPU via InferenceAgent, extended with **obs_spec/mechanics-version-keyed adapters** (`vector61`, `raster31v2`) so every historical champion stays loadable forever as an eval anchor; contract v2 validates obs spec hash, channel list, scales, action semantics, mechanics version, γ/λ, algo id.
- `torch.set_num_threads(1)` at startup in `web/serve.py` (measured 3× oversubscription penalty); inference inline in the game-loop coroutine; all snakes batched into one forward (**≤8 ms for 12 snakes** gate); never MPS (measured 3-21× slower); skip compile/ONNX/CoreML (measured zero-to-negative).
- `web/backend/session.py` maintains the live game's incremental occupancy + coarse grids (sub-ms) and calls the SHARED featurizer once per frame — the same module the trainer uses, so serve-time observations are train-time observations by construction. Side effect: the Phase-0 actor fixes and shared featurizer cut the real Mac frame cost driver (Python feature building, ~1.5 ms/snake today), so the web game gets smoother immediately.
- **UI:** Q-bars, dueling V+A split, decision traces, and activation views work **unchanged** (PQN keeps Q-values and the dueling head; conv feature maps slot into the activations panel). New: the **ego-raster viewer** — render the planes the snake actually sees, rotating with it — the best "watch it think" artifact this project has had, free because it IS the observation. Plus kill-feed/encircle-event overlays from the probes. The PPO-fallback adapter is specified in §3.5 so even the worst case has a designed UI story.

---

## 7. Telemetry and run-health tripwires (mid-campaign abort criteria — blind-spot fix)

Standing dashboards: per-term reward shares, max|Q|, loss/grad-norm, ε schedule vs realized action entropy, kills/ep and kill-opportunity trend, boost fraction/ROI, opponent-exposure mix, throughput, eval-in-sim score per generation. **Automatic tripwires (checkpoint → halt → alert):**
- NaN/inf in loss or Q.
- max|Q| > 5× the reward-scale-derived bound, persisting 3 consecutive checks (the no-clip regime's alarm).
- Behavior-policy action distribution collapse (mode share >95% ε-adjusted).
- Any single shaped term >70% of |return|.
- Median kills/ep pinned at 0 beyond a set step count while kill-opportunity telemetry is positive (post-economics-fix).
- Throughput drop >30% for 10 min; eval-in-sim regression >2 CI vs previous generation.
The project's own history (the best-save freeze that silently wasted an H100 run) is the argument: rented hours die by tripwire, not by morning discovery.

---

## 8. Disposition table (blind-spot fix: nothing left dangling)

> **September 2026 status.** This table records the blueprint's conditional
> migration plan, not completed deletions. Apex, curriculum, and offline tooling
> remain live while raster/PQN is unpromoted. Their removal remains contingent on
> a champion-gated raster win; current status is maintained in
> [Project history and durable findings](project_history_and_findings.md).

| Asset | Disposition |
|---|---|
| Historical Colab grid fork | Removed after its grid-contract divergence was recorded in the project history; it is not a continuous-dynamics precedent. |
| Ape-X stack (`apex_actor/buffer/learner`, PER, SumTree) | Retained until a champion-gated raster win; only then may its deletion and test retirement be considered. |
| `curriculum.py` | Planned for conditional P5 retirement after a champion-gated migration; currently live. |
| `memory_db_handler.py`, `generate_experiences.py`, `offline_train.py` | Proposed to freeze or retire only with the conditional Apex migration; current paths remain live where present. |
| `rebase_checkpoint.py`, `widen_input.py` | Retired (vector-widening warm starts don't apply across the obs change); kept in history. |
| `evaluate_checkpoints.py` | Folded into the repaired tournament_eval (or updated to contract v2) at P0. |
| 61-D featurizer | Kept as the `vector61` obs adapter (eval-only) so champion_a5 and all prior champions remain loadable forever. |
| Test suite (45 files, ~1100 tests) | **Explicit 4-6 pd budget**, partitioned: mechanics tests re-fixtured for v2 at P1; new parity/featurizer/mask/PQN tests at P2-P3; Ape-X/buffer tests kept green until their P5 deletion; contract/InferenceAgent tests extended for adapters. |
| `scores.db` leaderboard | Schema v2 with `mechanics_version`; v1 scores preserved read-only; reset announced. |

---

## 9. Judge objections — explicit resolutions

**Science lens (objections to the chassis):** (1) *Perceptual horizon = the discredited 300px cutoff* → **accepted, fixed** by the dual-scale strategic planes (~1250px) + nearest-2-enemy scalars, at near-zero CPU cost via per-env coarse grids ego-aligned on GPU. (2) *Grid movement caps the ceiling; deliverable is a boxing agent, not a slither agent* → **accepted as a named, priced cap**, resolved by the staged P6 decision with a designed movement-agnostic stack; the alternative (COIL) failed engineering review on a falsified existence proof. (3) *PQN thinnest track record; Q(λ) truncation subtle; PPO fallback breaks the UI justification* → **accepted, mitigated**: per-agent termination handling specified (§3.1), tripwires defined, and the PPO UI adapter designed up front so the fallback no longer breaks the product story.

**Engineering lens:** (1) *Bit-exact parity RNG alignment is a time sink* → **accepted, scoped**: single owned RNG stream, documented draw order, integer math; inside P2's 10-14 pd with the parity test as the exit criterion. (2) *"UI unchanged iff PQN holds"* → resolved via the pre-specified adapter. (3) *Crop-gather 4× uncertainty; worker scaling unmeasured* → the Mac gate, the cloud smoke bench, and the ordered escape hatches exist precisely for this; a shortfall to 50k/s still yields hours-scale generations. (4) *Four simultaneous bets* → phase gates + the MLP control arm + champion-gated deletion mean each bet is separable and reversible; nothing is deleted before its replacement wins.

**Economics lens:** (1) *Mac-to-cloud derate unstated* → now stated (2-3×) and gated (30-min smoke bench). (2) *"ONE H2D copy" wrong* → corrected; per-step action-selection round trip budgeted. (3) *PQN precedent thin* → fallback priced. (4) *Grid ceiling defers the headline behavior* → P6 decision memo with encircle-detector evidence. Additionally adopted: spot instances + preemption-safe checkpoints, eval-in-fast-sim, self-play tax inside gated numbers, and person-day-first sequencing (the ~3-day actor fixes land in P0 because they accelerate every later phase and the live product).

---

## 10. Roadmap

**P0 — Gate repair + telemetry + cheap actor fixes (existing stack; 5-7 pd)**
Repair tournament_eval (mass integral, paired deltas, multi-opponent + anchors, config fix); calibration check; power pilot; behavioral probes incl. encircle detector v0; per-term reward accounting; OpenSkill nightly; obs-histogram diff tool; **Refit's actor fixes** (next_state/next_mask carry-forward, one shared simulate-action function, batched 6-snake forwards — halves the measured ~1.5 ms/snake feature cost, speeds P1 and the live web game); freeze legacy offline tooling.
*Exit:* repaired gate runs end-to-end; champion_a5 re-baselined with CIs over ≥40 paired seeds; **gate reproduces A5 > anchor > random-safe with paired significance**; measured actor speedup recorded.

**P1 — Mechanics v2 + 3-arm kill diagnosis (6-8 pd + ~1-2 days unattended CPU)**
Cell-exact canonicalization; boost trail pellets; 100% corpse above cap; size-resolved head-ons; same-frame credit; PBRS reward v2; alive<3 training reset; ScoreStore v2 + leaderboard versioning; shared constants + event-reward modules; mechanics tests re-fixtured. Run arms A/B/C locally on the accelerated Apex stack.
*Exit:* Play mode feels right (product sign-off); A5 re-baselined under v2; **diagnosis verdict recorded with its P4 budget-allocation rule**; CI green on re-fixtured tests.

**P2 — Vectorized sim + dual-scale featurizer + parity (10-14 pd)**
`src/simd_env` batched cell-exact stepping; incremental occupancy + coarse grids; coded-plane tactical crops + GPU-side strategic alignment; 26 scalars; in-step masks (one shared fatality function); golden-replay parity CI; `tournament_eval --engine simd`; Mac throughput bench.
*Exit:* **bit-exact parity over 10k frames × 20 seeds** (positions, deaths, kill attribution, food, rewards, obs, masks); **≥40k agent-steps/s single process on the Mac incl. obs+masks** (hard go/no-go for renting); featurizer identical between batch and live paths.

**P3 — PQN trainer + pool self-play + contract v2 (9-12 pd)**
Q(λ) masked targets with per-agent termination handling; LayerNorm dueling net; flip augmentation; uint8→GPU pipeline; mixed-policy K+1 execution; 80/20 pool + seeded resets; checkpoint contract v2 + obs_spec adapters in InferenceAgent; tripwires implemented and dry-run (fault injection); spot-safe checkpointing; PPO fallback branch scaffolded.
*Exit:* a 50-100M-step local run **decisively beats the scripted anchor** on the repaired gate; checkpoint round-trips through InferenceAgent and tournament_eval; tripwires fire correctly on injected faults.

**P4 — GPU campaign (4-6 pd attended + 16-24 GPU-h, ~$40-120 spot)**
30-min cloud-CPU derate bench; 1h soak (**Gate 2: ≥100k/s end-to-end incl. pool forwards**); sweep (lr × λ × kill-scale × aux-head, MLP control arm, **replicate seeds on decisive arms**); 3 generations × ~1B with in-sim gate checks between; abort tripwires armed.
*Exit:* a champion that beats re-baselined A5(v2) AND the anchor with paired 95% significance; **kills/episode > 0.5 median**; boost fraction 5-30% with rational trail-adjusted ROI; encircle detector fires at a nonzero rate; spend ≤24 GPU-h. If the CNN loses to its own MLP control, **promote the better one — the gate decides.**

**P5 — Deployment + UI + champion-gated deletion (6-8 pd)**
Session.py on the shared featurizer + incremental grids; `set_num_threads(1)`; ego-raster viewer + probe overlays; human-play telemetry channel live; serving spot-check; **then, only after a champion-gated raster win,** delete the Ape-X stack, PER/SumTree, curriculum.py, and frozen offline tooling per §8.
*Exit:* 12 snakes ≥30 Hz with **≤8 ms inference** for all snakes batched; all visualizations working; Play E2E with versioned leaderboard; CI green after deletions; old champions load via adapters.

**P6 — Movement v2 decision point (optional; memo 1-2 pd; if green-lit ~15-20 pd)**
Written go/no-go on continuous-heading movement using v1 champion behavior, encircle-detector data, and human feedback. If go: scope = sim dynamics core (integer cells → continuous positions + turn-rate limit, same 6 actions) + rasterizer geometry (rot90 → affine sampling) + Play input (target-heading steering); parity becomes tolerance-based positions + event-exact deaths; the GPU-raster hatch becomes the training sim (COIL's endgame, entered with the trainer, gate, self-play, and serving already proven). Also on this menu: QR-distributional head (enriches the UI with a risk strip), shallow-lookahead serving module.

**Totals: ~40-55 person-days (8-10 calendar weeks solo), <$150 GPU. First benchmarkable deliverable (repaired gate + re-baselined champion) inside week 1; a working system at every phase boundary.**

---

## 11. Risks

1. **NumPy rasterizer misses throughput** (dominant, least-certain cost; 4× uncertainty band). Mitigated by the pre-rental Mac gate, the pre-campaign cloud gate, and three ordered escape hatches; a 3× shortfall still keeps generations at ~7h — degraded, not dead.
2. **The CNN loses to the MLP.** A real possibility; the P4 control arm with replicate seeds makes the comparison clean for the first time, and the trainer/sim/gate are representation-agnostic — either winner ships.
3. **PQN instability at scale** (thin precedent; Q(λ) truncation subtleties). LayerNorm recipe, conservative lr, λ sweep, tripwires; PPO fallback priced with its UI adapter pre-designed.
4. **Bit-exact parity slips on RNG alignment.** Scoped by single-stream RNG design; worst case falls back to event-exact + per-field tolerances with the KS check carrying distributional guarantees — a documented, lesser standard, not silent drift.
5. **Mechanics v2 changes product feel / invalidates history.** P1 product sign-off before training investment; versioned leaderboard; A5 re-baselined under v2 so the bar stays honest.
6. **Kill behavior still fails to emerge** (if data volume or opponent quality dominate). The P1 3-arm verdict reallocates P4 budget in advance; seeded resets and scripted-aggressor pool members are the curriculum levers; probes make failure visible in the first sweep hours, not after the campaign.
7. **New reward exploits** (e.g., trail-farming loops). Per-term accounting + probe bands are standing telemetry from day one; the potential-based structure and in-world economics minimize the exploit surface; the sweep probes trail economics explicitly.
8. **Cloud-core derate worse than 3×.** The 30-min smoke bench catches it before campaign spend; the fix is more workers or the GPU-raster hatch, both pre-planned.
9. **Grid ceiling turns out product-fatal** (true coiling is non-negotiable). Then P6 green-lights the continuous rewrite with the trainer, gate, self-play, serving, and telemetry already built — the deliberate ~15-20 pd residual instead of COIL's everything-at-once bet.
10. **Scope creep in mechanics tuning.** Constants live in one shared module; P1 is timeboxed to its exit criteria; later tuning is a red-CI parity event, not a re-port.

---

## Appendix A — Decision points reserved for the project owner

The blueprint is execution-ready, but three calls are product judgments:

1. **Movement model (blueprint §0.1, P6).** v1 stays 4-cardinal grid (parity, masks, tests, Play mode all survive; cut-off/boxing aggression is achievable). True smooth coiling requires the continuous-heading v2 (~15–20 pd). The stack is built movement-agnostic so this stays a scoped later decision — but if smooth coiling is non-negotiable for the product, say so before P2 and the P6 work moves earlier.
2. **Mechanics v2 feel (P1 exit).** Boost trail pellets, 100% corpse, size-resolved head-ons change how the game plays for humans too. P1 requires product sign-off before training investment.
3. **Leaderboard reset.** Mechanics v2 invalidates score comparability; the plan versions the ScoreStore (v1 read-only) and announces a fresh board.

## Appendix B — Adversarial verification verdicts

Every load-bearing claim was attacked by an independent verifier with repo, git-history, and web access. Summary:

| # | Claim | Verdict |
|---|-------|---------|
| 1 | "CNN variants were evaluated and lost every paired benchmark" (CLAUDE.md/README/docs) | **REFUTED** — documentation artifact. CNN code existed 7 days between squashed commits, never trained, zero checkpoints. Only GRU/DRQN actually lost benchmarks. |
| 2 | NumPy-vectorized env reaches 50–200k agent-steps/s | **PLAUSIBLE** — self-flagged extrapolation; hence the hard pre-rental gates (≥40k/s Mac, ≥100k/s cloud) before any spend. |
| 3 | Historical Colab fork is a grid env; no batched continuous-heading precedent exists | **CONFIRMED** in the pre-consolidation source preserved by the project-history provenance table — justifies staging the movement upgrade. |
| 4 | ≥40k agent-steps/s single-process on this MacBook incl. obs+masks | **PLAUSIBLE, Mac leg CONFIRMED by direct measurement** — verifier wrote and ran an independent full-spec prototype (2,048 agents, real board geometry, rot90 ego-rasters). Cloud leg holds only with forwards/learner on GPU (as budgeted). |
| 5 | ~0-kills is primarily reward/mechanics economics | **PLAUSIBLE** — every factual premise verified in code (kill ≤ 1.67 pellets, same-frame credit dropped, corpse under cap); causal share decided by the P1 3-arm experiment. |
| 6 | Bit-exact parity between Python game and batched sim is achievable | **CONFIRMED** — all v1 dynamics are integer cell math; radius tests verified exactly equivalent to integer compares for seg=10. |
| 7 | PQN + dueling head preserves all Q-value visualizations; ≤8 ms for 12 snakes | **CONFIRMED** — serialize.py needs zero changes; perf rebuilt and measured. |
| 8 | Current gate can promote regressions; gate repair must precede comparisons | **CONFIRMED** — worse than claimed (dead frames drop out of denominator entirely). |
| 9 | 3.5–4B transition campaign fits in 16–24 GPU-h at gate rate, ~$40–120 spot | **CONFIRMED** — arithmetic recomputed; July 2026 spot rates $0.15–0.70/hr make it ~$4–17. Caveat: in like-for-like agent-transition units the campaign is ~2× the Battlesnake budget, not 7×. |
| 10 | Prior CNN/GRU eliminations carry no evidential weight (five discredited conditions) | **CONFIRMED a fortiori** — no CNN was ever trained at all. |

**Follow-ups implied by verification:** retain the corrected CNN statement in the
README, algorithm document, and project-history record; treat claims 2/4/5 as
gated bets (the blueprint's Gate 1/Gate 2 and the P1 3-arm experiment exist
precisely to resolve them before money is spent).

## Appendix C — Judge panel outcome

Three lenses scored three competing proposals; each lens picked a different winner (science → clean-slate COIL; engineering → Ape-X Refit; economics → Pragmatic Modernization). Pragmatic Modernization won on aggregate (248/300, second place in both lenses it didn't win, no fatal objection). COIL's chassis was rejected on a **falsified existence proof** (its "the colab fork proves continuous dynamics tensorize" argument — the fork is a grid env, see Appendix B #3); Refit's throughput claim collapsed from 800–1200 to ~210 learner-steps/s under scrutiny while conceding the representation ceiling. COIL's dual-scale perception, GPU-raster escape hatch, encircle detector, and single-sim endgame were grafted in; Refit contributed the cheap actor fixes, the 3-arm kill diagnosis, and the telemetry/tripwire discipline.
