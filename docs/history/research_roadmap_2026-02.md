# Snake DQN Improvement Roadmap

> **Historical audit from 2026-02.** Items not acted on are accepted technical debt. Not an active roadmap.

**Synthesized from 9-agent deep-dive research audit (Feb 2026)**

---

## Executive Summary

The codebase is well-structured with clean dependency injection, immutable configs, and a solid Apex DQN foundation. However, several issues are preventing the agent from learning effective slither.io strategies. The problems fall into three categories:

1. **Bugs silently degrading training** (double collision, reward skew, epsilon decay)
2. **Missing slither.io mechanics** (no speed boost, no relative actions, limited state)
3. **Architectural limitations** (no temporal memory, redundant network code, no curriculum)

The roadmap below is prioritized by **impact / effort ratio**. Items near the top deliver the most learning improvement per hour of development.

---

## Phase 1: Quick Wins (High Impact, Low Effort)

*Each item: 1-4 hours. These fix bugs or tune values that are actively harming training.*

### 1.1 Fix Double Collision Detection [CRITICAL]
- **Source**: #2 Environment, validated in code
- **Problem**: `AISnake.update()` calls `self._check_collision()` (line 187) which calls `self.die()` inside `Snake.move()` (line 413). Then `GameState.handle_collisions()` (line 179) runs `GameLogic.check_collisions()` on the SAME frame. Snakes can die twice or have inconsistent death states.
- **Fix**: Remove `_check_collision()` call from `AISnake.update()`. Let `GameState.handle_collisions()` be the single source of truth. Adjust reward calculation to get collision info from the centralized system.
- **Impact**: Eliminates corrupt training signals from ghost deaths.

### 1.2 Rebalance Reward Scale [CRITICAL]
- **Source**: #3 Reward
- **Problem**: Wall penalty (-0.6) fires every frame near walls and is 6x the food signal (+0.1 shaping). The agent learns "avoid walls" much faster than "seek food", creating timid circling behavior.
- **Fix**:
  - Reduce `wall_danger` from -0.6 to -0.15 (match the current warning level)
  - Or better: use smooth gradient from 0 to -0.15 (already partially implemented)
  - Remove stepped wall thresholds (danger/warning/caution/awareness) - use single smooth gradient
  - Add small survival reward (+0.01/step when alive) to prevent "dying is cheap" mindset
- **Impact**: Shifts the learning signal from "wall avoidance" to "food seeking".

### 1.3 Slow Epsilon Decay [HIGH]
- **Source**: #6 Training, #9 Exploration
- **Problem**: `epsilon_decay: 0.9996` reaches floor (~0.02) in only ~15K steps / ~10K frames. In single-process mode with 4 snakes, this means exploration ends within the first ~15 episodes. The agent locks in whatever behavior it found early.
- **Fix**: Change `epsilon_decay` to 0.99995 (reaches floor at ~200K steps, giving ~150 episodes of meaningful exploration)
- **Impact**: Dramatic improvement in strategy diversity and late-game learning.

### 1.4 Increase Min Buffer Before Training [HIGH]
- **Source**: #6 Training
- **Problem**: Training starts after only 128 experiences (= `BATCH_SIZE`). Early batches are nearly identical transitions, causing severe overfitting to initial behavior.
- **Fix**: Set minimum buffer to 10,000 experiences before first training step (add `min_buffer_size` check in `train_step()`)
- **Impact**: Prevents early overfitting, gives more diverse initial learning.

### 1.5 Fix Hidden Size Mismatch [MEDIUM]
- **Source**: #5 Architecture
- **Problem**: Config says `hidden_size: 256` but `DuelingDQN.__init__` defaults to 512. Depending on load order, the network may use 256 (underpowered for 47-D input) or 512 (ignoring config).
- **Fix**: Set config to `hidden_size: 512` to match code default. The 47-D input warrants 512.
- **Impact**: Ensures consistent network capacity.

### 1.6 Enable Parameter Sharing Across Snakes [HIGH]
- **Source**: #8 Multi-Agent
- **Problem**: Each of the 4 snakes has its OWN independent policy, replay buffer, and network weights. This is 4x memory waste and the weakest multi-agent paradigm (independent learners). Snakes can't learn from each other's experiences.
- **Fix**: All 4 snakes share a single `ApexPolicy` instance (one network, one replay buffer). Differentiate via per-snake epsilon values (which is exactly how Apex is designed — varied exploration across actors). In `SnakeFactory`, create one policy and pass it to all snakes.
- **Impact**: 4x more training data per gradient step, 4x less memory, diverse exploration via epsilon variation. This is how Apex DQN is meant to work.
- **Effort**: Low-Medium — mostly wiring changes in `SnakeFactory` and `GameState`.

### 1.7 Remove Dead Reward Config [LOW]
- **Source**: #3 Reward
- **Problem**: `food_length_mult` (0.067) and `food_length_exp` (0.003) are defined in config but never used in `calculate_reward()`. `reward_max` (2.0) / `reward_min` (-10.0) clamping never triggers because shaping rewards are tiny. This is confusing dead config.
- **Fix**: Either remove these fields or wire them in. Recommend removing since flat food reward is simpler.
- **Impact**: Code clarity, prevents confusion.

---

## Phase 2: Medium-Term Improvements (High Impact, Moderate Effort)

*Each item: 1-3 days. These add missing capabilities that unlock new strategies.*

### 2.1 Switch to Relative Action Space [HIGH]
- **Source**: #7 Action Space
- **Problem**: 4 cardinal actions (up/right/down/left) means the agent must learn the same strategy 4x (once per facing direction). This quadruples sample complexity.
- **Fix**: Change to 3 relative actions: `[turn_left, go_straight, turn_right]`. Map to absolute direction based on current heading. Update `GameConfig.ACTIONS`, `GameConfig.OUTPUT_SIZE`, network output, and action masking.
- **Impact**: ~4x sample efficiency improvement. Single biggest bang-for-buck change after bug fixes.
- **Effort**: Medium - touches action selection, network output, state representation.

### 2.2 Expand State to ~55D with Per-Action Danger [HIGH]
- **Source**: #1 State Rep
- **Problem**: Current 16-sector danger map doesn't tell the agent "which action leads to danger". The agent must learn to map sectors to actions.
- **Fix**: Add 4 features (or 3 if relative actions): immediate danger score for each possible action direction. Computed as: "if I move this way, how close is the nearest obstacle?" Normalized 0-1.
- **Impact**: Directly actionable danger signals dramatically speed up wall/obstacle avoidance learning.
- **Effort**: Moderate - add to `get_state()`, update `INPUT_SIZE`, retrain.

### 2.3 Add Speed Boost Mechanic [HIGH]
- **Source**: #7 Action Space, #4 Slither Expert
- **Problem**: Real slither.io's core mechanic is speed boost (burn length for speed). This creates the risk/reward tradeoff that makes the game interesting. Without it, there's no strategic depth beyond "eat food, avoid walls". The slither-expert identifies this as the single most impactful action space change for enabling cut-off strategies.
- **Fix**: Add boost modifier to actions (e.g., 3 relative actions × 2 speed modes = 6 actions, or simpler: 3 relative + 1 boost toggle = 4 actions). When boosting: move 2x speed, lose 1 length per N frames. Requires minimum length to boost.
- **Impact**: Unlocks aggressive play, chasing, escaping, cut-off strategies — the core slither.io gameplay loop.
- **Effort**: Moderate - new mechanic in snake, state representation, action space.

### 2.4 Implement Curriculum Learning [HIGH]
- **Source**: #2 Environment
- **Recommended phases**:
  1. Solo snake, small board, lots of food (learn movement + eating)
  2. Solo snake, full board, normal food (learn wall avoidance at scale)
  3. 2 snakes, full board (learn basic avoidance)
  4. 4 snakes, full board (learn competitive dynamics)
  5. 4 snakes, speed boost enabled (learn advanced strategies)
- **Fix**: Add `CurriculumManager` that adjusts `num_snakes`, board size, food density, and enabled mechanics based on training progress (e.g., average episode length milestones).
- **Impact**: Prevents early catastrophic forgetting and lets the agent build skills incrementally.
- **Effort**: Moderate - new class, config integration, training loop changes.

### 2.5 Fix Kill Attribution + Scale Kill Reward [MEDIUM]
- **Source**: #3 Reward, #8 Multi-Agent, #4 Slither Expert
- **Problem**: Kill detection in `_calculate_interaction_reward()` uses proximity (within 2 segment sizes). This produces false positives (nearby snake dies from wall) and false negatives (we killed it but moved away). Kill reward (+1.0) is also too weak vs food (+3.0) to incentivize aggressive play.
- **Fix**: Track actual collision pairs in `GameState.handle_collisions()` and expose `last_frame_kills: Dict[int, int]` (killer_id -> victim_id). Scale kill reward by victim size (killing a 50-segment snake should reward more than killing a 3-segment one). Suggested: `kill_reward = 1.0 + 0.05 * victim_length`, capped at 5.0.
- **Impact**: Enables learning aggressive strategies with accurate, properly-scaled incentives.

### 2.6 Enrich Enemy State Features [MEDIUM]
- **Source**: #1 State Rep, #4 Slither Expert
- **Problem**: Agent sees only nearest enemy's position and size. Blind to enemy heading (can't predict movement), blind to other 2 opponents, and has no concept of kill opportunities.
- **Fix (prioritized by impact)**:
  - Enemy heading: +2D (dx/dy unit vector of nearest enemy's direction)
  - 2nd nearest enemy: +3D (rel_x, rel_y, rel_size) — same format as current
  - Enemy distance trend: +1D (closing vs separating, from frame-over-frame delta)
  - Kill opportunity score: +1D (are we adjacent to an enemy's path?)
- **Total**: +7D minimum (heading + 2nd enemy + trend), up to +10D with kill opportunity
- **Impact**: Enables predictive avoidance, interception, and cut-off strategies.

### 2.7 Unify DuelingDQN / ApexNetwork [LOW]
- **Source**: #5 Architecture
- **Problem**: `DuelingDQN` in `apex_policy.py` duplicates `ApexNetwork` in `model/apex_network.py`. Two implementations to maintain.
- **Fix**: Delete `DuelingDQN` from `apex_policy.py`, use `ApexNetwork` (which uses the mixin architecture). Update imports.
- **Impact**: Code quality, single source of truth for network architecture.

---

## Phase 3: Major Refactors (High Impact, High Effort)

*Each item: 1-2 weeks. These are architectural changes that unlock the next tier of performance.*

### 3.1 Add Temporal Memory (GRU/DRQN) [HIGH]
- **Source**: #5 Architecture
- **Problem**: Single-frame input means the agent has no memory. It can't learn patterns like "this snake is circling toward me" or "I've been going in circles".
- **Fix**: Add optional GRU layer after feature extraction. Switch from experience replay to sequence replay (store and sample trajectories of length T). Use `DRQN`-style training with burn-in.
- **Impact**: Enables learning temporal strategies (ambush, pursuit, escape patterns).
- **Effort**: High - requires sequence buffer, modified training loop, new network variant.
- **Alternative**: Frame stacking (last 4 frames concatenated) is simpler but less powerful. Could be a stepping stone.

### 3.2 Sum-Tree Priority Buffer [MEDIUM]
- **Source**: Deferred from original audit (#9)
- **Problem**: Current O(N) sampling scans entire buffer. At 1M capacity, this becomes a bottleneck.
- **Fix**: Implement sum-tree data structure for O(log N) priority sampling.
- **Impact**: 10-100x sampling speedup at scale, enables larger buffers.
- **Effort**: High - careful implementation needed for correctness.

### 3.3 True Distributed Apex Training [MEDIUM]
- **Source**: #6 Training, #9 Exploration
- **Problem**: Single-process mode doesn't leverage Apex's key advantage: diverse exploration via multiple actors with different epsilons.
- **Fix**: Implement `mp.Process`-based actors feeding a `SharedPrioritizedBuffer`. Learner pulls batches on GPU. Already partially scaffolded in codebase (`apex_actor.py`, `apex_learner.py`, `apex_buffer.py`).
- **Impact**: Linear scaling with CPU cores, much better exploration coverage.
- **Effort**: High - distributed systems are tricky to debug.

### 3.4 Remove Fixed Walls / Add Wrapping or Circular Arena [LOW]
- **Source**: #2 Environment, #4 Slither Expert (summary)
- **Problem**: Fixed rectangular walls are a major deviation from slither.io (which has no walls / circular boundary). Agents spend disproportionate effort on wall avoidance.
- **Fix**: Implement either wrapping boundaries (pacman-style) or circular arena with soft boundary.
- **Impact**: More slither.io-like behavior, reduces wall avoidance dominance.
- **Effort**: High - affects collision, state representation, rendering.

---

## Implementation Order (Recommended)

```
Week 1: Phase 1 (all quick wins)
  Day 1: Fix double collision (#1.1)
  Day 1: Rebalance rewards (#1.2)
  Day 1: Slow epsilon (#1.3) + min buffer (#1.4)
  Day 2: Fix hidden size (#1.5) + remove dead config (#1.7)
  Day 2: Parameter sharing across snakes (#1.6) ← highest-value quick win
  Day 3: Run baseline training, measure improvement vs pre-fix

Week 2-3: Phase 2A (core gameplay)
  Days 1-3: Relative action space (#2.1) ← biggest sample efficiency gain
  Days 4-5: Per-action danger signals (#2.2)
  Day 5: Validate with training run

Week 3-4: Phase 2B (mechanics + curriculum)
  Days 1-3: Speed boost mechanic (#2.3)
  Days 4-5: Curriculum learning (#2.4)
  Day 5: Fix kill attribution + scale reward (#2.5)

Week 5-6: Phase 2C (state + architecture cleanup)
  Days 1-2: Enrich enemy state features (#2.6)
  Day 3: Network unification (#2.7)
  Days 4-5: Validate full Phase 2 with extended training run

Week 7+: Phase 3
  GRU/DRQN (#3.1) - biggest long-term payoff
  Sum-tree buffer (#3.2) - if scaling becomes bottleneck
  Distributed training (#3.3) - if hardware available
  Circular arena (#3.4) - if targeting slither.io fidelity
```

---

## Cross-Cutting Concerns

### Config Consistency
- Hidden size: standardize on 512 everywhere
- Remove all stale `input_size: 45` references
- Audit that `configs/production.yaml` (if it exists) matches `default.yaml`

### Testing Strategy
- Each Phase 1 fix should include a unit test proving the bug is fixed
- Phase 2 items need integration tests (run 100 episodes, verify no crashes)
- Phase 3 items need performance benchmarks (training curves, wall-clock time)

### Metrics to Track
- **Food per episode** (primary learning metric)
- **Average episode length** (survival skill)
- **Kill/death ratio** (competitive skill, should improve after #1.6, #2.5)
- **Wall death %** (should decrease after #1.1, #1.2)
- **Exploration coverage** (state space visited, should increase after #1.3)
- **Training throughput** (transitions/sec, should 4x after #1.6 parameter sharing)
- **Strategy diversity** (action entropy over episodes, should stay higher with #1.3)

---

## What NOT to Do

- **Don't add NoisyNets** (#9 Exploration): Incompatible with Apex's per-actor epsilon design. The current epsilon-greedy with varied epsilons is correct for Apex.
- **Don't add frame stacking AND GRU**: Pick one. GRU is strictly better but harder to implement. Frame stacking is the simpler alternative.
- **Don't increase to 8 directions**: 4 cardinal (or 3 relative + boost) is sufficient. 8 directions would require diagonal movement which changes game physics fundamentally.
- **Don't use Adam instead of AdamW**: The weight decay in AdamW is intentional regularization. The #6 Training report flagged this as a "mismatch" but AdamW is actually the better choice.
- **Don't expand state to 73D all at once** (#4 Slither Expert): The expert suggested 26 new features. Many are redundant with existing 16-sector maps. Add incrementally: per-action danger → enemy heading → 2nd enemy. Validate each addition with training curves before adding more.
- **Don't keep independent policies** (#8 Multi-Agent): Parameter sharing is not optional — it's fundamental to how Apex DQN works. Independent learners is the single biggest architectural mistake in the current design.

---

*Generated by synthesizer agent from findings of: state-rep-analyst, env-analyst, reward-analyst, slither-expert, arch-analyst, training-analyst, action-analyst, multiagent-analyst, exploration-analyst*

---

## Appendix: Review Team Supplementary Findings

*10-agent verification and gap analysis (Feb 2026). Cross-references, verifies, and extends the roadmap above.*

### Bug Verification Results (11/12 confirmed, 4 new found)

| # | Bug | Verified? | Notes |
|---|-----|-----------|-------|
| 1.1 | Double collision | YES — actually **TRIPLE** | `Snake.move()` has its own `self.die()` at line 413, making 3 death paths |
| 1.2 | UI double-respawn | YES | Timer decremented 2x/frame in GUI mode |
| — | Food > MAX_FOOD | YES | Death drops bypass cap (arguably a feature) |
| — | Wall boundary inconsistency | PARTIAL | Latent bug, not active with current factory |
| — | max_length not enforced | YES | Normalized length can exceed 1.0 |
| — | input_size=45 in configs | YES | Also in 6+ Python default params (see Config Audit below) |
| 1.5 | Hidden size mismatch | YES (nuanced) | 512 default never executes; 256 always passed explicitly |
| — | MultiStepBuffer beta_increment | YES | 0.0001 hardcoded, config says 0.000001 |
| — | Duplicate network classes | YES | DuelingDQN duplicates ApexNetwork |
| — | Kill attribution proximity | YES | Heuristic, false positives possible |
| — | Dead config params | YES | food_length_mult, food_length_exp, 6 wall threshold configs unused |
| NEW | maintain_count double-call | NEW | GUI mode calls food maintenance twice per frame |
| NEW | production.yaml `state:` section | NEW | Silently ignored — `max_length: 150` never takes effect |
| NEW | Initial food spawns inside snakes | NEW | `_spawn_initial()` doesn't check snake positions |
| NEW | Respawn doesn't avoid other snakes | NEW | `get_random_position()` has no overlap check |

### Exhaustive Config Audit — input_size=45 in 6+ Places

The `input_size=45` bug is far more widespread than originally identified:
- `configs/production.yaml:16`, `configs/training_fast.yaml:16`
- `ApexLearnerConfig` default (`apex_learner.py:55`)
- `create_apex_learner()` factory (`apex_learner.py:681`)
- `ApexNetwork` default (`apex_network.py:50`) + docstrings
- `create_apex_network_pair()` and `create_apex_actor_network()` defaults
- `ApexConfig.state_dim` (`apex_train.py:72`)
- `colab_loader.py:59` fallback default
- `network_visualizer.py` comment says "45 neurons"

### Config Audit — Three Different beta_increment Defaults

| Location | Default | Config Value |
|----------|---------|-------------|
| `PrioritizedReplayBuffer` | 0.001 | 0.000001 |
| `MultiStepBuffer` | 0.0001 | 0.000001 |
| `default.yaml` | 0.000001 | — |

Also: `priority_eps` is 1e-5 in `PrioritizedReplayBuffer` but 1e-6 in `ApexSettings`. `num_actors=8` hardcoded in `apex_policy.py` but config says 64.

### Colab vs Main Codebase — Critical Drift Found

Despite claims of unification, the review found real differences:
1. **Danger map algorithm mismatch**: Colab uses ray-casting projection; main uses Euclidean pixel-distance. Same indices, different values for same game state.
2. **`colab_loader.py` broken load path**: Looks for `colab_architecture` key but export uses `config` key, falls back to `input_size=45`.
3. **Wall proximity threshold 5x mismatch**: Colab activates at ~3% of grid; main at 15%.
4. **Reward clamp ranges differ**: Colab [-5, 5] vs main [-10, 2].

### Test Coverage — Massive Gaps

**Zero tests for**: replay buffer, reward shaping components, state vector values, DuelingDQN model, training step/gradient updates, action masking, FoodManager, GameState lifecycle, SnakeFactory.

Existing tests check shapes and types but almost never check values. None of the bugs found by either research team would be caught by the current test suite.

### Checkpoint System Issues

1. **No input_size validation on load** — `strict=False` everywhere silently loads corrupted weights when state dim changes
2. **`--load` silently ignored in headless mode** — no warning printed
3. **`load_best_snake()` passes `self` (GUI widget) as policy arg** — would crash if called
4. **`SharedReplayBuffer._lock = None`** — not thread-safe despite multi-actor design
5. **Silent `except Exception: pass`** in memory loading swallows all errors

### Edge Cases & Race Conditions

1. **Update order race condition**: Later snakes see earlier snakes' NEW positions but later snakes' OLD positions. Creates systematic advantage for higher-indexed snakes.
2. **Stale priority indices**: Deque-based PER buffer can have indices pointing to wrong experiences after overflow between `sample()` and `update_priorities()`.
3. **N-step discount over-discounts partial returns**: Uses fixed `gamma^n` even for partial sequences at episode boundaries.
4. **Kill rewards depend on snake iteration order**: Snakes earlier in list can never get kill credit for snakes later in list dying on same frame.
5. **Eat-food-then-die loses food reward**: Death check comes first in `calculate_reward()`, so food reward is lost on death frame.
6. **Food drops missing for self-caused deaths**: Snakes dying inside `update()` via `_check_collision()` skip `_drop_food_from_snake()`.

### Import Graph — Layer Violations & Dead Code

- **`apex_actor.py` imports `GameState`** — training layer depends on game layer (should use DI)
- **Dead files**: `utils/logger.py` and `utils/cli.py` are never imported anywhere
- **Dead functions**: `normalize_batch_rewards()`, `clip_rewards()`, `build_tensor_batch()` in `tensor_utils.py`
- **`slitherio.py` DI violation**: `load_best_snake()` constructs `AISnake` directly, passing `self` (QWidget) as policy
- **Duplicate `TrainingDashboard`**: Rich terminal version in `cli.py` + PyQt5 version in `training_dashboard.py`

### Missing from Roadmap — New Items to Add

**Phase 0 (Pre-requisites):**
- Add `pyyaml` to `requirements.txt` (currently only in dev — `--config` crashes in production)
- Make `rich` import conditional in `cli.py` (or add to `requirements.txt`)
- Add `mp.set_start_method('spawn')` to headless mode (macOS + PyQt5 + fork = deadlock)

**Phase 1 Additions:**
- Wire up existing `APEX_MIN_BUFFER_SIZE` config in `ApexPolicy.train_step()` (config exists but unused)
- Consolidate 3 independent QTimers (radar 50ms, inspector 100ms, dashboard 500ms) into one
- Cap `MetricsTracker` memory (currently `defaultdict(list)` grows unbounded)
- Fix O(N) `max()` scan on every `PrioritizedReplayBuffer.add()` call (not just sample)

**Dependency Corrections:**
- §2.7 (network unification) MUST come before §1.5 (hidden size) — can't standardize size without knowing which class survives
- §1.6 (parameter sharing) SHOULD come before §1.3 (epsilon) — behavior differs with shared vs independent
- §2.1 (relative actions) MUST come before §2.2 (per-action danger) — danger dimensions depend on action count
- §2.3 (speed boost) depends on §2.1 (relative actions) — action space design depends on it
- §1.1 (collision fix) MUST come before §2.5 (kill attribution) — can't attribute kills if collisions are processed in 3 places

**"What NOT to Do" Additions:**
- Don't change state dimensions without a migration plan for existing checkpoints
- Don't fix hidden size (§1.5) before unifying networks (§2.7)
- Don't use `torch.load(weights_only=False)` for untrusted checkpoints (security risk)

### Third Training Code Path: OnlineTrainer

The roadmap discusses DuelingDQN vs ApexNetwork (§2.7) but misses that `OnlineTrainer` is a **third** complete training implementation. It duplicates ApexPolicy's logic with different return calculations (single-step vs n-step). Should be consolidated or clearly differentiated.

---

*Verified by review team: bug-verifier, config-auditor, test-analyst, colab-checker, state-tracer, reward-tracer, import-analyst, edge-case-hunter, checkpoint-reviewer, roadmap-reviewer*
