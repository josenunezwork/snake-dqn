# ML Algorithm Design — Apex DQN for Multi-Snake RL

This document describes the reinforcement-learning algorithm the project uses,
how it is implemented, and the design decisions behind it. It is a *design*
reference — no training is required to read or use anything here. For running
trained models see [`InferenceAgent`](../src/model/inference_agent.py); for the
state-vector layout see [`SnakeStateMixin.get_state`](../src/game/snake_state.py).

**Scope.** This document covers the **Apex vector stack**: `ApexNetwork` +
the 58/61-D hand-crafted state. That stack is no longer the whole codebase —
the redesign branch adds a second, independent track (`PQNTrainer`,
`RasterDuelingNetwork` on the vectorized `BatchSim`) built around ego-raster
observations. For that design and the rationale for re-opening architecture
questions, see
[`docs/ml_redesign_blueprint_2026-07.md`](ml_redesign_blueprint_2026-07.md).
Within *this* stack the consolidation to a single algorithm and network still
holds, and is itself a design choice: it keeps the surface area small enough to
reason about and tune.

**Historical record — read before citing "we already tested that."** GRU/DRQN
was trained and did lose paired, frozen-opponent benchmarks to the feedforward
model. The **CNN was never trained**: it was written and then deleted during
consolidation, with zero checkpoints ever produced, so it lost nothing. Earlier
drafts of this file claimed both variants "lost every benchmark" — refuted by
this repo's own verification appendix (blueprint Appendix B, claims 1 and 10).
The **CNN question is open**, and the raster/PQN stack is re-testing it. Note
also that the GRU/DRQN elimination was judged under the *old* promotion gate,
which has since been repaired (see §8) — so it carries less evidential weight
than a clean result would.

---

## 1. Problem shape

Several snakes share one arena. Each is an agent that, every frame, observes a
fixed-length feature vector and picks one of six discrete actions. Food grows a
snake; colliding with a wall, itself, or another snake's body kills it. Killing
another snake is rewarded. This is a **multi-agent, partially-observed,
discrete-action** control problem, which is a natural fit for value-based deep
RL.

All AI snakes share a **single policy** (parameter sharing). One network is
trained on the pooled experience of every snake; at play time every AI snake
runs the same weights. This is what makes a 6-snake arena trainable from one
GPU learner.

### Action space (6 actions)

Relative to the snake's current heading, so the policy never has to reason about
absolute compass directions:

| Action | Meaning            | Speed  |
|:------:|--------------------|--------|
| 0      | turn left          | normal |
| 1      | go straight        | normal |
| 2      | turn right         | normal |
| 3      | turn left          | boost  |
| 4      | go straight        | boost  |
| 5      | turn right         | boost  |

Boost trades length for speed (see `boost_length_cost_frames` /
`min_boost_length`). The human-play mode maps arrow/WASD keys onto the same
turn semantics via [`HumanSnake.apply_direction_input`](../src/game/human_snake.py).

---

## 2. Network — feedforward Dueling DQN

Implemented in [`ApexNetwork`](../src/model/apex_network.py).

```
state (58 or 61)
      │
      ▼
 shared feature extractor:  Linear→ReLU (H) → Linear→ReLU (H/2)
      │
      ├──────────────► value stream:      Linear→ReLU(H) → Linear(1)      V(s)
      │
      └──────────────► advantage stream:  Linear→ReLU(H) → Linear(6)      A(s,a)
                                   │
                                   ▼
              Q(s,a) = V(s) + ( A(s,a) − mean_a A(s,a) )
```

with `H = hidden_size = 512` (so the streams operate on a 256-D feature).

**Why dueling.** Most states in Snake have a dominant "how good is it to be here"
signal (am I boxed in? is food near?) that is independent of which of six turns I
take. Splitting `V(s)` from `A(s,a)` lets the network learn state value from every
transition, not only from the action actually taken, and the mean-subtraction
keeps the decomposition identifiable. The dueling combine lives in one helper,
`dueling_q(value, advantage)`, shared by `forward` and the visualization path.

**Why feedforward (not recurrent/convolutional).** The state vector is already a
hand-engineered, egocentric summary (danger sectors, nearest-enemy features,
free-space) — the temporal and spatial structure a GRU or CNN would have to
rediscover is pre-baked. That is the *a priori* argument. The *empirical* support
covers the recurrent case only: the feedforward + 61-D model beat GRU/DRQN on
frozen-opponent, paired-seed benchmarks, and DRQN was deleted. The convolutional
variant was deleted **untested** (never trained, no checkpoints), so nothing here
should be read as a CNN result — the raster/PQN track is running that experiment
properly.

---

## 3. State representation

Two selectable encodings (config `use_free_space`):

- **58-D** hand-crafted egocentric vector — direction one-hot, normalized length,
  food direction/distance/density sectors, danger sectors, wall distances,
  two-nearest-enemy features, kill-opportunity, per-action danger, boost-available.
  Feature construction and ordering live in [`SnakeStateMixin.get_state`](../src/game/snake_state.py).
- **61-D** = 58-D **+ 3 "don't-trap-yourself" free-space features**. These count
  reachable open space after a left / straight / right turn (a cheap flood-fill
  proxy), which directly attacks the dominant failure mode of the 58-D model:
  self-collision by coiling into a dead end.

The 61-D free-space model is the production winner
([`configs/free_space_v2.yaml`](../configs/free_space_v2.yaml),
`saved_snakes/champion_a5_freespace_20260621.pth`). The free-space fix is a
**state** fix, not a reward fix — the information the agent needed to avoid
trapping itself simply wasn't in the observation before.

---

## 4. Learning algorithm — Apex DQN

"Apex" (Ape-X) = **distributed prioritized experience replay** around a DQN core.
The value-learning ingredients, and where each lives:

### 4.1 TD target — Double DQN + n-step

[`src/training/td_targets.py`](../src/training/td_targets.py):

```
a*      = argmax_a  Q_online(s_{t+n}, a)          # action chosen by the online net
next_q  = Q_target(s_{t+n}, a*)                    # value read from the target net
target  = Σ_{k=0..n-1} γ^k r_{t+k}  +  (1 − done) · γ^n · next_q
```

- **Double DQN** (`double_dqn_next_q`): the online network selects the bootstrap
  action, the target network evaluates it. Decoupling selection from evaluation
  removes the maximization bias of vanilla DQN, which matters here because
  optimistic Q-values encourage exactly the reckless boosts we want to avoid.
- **n-step returns** (`n_step_td_target`, default `n = 3`): propagate reward over
  a short horizon before bootstrapping. Faster credit assignment for
  "ate food 3 frames after turning" without the variance of full Monte-Carlo.
  The bootstrap length is *per-transition* (episodes that end early carry a
  shorter horizon), so terminal transitions are handled correctly.
- Targets are clamped to `±q_clip` for stability.

### 4.2 Prioritized replay (PER)

[`SumTree`](../src/training/sum_tree.py) gives O(log N) sampling proportional to
TD-error priority, with importance-sampling weights to correct the induced bias
(`priority_epsilon` keeps every transition reachable). Transitions the network is
most wrong about are replayed most — the arena produces long stretches of "boring"
survival frames, and PER stops those from drowning out the rare death/kill frames
that actually carry the learning signal.

### 4.3 Target network

A periodically hard-updated copy of the online network
(`target_update_freq ≈ 2500` learner steps) supplies the bootstrap value,
stabilizing the regression target.

### 4.4 Action masking — "don't walk into a wall"

[`src/training/action_mask.py`](../src/training/action_mask.py). The per-action
danger features in the state identify immediately-fatal moves. A validity mask is
applied **both**:

- at **action selection** (never greedily pick a certain-death action when a safe
  one exists), and
- inside the **TD target** for the next state (don't bootstrap off the value of a
  move the agent would never be allowed to make).

Masking both places keeps the behavior policy and the learning target consistent.
`InferenceAgent.act_safe()` exposes the same idea for read-only serving.

---

## 5. Distributed topology

```
   ┌───────── actor 0 (CPU, ε₀) ─────────┐
   │  ...                                │   experiences        ┌── BufferProcess ──┐
   ├───────── actor k (CPU, εₖ) ─────────┼───────────────────► │ SumTree PER store  │
   │  ...                                │                      └─────────┬─────────┘
   └───────── actor N (CPU, ε_N) ────────┘                                │ prioritized
                    ▲                                                     ▼  minibatches
                    │  weight broadcast (periodic)          ┌──────── Learner (GPU) ───────┐
                    └───────────────────────────────────────┤ Double-DQN + n-step + PER     │
                                                            └──────────────────────────────┘
```

- **Actors** ([`apex_actor.py`](../src/training/apex_actor.py)) run on CPU with
  **different epsilons** (`epsilon_base`, `epsilon_alpha`) so the buffer sees a
  spectrum from near-greedy to exploratory — Apex's exploration-diversity trick.
- **Buffer** ([`apex_buffer.py`](../src/training/apex_buffer.py)) owns the SumTree
  and serves prioritized minibatches over IPC.
- **Learner** ([`apex_learner.py`](../src/training/apex_learner.py)) does all
  gradient work on GPU and broadcasts fresh weights back to the actors.

A single-process local path exists for Mac dev (the shared-policy game loop in
[`game_state.py`](../src/game/game_state.py) trains one policy directly).

---

## 6. Reward shaping & curriculum

[`CurriculumManager`](../src/training/curriculum.py) ramps difficulty through four
phases — **survival → food-seeking → enemy-awareness → kill-optimization** —
by adjusting reward weights, food density, and opponent count. The intent is to
avoid the cold-start trap where a randomly-initialized snake dies before it ever
sees food, and to defer the hard multi-agent objectives until the basics are
learned. Kill attribution (collision-pair tracking) scales the killer's reward by
the victim's length.

The **reward contract** ([`src/core/reward_contract.py`](../src/core/reward_contract.py))
and **checkpoint contract** ([`src/training/checkpoint_contract.py`](../src/training/checkpoint_contract.py))
pin down the reward constants and architecture dims a checkpoint was trained
under, so a resume can't silently mix incompatible `gamma` / `n_step` / `hidden`.

---

## 7. Key hyperparameters (production 61-D winner)

From [`configs/free_space_v2.yaml`](../configs/free_space_v2.yaml):

| Knob                 | Value    | Notes                                   |
|----------------------|----------|-----------------------------------------|
| `input_size`         | 61       | 58-D + 3 free-space features            |
| `hidden_size`        | 512      | streams operate on 256-D                |
| `output_size`        | 6        | 3 turns × 2 speed modes                 |
| `gamma`              | 0.99     | discount                                |
| `n_step`             | 3        | n-step return horizon                   |
| `learning_rate`      | 1e-4     | Adam                                     |
| `batch_size`         | 256      |                                         |
| `buffer_size`        | 200k     | PER capacity (1M in full distributed)   |
| `min_buffer_size`    | 10k      | warmup before learning                  |
| `target_update_freq` | 2500     | learner steps between hard updates      |

---

## 8. Inference / serving path

Training state (optimizer, target net, replay buffer, epsilon schedule) is not
needed to *use* a trained snake. Two options:

- [`ApexPolicy(training=False)`](../src/training/apex_policy.py) — the training
  class in eval mode; what the live web session currently loads.
- [`InferenceAgent`](../src/model/inference_agent.py) — a minimal, forward-only
  wrapper: load a checkpoint, get Q-values / a greedy (optionally masked) action /
  activations. No training machinery, cheap to build, ideal for eval scripts,
  tournaments, and read-only serving:

  ```python
  from src.model.inference_agent import InferenceAgent
  agent = InferenceAgent.from_checkpoint("saved_snakes/champion_a5_freespace_20260621.pth")
  action = agent.act_safe(state_vector)   # greedy with don't-trap masking
  ```

Promotion is gated by [`tournament_eval.py`](../src/scripts/tournament_eval.py)
— **not** by training reward, which self-play inflates (Red Queen effect).

The gate's headline metric is the **mass integral**: mean per-frame mass over the
**total** episode horizon, dead frames contributing **0**. It replaced an
alive-frames-only `mean_mass` under which dying rich outranked surviving — i.e.
the old gate could promote regressions, so results judged under it (including the
GRU/DRQN elimination) are weaker evidence than they look. `mean_mass_alive`
survives as a legacy diagnostic; nothing gates on it.

The rule, enforced in `promotion_decision()` and reported through the `--gate`
exit code (0 = promote, 1 = reject): promote iff the **paired** per-seed
mass-integral delta vs `--baseline` is **> 0 at 95% CI on ≥ 2 distinct opponent
mixes** AND shows **no regression vs the scripted anchor mix**. Candidate and
baseline share seeds (common random numbers); opponents are drawn from three
mixes (`frozen` / `scripted` / `mixed`) rather than clones of one checkpoint.

---

## 9. Design directions worth exploring (no training required to design)

Ideas that fit the current architecture, listed as design work rather than
training runs:

1. **Distributional value head (C51/QR-DQN).** Replace the scalar `V/A` heads with
   a return *distribution*. Snake returns are heavy-tailed (a lucky kill chain vs.
   an early death); modeling the distribution often improves the mean policy and
   gives a natural risk-sensitivity knob for boosting. `ApexNetwork.num_atoms`
   already anticipates this (currently 1).
2. **Learned free-space features.** The 61-D free-space features are a hand-coded
   flood-fill proxy. A tiny auxiliary head predicting reachable area could learn a
   sharper signal — but only if it clears the repaired promotion gate (§8), the
   same bar GRU/DRQN failed (under the gate's older, weaker form).
3. **Munchausen DQN.** A one-line target augmentation (add a scaled log-policy
   term) that is cheap to implement and frequently a free win for discrete DQN.
4. **Opponent-aware evaluation heads.** Currently every snake shares one policy;
   a small conditioning input (e.g. opponent aggression estimate) could let one
   network express population diversity without N networks.

Any of these should be prototyped behind the checkpoint/reward contracts and
judged on `tournament_eval.py`, never on training reward.
```
