# State and system review — 2026-09-06

## Decision

**Keep the incumbent vector61 Apex checkpoint frozen. Continue investing in the
ego-raster plus `BatchSim` direction, but do not treat the current raster/PQN
path as a semantically validated replacement.** The engineering chassis is
promising: a vectorized cell simulator, integer ego rotation, masked six-action
control, and a spatial network are a credible way to test this game. The next
work is to repair versioned contracts, then run small single-axis comparisons.
It is not a case for a larger CNN, GRU, Transformer, automatic PPO switch, or
promotion of a raster checkpoint.

The completed sampler screen remains unchanged and inconclusive. Its five
independent seed pairs found complete-epoch minus legacy mass integral of
-1.3694 with 95% CI [-4.748, +2.009], so it establishes neither sampler
superiority nor an algorithm ranking. See
[the sampler results](pqn_sampler_results_2026-09-06.md) for its 10 arms,
5,004,509 transitions, and held-out SIMD screen.

This review addresses a different question: whether the state, transition,
reward, training, evaluation, and serving systems agree on a reproducible
learning problem. Its reader should understand basic RL returns, observation
aliases, and paired evaluation. The audit worktree began at `c1cf7e8`; its
baseline is `origin/main` `950d583`. Product source was read, not modified.
The evidence bundle pins the audited source and records individual probes under
[`docs/experiments/state_system_audit_2026-09-06`](experiments/state_system_audit_2026-09-06/).

## What is promising

The incumbent is a 61-feature feed-forward Apex DQN. The candidate is a
dual-scale raster dueling network trained by synchronous Q(lambda) over
`BatchSim`. Both use relative left/straight/right plus boost actions. This is a
good experimental seam: the candidate can change spatial representation and
learner without requiring an incompatible action interface or an entirely new
product inspector.

`BatchSim` is a valuable asset, not a claim that every engine contract is
settled. It provides a cheap, vectorized dynamics loop and has a parity harness
([`parity.py`](../src/simd_env/parity.py)). The raster can use exact cardinal
rotation with integer cells and no interpolation. The network retains Q values
and dueling streams, so existing decision traces and visualization can remain
useful while observations are investigated.

The audit does not classify finite spatial horizons, enemy summaries, or a
partially observed multi-agent game as defects by themselves. A finite local
view may be the desired product constraint; a hidden variable becomes a learning
priority only after its prevalence and action consequence are measured. The
findings below distinguish such design limits from proven cross-system
contradictions.

## Representation map

| Surface | What it contains | Strength | Observed limit or question | Consequence now |
| --- | --- | --- | --- | --- |
| Vector61 | Bounded direction, food, sectors, danger, walls, nearest/second enemy summaries, per-action danger, boost and three free-space values | Compact, replay-compatible, logical mass and an additive 58→61 checkpoint path | Compresses many entities; second-enemy heading is absent; world-frame spatial features pair with relative actions; danger/free-space/mass features can saturate | Preserve it as the incumbent baseline; measure an ego-vector rotation ablation before calling it deficient |
| Raster tactical | Nine ego-aligned integer planes over a forward-biased 31×31 window | Cardinal rotation is exact; it retains local occupancy, food classes, walls and body ordering | 23 cells ahead and 7 behind; its declared type priority can be overwritten; local body order is not a true vacancy-time model | Repair paint precedence and test action impact before adding channels |
| Raster strategic | Three 25×25 coarse planes, five raw cells per bin | Cheap broad spatial context | Its ±62-cell span does not cover a 145-cell-wide board; the final strategic convolution has a 7×7 coarse-cell / 35 raw-cell receptive field before global dense mixing | Measure strategic-horizon action regret before changing capacity |
| Raster scalars | 26 own-state, hunger/boost/progress, wall, food and two-enemy summaries plus world x/y and arena flag | Carries useful beyond-raster summaries and observable counters | It has two enemy summaries, one far-food fallback, and world coordinates that complicate reflection augmentation | Version any feature/augmentation change and test it separately |
| Network | Two conv trunks, FC fusion and dueling head; 1,435,751 parameters | Shares downstream Q visualization; 73.1% of parameters are tactical flatten→FC | Tactical conv receptive field is 9×9; no conv LayerNorm; 73.1% of parameters sit in tactical FC | Treat normalization/topology as ablations, not an assumed bottleneck |

The supporting network analysis is
[`raster/network_analysis.json`](experiments/state_system_audit_2026-09-06/raster/network_analysis.json).
The present model applies LayerNorm after tactical, strategic and fused fully
connected layers, not the convolutional stacks or dueling hidden layers
([`raster_network.py`](../src/model/raster_network.py)). The trainer uses Adam
and Huber loss ([`pqn_trainer.py`](../src/training/pqn_trainer.py)). These are
intentional deviations from the pinned author PQN implementation, which applies
[LayerNorm more broadly](https://github.com/mttga/purejaxql/blob/47af6d7b35c89ddfe633aaf7341bdb8964cb7cce/purejaxql/pqn_atari.py#L25-L67),
uses [RAdam](https://github.com/mttga/purejaxql/blob/47af6d7b35c89ddfe633aaf7341bdb8964cb7cce/purejaxql/pqn_atari.py#L177-L183),
and uses a [half-squared TD loss](https://github.com/mttga/purejaxql/blob/47af6d7b35c89ddfe633aaf7341bdb8964cb7cce/purejaxql/pqn_atari.py#L291-L305).
They are not proof that importing those choices improves this game.

### What the alias probes establish—and what they do not

The vector probe found equal current observations with different hidden boost
phase or hunger counter values. Starting from the same logical length, boost
phase 0 versus 2 yields next lengths 6 versus 5; hunger 0 versus 600 under v1
reward yields -0.071034 versus -0.171034. The raster probe likewise constructed equal hero observations with
different remote enemy headings or ambient-versus-corpse food class and then
different next observations under the same joint actions. See
[`vector/vector_contract_probe.json`](experiments/state_system_audit_2026-09-06/vector/vector_contract_probe.json),
[`raster/remote_heading_transition.json`](experiments/state_system_audit_2026-09-06/raster/remote_heading_transition.json),
and [`raster/food_class_transition.json`](experiments/state_system_audit_2026-09-06/raster/food_class_transition.json).

Those are formal observation aliases. They do **not** establish a different
optimal action, show that a longer history wins, or require recurrence. They
are a reason to measure (a) how often each alias occurs and (b) whether an
oracle with the omitted variable changes action ordering. World-coordinate
features plus relative actions can make rotation a learnable nuisance; they do
not mean heading is missing.

The current flip augmentation has a separate physical-realizability problem.
It mirrors ego rasters and lateral fields while leaving absolute world x/y
unchanged. A full-world reflection needs the heading to decide whether x or y
is complemented; the canonical observation has discarded that transform choice.
Test a physically realizable transform, coordinate removal, or disabled
augmentation under a versioned observation specification rather than selecting
one repair by intuition. The targeted reproduction remains documented in
[`prior_audit/audit_reproductions.json`](experiments/state_system_audit_2026-09-06/prior_audit/audit_reproductions.json),
which records the source hashes used for that prior reproduction.

Distributed Apex is a separate representation-distribution concern. Its default
actor board scale is .2 and food scale .5: roughly 290×166 against the 1450×830
full board while the 300-pixel danger radius stays fixed. In one seed-906,
120-frame probe, 44 of 61 features had KS distance above .25 (maximum .943)
across 698 versus 720 observations. This is a single-trajectory diagnostic,
not a population confidence interval. Normal headless defaults are full scale,
and the A5 checkpoint records `distributed=False`; its historical board scale
is not proved by the checkpoint. See
[`vector/actor_vs_full_obs.md`](experiments/state_system_audit_2026-09-06/vector/actor_vs_full_obs.md)
and [`apex_train.py`](../src/scripts/apex_train.py#L1150-L1165),
[`apex_actor.py`](../src/training/apex_actor.py#L500-L507).

## Prioritized correctness findings

These are defects because they make an intended semantic relation false, not
because they make the game simpler or partially observed. Severity reflects the
risk to serving, target construction, reproducibility, or gate authority.

### S01 — Raster Play serves Q rows to the wrong snakes

The web raster-policy path produces Q rows for all alive snakes while the live
game calls policy actions only for AI snakes. With a human in slot 0, the first
AI is slot 1 but receives Q row 0; later AI snakes are shifted as well. This is
a deterministic serving defect, reproduced against a real `GameSession`.

- Evidence: [`serving/serving_contract_probe.json`](experiments/state_system_audit_2026-09-06/serving/serving_contract_probe.json).
- Source: [`raster_policy.py`](../web/backend/raster_policy.py#L185-L198) and
  [`game_state.py`](../src/game/game_state.py#L345-L355).
- Repair gate: construct rows for the exact AI-slot list or return a
  slot-indexed mapping; add human-plus-multiple-AI row identity tests for both
  raster and vector serving.

### S02 — Tactical raster paint order violates its own precedence and slot symmetry

The tactical producer says a higher type code wins, but a later prediction code
4 can overwrite enemy body code 6 or a wall. Swapping only enemy IDs in the
same physical world can therefore change the hero tactical observation. The
CPU and Torch producers can agree because they implement the same order; byte
parity does not validate the intended semantic precedence.

- Evidence: [`raster/enemy_slot_permutation.json`](experiments/state_system_audit_2026-09-06/raster/enemy_slot_permutation.json) and
  [`raster/probes.json`](experiments/state_system_audit_2026-09-06/raster/probes.json).
- Source: [`featurizer.py`](../src/simd_env/featurizer.py#L252-L435) and
  [`gpu_featurizer.py`](../src/simd_env/gpu_featurizer.py#L282).
- Repair gate: define one explicit per-cell reduction priority, test every
  collision pair and enemy-ID permutation, then assert NumPy/Torch parity
  against that oracle rather than only each other.

### S03 — `train_pqn --config` silently ignores much of shared world and hardware YAML

PQN reads its `pqn:` block plus a small shared subset, but ignores geometry,
food limits, segment/wall dimensions, boost timings, frame rate, and YAML
hardware device. A probe requesting a 777×666, segment-7, wall-3, food-17/19,
boost-9/11, 12-fps world instead built the `BatchSimConfig` defaults
1450×830, segment 10, wall 10, food 250/300, boost 5/3, and frame rate 1.
`training_fast.yaml` shows the same class of ignored geometry/food settings.
Device resolution happens before the YAML fields are mapped.

- Evidence: [`config/config_system_probe.json`](experiments/state_system_audit_2026-09-06/config/config_system_probe.json) and
  [`config/source_line_refs.txt`](experiments/state_system_audit_2026-09-06/config/source_line_refs.txt).
- Source: [`train_pqn.py`](../src/scripts/train_pqn.py#L84-L139),
  [`train_pqn.py`](../src/scripts/train_pqn.py#L381-L397), and
  [`pqn_trainer.py`](../src/training/pqn_trainer.py#L301-L335).
- Repair gate: reject unsupported shared keys or map and serialize every
  effective field before construction; resolve the documented YAML device
  authority before the trainer is built; checkpoint a complete effective world.

The mechanics-v2 sampler campaign is not invalidated by this finding: it used
the `BatchSim` world defaults and frame rate 1 consistently in both arms. The
finding prevents general claims that any supplied shared YAML describes a PQN
world.

### S04 — `reward_version` is metadata, not a BatchSim objective switch

`PQNConfig.reward_version` can say 1 or 2 and checkpoints record that label,
but `BatchSim` has no corresponding reward-version field and produces the v2
growth result in both cases. The objective probe observed +0.0994 in both
labels. This lets an artifact claim one reward contract while optimizing
another.

- Evidence: [`pqn/objective_contract_probe.json`](experiments/state_system_audit_2026-09-06/pqn/objective_contract_probe.json).
- Source: [`pqn_trainer.py`](../src/training/pqn_trainer.py) and
  [`batch_sim.py`](../src/simd_env/batch_sim.py).
- Repair gate: reject unsupported reward versions or dispatch them honestly;
  version the objective semantics and do not relabel old v2 checkpoints as v1.

### S05 — Action masks are neither a joint-transition death oracle nor a complete legal domain

The safety computation can reject a straight action into an enemy cell that
vacates this frame, and permit a converging head-on whose heads end in the same
cell. The vector actor
records the exact collected mask faithfully in replay, but correct recording
does not make the advice perfect safety. The earlier raster/PQN audit also
showed an all-false advisory mask that still admits a live boost action; using
that event as an immediate `death_value` target corrupts learning.

- Evidence: [`vector/vector_contract_probe.json`](experiments/state_system_audit_2026-09-06/vector/vector_contract_probe.json) and
  [`prior_audit/audit_reproductions.json`](experiments/state_system_audit_2026-09-06/prior_audit/audit_reproductions.json)
  (prior source-hash evidence).
- Source: [`ai_snake.py`](../src/game/ai_snake.py#L24-L98),
  [`game_state.py`](../src/game/game_state.py#L274-L355), and
  [`snake_state.py`](../src/game/snake_state.py#L419-L528).
- Repair gate: distinguish always-nonempty domain legality from safety advice;
  use actual `done` as death termination; decide and version the target mask
  policy. Do not blindly zero all next Q values: a legal-only max is a separate
  overestimation ablation, not a free correction.

### S06 — PQN can save a poisoned checkpoint after detecting a non-finite update

The SGD update happens before tripwire evaluation. A synthetic NaN-target probe
made all 30 parameter tensors non-finite; the loop then dropped incident
history and unconditionally overwrote `latest_pqn.pth` with the bad state.
No campaign NaN was observed—this is fault injection, not a reported training
event.

- Evidence: [`pqn/objective_contract_probe.json`](experiments/state_system_audit_2026-09-06/pqn/objective_contract_probe.json) and
  [`verification/major_claims_probe.json`](experiments/state_system_audit_2026-09-06/verification/major_claims_probe.json).
- Source: [`pqn_trainer.py`](../src/training/pqn_trainer.py#L725-L765),
  [`pqn_trainer.py`](../src/training/pqn_trainer.py#L874-L944), and
  [`train_pqn.py`](../src/scripts/train_pqn.py#L484-L506).
- Repair gate: validate targets, loss, and gradients before `optimizer.step`,
  then validate resulting parameters and optimizer state afterward. On either
  failure preserve or roll back to a known-good state and write a durable
  incident record instead of replacing the recovery artifact.

### S07 — Distributed Ape-X PER priorities can update a reused ring slot

The distributed buffer samples bare indices. A delayed priority update can
arrive after the ring has reused that index, overwriting the current row's
priority. A capacity-two probe sampled old markers 1/2, replaced them with
101/102, then observed the old priorities 20/30 applied to the new entries.

- Evidence: [`apex/stale_per_generation_probe.json`](experiments/state_system_audit_2026-09-06/apex/stale_per_generation_probe.json).
- Source: [`apex_buffer.py`](../src/training/apex_buffer.py#L573-L677) and
  [`sum_tree.py`](../src/training/sum_tree.py#L55-L82).
- Repair gate: return slot generation tokens with samples, reject stale updates
  and count them; then bound queued training pressure. Frequency in a realistic
  run is not measured by this probe.

## Contracts that can make experiments incomparable

### Training and reset semantics

Self-play policy IDs are redrawn at every PQN rollout. The probe records an
all-alive environment whose IDs change from `[-1,0,-1,-1,0,2]` to
`[-1,-1,-1,-1,0,0]`, changing hero eligibility within an episode. Q(lambda)
stops at rollout edges, so this does not claim a return recursively crosses a
switch. It does mean the policy environment is not frozen per episode. Decide
whether that is intentional; if it is, name it and report exposure by policy.

`BatchSim` construction and its later reset consume food draws differently from
`GameState`: initial worlds match, then independently corroborated seeds 41 and
314159 diverge on their second reset. The parity harness currently covers one
episode. Post-population-floor worlds can also continue stepping and advancing
RNG before a later reset. This proves a multi-episode seeded-parity gap, not a
marginal spawn bias or a claim that the one-episode harness is useless.

- Evidence: [`pqn/objective_contract_probe.json`](experiments/state_system_audit_2026-09-06/pqn/objective_contract_probe.json) and
  [`dynamics/dynamics_contract_probe.json`](experiments/state_system_audit_2026-09-06/dynamics/dynamics_contract_probe.json).
- Repair gate: freeze finished environments or reset asynchronously, specify
  reset RNG consumption, and add multi-episode parity cases across reset,
  floor, food and respawn events.

The assignment, reset, and post-floor behavior are in
[`pqn_trainer.py`](../src/training/pqn_trainer.py#L441-L457),
[`batch_sim.py`](../src/simd_env/batch_sim.py#L269-L287), and
[`game_state.py`](../src/game/game_state.py#L117-L133).

Resume must distinguish exact continuation from a warm start. PQN restores
optimizer and step counters but accepts changed learning rate and epsilon
horizon; its sampler provenance is better protected than its pool/RNG/reward
coefficient state. The probe requested learning rate .0002, but optimizer
restoration made the active rate .001. It also accepted `max_frames` 100→40
and `eps_decay_steps` 100→10. At restored step 50, the new epsilon schedule
returned 0 instead of the source schedule's .5.
For Apex, `--total-steps` is optimizer updates while PQN counts hero
transitions. Log both requested and effective incremental budgets, optimizer
updates, frames, and hero rows before comparing learners.

The relevant resume and optimizer paths are
[`train_pqn.py`](../src/scripts/train_pqn.py#L133-L269),
[`apex_train.py`](../src/scripts/apex_train.py#L1285-L1428), and
[`apex_learner.py`](../src/training/apex_learner.py).

### Apex and offline replay are distinct recipes

Apex has no CLI seed or configuration run seed. Actor processes seed Torch and
NumPy from wall-clock-derived values but leave Python `random`, used in
exploration/spawning, outside that contract. PQN `--seed` also leaves Torch
network initialization unseeded; the sampler diagnostic explicitly seeds all
three RNGs and is the appropriate pattern for causal runs.

Offline/local Apex and distributed Apex are not one interchangeable algorithm
recipe: local/offline uses AdamW, weight decay 1e-5, target clip 50 and a
different PER beta clock, while distributed uses Adam, eps 1.5e-4, clip 100
and its own clock. Local curriculum behavior is not carried into the distributed
actor. Persist the full recipe in every checkpoint and label comparisons by
recipe, rather than by the generic word “Apex.”

These recipe differences are implemented in
[`apex_policy.py`](../src/training/apex_policy.py#L147-L540),
[`apex_learner.py`](../src/training/apex_learner.py#L171-L334), and
[`apex_actor.py`](../src/training/apex_actor.py#L464-L475).

Offline replay validation accepts an empty metadata mapping; missing gamma,
n-step, state width, mechanics, arena, observation/action contract can pass.
Experience generation omits several of those fields, and the offline path does
not honor mechanics-v2 population-floor episode ending. That permits appending
or training incompatible data. The existing actor-failure wait risk and
optimizer-update step budget remain separate Apex operational issues already
recorded in [the RL review](rl_review_2026-09-06.md).

The permissive offline metadata and episode-end paths are
[`replay_quality.py`](../src/data/replay_quality.py#L251),
[`generate_experiences.py`](../src/scripts/generate_experiences.py#L387-L1882),
and [`game_state.py`](../src/game/game_state.py#L1077).

### Evaluation and serving do not yet define one deployment world

The live tournament path and SIMD path differ in conditions that affect the
world and headline metric. Live evaluation runs train mode with respawn enabled;
SIMD evaluation uses a different mode. A corpse pellet at ambient capacity leaves
one versus two ambient/total food outcomes in the probe. The live engine uses
segment count for mass while SIMD uses logical length; immediately after growth,
logical length is two while physical segments are still one. The SIMD adapter
also omits frame rate, so default live is 100 and SIMD is 1.

- Evidence: [`evaluation/eval_contract_repro.json`](experiments/state_system_audit_2026-09-06/evaluation/eval_contract_repro.json) and
  [`evaluation/food_mode_repro.json`](experiments/state_system_audit_2026-09-06/evaluation/food_mode_repro.json).
- Repair gate: decide the intended deployment world and publish an engine/event
  metric parity contract. Do not merely force a flag without that decision.

The live/SIMD construction and mass paths are
[`tournament_eval.py`](../src/scripts/tournament_eval.py#L315-L396),
[`eval_engine.py`](../src/simd_env/eval_engine.py#L253-L428), and
[`batch_sim.py`](../src/simd_env/batch_sim.py#L179).

The default tournament config is v1 while the web default is v2. Serving ignores
checkpoint mechanics metadata, so a v1 checkpoint can be served as v2. The
observation schema records shapes but not feature names/scales, action meaning,
normalizations, or a digest; the eval `max_frames` parameter is unused in one
SIMD policy path and serving has an independent 5000-frame default. Circular
geometry is not represented in `ObsInputs`, raster walls remain rectangular,
and `BatchSim` rejects circular arenas even though raster live code can accept
the flag. Reject unsupported geometry and use a versioned frozen adapter for
legacy checkpoints rather than silently reinterpreting them.

Serving does have useful hardening: normal inference uses evaluation/no-grad
mode, unknown observation specs and incompatible shapes fail, atomic checkpoint
writes exist, and catalog containment is checked. The raster row-dispatch issue
above still blocks Play-mode confidence. A bare 58-weight compatibility edge
also remains conditional: `InferenceAgent` infers 58, while a web helper can
default to 61. An ignored missing champion path can silently leave Watch using
random behavior; the main checkout currently has the file, while a fresh
worktree may not. Treat that as an installation/serving contract to make
explicit, not a claim about a live user session.

Serving contract sources include
[`inference_agent.py`](../src/model/inference_agent.py),
[`session.py`](../web/backend/session.py), and
[`checkpoint_io.py`](../src/model/checkpoint_io.py).

### Gate statistics, selection, and probe naming

The gate's paired per-mix t intervals and duplicate-seed rejection are valuable,
but its pilot estimates sample size from baseline SD rather than paired-delta
SD. A baseline-only sample cannot identify candidate variance. It also allows a
two-world zero-variance positive example to pass the current rule, while the
blueprint expects a much larger floor. The t calculation can be valid at n=2;
the campaign is still too weak for a credible promotion claim.

The generic `PROMOTE` result can compare scripted baselines and does not write a
champion. It is not evidence of beating A5. Fixed opponent order can confound
slots, and repeated policy specifications can fill a roster. A promotion receipt
should snapshot checkpoint bytes and SHA, incumbent identity, config/source
identity, roster and ordering, seed count, held-out status, and each metric.
Checkpoint reload happens per seed/mix; a concurrently rewritten rolling path
is a static conditional risk for mixed bytes, not observed hash drift in the
sampler campaign.

The relevant gate and selection paths are
[`tournament_eval.py`](../src/scripts/tournament_eval.py#L248-L871),
[`eval_stats.py`](../src/scripts/eval_stats.py), and
[`evaluate_checkpoints.py`](../src/scripts/evaluate_checkpoints.py#L229-L257).

The checkpoint evaluation helper's `--copy-best-to` operation ranks a short reward screen rather
than a tournament and can overwrite a chosen champion destination. Several
behavior probes need clearer names: entrapment is dead-frame-only, SIMD zero can
mean unavailable, food-positive net length misses growth-plus-burn, and boost
denominators differ by engine. Keep them engine-specific until their event
definitions and denominators agree.

### Instrumentation limits

The histogram drift probe accepts all-NaN/all-inf input and equal marginal
distributions with different joint structure. It is a synthetic limitation, not
an observed campaign non-finite observation. The collector is vector-only with
random-safe `GameState`, not raster or trained serving; its CI path does not
invoke config-drift protection. Existing observation-parity tests use a
controlled Python reference and shared featurizer frames. They should not be
described as complete runtime or all-engine semantic parity.

- Evidence: [`instrumentation/probe_drift_contract.json`](experiments/state_system_audit_2026-09-06/instrumentation/probe_drift_contract.json).
- Repair gate: reject non-finite inputs, add joint/conditional checks where
  appropriate, record config identity, and label the collector’s supported
  observation and serving scope.

## Deliberate gameplay and design limits

Some surprising fixtures are documented update-order choices rather than live
versus-SIMD parity failures. A contested pellet before a head-on can give ID 0
the pellet first, converting equal 6/6 into 7/6 and leaving that slot alive.
An equal three-way head contact can leave the high-index snake alive after the
first pair dies. A boost’s first substep can enter a body that is removed on the
second substep without collision. In the retained fixture the mask marks normal
and straight-boost actions fatal, yet a forced boost survives because collision
uses the final body after both tail pops. That is a conservative
mask-versus-execution mismatch and a gameplay design decision, not a claim that
the engines disagree. These may be fairness or temporal-physics choices worth
revisiting, but changing them must be a versioned rule change in both engines,
not an unreviewed “bug fix.” See
[`dynamics/dynamics_contract_probe.json`](experiments/state_system_audit_2026-09-06/dynamics/dynamics_contract_probe.json).

Similarly, a 31×31 tactical view, two-enemy summaries, and a strategic coarse
map are finite representations. The concrete cheap question is whether the
omitted information is prevalent and changes best-action ordering. The existing
evidence does not answer that question, so it does not justify recurrence or a
larger spatial network first.

## Minimal research plan

1. **Repair and freeze semantics.** Version the actual-target/advice-mask,
   raster paint, flip/world-coordinate, reward, geometry, reset and checkpoint
   contracts. Add effective-config, recipe, seed and input-SHA manifests.
2. **Freeze one held-out corpus.** Record world seeds, opponents, action traces,
   intended engine/world, and event metrics before comparing anything. Use a
   uniform clock vocabulary for hero transitions, environment frames, optimizer
   updates, and wall time.
3. **Measure observation value before expansion.** On split seeds/opponents,
   run action-conditional 2/4/8-history probes against oracle counters and
   report alias prevalence and best-action changes. This is the test that can
   justify memory or feature work.
4. **Run cheap feature-family controls.** Compare ego-vector rotation,
   strategic horizon/far-food, enemy identity symmetry, and the versioned flip
   treatment one axis at a time. If identity loss is measured, a modest
   permutation-invariant enemy encoder is a reasonable next candidate.
   [Deep Sets](https://arxiv.org/abs/1703.06114) and
   [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) support the
   architecture family, not a guaranteed Snake improvement.
5. **Use recurrence only after a history gap.** DRQN provides a precedent for
   recurrent partial-observation control ([Hausknecht and Stone](https://arxiv.org/abs/1507.06527)); it does not show that this game needs a GRU. Likewise,
   state-abstraction work distinguishes preservation conditions from an alias
   merely existing ([Li, Walsh, and Littman](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/camera-ready-9.pdf)).
6. **Add masked shared PPO as an algorithm control only after the same world,
   reward, observation, masks and gate are frozen.** PPO is a credible future
   baseline ([Schulman et al.](https://arxiv.org/abs/1707.06347)); it is not
   claimed better here. Keep PQN as a credible value-learning candidate.

Equivariance is also a useful ablation lens: discrete rotations/reflections can
be encoded or learned, and [Cohen and Welling](https://proceedings.mlr.press/v48/cohenc16.html)
provide the general motivation. The paper does not prove that a group-equivariant
model beats the current architecture on Snake.

## Evidence, tests, and resource boundary

This is a read-only source audit with targeted deterministic probes. The
completed sampler campaign remains the only substantial new training evidence;
this audit starts no new large training. Its configuration/world observations
must not be generalized to every config: mechanics-v2’s sampler recipe used
the same SIMD world defaults and frame rate in both arms.

Focused source-test shards reported 56 vector tests in 0.54 seconds, 33 raster
tests in 18.16 seconds, 95 dynamics tests with three slow tests deselected in
9.12 seconds, and 53 serving tests with two skips in 16.41 seconds. They
overlap, so they must not be summed into a suite total. CPU and Torch producer
parity can pass a shared semantic bug, as S02 demonstrates.

Portable evidence is organized by lane under
[`docs/experiments/state_system_audit_2026-09-06`](experiments/state_system_audit_2026-09-06/).
The audit manifest records the source revision and file hashes; individual
artifact links above are the evidence for their narrower claims. The incumbent,
production defaults, main checkout, and promotion state are unchanged.

An independent verifier repeated the serving row-shift through actual
`GameState.update`, the raster enemy-order paint overwrite, metadata-only reward
label, same-episode policy redraw, and synthetic poison-checkpoint path. It also
rechecked the objective contract and source hashes. Those confirmations are
independent corroboration of deterministic probes, not additional training;
see [`verification/major_claims_probe.json`](experiments/state_system_audit_2026-09-06/verification/major_claims_probe.json)
and [`verification/objective_contract_probe_recheck.json`](experiments/state_system_audit_2026-09-06/verification/objective_contract_probe_recheck.json).
