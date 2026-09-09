# PQN per-environment autoreset v1 interface contract

Date: 2026-09-09
Status: B0 interface freeze for implementation
Base: `1451128e0ded7c9863dd35f212c006a363f22148`

This contract implements the B packets in
`docs/plans/rl_repair_2026-09-08-followup.md`. It freezes the interfaces shared by
the B1 simulator, B2 self-play, and B3 trainer writers. It does not alter the accepted
X0 artifacts or authorize a learner or evaluation run.

## Non-negotiable behavior

The global `PQNConfig.recipe` default remains `legacy`; all existing recipe defaults stay
unchanged. A legacy configuration accepts only
`batch_barrier_v1 + continuous_env_rng_v1 + scheduled_v1`, treats those values as inert
compatibility defaults, and emits the exact existing legacy checkpoint/telemetry schemas.
It rejects every non-default B-mode value during `PQNConfig` construction. Its existing
continuation validator remains the legacy authority and does not require or synthesize a
lifecycle descriptor. Within the opt-in `corrected-v3` profile, existing behavior remains:

```text
recipe                  corrected-v3
episode_reset_mode      batch_barrier_v1
episode_seed_mode       continuous_env_rng_v1
runtime reset_strategy  batch_episode
```

That corrected-v3 default path must retain the trajectories, RNG call order, target values, policy
assignments, leases, update behavior, and existing telemetry values at this base.
Only newly saved corrected-v3 checkpoints receive lifecycle metadata, so their descriptor
bytes differ. The adapter below preserves the identity of previously saved corrected-v3
descriptors; generic legacy checkpoints retain their current bytes and continuation path.

The opt-in pair is:

```text
episode_reset_mode      per_env_autoreset_v1
episode_seed_mode       derived_env_episode_v1
runtime reset_strategy  per_env_rollout_boundary
```

`per_env_autoreset_v1` is valid only with `recipe=corrected-v3`,
`obs_spec=raster31v3`, mechanics/reward v2, no respawn, and
`derived_env_episode_v1`. A batch-barrier arm may select
`derived_env_episode_v1` to provide the matched experimental control. A per-environment
arm with `continuous_env_rng_v1` is rejected during `PQNConfig` construction.

Environment completion does not become snake death:

- A collision death stores its last reward once with `done=True`; its target is the
  reward with no bootstrap.
- A surviving snake on a population-floor or frame-cap step stores a valid transition
  with `done=False`; its target bootstraps from that episode's final post-step
  observation and resolved action mask.
- A completed environment is frozen for the remainder of the current rollout. Target
  computation and telemetry finish before the lane is reset at the next `_rollout()`
  entry. No reset-only or cross-episode transition is stored.
- `valid=False` prevents Q(lambda) carry. The first transition in episode `k+1` cannot
  affect a target in episode `k`.

The current `_compute_targets()` implementation is feasible without a same-step final
observation side channel. If completion occurs before the rollout edge, the frozen lane's
`hero_q[t+1]` is a forward of the unchanged final state; if completion occurs at the edge,
`q_final` is that state. In both cases `next_mask[t]` belongs to the final pre-reset state.

## B1: `BatchSim` masked reset API

B1 owns only `src/simd_env/batch_sim.py` and `tests/test_simd_reset_contract.py`.
The public API is:

```python
def reset_envs(
    self,
    env_mask: np.ndarray,
    *,
    seeds: Optional[Sequence[int]] = None,
) -> None:
    ...
```

### Input and atomicity

- `env_mask` must already be a NumPy array with dtype `np.bool_` and exact shape
  `(self.E,)`. Lists, integer arrays, scalars, and broadcastable shapes are rejected.
- When supplied, `seeds` contains one seed per selected environment in ascending
  `np.flatnonzero(env_mask)` order. Its length must equal `env_mask.sum()`. Each value
  must be a non-boolean integer in `[0, 2**64)`.
- An all-false mask is a valid no-op only with `seeds=None` or an empty sequence. A
  non-empty seed sequence is rejected.
- Mask, seed values, count, and replacement `EnvRng` construction are validated before
  any simulator field is mutated. Failure leaves every lane and RNG unchanged.
- Supplied seeds replace only `_rngs[e]` and `_seeds[e]` for selected rows. Omitted
  seeds continue the selected rows' existing independent RNG streams.

### Constructor and soft-reset draw order

`BatchSim.__init__` continues to construct all environment RNGs and call the existing
no-argument `reset()`. That first all-lane reset consumes the discarded
`FoodManager.__init__` food draws exactly once through the existing batch-global
`_constructor_food_draws_pending` flag. This preserves live-parity episode-0 worlds.

`reset_envs()` is an episode soft reset. Even when it installs a newly derived episode
seed, it does **not** replay constructor-only discarded food draws. The new RNG begins
with snake placement and then initial food placement. Calling `reset_envs()` with an
all-true mask is therefore not required to match constructing a new `BatchSim`; it must
match a seeded soft reset. The existing `reset()` retains its exact first-call and later
all-lane behavior and remains the default-path API.

An internal helper may be factored as:

```python
def _reset_selected_envs(
    self,
    env_mask: np.ndarray,
    *,
    replacement_rngs: Optional[Mapping[int, EnvRng]],
    consume_constructor_food_draws: bool,
) -> None:
    ...
```

The public API exposes no constructor-draw switch.

### Selected state and continuing-lane invariant

For selected rows only, reset all episode-scoped state: `bodies`, `head_ptr`,
`seg_count`, `length`, `alive`, `direction`, `boost_frames`, `frames_since_food`,
`respawn_timer`, `_reward_prev_length`, `_boosted_this_step`, `_trav`, `_trav_valid`,
ordered `food_cells`, `food_set`, `corpse_cells`, `frame`, `_last_reward`, `_last_mask`,
`_last_legal_mask`, `_last_resolved_mask`, `_last_done`, `_last_transition_valid`,
`_last_food_ate`, `_last_death_cause`, `_last_kills`, and every selected
`_last_kill_victim_len` object cell. Rebuild traversal and masks only for selected rows.

Every unselected row must remain byte/value identical, including NumPy slices, ordered
lists, sets, object-array contents, `_seeds[e]`, and the underlying
`random.Random.getstate()` of `_rngs[e]`. The selected population floor clears, frame is
zero, initial observation/masks are valid, and per-step outputs represent no transition.

## B2: assignment, exploration, and lease APIs

B2 owns only `src/training/pqn_selfplay.py` and
`tests/test_pqn_selfplay_per_env.py`. It retains the existing `assign_policy_ids()` and
the legacy branch of `batched_act()` exactly.

### Selected-row assignment

```python
def assign_policy_ids_per_env(
    policy_ids: np.ndarray,
    *,
    env_indices: Sequence[int],
    episode_ids: np.ndarray,
    pool_ids: Sequence[int],
    hero_frac: float,
    run_seed: int,
    hero_slot0: bool = True,
) -> np.ndarray:
    ...
```

- `policy_ids` is a signed-integer `(E,S)` array; `episode_ids` is an exact `np.uint64`
  `(E,)` array. `env_indices` contains unique, non-boolean integers in `[0,E)`; input
  order is ignored and validated indices are processed in ascending order. `run_seed` is
  a strict unsigned 64-bit integer, `hero_frac` is finite in `[0,1]`, and `pool_ids`
  contains unique, non-boolean, nonnegative integer snapshot IDs. The hero sentinel
  `HERO_POLICY_ID == -1` and every other negative value are rejected, preventing Python
  negative indexing from naming a snapshot. B3 supplies only IDs returned by
  `PinnedOpponentPool.policy_ids()`.
- Validate the complete request before deriving a seed or drawing. Return a copy. Every
  unselected row is byte-identical to the input.
- For selected row `e` at episode `k`, construct a temporary generator with
  `derive_seed(run_seed, f"pqn/assignment/env/{e}/episode/{k}")`. Perform the existing
  hero selector and frozen-choice draws for all `S` cells, then force slot 0 to hero when
  requested. An empty pool produces an all-hero row.
- Assignment is stateless because it occurs once per `(environment, episode)`; call order
  among selected rows cannot affect another row.

### Stateful per-environment exploration

```python
@dataclass
class PerEnvExplorationDecisions:
    epsilon: float
    eligible: np.ndarray
    explore: np.ndarray
    random_actions: np.ndarray


def sample_per_env_exploration(
    policy_ids: np.ndarray,
    valid_masks: np.ndarray,
    eligible: np.ndarray,
    epsilon: float,
    action_rngs: Mapping[int, np.random.Generator],
    *,
    env_indices: Optional[Sequence[int]] = None,
) -> PerEnvExplorationDecisions:
    ...
```

- Shapes are `(E,S)`, `(E,S,6)`, and `(E,S)` respectively. Masks must have boolean
  dtype; policy IDs and random actions are integer; `action_rngs` maps exactly every key
  in `range(E)` to one NumPy generator, and the `E` generator object identities are all
  distinct; epsilon is finite in `[0,1]`. Optional unique,
  in-range `env_indices` are normalized to ascending order and limit consumption; omission
  selects every environment. Validate all fields before any RNG call.
- B3 supplies `eligible_slot_mask = live_env[:, None] & sim.get_alive()` at step entry.
  Thus a collision-causing death step remains eligible, while dead-at-entry and completed
  or frame-capped lanes consume no action RNG.
- Iterate environments and slots in ascending order. For each eligible hero slot only,
  consume one epsilon draw. Consume one integer draw only if exploration succeeds, from
  the resolved valid actions, or uniformly from all six actions when that row has no valid
  action. Non-hero and ineligible cells have `eligible=False`, `explore=False`, and
  `random_actions=-1`.
- Epsilon zero consumes no RNG and returns `explore=False` and `random_actions=-1` in
  every cell. The helper never derives or recreates an action generator.
  B3 owns one persistent generator per environment and replaces it only at an episode
  boundary using `derive_seed(run_seed, f"pqn/action/env/{e}/episode/{k}")`.

Extend acting compatibly:

```python
def batched_act(
    hero: RasterDuelingNetwork,
    pool: PolicyGetter,
    policy_ids: np.ndarray,
    obs: Dict[str, torch.Tensor],
    mask: torch.Tensor,
    epsilon: float,
    rng: Optional[np.random.Generator],
    device: torch.device,
    exploration: Optional[PerEnvExplorationDecisions] = None,
) -> Tuple[np.ndarray, torch.Tensor]:
    ...
```

With `exploration=None`, `rng` is required and the current vectorized draw order and
actions remain exact. With precomputed decisions, `rng` must be `None`; epsilon, shapes,
dtypes, hero/eligible subset, random-action legality, and the invariant that
`epsilon == 0` implies no exploring cell are checked before any model
forward. Acting retains one hero forward and one forward per distinct frozen identity,
then applies only the precomputed exploring-hero overrides. It consumes no shared rollout
RNG.

The fixed-policy path computes exploration eligibility from the original `policy_ids`,
before its existing all-hero `act_ids` Q-forward workaround. In derived mode it passes the
same actual epsilon stored in the decisions to `batched_act(..., rng=None,
exploration=decisions)`, skips the old manual shared-RNG hero exploration loop, and then
overwrites frozen-source slots last. The continuous default retains its current
`epsilon=0` acting call plus manual shared-RNG hero loop byte-for-byte.

### Leases

No weaker eviction or replacement API is introduced. B3 acquires one existing
`OpponentLease` per environment using `pool.acquire(policy_ids[e])` and passes the shared
pool to `batched_act()` while those leases pin every referenced identity. Existing
idempotent `OpponentLease.close()` remains the release operation. Add one read-only audit
surface:

```python
@property
def active_pin_count(self) -> int:
    ...
```

It returns the sum of pins across resident `PinnedOpponentPool` snapshots. It never
changes admission or residency.

Per-lane RNG invariance means that, with fixed network weights and the same observations,
resetting lane 0 does not alter lane 1's assignment or random draws. Shared SGD changes
the hero weights and can couple later trajectories; the implementation must not claim
whole-training trajectory independence.

## B3: trainer and serialized contract

### Ownership expansion

B3 is one trunk with three non-overlapping writers after B1 and B2 integrate:

- **B3C shared contract**, first: new `src/training/pqn_lifecycle.py`,
  `src/core/runtime_contract.py` for shared closed reset constants, and new
  `tests/test_pqn_lifecycle.py`.
- **B3T trainer/config**, after B3C: the core files below.
- **B3S serving/strict consumer**, after B3C: the consumer files below.

B3T owns:

- `src/training/pqn_trainer.py`
- `src/scripts/train_pqn.py`
- `src/core/config_loader.py`
- `src/core/game_config.py`
- `tests/test_config_pqn_block.py`
- `tests/test_pqn_episode_contract.py`
- `tests/test_pqn_contract_regressions.py`
- `tests/test_pqn_resume_state.py`
- `tests/test_train_pqn.py`
- new `tests/test_pqn_per_env_autoreset.py`.

Truthful serving of a per-environment-trained checkpoint requires B3S to own:

- `web/backend/session.py` and its raster-serving contract tests;
- `src/evaluation/strict_promotion.py` and `tests/test_strict_artifacts.py` for the
  strict candidate/serving consumer;
- `tests/test_serving_runtime_contract.py`, `tests/test_v3_serving_identity_contract.py`,
  and `tests/test_web_raster_serving.py`.

`src/model/inference_agent.py`, `src/evaluation/serving_episode.py`, raster policy code,
frontend code, and model weights need no change. `InferenceAgent` validates observation and
model-head identities; `GameSession` owns source-runtime acceptance. Any needed edit outside
the files above is a new root-adjudicated packet, not an implicit B3 expansion.

### Configuration

Add these `PQNConfig` fields:

```python
episode_reset_mode: str = "batch_barrier_v1"
episode_seed_mode: str = "continuous_env_rng_v1"
pool_admission_mode: str = "scheduled_v1"
initial_opponent_checkpoint_sha256: Optional[str] = None
```

Closed values are the two reset modes, the two seed modes, and
`{"scheduled_v1", "disabled_v1"}`. A declared checkpoint hash is exactly 64 lowercase
hex characters and requires corrected-v3 snapshot-pool mode with capacity at least one.
It is an execution identity and is excluded from the YAML schema/parity surface.
`pool_admission_mode` and the two lifecycle modes are optional YAML fields and participate
in field-source resolution. Global legacy recipe resolution accepts only their inert
defaults and preserves the pre-B checkpoint schema.

Add CLI choices `--episode-reset-mode`, `--episode-seed-mode`, and
`--pool-admission-mode`. `initial_opponent_checkpoint_sha256` has no generic YAML or
unverified-path input. The B5 execution runner sets it only after reading and hashing
immutable checkpoint bytes and validating their observation/model-head contract.

### Initial fixed snapshot seam for B5

Expose a pre-episode public operation:

```python
def preload_opponent_snapshot(
    self,
    source_network: RasterDuelingNetwork,
    *,
    checkpoint_sha256: str,
    model_head_digest: str,
) -> int:
    ...
```

It is permitted only when the pool is empty, `update_idx == agent_steps == 0`, all sim
frames and episode IDs are zero, no policy assignment or lease exists, and the supplied
checkpoint hash equals `cfg.initial_opponent_checkpoint_sha256`. Validate a
declared `pqn/dueling_q` six-action raster31v3 model-head digest and the complete request
before cloning. Admit
through `PinnedOpponentPool.add_snapshot()`, return its stable ID, and record the source
checkpoint SHA, model-head digest, and realized snapshot content hash in the policy-source
descriptor. A configured initial hash without a successful preload makes the first
`_rollout()` fail.

This method cannot prove that an arbitrary in-memory network came from the named file.
The B5 runner owns that proof: it loads the preflighted checkpoint into the learner before
calling this method, hashes the serialized state supplied for cloning, and records both
the checkpoint-byte SHA and realized pool snapshot-state hash. A mismatch fails before
episode-0 assignment.

This permits the matched barrier and autoreset arms to apply the same immutable checkpoint
as weights-only learner initialization, then clone those exact loaded weights into the
pool before episode-0 assignment. `pool_capacity=1` and
`pool_admission_mode=disabled_v1` keep that one snapshot resident. No mid-episode pool
mutation or test-only private-field injection is needed.

### Episode 0

Episode IDs are exact `np.uint64` and start at zero. Continuous mode constructs
`BatchSim` with the current `[config.seed + e]` seed list and preserves the current first
world and constructor-only food draws exactly. Derived mode computes every
`derive_seed(config.seed, f"pqn/world/env/{e}/episode/0")` before simulator construction
and supplies that list to the existing `BatchSim(..., seeds=...)` constructor. Thus the
derived episode-0 world also receives constructor semantics, including its one discarded
`FoodManager` draw sequence; it is not created by constructing a throwaway world followed
by `reset_envs()`.

Derived mode also constructs one distinct action generator per environment from
`derive_seed(config.seed, f"pqn/action/env/{e}/episode/0")`. Continuous mode creates no
per-environment action generators and retains `self.rng`. After a required initial snapshot
preload succeeds, the first `_rollout()` assigns all rows for episode 0 and acquires all
leases without resetting the simulator, incrementing an episode ID, or incrementing a
reset count. Without a declared initial snapshot, the same first assignment occurs against
the pool state produced by the existing initialization path.

### Lifecycle descriptor and compatibility

B3T obtains the descriptor from B3C's
`build_pqn_episode_lifecycle_contract(config.episode_reset_mode,
config.episode_seed_mode)`; no trainer-local formula is permitted. Its exact top-level
identity is:

```json
{
  "schema_version": "pqn-episode-lifecycle/v1",
  "episode_reset_mode": "batch_barrier_v1 | per_env_autoreset_v1",
  "episode_seed_mode": "continuous_env_rng_v1 | derived_env_episode_v1",
  "reset_timing": "shared_batch_boundary | selected_envs_at_next_rollout_boundary",
  "completion": {
    "collision_death": "done_true_reward_only",
    "population_floor": "done_false_final_successor_bootstrap",
    "frame_cap": "done_false_final_successor_bootstrap"
  },
  "invalid_transition_carry": "break",
  "assignment_lifetime": "batch_episode | environment_episode",
  "lease_lifetime": "batch_episode | environment_episode",
  "world_seed_stream": "config_seed_plus_env_continuous | pqn/world/env/{e}/episode/{k}",
  "assignment_seed_stream": "shared_rollout_rng | pqn/assignment/env/{e}/episode/{k}",
  "action_seed_stream": "shared_rollout_rng | pqn/action/env/{e}/episode/{k}"
}
```

Serialize it as `episode_lifecycle_contract` and
`episode_lifecycle_contract_digest`. New corrected-v3 target and sampler descriptors use
the exact versions `pqn-qlambda-corrected-v3-lifecycle-v1` and
`pqn-sampler-corrected-v3-lifecycle-v1`, contain that digest, and retain every existing
target/optimizer formula. Their reset, assignment-lifetime, pinning, and RNG values reflect
the selected modes. `RunProvenance` remains schema v1: its existing runtime, target, and
sampler digests bind the new lifecycle semantics.

New corrected-v3 checkpoints also serialize this exact static policy-source contract and
its digest:

```json
{
  "schema_version": "pqn-rollout-policy-source/v1",
  "rollout_policy_mode": "snapshot_pool | fixed",
  "fixed_policy_identity": "string-or-null",
  "assignment_lifetime": "batch_episode | environment_episode",
  "lease_lifetime": "batch_episode | environment_episode",
  "pool_admission_mode": "scheduled_v1 | disabled_v1",
  "initial_opponent_checkpoint_sha256": "sha256-or-null",
  "initial_opponent_model_head_digest": "sha256-or-null",
  "initial_opponent_snapshot_state_sha256": "sha256-or-null",
  "episode_lifecycle_contract_digest": "sha256"
}
```

The sampler descriptor contains both contract digests. Its pool mutation text states that
admission runs after a successful update, defers while all eviction candidates are pinned,
and a deferred scheduled attempt is not replayed at an off-schedule update. Each realized
`rollout_policy_source` record also carries the static policy-source digest. This records
the actual possible scheduled-admission loss instead of implying every due snapshot is
eventually admitted.

`RuntimeModeContract` remains the existing six-field descriptor. Share constants for
`batch_episode` and `per_env_rollout_boundary`; do not add fields or change old digest
normalization. New barrier checkpoints use the former; per-environment checkpoints use
the latter.

B3C owns the only cross-descriptor implementation in the Torch-free
`src/training/pqn_lifecycle.py`:

```python
@dataclass(frozen=True)
class ValidatedPQNLifecycle:
    descriptor: Mapping[str, Any]
    digest: str
    policy_source_descriptor: Mapping[str, Any]
    policy_source_digest: str
    compatibility: Optional[Mapping[str, Any]]


def build_pqn_episode_lifecycle_contract(
    episode_reset_mode: str,
    episode_seed_mode: str,
) -> Dict[str, Any]:
    ...


def validate_pqn_episode_lifecycle_metadata(
    metadata: Mapping[str, Any],
    *,
    allow_corrected_v3_adapter: bool,
) -> ValidatedPQNLifecycle:
    ...
```

For native metadata the validator requires exact lifecycle and policy-source key sets,
canonical digests, and supported versions. It then enforces all of these crosslinks before
returning:

- runtime reset strategy equals the lifecycle reset mode's closed mapping;
- target and sampler carry the lifecycle digest, and sampler carries the policy-source
  digest;
- policy source carries the lifecycle digest and its reset-dependent assignment/lease
  lifetimes agree with the lifecycle descriptor;
- each realized rollout-policy-source record carries the same policy-source digest;
- `RunProvenance.from_metadata()` succeeds and its runtime, target, and sampler digests
  equal the corresponding verified top-level digests.

Validation is read-only and completes before trainer continuation, inference loading, or
serving world construction. B3T calls it for new and pre-lifecycle corrected-v3
continuation. B3S calls the same helper when opening candidate checkpoint bytes; neither
consumer reimplements or partially applies these formulas.

Implement an explicit adapter for a corrected-v3 checkpoint that has no lifecycle descriptor only
when all of these raw facts match the known corrected-v3 legacy schema: verified original
runtime/target/sampler digests, runtime `batch_episode`, target version
`pqn-qlambda-corrected-v3`, sampler version `pqn-sampler-corrected-v3`, and the complete
expected old descriptors. The adapter returns effective
`batch_barrier_v1 + continuous_env_rng_v1` plus an adapter version and the original three
digests. It never rewrites checkpoint mappings or recomputes their recorded provenance.
Unknown or partial legacy schemas fail rather than acquiring invented lifecycle metadata.

The returned compatibility mapping is exactly:

```json
{
  "schema_version": "pqn-episode-lifecycle-legacy-adapter/v1",
  "source_schema": "corrected-v3-pre-lifecycle",
  "original_runtime_contract_digest": "sha256",
  "original_target_contract_digest": "sha256",
  "original_sampler_contract_digest": "sha256"
}
```

For that adapter only, `policy_source_descriptor` is an explicit derived record rather than
a claim that the old checkpoint stored the new schema:

```json
{
  "schema_version": "pqn-rollout-policy-source-legacy-adapter/v1",
  "source_schema": "corrected-v3-pre-lifecycle",
  "source_rollout_policy_source": {
    "mode": "snapshot_pool | fixed",
    "identity": "verified old identity"
  },
  "source_rollout_policy_source_digest": "sha256",
  "source_sampler_contract_digest": "original sha256",
  "effective_episode_lifecycle_contract_digest": "adapter result sha256"
}
```

The validator computes this adapter record only after the original sampler,
rollout-policy-source mapping, and schema-v1 RunProvenance crosslinks pass. Its new digest
identifies the adapter result; it does not replace or masquerade as a digest stored by the
old checkpoint.

`allow_corrected_v3_adapter=False` rejects it. A generic `recipe=legacy` checkpoint never
enters this helper and continues through the unchanged legacy validator.

New-to-new continuation compares exact descriptor digests. Adapted legacy continuation is
allowed only into the default modes. `per_env_autoreset_v1` or
`derived_env_episode_v1` rejects optimizer continuation before trainer/world construction,
because simulator state, episode RNGs, assignments, pool contents, leases, and MPS RNG are
not restorable. Exact resume remains unsupported. Weights-only loading is allowed and
starts episode 0 with fresh per-environment state, recording the immutable parent checkpoint
SHA and selected lifecycle/seed contract.

### Trainer state and operation order

The default branch retains `_episode_lease` and `_episode_reset_count` for compatibility.
The per-environment branch adds an `E`-length lease list, episode reset-count array, current
world-seed array, and persistent action-generator list. `_episode_policy_ids` remains
`(E,S)`, `_episode_ids` remains `(E,)`, and `_episode_finished_env` is the pending reset
mask.

At per-environment `_rollout()` entry:

1. Copy the pending mask and validate completed-lane final state still exists.
2. Close only those lanes' old leases and clear those entries.
3. Increment only their episode IDs and reset counts.
4. Derive world and action seeds for the new IDs; call `sim.reset_envs()` once with
   selected seeds in ascending lane order.
5. Replace only selected assignment rows and acquire their new per-row leases.
6. Clear only selected pending flags, then collect a fixed `cfg.rollout_len` rollout.

Those six steps apply only when the pending mask is non-empty after episode 0. The first
rollout follows the episode-0 construction sequence above: validate required preload,
assign every row at ID zero, acquire every row lease, and collect without a reset or ID
increment. The batch-barrier derived path uses the same derived constructor/action seeds
for episode 0; at later shared boundaries it increments every ID and uses one all-true
seeded soft reset. The continuous batch path retains its current `sim.reset()` and shared
rollout RNG order exactly.

The per-environment path does not use the current
`max_frames - sim.frame.max()` rollout length. It always allocates `rollout_len` and uses
the existing per-lane `live_env &= sim.frame < max_frames` check; a lane reaching its cap
mid-rollout freezes until the next boundary. The default batch path retains the current
short-tail behavior exactly.

On each derived-mode collection step, compute `live_env` and
`eligible_slot_mask = live_env[:, None] & sim.get_alive()` before sampling exploration or
acting. Precompute decisions from the original assignment grid, call `batched_act`, apply
the fixed-source override last when applicable, and then call `sim.step(...,
active_env_mask=live_env)`. All request validation occurs before the first RNG draw or
network forward. This ordering gives the collision-causing action one draw while consuming
none for a lane already dead, capped, floor-complete, or pending reset.

After `_rollout()` returns, `_compute_targets()`, SGD, telemetry, and snapshot admission
run in their current order. Old completed leases remain pinned through the preceding
end-of-update admission, so a due admission may safely defer. Reset/reassignment happens
at the next update entry; there is no eviction window between selecting a policy ID and
acquiring its lease.

Add idempotent `PQNTrainer.close()`. It snapshots every non-null owned lease reference,
attempts `close()` on each even when an earlier close raises, clears all owned references,
and collects cleanup exceptions. Only when the pool is a `PinnedOpponentPool` does it then
require `pool.active_pin_count == 0`; the legacy mutable `OpponentPool` has no pin audit and
continues through its existing path. One cleanup failure is raised directly and multiple
cleanup failures are raised together as a `BaseExceptionGroup`. A second call has no
leases to close and succeeds only when the applicable pin audit is zero.

The CLI calls `close()` on success, tripwire, ordinary exception, interrupt, and resource
stop. With no training failure, a cleanup failure makes the command fail. With a primary
training failure, the CLI retains that primary exception and traceback as the top-level
failure and chains the cleanup failure or group as its cause; cleanup cannot replace the
training classification, and the primary error cannot prevent attempts to close later
leases. Successful cleanup re-raises the original primary unchanged.

`pool_admission_mode=scheduled_v1` retains current positive-update-index interval behavior;
`disabled_v1` performs no learned snapshot admission. Pool mode, initial snapshot identity,
and realized exposure are serialized in sampler/policy-source metadata.

### Telemetry and checkpoint state

Version new telemetry records and add: lifecycle/reset/seed modes, per-environment episode
IDs and reset counts, the exact reset lane indices for this update, newly completed lane
count, current derived world-seed identities, assignment identities, per-identity exposure,
useful/valid/capacity counts, and optimizer sampling counts. Existing fields and values are
unchanged in the default mode. Existing scalar `episode_reset_count` remains the number of
rollout boundaries at which a reset occurred; the new array is authoritative for per-lane
counts. Document `completed_episodes` as newly completed environments, matching source.

Checkpoint metadata includes the lifecycle descriptor/digest, modes, observed episode IDs
and reset counts, current seed identities, assignment identities, policy-source identity,
and `restorable_environment_state=false`. These observed fields establish lineage and are
not consumed as exact resume state.

### Serving and strict-consumer acceptance

The shared serving runtime accepts exactly `batch_episode` and
`per_env_rollout_boundary` for a `pqn_train` source checkpoint, while deployed Watch/Play
remains manual-reset live `GameState`. Before policy construction,
`web/backend/session.py` validates the exact six-field source runtime and digest and calls
`validate_pqn_episode_lifecycle_metadata()`. Its serving contract and deployment target
manifest include the validator's lifecycle descriptor/digest, compatibility mapping or
explicit `null`, and policy-source descriptor/digest. It preserves those source facts and
does not emulate training autoreset.

The strict request's `candidate_contracts` gains exact `episode_lifecycle` and
`policy_source` descriptor/digest records. During request freezing, the consumer opens the
candidate path already bound by the E0 byte snapshot, applies the shared validator, and
requires its complete result to equal those frozen records. The serving contract and target
manifest must then reproduce them exactly. The strict consumer therefore expands its
hard-coded runtime expectation to the closed two-value set only after lifecycle validation;
allowing a reset string alone is insufficient. It still requires every other runtime field,
checkpoint/runtime/target/sampler/policy-source digest, schema-v1 RunProvenance digest,
source closure, deployment manifest, and serving receipt to agree. Unknown reset values,
partial adapters, and crosslink substitutions fail. No source descriptor or digest is
rewritten by serving.

Acceptance requires real checkpoints for both reset strategies to load through
`InferenceAgent`, Watch, and Play; the source-runtime digest must survive into serving
evidence; wrong or forged lifecycle/runtime/RunProvenance digests must fail before session
world construction; the strict artifact consumer must accept the valid per-environment
fixture and reject a reset-strategy substitution. Existing batch-barrier serving and strict
fixtures remain green.

## B3 acceptance matrix

The B3 verifier must cover these gates before any matched screen:

1. Default barrier plus continuous RNG matches base golden worlds, actions, targets,
   assignments, lease timing, and existing telemetry values. Generic legacy saves and
   continuation remain byte/schema compatible and reject non-default B modes.
2. Barrier plus derived RNG and per-environment plus derived RNG resolve from YAML/CLI and
   serialize distinct, self-consistent lifecycle/runtime/target/sampler digests.
3. Per-environment plus continuous RNG, unknown mode values, malformed hashes, and
   conflicting legacy descriptors fail before world construction.
4. One floor, simultaneous floors, all floors, cap completion, and death on the floor step
   obey the reset and target rules above. No Q(lambda) return crosses an episode ID.
5. Resetting one lane leaves a continuing lane's world and RNG byte-identical; with frozen
   weights/context, assignment and exploration draws also remain identical. Aliasing one
   action generator across two lanes, any negative pool ID, and an epsilon-zero decision
   containing `explore=True` all fail before RNG use or forward.
6. Episode policy identity is stable, selected rows alone refresh, every lease closes once,
   pinned identities cannot be evicted, and success/failure cleanup ends with zero active
   pins.
7. Derived episode 0 uses constructor seeds and constructor-only draws without a preliminary
   reset; its action generators and all-row assignment use episode ID zero. The configured
   initial checkpoint is preloaded before that assignment. The fixture separately proves
   checkpoint-byte SHA, loaded learner-state SHA, and realized snapshot-state SHA, and the
   snapshot remains the only resident member when admission is disabled.
8. Old known corrected-v3 checkpoints retain original bytes/digests and adapt only to the
   default modes. New default and per-environment checkpoints validate internally. Cross-mode
   continuation and per-environment/derived continuation fail; weights-only parent lineage
   is accurate.
9. Inference, Watch, Play, serving receipts, and strict promotion artifacts accept both
   truthful source reset strategies and reject forged, unknown, or crosslinked-only-in-part
   lifecycle/runtime/target/sampler/policy-source records without changing deployment
   runtime behavior.
10. Cleanup fault injection makes every owned lease close attempt run. Legacy pools skip
    pin audit, pinned pools finish at zero pins, and a simultaneous training plus cleanup
    failure reports the training exception as primary with cleanup chained.

Focused B1/B2/B3 tests run only in allocated CPU lanes after integration. The existing slow
SIMD/live parity cases and affected serving/strict-artifact union are required before B4.
A benchmark, learner, or policy-quality claim is outside this interface gate.

## Risks retained by design

- A completed lane can waste at most `rollout_len - 1` positions; same-step reset remains a
  possible later optimization with a larger final-observation risk surface.
- Independent resets change the training distribution. This contract establishes semantics,
  lineage, and utilization measurement, not learning superiority.
- The B5 fixed one-snapshot screen does not qualify asynchronous learned-pool admission.
- Exact environment continuation remains unsupported; an interrupted experimental arm is
  retained incomplete rather than resumed under fabricated state.
