"""Fast batched eval engine for the promotion gate (blueprint §5.4).

Runs the paired candidate-vs-baseline rollouts of
:mod:`src.scripts.tournament_eval` inside the vectorized
:class:`~src.simd_env.batch_sim.BatchSim` instead of the live
:class:`~src.game.game_state.GameState`. Every eval seed becomes one parallel
environment, so a whole mix's worth of paired rollouts advances one frame per
:meth:`BatchSim.step`, making gating ~100x cheaper than the live engine
(blueprint §0.7 / §5.4).

The hero is arena slot 0 (death terminal — its mass integral integrates 0 over
dead frames, matching the live engine's ``auto_respawn=False`` hero). Opponents
fill slots 1..S-1 and DO respawn in eval mode so arena pressure stays constant.
The SAME gate metrics are computed from the batch arrays: the mass integral over
total frames, survival fraction, deaths, kills — plus the probes that are
feasible from the batch state (boost-frame fraction, food eaten via length
deltas, peak length, death cause).

Policy wiring
-------------
Agents act via a :class:`SimdPolicy` — a batched, per-frame action selector over
the ``(num_agents, 6)`` safe-action masks the sim already computes. Three
policies are provided: two scripted anchors (``greedy_food`` mask+food heuristic,
``random_safe`` mask-follower) and :class:`NetworkSimdPolicy`, which serves a
raster (obs_spec ``raster31v2``) checkpoint through the shared featurizer. The
61-D ``vector61`` champions are NOT featurized by the batch sim, so a vector
checkpoint under ``--engine simd`` raises with a clear message pointing at
``--engine live``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from src.core.runtime_contract import EffectiveWorldConfig, RuntimeModeContract
from src.evaluation.anchors import AnchorContext, ScriptedAnchor
from src.evaluation.metrics import (
    EvaluationMetricsAccumulator,
    PostStepState,
    StepEvents,
)
from src.evaluation.protocol import EvaluationProfile
from src.simd_env.batch_sim import (
    CARDINAL,
    DEATH_BODY,
    DEATH_HEAD,
    DEATH_NONE,
    DEATH_SELF,
    DEATH_WALL,
    BatchSim,
    BatchSimConfig,
)

# (kind, ref) agent spec, matching tournament_eval.AgentSpec.
AgentSpec = Tuple[str, str]

# Death-cause code -> probe label (matches BehaviorProbes' vocabulary).
_DEATH_CAUSE_LABEL = {
    DEATH_WALL: "wall",
    DEATH_SELF: "self",
    DEATH_HEAD: "head_on",
    DEATH_BODY: "enemy_body",
}


def _slot_content_hash(kind: str, reference: str) -> str:
    if kind == "checkpoint":
        digest = hashlib.sha256()
        with Path(reference).open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    from src.core.runtime_contract import canonical_digest
    from src.evaluation.anchors import SCRIPTED_ANCHOR_VERSION

    return canonical_digest(
        {"scripted_anchor": reference, "anchor_version": SCRIPTED_ANCHOR_VERSION}
    )


def _roster_identity(seed: int, specs: Sequence[AgentSpec], mix_id: str) -> Dict[str, object]:
    from src.core.runtime_contract import canonical_digest

    hashes = [_slot_content_hash(kind, ref) for kind, ref in specs]
    return {
        "seed_namespace": "evaluation-world/v1",
        "seed": int(seed),
        "mix_id": mix_id,
        "ordered_slot_content_hashes": hashes,
        "roster_id": canonical_digest({"slots": hashes}),
    }


def validate_v3_checkpoint_for_profile(checkpoint_path: str, profile: EvaluationProfile) -> None:
    """Reject a v3 checkpoint whose declared source world differs from evaluation."""
    import torch

    from src.core.runtime_contract import EffectiveWorldConfig
    from src.model.obs_spec import RASTER31V3_CONTRACT

    if profile.world.arena_type != "rectangular":
        raise ValueError("raster31v3 evaluation supports rectangular worlds only")
    if set(profile.world.normalization) != {"max_frames", "starvation_max", "max_length"}:
        raise ValueError("raster31v3 evaluation profile requires explicit normalization")
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if blob.get("obs_contract_digest") != RASTER31V3_CONTRACT.digest:
        raise ValueError("raster31v3 checkpoint observation contract does not match evaluator")
    raw_world = blob.get("effective_world")
    world_fields = set(EffectiveWorldConfig.__dataclass_fields__)
    if not isinstance(raw_world, dict) or set(raw_world) != world_fields:
        raise ValueError("raster31v3 checkpoint requires a complete effective_world descriptor")
    source_world = EffectiveWorldConfig(**raw_world)
    if blob.get("effective_world_digest") != source_world.digest:
        raise ValueError("raster31v3 checkpoint effective_world_digest does not match descriptor")
    if source_world.digest != profile.world.digest:
        raise ValueError("raster31v3 checkpoint world does not match evaluation profile")
    raw_runtime = blob.get("runtime_contract")
    runtime_fields = set(RuntimeModeContract.__dataclass_fields__)
    if not isinstance(raw_runtime, dict) or set(raw_runtime) != runtime_fields:
        raise ValueError("raster31v3 checkpoint requires a complete runtime_contract")
    for key in ("training", "respawn", "hero_terminal", "population_floor"):
        if not isinstance(raw_runtime[key], bool):
            raise ValueError("raster31v3 checkpoint runtime_contract has invalid boolean fields")
    if not isinstance(raw_runtime["mode"], str) or not isinstance(
        raw_runtime["reset_strategy"], str
    ):
        raise ValueError("raster31v3 checkpoint runtime_contract has invalid string fields")
    source_runtime = RuntimeModeContract(**raw_runtime)
    if blob.get("runtime_contract_digest") != source_runtime.digest:
        raise ValueError("raster31v3 checkpoint runtime_contract_digest does not match descriptor")
    if not isinstance(blob.get("model_head"), dict):
        raise ValueError("raster31v3 checkpoint requires an explicit model head contract")


class SimdPolicy:
    """Batched per-frame action selector over an ``(N, 6)`` safe-action mask.

    Implementations pick one action in ``[0, 5]`` per agent given the current
    mask and (optionally) the sim, without mutating sim state. This is the
    single hook a network policy will replace (see :class:`NetworkSimdPolicy`).
    """

    def actions(self, masks: np.ndarray, sim: BatchSim, slots: np.ndarray) -> np.ndarray:
        """Return an integer action per agent row of ``masks`` (shape ``(N,)``).

        Args:
            masks: ``(N, 6)`` bool safe-action mask (True == safe) for the N
                agents this policy controls.
            sim: The batch sim (read-only) for policies that need world state.
            slots: ``(N, 2)`` int array of ``(env, snake)`` indices this call's
                masks correspond to, in row order.

        Returns:
            ``(N,)`` int actions in ``[0, 5]``.
        """
        raise NotImplementedError


class RandomSafeSimdPolicy(SimdPolicy):
    """Uniform random over safe non-boost actions; straight if none are safe.

    Mirrors the live ``scripted:random_safe`` anchor (never boosts). Seeded for
    determinism given the seed.
    """

    def __init__(self, seed: int) -> None:
        """Build a random-safe policy.

        Args:
            seed: RNG seed (determinism given seed).
        """
        self._gen = np.random.default_rng(seed ^ 0x5A4E_5A4E)

    def actions(self, masks: np.ndarray, sim: BatchSim, slots: np.ndarray) -> np.ndarray:
        n = masks.shape[0]
        out = np.ones(n, dtype=np.int64)  # default straight
        # Only normal-speed actions [0, 1, 2] are eligible.
        safe_norm = masks[:, :3]
        for i in range(n):
            safe = np.nonzero(safe_norm[i])[0]
            if safe.size:
                out[i] = int(self._gen.choice(safe))
        return out


class GreedyFoodSimdPolicy(SimdPolicy):
    """Safe move minimizing cell distance to the nearest food; never boosts.

    A vectorization-friendly analogue of the live ``scripted:greedy_food``
    anchor: among the safe non-boost actions it picks the one whose resulting
    head cell is closest to the nearest food pellet, tie-breaking
    straight > left > right. (It does NOT reproduce the live anchor's flood-fill
    "don't trap yourself" veto — that BFS is out of scope for the batched path;
    documented as a behavioral substitution, see the module report.)
    """

    # Tie-break priority (lower wins): straight(1) > left(0) > right(2).
    _PRIORITY = np.array([1, 0, 2], dtype=np.int64)  # index by relative action

    def actions(self, masks: np.ndarray, sim: BatchSim, slots: np.ndarray) -> np.ndarray:
        n = masks.shape[0]
        out = np.ones(n, dtype=np.int64)
        heads = sim.heads()  # (E, S, 2)
        directions = sim.direction  # (E, S)
        for i in range(n):
            e, s = int(slots[i, 0]), int(slots[i, 1])
            safe = [rel for rel in range(3) if masks[i, rel]]
            if not safe:
                out[i] = 1  # deterministic fatal fallback: straight
                continue
            food = sim.get_food(e)
            if not food:
                out[i] = int(min(safe, key=lambda a: self._PRIORITY[a]))
                continue
            food_arr = np.asarray(food, dtype=np.int64)  # (F, 2)
            hx, hy = int(heads[e, s, 0]), int(heads[e, s, 1])
            cur_dir = int(directions[e, s])

            def next_dist(rel: int) -> float:
                delta = -1 if rel == 0 else (1 if rel == 2 else 0)
                ndir = (cur_dir + delta) % 4
                dv = CARDINAL[ndir]
                nx, ny = hx + int(dv[0]), hy + int(dv[1])
                d = food_arr - np.array([nx, ny])
                return float(np.min(d[:, 0] ** 2 + d[:, 1] ** 2))

            out[i] = int(min(safe, key=lambda a: (next_dist(a), self._PRIORITY[a])))
        return out


class _ProfileAnchorSimdPolicy(SimdPolicy):
    """Profile-scoped scripted anchor keyed by world identity, never row order."""

    def __init__(self, kind: str, world_seeds: Sequence[int]) -> None:
        self._anchor = ScriptedAnchor(kind)
        self._world_seeds = tuple(int(seed) for seed in world_seeds)

    def actions(self, masks: np.ndarray, sim: BatchSim, slots: np.ndarray) -> np.ndarray:
        heads = sim.get_heads()
        headings = sim.get_direction_vectors()
        out = np.ones(len(slots), dtype=np.int64)
        for row, (env, slot) in enumerate(slots):
            env_i, slot_i = int(env), int(slot)
            out[row] = self._anchor.action(
                AnchorContext(
                    world_seed=self._world_seeds[env_i],
                    slot=slot_i,
                    frame=int(sim.frame[env_i]) - 1,
                    head_cell=tuple(int(v) for v in heads[env_i, slot_i]),
                    heading=tuple(int(v) for v in headings[env_i, slot_i]),
                    food_cells=tuple(tuple(int(v) for v in cell) for cell in sim.get_food(env_i)),
                    allowed_mask=masks[row],
                )
            )
        return out


class NetworkSimdPolicy(SimdPolicy):
    """Batched raster-network policy (obs_spec ``raster31v2``) for the simd gate.

    Loads a raster checkpoint forward-only and, each frame, builds the dual-scale
    ego-raster observation for the whole batch through the SAME featurizer the
    trainer uses (:func:`~src.simd_env.featurizer.obs_inputs_from_batch_sim` +
    :func:`~src.simd_env.featurizer.build_observations`), runs the network, and
    returns masked-greedy actions for the slots it controls. Only ``raster31v2``
    checkpoints are supported here; 61-D ``vector61`` champions are not
    featurized by the batch sim and must use ``--engine live``.
    """

    def __init__(self, checkpoint_path: str, profile: EvaluationProfile | None = None) -> None:
        """Load a raster checkpoint for batched, no-grad greedy action selection.

        Args:
            checkpoint_path: Path to a ``raster31v2`` ``.pth`` checkpoint.

        Raises:
            ValueError: If the checkpoint is not a raster (``raster31v2``) model.
        """
        import torch

        from src.model.inference_agent import InferenceAgent
        from src.model.obs_spec import RASTER31V2, RASTER31V3

        self._torch = torch
        agent = InferenceAgent.from_checkpoint(checkpoint_path)
        obs_spec = getattr(agent, "obs_spec", None)
        if obs_spec not in (RASTER31V2, RASTER31V3):
            raise ValueError(
                f"--engine simd checkpoint policy needs a raster model; "
                f"{checkpoint_path!r} is '{obs_spec or '?'}'. "
                "Use --engine live for 61-D vector champions."
            )
        if obs_spec == RASTER31V3 and profile is None:
            raise ValueError("raster31v3 SIMD evaluation requires an explicit EvaluationProfile")
        if profile is not None and obs_spec == RASTER31V3:
            validate_v3_checkpoint_for_profile(checkpoint_path, profile)
        self._agent = agent
        self._obs_spec = obs_spec
        self._profile = profile
        self._network = agent.network
        self._device = agent.device

    def actions(self, masks: np.ndarray, sim: BatchSim, slots: np.ndarray) -> np.ndarray:
        from src.model.raster_network import raster_tensors_from_obs
        from src.simd_env.featurizer import (
            build_observations,
            obs_inputs_from_batch_sim,
        )

        torch = self._torch
        # Featurize the whole batch once, then forward only the slots we control.
        # A shared checkpoint policy receives every live roster row it controls
        # in one call from ``run_simd_eval``.
        if self._profile is None:
            obs = build_observations(obs_inputs_from_batch_sim(sim), obs_spec=self._obs_spec)
        else:
            norm = self._profile.world.normalization
            inputs = obs_inputs_from_batch_sim(
                sim,
                max_frames=int(self._profile.observation_progress_horizon),
                starvation_max=int(norm["starvation_max"]),
                max_length=int(norm["max_length"]),
            )
            obs = build_observations(
                inputs,
                mask=sim.get_resolved_action_mask(),
                obs_spec=self._obs_spec,
            )
        tensors = raster_tensors_from_obs(obs, device=self._device)  # (E*S, ...)
        S = int(sim.S)
        flat = slots[:, 0] * S + slots[:, 1]  # (N,) row index into E*S
        flat_t = torch.as_tensor(flat, device=self._device, dtype=torch.long)
        with torch.no_grad():
            q = self._network(
                tensors["tactical"][flat_t],
                tensors["strategic"][flat_t],
                tensors["scalars"][flat_t],
            )  # (N, 6)
        q_rows = q
        mask_t = torch.as_tensor(masks, dtype=torch.bool, device=q.device)  # (N, 6)
        neg_inf = torch.finfo(q_rows.dtype).min
        masked = torch.where(mask_t, q_rows, torch.full_like(q_rows, neg_inf))
        # Rows with no safe action fall back to the raw argmax (better a wall).
        any_safe = mask_t.any(dim=1, keepdim=True)
        masked = torch.where(any_safe, masked, q_rows)
        return masked.argmax(dim=1).cpu().numpy().astype(np.int64)


def build_simd_policy(
    spec: AgentSpec,
    seed: int,
    *,
    profile: EvaluationProfile | None = None,
    world_seeds: Sequence[int] | None = None,
) -> SimdPolicy:
    """Build a :class:`SimdPolicy` for an agent spec.

    Args:
        spec: ``("scripted", kind)`` for the anchors, or ``("checkpoint", path)``
            for a ``raster31v2`` network policy.
        seed: Seed for stochastic scripted policies.

    Returns:
        A :class:`SimdPolicy`.

    Raises:
        ValueError: For a ``vector61`` checkpoint (use ``--engine live``) or an
            unknown scripted kind.
    """
    kind, ref = spec
    if kind == "checkpoint":
        return NetworkSimdPolicy(ref, profile=profile)
    if kind != "scripted":
        raise ValueError(f"unknown agent kind {kind!r}")
    if profile is not None:
        if world_seeds is None:
            raise ValueError("profile anchors require the complete world seed assignment")
        return _ProfileAnchorSimdPolicy(ref, world_seeds)
    if ref == "greedy_food":
        return GreedyFoodSimdPolicy()
    if ref == "random_safe":
        return RandomSafeSimdPolicy(seed)
    raise ValueError(f"unknown scripted kind {ref!r}")


def _config_from_game_config(
    num_snakes: int, gamma: float, profile: EvaluationProfile | None = None
) -> BatchSimConfig:
    """Build a :class:`BatchSimConfig` from the active immutable ``GameConfig``.

    Reads the same arena/food/boost/mechanics knobs the live engine's
    ``create_training_game_state`` would, so the simd engine evaluates on the
    same arena the ``--config`` selects.

    Args:
        num_snakes: Number of arena slots (hero + opponents).
        gamma: Discount used by the reward's PBRS term.

    Returns:
        A :class:`BatchSimConfig` mirroring the active game config.
    """
    from src.core.game_config import GameConfig

    world = profile.world if profile is not None else None
    if world is not None and world.num_snakes != num_snakes:
        raise ValueError("evaluation profile world roster does not match opponent assignment")
    return BatchSimConfig(
        num_envs=1,  # overwritten per call
        num_snakes=num_snakes,
        game_width=int(world.width if world else GameConfig.WIDTH),
        game_height=int(world.height if world else GameConfig.HEIGHT),
        segment_size=int(world.segment_size if world else GameConfig.SEGMENT_SIZE),
        wall_thickness=int(world.wall_thickness if world else GameConfig.WALL_THICKNESS),
        initial_food=int(world.initial_food if world else GameConfig.INITIAL_FOOD),
        max_food=int(world.max_food if world else GameConfig.MAX_FOOD),
        min_boost_length=int(world.min_boost_length if world else GameConfig.MIN_BOOST_LENGTH),
        boost_length_cost_frames=int(
            world.boost_length_cost_frames if world else GameConfig.BOOST_LENGTH_COST_FRAMES
        ),
        mechanics_version=int(world.mechanics_version if world else GameConfig.MECHANICS_VERSION),
        gamma=float(gamma),
        arena_type=str(world.arena_type if world else GameConfig.ARENA_TYPE),
        max_capacity=int(world.max_capacity if world else 400),
        frame_rate=int(world.frame_rate if world else GameConfig.FRAME_RATE),
        kill_scale=float(world.kill_scale if world else 0.3),
        death_value=float(world.death_value if world else -3.0),
    )


def _dispatch_actions(
    sim: BatchSim,
    masks: np.ndarray,
    actions: np.ndarray,
    hero_policies: Sequence[SimdPolicy],
    opp_policies: Sequence[Sequence[SimdPolicy]],
) -> None:
    """Fill actions with a legacy-order scripted dispatch and grouped networks.

    Stateful scripted policies deliberately retain the previous one-call-per-live
    row ordering.  Checkpoint policies are stateless, so each shared instance is
    called once with all of its live ``(env, slot)`` rows for this frame.
    """
    alive = sim.get_alive()
    network_rows: Dict[NetworkSimdPolicy, List[Tuple[int, int]]] = {}

    def dispatch(policy: SimdPolicy, env: int, slot: int) -> None:
        if isinstance(policy, NetworkSimdPolicy):
            network_rows.setdefault(policy, []).append((env, slot))
            return
        result = policy.actions(
            masks[env : env + 1, slot, :], sim, np.array([[env, slot]], dtype=np.int64)
        )
        actions[env, slot] = int(result[0])

    # Preserve the pre-existing ordering for RNG-bearing scripted policies:
    # hero envs first, then each opponent slot across envs.
    for env, policy in enumerate(hero_policies):
        if alive[env, 0]:
            dispatch(policy, env, 0)
    for slot in range(1, int(sim.S)):
        for env, row in enumerate(opp_policies):
            if alive[env, slot]:
                dispatch(row[slot - 1], env, slot)

    for policy, rows in network_rows.items():
        slots = np.asarray(rows, dtype=np.int64)
        selected_masks = masks[slots[:, 0], slots[:, 1], :]
        actions[slots[:, 0], slots[:, 1]] = policy.actions(selected_masks, sim, slots)


class _TerminalHeroBatchSim(BatchSim):
    """Eval sim where arena slot 0 is terminal but opponents still respawn.

    The live gate makes the hero terminal per-snake (``hero.auto_respawn = False``,
    honoured by :meth:`~src.game.game_state.GameState.update`). :class:`BatchSim`
    only exposes the sim-wide ``allow_respawn``, so slot 0 is exempted from the
    respawn sweep here instead.
    """

    # Parked into a terminal hero's respawn timer: large enough that the base
    # sweep's per-frame decrement can never reach zero within an eval horizon.
    _NEVER = 1 << 40

    def _respawn_dead(self) -> None:
        hero_timer = self.respawn_timer[:, 0].copy()
        # A timer above zero makes the base sweep skip slot 0 before it draws a
        # respawn position, so a dead hero perturbs neither the env RNG nor the
        # world; restoring it afterwards keeps the hero from burning the timer.
        self.respawn_timer[:, 0] = self._NEVER
        try:
            super()._respawn_dead()
        finally:
            self.respawn_timer[:, 0] = hero_timer


def run_simd_eval(
    hero_spec: AgentSpec,
    opponent_specs: Sequence[AgentSpec],
    frames: int,
    seeds: Sequence[int],
    gamma: float = 0.99,
    max_frames: int = 5000,
    profile: EvaluationProfile | None = None,
    opponent_specs_by_world: Mapping[int, Sequence[AgentSpec]] | None = None,
    world_identities: Mapping[int, Dict[str, object]] | None = None,
    mix_id: str = "unspecified",
) -> List[Dict[str, object]]:
    """Run one hero over all ``seeds`` of one opponent mix in a single batch.

    Every seed is one parallel environment (``E == len(seeds)``); the hero is
    arena slot 0 with terminal death, opponents fill slots 1..S-1 and respawn.
    All envs advance in lockstep for ``frames`` steps; the gate metrics are read
    from the batch arrays each step.

    Args:
        hero_spec: Candidate/baseline agent for slot 0 (scripted only for now).
        opponent_specs: One spec per opponent slot (``len == num_snakes - 1``).
        frames: Total episode horizon (mass-integral denominator).
        seeds: World seeds; each becomes one parallel env (paired across heroes).
        gamma: Discount for the sim's PBRS reward (unused by metrics but kept
            for parity with the live reward path).
        max_frames: Episode-length cap for any frame-progress bookkeeping.

    Returns:
        One per-seed metric dict per seed, in ``seeds`` order, with the same
        keys the live engine's ``rollout`` emits: ``seed``, ``mass_integral``,
        ``max_mass``, ``mean_mass_alive``, ``kills``, ``deaths``,
        ``survival_fraction`` and a ``probes`` sub-dict.

    Raises:
        ValueError: If ``frames`` <= 0 or ``seeds`` is empty.
    """
    if frames <= 0:
        raise ValueError(f"frames must be positive, got {frames}")
    seeds = list(seeds)
    if not seeds:
        raise ValueError("at least one seed is required")
    if profile is not None:
        if profile.legacy_diagnostic:
            raise ValueError("legacy diagnostic profiles must use the legacy evaluation path")
        if frames != profile.scored_horizon:
            raise ValueError("frames must equal the explicit profile scored_horizon")
        if (
            profile.runtime.training
            or not profile.runtime.respawn
            or not profile.runtime.hero_terminal
        ):
            raise ValueError("SIMD promotion evaluation requires Watch respawn with terminal hero")

    assigned_specs = [list(opponent_specs) for _ in seeds]
    if opponent_specs_by_world is not None:
        if set(opponent_specs_by_world) != set(seeds):
            raise ValueError("opponent_specs_by_world must provide exactly one roster per seed")
        assigned_specs = [list(opponent_specs_by_world[seed]) for seed in seeds]
        if any(len(row) != len(assigned_specs[0]) for row in assigned_specs):
            raise ValueError("all materialized world rosters must have the same slot count")
    derived_identities = (
        {
            int(seed): _roster_identity(int(seed), assigned_specs[index], mix_id)
            for index, seed in enumerate(seeds)
        }
        if profile is not None
        else {}
    )
    if world_identities is not None:
        if profile is None:
            raise ValueError("world_identities require an explicit evaluation profile")
        if set(world_identities) != set(seeds):
            raise ValueError("world_identities must provide exactly one identity per seed")
        if any(dict(world_identities[seed]) != derived_identities[seed] for seed in seeds):
            raise ValueError("world_identities do not match the materialized opponent rosters")
    num_snakes = len(assigned_specs[0]) + 1
    cfg = _config_from_game_config(num_snakes, gamma, profile)
    E = len(seeds)
    cfg = BatchSimConfig(**{**cfg.__dict__, "num_envs": E})

    # train_mode would make deaths terminal for everyone, but the gate wants
    # opponents to respawn — so run in eval (non-train) mode, where only slot 0
    # is held terminal (withholding a dead hero's actions would NOT do it: the
    # non-train respawn sweep resurrects any dead slot).
    sim = _TerminalHeroBatchSim(cfg, seeds=seeds, train_mode=False)

    # Build one policy per (spec, env-seed) for scripted agents: their RNGs must
    # remain per world/slot.  A checkpoint model is stateless at evaluation time,
    # so identical paths share one policy instance and one forward per frame.
    checkpoint_cache: Dict[str, NetworkSimdPolicy] = {}

    def policy_for(spec: AgentSpec, seed: int) -> SimdPolicy:
        if spec[0] != "checkpoint":
            return (
                build_simd_policy(spec, seed)
                if profile is None
                else build_simd_policy(spec, seed, profile=profile, world_seeds=seeds)
            )
        cached = checkpoint_cache.get(spec[1])
        if cached is None:
            built = (
                build_simd_policy(spec, seed)
                if profile is None
                else build_simd_policy(spec, seed, profile=profile, world_seeds=seeds)
            )
            if not isinstance(built, NetworkSimdPolicy):
                raise TypeError("checkpoint specs must build NetworkSimdPolicy instances")
            checkpoint_cache[spec[1]] = built
            cached = built
        return cached

    hero_policies = [policy_for(hero_spec, seed) for seed in seeds]
    opp_policies: List[List[SimdPolicy]] = []
    for seed in seeds:
        row: List[SimdPolicy] = []
        for opp_slot, spec in enumerate(assigned_specs[len(opp_policies)], start=1):
            row.append(policy_for(spec, seed * 1000 + opp_slot))
        opp_policies.append(row)

    # The profile path consumes only exact BatchSim transition facts.  Preserve
    # the named legacy diagnostic accounting below until E2 retires it.
    profile_accumulators = (
        [EvaluationMetricsAccumulator(profile.scored_horizon) for _ in seeds]
        if profile is not None
        else None
    )

    # --- Per-seed accumulators (index by env; legacy diagnostic path) ---
    mass_sum = np.zeros(E, dtype=np.float64)
    max_mass = sim.get_lengths()[:, 0].astype(np.float64)  # start length
    alive_frames = np.zeros(E, dtype=np.int64)
    deaths = np.zeros(E, dtype=np.int64)
    kills = np.zeros(E, dtype=np.int64)
    boost_frames_alive = np.zeros(E, dtype=np.int64)
    food_eaten = np.zeros(E, dtype=np.int64)
    peak_length = sim.get_lengths()[:, 0].astype(np.int64)
    death_cause_code = np.zeros(E, dtype=np.int64)  # last non-none cause
    prev_alive = sim.get_alive()[:, 0].copy()
    prev_len = sim.get_lengths()[:, 0].astype(np.int64).copy()

    env_idx = np.arange(E)

    for _ in range(frames):
        actions = np.ones((E, num_snakes), dtype=np.int64)
        if profile is not None:
            # The callback sees the exact Watch pre-action snapshot: frame
            # increment, ambient-food maintenance, and respawns are complete.
            def choose_actions(prepared_sim: BatchSim) -> np.ndarray:
                masks = prepared_sim.get_resolved_action_mask()
                _dispatch_actions(prepared_sim, masks, actions, hero_policies, opp_policies)
                return actions

            sim.step_with_policy(choose_actions)
        else:
            masks = sim.get_action_mask()
            _dispatch_actions(sim, masks, actions, hero_policies, opp_policies)
            sim.step(actions)
        if profile is not None and np.any(sim.get_lengths() >= profile.world.max_capacity):
            raise RuntimeError("evaluation world exceeded its declared max_capacity")

        alive0 = sim.get_alive()[:, 0]
        len0 = sim.get_lengths()[:, 0].astype(np.int64)
        cause0 = sim.get_death_cause()[:, 0]
        kills0 = sim.get_kill_credit()[:, 0]
        boosting0 = sim.get_boosted_this_step()[:, 0]  # boost 2nd-step engaged this frame
        exact_events = sim.get_step_events() if profile_accumulators is not None else None

        if profile_accumulators is not None:
            assert exact_events is not None
            for e in env_idx:
                valid = bool(exact_events["transition_valid"][e, 0])
                died = bool(exact_events["done"][e, 0]) if valid else False
                cause = int(exact_events["death_cause"][e, 0]) if valid else DEATH_NONE
                profile_accumulators[e].observe(
                    pre_alive=bool(prev_alive[e]),
                    post=PostStepState(alive=bool(alive0[e]), logical_mass=float(len0[e])),
                    events=StepEvents(
                        food_eaten=int(exact_events["food_ate"][e, 0]) if valid else 0,
                        boost_executed=bool(exact_events["boosted"][e, 0]) if valid else False,
                        kills=int(exact_events["kills"][e, 0]) if valid else 0,
                        death=died,
                        death_cause=_DEATH_CAUSE_LABEL.get(cause) if died else None,
                    ),
                    acted=valid,
                )

        # Mass integral: hero mass over frames it was alive (dead => 0).
        mass_sum += np.where(alive0, len0, 0)
        alive_frames += alive0.astype(np.int64)
        max_mass = np.where(alive0, np.maximum(max_mass, len0), max_mass)
        peak_length = np.maximum(peak_length, np.where(alive0, len0, peak_length))

        # Deaths: alive -> dead transition. The hero is terminal, so this can
        # fire at most once, matching the live rollout's deaths <= 1.
        died_now = prev_alive & (cause0 != DEATH_NONE)
        deaths += died_now.astype(np.int64)
        # Record the most recent death cause for the probe label.
        death_cause_code = np.where(died_now, cause0, death_cause_code)

        kills += kills0

        # Boost fraction over ALIVE frames.
        boost_frames_alive += (boosting0 & alive0).astype(np.int64)

        # Food eaten: a length increase while staying alive is a pellet (boost
        # burns DECREASE length; growth is monotone +1 per pellet). Count
        # positive length deltas on frames the hero survived.
        grew = alive0 & prev_alive & (len0 > prev_len)
        food_eaten += np.where(grew, len0 - prev_len, 0)

        prev_alive = alive0.copy()
        prev_len = len0.copy()

    # --- Assemble per-seed records (live-engine METRIC_KEYS parity) ---
    records: List[Dict[str, object]] = []
    for e in env_idx:
        if profile_accumulators is not None:
            record = profile_accumulators[e].result()
            record.update(
                {
                    "seed": int(seeds[e]),
                    "evaluation_profile": profile.descriptor(),
                    "evaluation_profile_digest": profile.digest,
                    "world_identity": derived_identities[int(seeds[e])],
                }
            )
            records.append(record)
            continue
        af = int(alive_frames[e])
        records.append(
            {
                "seed": int(seeds[e]),
                "mass_integral": float(mass_sum[e]) / float(frames),
                "max_mass": float(max_mass[e]),
                "mean_mass_alive": float(mass_sum[e] / af) if af else 0.0,
                "kills": float(kills[e]),
                "deaths": float(deaths[e]),
                "survival_fraction": float(af / frames),
                "probes": {
                    "death_cause": (
                        _DEATH_CAUSE_LABEL.get(int(death_cause_code[e]))
                        if death_cause_code[e] != DEATH_NONE
                        else None
                    ),
                    "boost_frame_fraction": float(boost_frames_alive[e] / af) if af else 0.0,
                    "food_eaten": int(food_eaten[e]),
                    # Kill-opportunity + entrapment need the live BehaviorProbes'
                    # geometry; not reconstructed from the batch arrays here.
                    "kill_opportunity_count": 0,
                    "entrapment_event": False,
                    "peak_length": int(peak_length[e]),
                },
            }
        )
    return records
