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

from typing import Dict, List, Sequence, Tuple

import numpy as np

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

    def __init__(self, checkpoint_path: str) -> None:
        """Load a raster checkpoint for batched, no-grad greedy action selection.

        Args:
            checkpoint_path: Path to a ``raster31v2`` ``.pth`` checkpoint.

        Raises:
            ValueError: If the checkpoint is not a raster (``raster31v2``) model.
        """
        import torch

        from src.model.inference_agent import InferenceAgent
        from src.model.obs_spec import RASTER31V2

        self._torch = torch
        agent = InferenceAgent.from_checkpoint(checkpoint_path)
        if getattr(agent, "obs_spec", None) != RASTER31V2:
            raise ValueError(
                f"--engine simd checkpoint policy needs a '{RASTER31V2}' model; "
                f"{checkpoint_path!r} is '{getattr(agent, 'obs_spec', '?')}'. "
                "Use --engine live for 61-D vector champions."
            )
        self._agent = agent
        self._network = agent.network
        self._device = agent.device

    def actions(self, masks: np.ndarray, sim: BatchSim, slots: np.ndarray) -> np.ndarray:
        from src.model.raster_network import raster_tensors_from_obs
        from src.simd_env.featurizer import (
            build_observations,
            obs_inputs_from_batch_sim,
        )

        torch = self._torch
        # Featurize the whole batch once, then index the slots we control.
        obs = build_observations(obs_inputs_from_batch_sim(sim))
        tensors = raster_tensors_from_obs(obs, device=self._device)  # (E*S, ...)
        S = int(sim.S)
        with torch.no_grad():
            q = self._network(
                tensors["tactical"], tensors["strategic"], tensors["scalars"]
            )  # (E*S, 6)
        flat = slots[:, 0] * S + slots[:, 1]  # (N,) row index into E*S
        q_rows = q[torch.as_tensor(flat, device=q.device, dtype=torch.long)]  # (N, 6)
        mask_t = torch.as_tensor(masks, dtype=torch.bool, device=q.device)  # (N, 6)
        neg_inf = torch.finfo(q_rows.dtype).min
        masked = torch.where(mask_t, q_rows, torch.full_like(q_rows, neg_inf))
        # Rows with no safe action fall back to the raw argmax (better a wall).
        any_safe = mask_t.any(dim=1, keepdim=True)
        masked = torch.where(any_safe, masked, q_rows)
        return masked.argmax(dim=1).cpu().numpy().astype(np.int64)


def build_simd_policy(spec: AgentSpec, seed: int) -> SimdPolicy:
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
        return NetworkSimdPolicy(ref)
    if kind != "scripted":
        raise ValueError(f"unknown agent kind {kind!r}")
    if ref == "greedy_food":
        return GreedyFoodSimdPolicy()
    if ref == "random_safe":
        return RandomSafeSimdPolicy(seed)
    raise ValueError(f"unknown scripted kind {ref!r}")


def _config_from_game_config(num_snakes: int, gamma: float) -> BatchSimConfig:
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

    return BatchSimConfig(
        num_envs=1,  # overwritten per call
        num_snakes=num_snakes,
        game_width=int(GameConfig.WIDTH),
        game_height=int(GameConfig.HEIGHT),
        segment_size=int(GameConfig.SEGMENT_SIZE),
        wall_thickness=int(GameConfig.WALL_THICKNESS),
        initial_food=int(GameConfig.INITIAL_FOOD),
        max_food=int(GameConfig.MAX_FOOD),
        min_boost_length=int(GameConfig.MIN_BOOST_LENGTH),
        boost_length_cost_frames=int(GameConfig.BOOST_LENGTH_COST_FRAMES),
        mechanics_version=int(GameConfig.MECHANICS_VERSION),
        gamma=float(gamma),
        arena_type=str(GameConfig.ARENA_TYPE),
    )


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

    num_snakes = len(opponent_specs) + 1
    cfg = _config_from_game_config(num_snakes, gamma)
    E = len(seeds)
    cfg = BatchSimConfig(**{**cfg.__dict__, "num_envs": E})

    # train_mode would make deaths terminal for everyone, but the gate wants
    # opponents to respawn — so run in eval (non-train) mode, where only slot 0
    # is held terminal (withholding a dead hero's actions would NOT do it: the
    # non-train respawn sweep resurrects any dead slot).
    sim = _TerminalHeroBatchSim(cfg, seeds=seeds, train_mode=False)

    # Build one policy per (spec, env-seed): scripted RNG policies must be seeded
    # per seed so paired heroes are deterministic given the seed, exactly like
    # the live engine's set_seed(seed) before each rollout.
    hero_policies = [build_simd_policy(hero_spec, seed) for seed in seeds]
    opp_policies: List[List[SimdPolicy]] = []
    for seed in seeds:
        row: List[SimdPolicy] = []
        for opp_slot, spec in enumerate(opponent_specs, start=1):
            row.append(build_simd_policy(spec, seed * 1000 + opp_slot))
        opp_policies.append(row)

    # --- Per-seed accumulators (index by env) ---
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
        masks = sim.get_action_mask()  # (E, S, 6)
        actions = np.ones((E, num_snakes), dtype=np.int64)

        # Hero (slot 0): one policy per env; feed each its env's mask row.
        hero_masks = masks[:, 0, :]  # (E, 6)
        for e in range(E):
            if not sim.get_alive()[e, 0]:
                continue
            a = hero_policies[e].actions(
                hero_masks[e : e + 1], sim, np.array([[e, 0]], dtype=np.int64)
            )
            actions[e, 0] = int(a[0])

        # Opponents (slots 1..S-1): one policy per (env, slot).
        for opp_slot in range(1, num_snakes):
            slot_masks = masks[:, opp_slot, :]  # (E, 6)
            for e in range(E):
                if not sim.get_alive()[e, opp_slot]:
                    continue
                a = opp_policies[e][opp_slot - 1].actions(
                    slot_masks[e : e + 1], sim, np.array([[e, opp_slot]], dtype=np.int64)
                )
                actions[e, opp_slot] = int(a[0])

        sim.step(actions)

        alive0 = sim.get_alive()[:, 0]
        len0 = sim.get_lengths()[:, 0].astype(np.int64)
        cause0 = sim.get_death_cause()[:, 0]
        kills0 = sim.get_kill_credit()[:, 0]
        boosting0 = sim.get_boosted_this_step()[:, 0]  # boost 2nd-step engaged this frame

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
