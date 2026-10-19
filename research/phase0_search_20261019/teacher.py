"""Exact-simulator one-ply search teacher for the Phase 0 premise test.

The teacher is deliberately an evaluation-side policy.  It consumes a world
captured at the strict evaluator's Watch seam and never mutates that source
world while it examines continuations.  Each continuation owns a cloned SIMD
world plus the Vector61 runtime carry state used by frozen opponents.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Spawned preflight workers import this module before they call configure_cpu().
# Cap native pools before NumPy/Torch can initialize their BLAS/OpenMP backends.
for _thread_env in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_thread_env] = "1"
os.environ["SNAKE_DQN_DEVICE"] = "cpu"

import numpy as np

from research.t3_rollout_labels_20261015 import bank_engine


@dataclass(frozen=True)
class SearchConfig:
    """Pre-registered local search settings."""

    rollouts: int = 8
    horizon: int = 128
    epsilon: float = 0.02
    margin_threshold: float = 0.0
    length_gate: int = 600
    leaf_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.rollouts < 1 or self.horizon < 1:
            raise ValueError("rollouts and horizon must be positive")
        if not np.isfinite(self.epsilon) or not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must lie in [0, 1]")
        if (
            self.length_gate < 0
            or not np.isfinite(self.margin_threshold)
            or not np.isfinite(self.leaf_scale)
            or self.leaf_scale < 0
        ):
            raise ValueError("invalid length_gate or leaf_scale")


def configure_cpu() -> None:
    """Keep Phase 0 local CPU work within the six-thread operating limit."""
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["SNAKE_DQN_DEVICE"] = "cpu"
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits this only before its first parallel operation.
        pass


def build_world(worldrow: dict[str, Any], checkpoint: str | Path) -> bank_engine.CapturedWorld:
    """Build one strict-source world from a persisted bank row.

    ``bank_engine.build_world`` captures the real strict evaluator inside its
    Watch selector, including the terminal-hero wrapper and opponent runtime.
    ``worldrow`` may use the Phase-0 names (``world_seed``, ``opponents``) or
    T3's persisted names (``seed``, ``specs``).
    """
    seed = int(worldrow.get("world_seed", worldrow.get("seed")))
    specs = worldrow.get("opponents", worldrow.get("specs"))
    if specs is None:
        raise ValueError("world row requires opponents/specs")
    resolved_specs = []
    # Phase-0's portable bank stores frozen checkpoint SHA identities.  T3's
    # strict-source builder accepts filesystem paths, so resolve only those
    # identities through the pinned strict context; already-path specs remain
    # untouched for hermetic fixtures.
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    lookup = gid._context(1)["lookup"]
    for kind, reference in specs:
        resolved_specs.append(
            list(lookup[reference])
            if kind == "checkpoint" and reference in lookup
            else [kind, reference]
        )
    plan = {
        "seed": seed,
        "mix": str(worldrow["mix"]),
        "specs": resolved_specs,
    }
    return bank_engine.build_world("b8_s0", plan, hero_checkpoint=Path(checkpoint))


def _mask_argmax(q: np.ndarray, mask: np.ndarray) -> int:
    allowed = np.asarray(mask, dtype=bool).copy()
    if not allowed.any():
        allowed[:3] = True
    return int(np.where(allowed, q.astype(np.float64), -np.inf).argmax())


def _q_and_legal(
    world: bank_engine.CapturedWorld, env: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

    sim = world.sim
    obs = build_ego_raster(
        sim, EgoRasterConfig(version="bv8"), backend="numba", rows=np.array([[env, 0]])
    )
    from src.simd_env.ego2s_policy import forward_q

    q = forward_q(world.hero.net, obs, "rowwise").numpy()[0].copy()
    legal = sim.get_legal_action_mask()[env, 0].astype(bool)
    advisory = sim.get_advisory_action_mask()[env, 0].astype(bool)
    # B8's M3-C selection is advisory with the normal-only empty fallback.
    select_mask = advisory.copy()
    if not select_mask.any():
        select_mask[:3] = legal[:3]
    return q, legal, select_mask


def _margin(q: np.ndarray, legal: np.ndarray) -> float:
    values = np.sort(np.asarray(q, dtype=np.float64)[np.asarray(legal, dtype=bool)])
    return float(values[-1] - values[-2]) if len(values) >= 2 else float("inf")


def _rng(seed: int, decision: int, action: int, rollout: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([seed, decision, action, rollout, 20261019])
    )


def _ordinary_actions(
    world: bank_engine.CapturedWorld, rng: np.random.Generator | None, epsilon: float
) -> np.ndarray:
    """Dispatch real opponents and a B8 epsilon-greedy hero at a Watch seam."""
    from src.simd_env.eval_engine import _dispatch_actions

    sim = world.sim
    q, legal, select_mask = _q_and_legal(world)
    hero_action = _mask_argmax(q, select_mask)
    if rng is not None and epsilon and float(rng.random()) < epsilon:
        hero_action = int(rng.choice(np.flatnonzero(legal)))
    actions = np.ones((1, sim.S), dtype=np.int64)
    _dispatch_actions(sim, sim.get_resolved_action_mask(), actions, [world.hero], [world.opponents])
    actions[0, 0] = hero_action
    return actions


def _continuation_score(
    source: bank_engine.CapturedWorld,
    action: int,
    rollout: int,
    config: SearchConfig,
    decision: int,
) -> float:
    """Score one independent forced-action branch without sharing mutable state."""
    branch = source.clone()
    rng = _rng(source.seed, decision, action, rollout)
    mass = 0.0

    # ``choose`` is called inside the live selector, after its public Watch
    # preparation/runtime.prepare.  The clone carries that exact selection
    # state, so direct ``step`` consumes one decision without preparing twice.
    bank_engine.step_prepared(branch, action)
    if bool(branch.sim.alive[0, 0]):
        mass += float(branch.sim.length[0, 0])
    for _ in range(1, config.horizon):
        if not bool(branch.sim.alive[0, 0]):
            break

        def choose(prepared: Any) -> np.ndarray:
            branch.runtime.prepare(prepared)
            return _ordinary_actions(branch, rng, config.epsilon)

        branch.sim.step_with_policy(choose)
        branch.runtime.observe_step(branch.sim)
        if bool(branch.sim.alive[0, 0]):
            mass += float(branch.sim.length[0, 0])
    if not bool(branch.sim.alive[0, 0]):
        return mass
    # ``get_*_mask`` is refreshed at the previous transition's end and the
    # raster featurizer reads the exact post-step GridBatchSim state.
    q, legal = _leaf_q(branch)
    return mass + config.leaf_scale * float(np.max(q[legal]))


def _replicate(source: bank_engine.CapturedWorld, count: int) -> Any:
    """Make K independent exact rows, including mutable Vector61 carry state."""
    from types import SimpleNamespace

    from research.b8_death_census_20261014 import census
    from research.t3_rollout_labels_20261015.engine import _restore_grid_views
    from src.simd_env.fast_anchors import FastProfileAnchorSimdPolicy

    sim = census.clone_sim(source.sim)
    for name, value in list(vars(sim).items()):
        if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == 1:
            if value.dtype.hasobject:
                copied = np.empty((count, *value.shape[1:]), dtype=value.dtype)
                for env in range(count):
                    copied[env] = copy.deepcopy(value[0])
                setattr(sim, name, copied)
            else:
                setattr(sim, name, np.repeat(value, count, axis=0))
        elif isinstance(value, list) and len(value) == 1:
            setattr(sim, name, [copy.deepcopy(value[0]) for _ in range(count)])
    sim.E = count
    sim._env_index = np.arange(count, dtype=np.int64)[:, None]
    sim._snake_index = np.arange(sim.S, dtype=np.int64)[None, :]
    sim._det = None
    sim._policy_phase_prepared = True
    sim._policy_step_in_progress = False
    if hasattr(sim, "_active_env_mask"):
        del sim._active_env_mask
    _restore_grid_views(sim)
    runtime = copy.copy(source.runtime)
    for name, value in vars(source.runtime).items():
        if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == 1:
            setattr(runtime, name, np.repeat(value, count, axis=0))
        elif name == "memory":
            runtime.memory = copy.deepcopy(value)
            for mem_name, mem_value in vars(runtime.memory).items():
                if isinstance(mem_value, np.ndarray) and mem_value.ndim and mem_value.shape[0] == 1:
                    setattr(runtime.memory, mem_name, np.repeat(mem_value, count, axis=0))
    heroes, opponents = [], []
    anchor_cache: dict[str, Any] = {}
    frozen_cache: dict[int, Any] = {}
    for _ in range(count):
        hero = copy.copy(source.hero)
        hero.__dict__ = dict(hero.__dict__)
        if hasattr(hero, "runtime"):
            hero.runtime = runtime
        if hasattr(hero, "forward"):
            hero.forward = "rowwise"
        heroes.append(hero)
        row = []
        for policy in source.opponents:
            if isinstance(policy, FastProfileAnchorSimdPolicy):
                policy = anchor_cache.setdefault(
                    policy.kind, FastProfileAnchorSimdPolicy(policy.kind, [source.seed] * count)
                )
            else:
                key = id(policy)
                if key not in frozen_cache:
                    clone = copy.copy(policy)
                    clone.__dict__ = dict(clone.__dict__)
                    if hasattr(clone, "runtime"):
                        clone.runtime = runtime
                    if hasattr(clone, "forward"):
                        clone.forward = "batched"
                    frozen_cache[key] = clone
                policy = frozen_cache[key]
            row.append(policy)
        opponents.append(row)
    return SimpleNamespace(sim=sim, runtime=runtime, heroes=heroes, opponents=opponents)


def _batched_scores(
    source: bank_engine.CapturedWorld, actions: list[int], config: SearchConfig, decision: int
) -> dict[int, float]:
    """Evaluate ``action x rollout`` independent rows in one SIMD continuation."""
    from src.simd_env.eval_engine import _dispatch_actions

    branches = [(action, rollout) for action in actions for rollout in range(config.rollouts)]
    replica = _replicate(source, len(branches))
    sim, runtime = replica.sim, replica.runtime
    rngs = [_rng(source.seed, decision, action, rollout) for action, rollout in branches]
    forced = np.asarray([action for action, _ in branches], dtype=np.int64)

    actions_grid = np.ones((sim.E, sim.S), dtype=np.int64)
    _dispatch_actions(
        sim, sim.get_resolved_action_mask(), actions_grid, replica.heroes, replica.opponents
    )
    actions_grid[:, 0] = forced
    sim.step(actions_grid)
    runtime.observe_step(sim)
    mass = np.where(sim.alive[:, 0], sim.length[:, 0], 0).astype(np.float64)
    active = sim.alive[:, 0].copy()
    for _ in range(1, config.horizon):
        if not active.any():
            break

        def choose(prepared: Any) -> np.ndarray:
            runtime.prepare(prepared)
            grid = np.ones((sim.E, sim.S), dtype=np.int64)
            _dispatch_actions(
                prepared,
                prepared.get_resolved_action_mask(),
                grid,
                replica.heroes,
                replica.opponents,
            )
            # Hero evaluation is intentionally rowwise; frozen opponents remain
            # batched through their shared runtime/policy instance.
            for env in np.flatnonzero(active):
                row = bank_engine.CapturedWorld(
                    prepared,
                    runtime,
                    replica.heroes[int(env)],
                    replica.opponents[int(env)],
                    source.seed,
                    source.mix,
                    source.kinds,
                )
                q, legal, select_mask = _q_and_legal(row, int(env))
                choice = _mask_argmax(q, select_mask)
                if config.epsilon and float(rngs[int(env)].random()) < config.epsilon:
                    choice = int(rngs[int(env)].choice(np.flatnonzero(legal)))
                grid[int(env), 0] = choice
            return grid

        sim.step_with_policy(choose, active_env_mask=active)
        runtime.observe_step(sim)
        mass += np.where(sim.alive[:, 0] & active, sim.length[:, 0], 0)
        active &= sim.alive[:, 0]
    # Read the leaf at the real next Watch boundary, then abort before movement.
    leaf_q = np.zeros((sim.E, 6), dtype=np.float64)
    leaf_legal = np.zeros((sim.E, 6), dtype=bool)
    live = sim.alive[:, 0].copy()
    if live.any():

        class _LeafReady(Exception):
            pass

        def leaf(prepared: Any) -> np.ndarray:
            runtime.prepare(prepared)
            for env in np.flatnonzero(live):
                row = bank_engine.CapturedWorld(
                    prepared,
                    runtime,
                    replica.heroes[int(env)],
                    replica.opponents[int(env)],
                    source.seed,
                    source.mix,
                    source.kinds,
                )
                leaf_q[int(env)], leaf_legal[int(env)], _ = _q_and_legal(row, int(env))
            raise _LeafReady

        try:
            sim.step_with_policy(leaf, active_env_mask=live)
        except _LeafReady:
            pass
    totals = mass + config.leaf_scale * np.where(
        live, np.array([np.max(q[m]) if m.any() else 0.0 for q, m in zip(leaf_q, leaf_legal)]), 0.0
    )
    return {
        action: float(
            np.mean([totals[index] for index, pair in enumerate(branches) if pair[0] == action])
        )
        for action in actions
    }


def _leaf_q(world: bank_engine.CapturedWorld) -> tuple[np.ndarray, np.ndarray]:
    """Read B8 at the next exact Watch boundary without consuming a transition."""
    captured: dict[str, Any] = {}

    class _LeafReady(Exception):
        pass

    def select(prepared: Any) -> np.ndarray:
        world.runtime.prepare(prepared)
        q, legal, _ = _q_and_legal(world)
        captured.update(q=q, legal=legal)
        raise _LeafReady

    try:
        world.sim.step_with_policy(select)
    except _LeafReady:
        pass
    if "q" not in captured:
        raise RuntimeError("could not prepare exact leaf boundary")
    return captured["q"], captured["legal"]


class SearchTeacher:
    """Choose legal hero actions from exact cloned continuations."""

    def __init__(self, config: SearchConfig = SearchConfig()) -> None:
        self.config = config

    def choose(self, world: bank_engine.CapturedWorld) -> tuple[int, dict[str, Any]]:
        """Return an action and diagnostics from an already-prepared Watch world."""
        q, legal, select_mask = _q_and_legal(world)
        base = _mask_argmax(q, select_mask)
        # The pre-registered trigger is Q margin over domain-legal moves;
        # B8's action itself retains its advisory-mask selection semantics.
        margin = _margin(q, legal)
        length = int(world.sim.length[0, 0])
        searched = bool(length >= self.config.length_gate or margin < self.config.margin_threshold)
        diag: dict[str, Any] = {
            "q": q.copy(),
            "mask": legal.copy(),
            "selection_mask": select_mask.copy(),
            "margin": margin,
            "searched": searched,
            "scores": {},
            "length": length,
        }
        if not searched:
            return base, diag
        decision = int(world.sim.frame[0]) - 1
        scores = _batched_scores(
            world, [int(action) for action in np.flatnonzero(legal)], self.config, decision
        )
        diag["scores"] = scores
        # dict insertion follows increasing action order, preserving first-index ties.
        return max(scores, key=scores.__getitem__), diag


def step_world(
    world: bank_engine.CapturedWorld,
    teacher: SearchTeacher | None = None,
    trace: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Advance one exact Watch transition, optionally replacing B8's hero action."""
    from src.simd_env.eval_engine import _dispatch_actions

    selected: dict[str, Any] = {}

    def choose(prepared: Any) -> np.ndarray:
        world.runtime.prepare(prepared)
        q, legal, select_mask = _q_and_legal(world)
        base = _mask_argmax(q, select_mask)
        diag = {
            "q": q.copy(),
            "mask": legal.copy(),
            "selection_mask": select_mask.copy(),
            "margin": _margin(q, legal),
            "searched": False,
            "scores": {},
            "length": int(prepared.length[0, 0]),
        }
        if teacher is not None:
            action, diag = teacher.choose(world)
        else:
            action = base
        actions = np.ones((1, prepared.S), dtype=np.int64)
        _dispatch_actions(
            prepared, prepared.get_resolved_action_mask(), actions, [world.hero], [world.opponents]
        )
        actions[0, 0] = action
        selected.update(diag, action=int(action), b8_action=int(base), frame=int(prepared.frame[0]))
        return actions

    world.sim.step_with_policy(choose)
    world.runtime.observe_step(world.sim)
    if trace is not None:
        trace(dict(selected))
    return selected
