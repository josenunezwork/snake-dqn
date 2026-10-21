"""CPU GridBatchSim rollout collection for the BL-E2 PPO pilot.

The learner owns the hero forward pass: workers only advance worlds, build B8-format
rasters, and run the immutable opponent roster.  This keeps the L4 work on the
learner and gives a single deterministic merge point for PPO batches.
"""

from __future__ import annotations

import functools
import hashlib
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CollectorConfig:
    """The fixed E2/E4 collection contract.

    ``excluded_seeds`` is intentionally supplied by the bank builder.  A rejected
    collision is a pre-game allocation skip, never a rollout transition.
    """

    worlds: int = 128
    processes: int = 6
    rollout_steps: int = 128
    horizon: int = 5000
    seed: int = 20261021
    roster: str | tuple[tuple[str, str], ...] = "frozen-equal-mixes"
    excluded_seeds: frozenset[int] = field(default_factory=frozenset)
    long_bodies: bool = False
    native_v8: bool = True
    preflight: bool = False

    def __post_init__(self) -> None:
        if self.worlds != 128 and not self.preflight:
            raise ValueError("E2/E4 collector is fixed at 128 total worlds")
        if not 1 <= self.processes <= self.worlds:
            raise ValueError("processes must be in [1, worlds]")
        if self.rollout_steps < 1 or self.horizon < 1:
            raise ValueError("rollout_steps and horizon must be positive")


@dataclass
class RolloutBatch:
    """A deterministic, world-major group of hero transitions."""

    local: np.ndarray
    global_: np.ndarray
    scalars: np.ndarray
    mask: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    reset: np.ndarray
    world_id: np.ndarray
    frame: np.ndarray
    length: np.ndarray
    mix: np.ndarray
    episode: np.ndarray
    active_transitions: int
    allocated_transitions: int
    wall_seconds: float


PolicyFn = Callable[[Mapping[str, np.ndarray], np.ndarray], Any]


def _thread_caps() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(name, "1")


def _world_seed(config: CollectorConfig, world_id: int, episode: int) -> int:
    """World-keyed root, stable when E4 changes six workers to twelve."""
    attempt = 0
    while True:
        value = int(
            np.random.SeedSequence([config.seed, 0xE2, world_id, episode, attempt]).generate_state(
                1
            )[0]
        )
        if value not in config.excluded_seeds:
            return value
        attempt += 1


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@functools.lru_cache(maxsize=None)
def _resolve_frozen_checkpoint(reference: str) -> str:
    """Resolve SHA identities only through byte-verified transferred checkpoints."""
    from pathlib import Path

    if Path(reference).is_file():
        return reference
    root = Path(
        os.environ.get("SNAKE_CKPT_DIR", "/Users/josenunez/Projects/ml/snake-dqn/saved_snakes")
    )
    for path in sorted(root.glob("*.pth")):
        if _sha256(str(path)) == reference:
            return str(path)
    raise FileNotFoundError(f"no byte-verified frozen checkpoint for SHA {reference}")


def _sim_config(worlds: int) -> Any:
    from research.apex_safety_20260926 import dev_screen as ds
    from src.core.config_loader import load_and_initialize_config
    from src.scripts.tournament_eval import evaluation_profile_for_name
    from src.simd_env.eval_engine import _config_from_game_config

    load_and_initialize_config(str(ds.DEFAULT_CONFIG))
    profile = evaluation_profile_for_name(ds.PROFILE_NAME, ds.HORIZON)
    if profile.digest != ds.PROFILE_DIGEST:
        raise ValueError("promotion profile digest differs")
    return replace(_config_from_game_config(6, 1.0, profile), num_envs=worlds)


def _prepare_sim(sim: Any) -> None:
    """Pause at the real Watch selector seam, after maintenance and respawns."""
    if getattr(sim, "_policy_phase_prepared", False):
        return

    class Prepared(Exception):
        pass

    def stop(_sim: Any) -> None:
        raise Prepared

    try:
        sim.step_with_policy(stop)
    except Prepared:
        sim._policy_phase_prepared = True


def _inject_long(sim: Any, selected: np.ndarray | None = None) -> None:
    """Install the established six-live-snake long fixture on selected worlds only."""
    import copy

    from src.simd_env.grid_scenarios import inject_serpentines

    if selected is None or np.all(selected):
        inject_serpentines([sim], (900, 600, 400, 300, 150, 150), box_height=8)
        return
    # The public helper operates on complete batches. Preserve every unselected
    # environment's arrays, ordered food indexes and RNG by restoring them after
    # its deterministic (RNG-free) geometry replacement.
    import types

    saved = {
        key: value if isinstance(value, types.ModuleType) else copy.deepcopy(value)
        for key, value in sim.__dict__.items()
    }
    inject_serpentines([sim], (900, 600, 400, 300, 150, 150), box_height=8)
    keep = ~np.asarray(selected, dtype=bool)
    for key, old in saved.items():
        current = sim.__dict__[key]
        if isinstance(old, np.ndarray) and old.ndim and old.shape[0] == sim.E:
            current[keep] = old[keep]
        elif isinstance(old, list) and len(old) == sim.E:
            for env in np.flatnonzero(keep):
                current[int(env)] = old[int(env)]


def _roster_rows(
    config: CollectorConfig, world_ids: np.ndarray
) -> list[tuple[tuple[str, str], ...]]:
    """Return fixed per-world rosters, independent of process partitioning."""
    if isinstance(config.roster, str):
        if config.roster != "frozen-equal-mixes":
            raise ValueError(f"unknown roster source {config.roster!r}")
        from research.t34_round0.bank import _rows

        # Build the full fixed allocation even for a two-world operational
        # preflight, so its global identities are a true prefix of E4/E2.
        counts = {"scripted": 43, "frozen": 43, "mixed": 42}
        rows = _rows("bl-ppo-e2-training-20261021/v1/collector", counts, config.horizon)
        if max(world_ids, default=-1) >= len(rows):
            raise ValueError("global world id exceeds fixed roster allocation")
        groups = {mix: [r for r in rows if r["mix"] == mix] for mix in counts}
        ordered = []
        for index in range(43):
            for mix in counts:
                if index < len(groups[mix]):
                    ordered.append(groups[mix][index])
        return [
            tuple((str(kind), str(ref)) for kind, ref in ordered[int(world)]["opponents"])
            for world in world_ids
        ]
    roster = tuple((str(kind), str(ref)) for kind, ref in config.roster)
    if len(roster) != 5:
        raise ValueError("explicit frozen roster must contain five opponent slots")
    return [roster] * len(world_ids)


class _FrozenRoster:
    """Per-environment frozen rosters, grouped only for safe batched dispatch."""

    def __init__(
        self,
        sim: Any,
        seeds: Sequence[int],
        rosters: Sequence[Sequence[tuple[str, str]]],
        *,
        native_v8: bool = True,
    ) -> None:
        from src.simd_env.fast_anchors import FastProfileAnchorSimdPolicy
        from src.simd_env.vector61_policy import Vector61Runtime, Vector61SimdPolicy

        if len(rosters) != sim.E or any(len(row) != sim.S - 1 for row in rosters):
            raise ValueError("every environment needs exactly one frozen member per opponent slot")
        self.rosters = tuple(tuple(tuple(member) for member in row) for row in rosters)
        self.scripted: dict[str, Any] = {}
        controlled = np.zeros((sim.E, sim.S), dtype=bool)
        for env, roster in enumerate(self.rosters):
            for slot, (kind, _ref) in enumerate(roster, start=1):
                if kind != "scripted":
                    controlled[env, slot] = True
        self.runtime = Vector61Runtime(backend="numba") if controlled.any() else None
        if self.runtime is not None:
            self.runtime.bind(sim, controlled)
        rows_by_ref: dict[tuple[str, int], list[tuple[int, int]]] = {}
        scripted_refs: set[str] = set()
        for env, roster in enumerate(self.rosters):
            for slot, (kind, ref) in enumerate(roster, start=1):
                if kind == "scripted":
                    scripted_refs.add(ref)
                elif kind in {"checkpoint", "vector61"}:
                    rows_by_ref.setdefault((ref, slot), []).append((env, slot))
                else:
                    raise ValueError(f"unsupported frozen roster kind {kind!r}")
        for ref in scripted_refs:
            self.scripted[ref] = FastProfileAnchorSimdPolicy(ref, seeds)
        assert self.runtime is not None or not rows_by_ref
        self.vector_rows = {
            ref: np.asarray(rows, dtype=np.int64) for ref, rows in rows_by_ref.items()
        }
        self.vector = {
            ref: Vector61SimdPolicy(
                _resolve_frozen_checkpoint(ref[0]),
                self.runtime,
                veto_slots=(ref[1],),
                veto_variant="v8",
                veto_lambda=8.0,
                forward="rowwise",
            )
            for ref in rows_by_ref
        }
        self.native_v8_actual = "stock"
        self.native_v8_stats: Any = None
        self.use_native_v8 = bool(native_v8)

    def actions(self, sim: Any, action_grid: np.ndarray) -> None:
        masks, alive = sim.get_resolved_action_mask(), sim.get_alive()
        if self.runtime is not None:
            self.runtime.prepare(sim)
        scripted_rows: dict[str, list[tuple[int, int]]] = {}
        for env, roster in enumerate(self.rosters):
            for slot, (kind, ref) in enumerate(roster, start=1):
                if alive[env, slot] and kind == "scripted":
                    scripted_rows.setdefault(ref, []).append((env, slot))
        for ref, pairs in scripted_rows.items():
            rows = np.asarray(pairs, dtype=np.int64)
            action_grid[rows[:, 0], rows[:, 1]] = self.scripted[ref].actions(
                masks[rows[:, 0], rows[:, 1]], sim, rows
            )
        try:
            if not self.use_native_v8:
                raise ImportError
            from research.t34_speed_s2.native_v8 import NativeV8Stats, native_scope

            self.native_v8_actual = "s2-native-v8"
            if self.native_v8_stats is None:
                self.native_v8_stats = NativeV8Stats()
            scope = native_scope(self.native_v8_stats, audit=False)
        except ImportError:
            import contextlib

            self.native_v8_actual, scope = "stock-v8", contextlib.nullcontext()
        with scope:
            for ref, candidates in self.vector_rows.items():
                rows = candidates[alive[candidates[:, 0], candidates[:, 1]]]
                if len(rows):
                    action_grid[rows[:, 0], rows[:, 1]] = self.vector[ref].actions(
                        masks[rows[:, 0], rows[:, 1]], sim, rows
                    )

    def observe_step(self, sim: Any) -> None:
        if self.runtime is not None:
            self.runtime.observe_step(sim)

    def __getstate__(self) -> dict[str, Any]:
        """Exclude the torch module handle while retaining every policy/runtime state."""
        import copy

        state = dict(self.__dict__)
        state["vector"] = {}
        for key, policy in self.vector.items():
            saved = copy.copy(policy)
            saved._torch = None
            # Native v8 caches only which simulator's immutable rule config it
            # already validated; revalidate after restoration, without retaining
            # a second simulator object and its imported kernel module.
            if hasattr(saved, "_rule_config_checked"):
                saved._rule_config_checked = None
            state["vector"][key] = saved
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        import torch

        self.__dict__.update(state)
        for policy in self.vector.values():
            policy._torch = torch

    def reset_envs(self, env_mask: np.ndarray, seeds: Sequence[int]) -> None:
        """Forget only reset lanes; continuing vector carry and veto state persists."""
        from src.simd_env.fast_anchors import FastProfileAnchorSimdPolicy

        self.scripted = {ref: FastProfileAnchorSimdPolicy(ref, seeds) for ref in self.scripted}
        if self.runtime is None:
            return
        self.runtime.memory.reset_where(
            np.broadcast_to(env_mask[:, None], self.runtime.controlled.shape)
        )
        self.runtime._carry_valid[env_mask] = False
        self.runtime.sel_ready[env_mask] = False
        self.runtime.sel_fresh[env_mask] = False
        self.runtime._alive_post[env_mask] = False
        for policy in self.vector.values():
            for env in np.flatnonzero(env_mask):
                policy.veto_counters.pop(int(env), None)
                policy.boost_counters.pop(int(env), None)
                policy.live_vetoes.pop(int(env), None)


class _GridShard:
    """One contiguous fixed-order world shard living in one CPU process."""

    def __init__(self, config: CollectorConfig, worker_id: int, world_ids: np.ndarray) -> None:
        _thread_caps()
        from src.simd_env.grid_sim import GridBatchSim

        self.config, self.worker_id = config, worker_id
        self.world_ids = np.asarray(world_ids, dtype=np.int64)
        self.episode_number = np.zeros(len(world_ids), dtype=np.int64)
        self.episode_seeds = [_world_seed(config, int(world), 0) for world in self.world_ids]
        self.sim = GridBatchSim(
            _sim_config(len(world_ids)),
            seeds=self.episode_seeds,
            train_mode=False,
            allow_respawn=True,
        )
        self.rosters = _roster_rows(config, self.world_ids)
        # The fixed roster allocation is the source of this label.  It is carried
        # with every hero transition so reporting never tries to infer a mix from
        # an opponent action or a post-hoc world ordering.
        self.mix = np.asarray(
            [
                (
                    "scripted"
                    if all(kind == "scripted" for kind, _ in roster)
                    else "frozen" if all(kind != "scripted" for kind, _ in roster) else "mixed"
                )
                for roster in self.rosters
            ],
            dtype="U8",
        )
        self.frozen = _FrozenRoster(
            self.sim, self.episode_seeds, self.rosters, native_v8=config.native_v8
        )
        if config.long_bodies:
            _inject_long(self.sim)
        self.reset_next = np.ones(len(world_ids), dtype=bool)
        self.episode_frame = np.zeros(len(world_ids), dtype=np.int32)

    def observe(self) -> dict[str, np.ndarray]:
        from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

        _prepare_sim(self.sim)
        rows = np.column_stack((np.arange(self.sim.E), np.zeros(self.sim.E, dtype=np.int64)))
        # B8 is bv8: seven local planes, four global planes, and sixty scalars
        # (the final thirty are native v8 feature planes), never the old b/30 view.
        obs = build_ego_raster(self.sim, EgoRasterConfig(version="bv8"), backend="numba", rows=rows)
        return {
            "local": np.ascontiguousarray(obs["local"]),
            "global": np.ascontiguousarray(obs["global"]),
            "scalars": np.ascontiguousarray(obs["scalars"]),
            "mask": np.ascontiguousarray(self.sim.get_advisory_action_mask()[:, 0]),
            "reset": self.reset_next.copy(),
            "world_id": self.world_ids.copy(),
            "frame": self.episode_frame.copy(),
            "length": self.sim.length[:, 0].copy(),
            "mix": self.mix.copy(),
            "episode": self.episode_number.copy(),
        }

    def _frozen_actions(self, actions: np.ndarray) -> None:
        rows = np.argwhere(self.sim.get_alive()[:, 1:])
        if not len(rows):
            return
        rows[:, 1] += 1
        self.frozen.actions(self.sim, actions)

    def step(self, hero_actions: np.ndarray) -> dict[str, np.ndarray]:
        hero_actions = np.asarray(hero_actions, dtype=np.int64).reshape(self.sim.E)
        before = self.observe()
        actions = np.ones((self.sim.E, self.sim.S), dtype=np.int64)
        actions[:, 0] = hero_actions
        self._frozen_actions(actions)
        self.sim.step(actions)
        self.frozen.observe_step(self.sim)
        done = self.sim.get_done()[:, 0]
        alive = self.sim.get_alive()[:, 0]
        # E2 reward is objective mass only: terminal frames contribute zero.
        reward = np.where(alive, self.sim.length[:, 0] / 1000.0, 0.0).astype(np.float32)
        self.episode_frame += 1
        horizon = self.episode_frame >= self.config.horizon
        terminal = done | horizon
        result = {
            **before,
            "action": hero_actions.copy(),
            "reward": reward,
            "done": terminal.copy(),
            # `done` alone conflates an H5000 survivor with a death on the
            # final action.  Endpoint ledgers must retain this distinction.
            "survived_horizon": (terminal & horizon & alive).copy(),
        }
        if terminal.any():
            selected = np.flatnonzero(terminal)
            self.episode_number[selected] += 1
            new = [
                _world_seed(self.config, int(self.world_ids[ix]), int(self.episode_number[ix]))
                for ix in selected
            ]
            self.sim.reset_envs(terminal.astype(bool), seeds=new)
            for ix, seed in zip(selected, new):
                self.episode_seeds[int(ix)] = seed
            # reset_envs intentionally leaves all non-terminal environments and
            # their vector61 carries/v8 hook state untouched.
            self.frozen.reset_envs(terminal, self.episode_seeds)
            if self.config.long_bodies:
                _inject_long(self.sim, terminal)
            self.episode_frame[terminal] = 0
        self.reset_next = terminal
        return result

    def snapshot(self) -> dict[str, Any]:
        # Pickle retains GridBatchSim arrays and EnvRng state; this is intentionally
        # a complete process-local checkpoint rather than a weights-only hint.
        import copy

        saved_sim = copy.copy(self.sim)
        kernel = getattr(saved_sim, "_kernels", None)
        if kernel is not None:
            saved_sim._kernels = None
        return {
            "sim": saved_sim,
            "kernel_module": getattr(kernel, "__name__", None),
            "frozen": self.frozen,
            "episode_number": self.episode_number.copy(),
            "episode_seeds": list(self.episode_seeds),
            "reset_next": self.reset_next.copy(),
            "episode_frame": self.episode_frame.copy(),
        }

    def restore(self, state: Mapping[str, Any]) -> None:
        self.sim = state["sim"]
        if state.get("kernel_module"):
            import importlib

            self.sim._kernels = importlib.import_module(state["kernel_module"])
        self.frozen = state["frozen"]
        self.episode_number = np.asarray(state["episode_number"], dtype=np.int64)
        self.episode_seeds = list(state["episode_seeds"])
        self.reset_next = np.asarray(state["reset_next"], dtype=bool)
        self.episode_frame = np.asarray(state["episode_frame"], dtype=np.int32)
        if (
            self.frozen.runtime is not None
            and self.frozen.runtime.controlled.shape[0] != self.sim.E
        ):
            raise ValueError("snapshot frozen runtime does not match simulator")


class TerminalEvalDriver:
    """One H5000 terminal GridBatchSim world for the fresh paired screen."""

    def __init__(self, seed: int, roster: Sequence[tuple[str, str]], horizon: int) -> None:
        from src.simd_env.grid_sim import GridBatchSim

        self.seed, self.horizon, self.frame = int(seed), int(horizon), 0
        self.sim = GridBatchSim(_sim_config(1), seeds=[seed], train_mode=False, allow_respawn=True)
        self.frozen = _FrozenRoster(self.sim, [seed], [roster])

    def observe_hero(self) -> dict[str, np.ndarray]:
        from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

        _prepare_sim(self.sim)
        rows = np.array([[0, 0]], dtype=np.int64)
        obs = build_ego_raster(self.sim, EgoRasterConfig(version="bv8"), backend="numba", rows=rows)
        mask = self.sim.get_advisory_action_mask()[0, 0].copy()
        if not mask.any():
            mask[:3] = True
        return {
            "local": obs["local"],
            "global": obs["global"],
            "scalars": obs["scalars"],
            "mask": mask[None, :],
            "reset": np.array([self.frame == 0]),
        }

    def step(self, action: int) -> dict[str, Any]:
        if self.frame >= self.horizon or not self.sim.get_alive()[0, 0]:
            raise RuntimeError("terminal driver is finished")
        actions = np.ones((1, self.sim.S), dtype=np.int64)
        actions[0, 0] = int(action)
        self.frozen.actions(self.sim, actions)
        self.sim.step(actions)
        self.frozen.observe_step(self.sim)
        self.frame += 1
        alive = bool(self.sim.get_alive()[0, 0])
        done = not alive or self.frame >= self.horizon
        return {
            "done": done,
            "mass": float(self.sim.length[0, 0]) if alive else 0.0,
            "alive": alive,
            "frame": self.frame,
        }


def make_eval_driver(
    seed: int, roster: Sequence[tuple[str, str]], horizon: int = 5000, **_: Any
) -> TerminalEvalDriver:
    """Build the public E2 screen driver; hero action is supplied by the evaluator."""
    return TerminalEvalDriver(seed, roster, horizon)


def _worker_main(conn: Any, config: CollectorConfig, worker_id: int, ids: np.ndarray) -> None:
    try:
        shard = _GridShard(config, worker_id, ids)
    except BaseException as exc:
        conn.send({"op": "init_error", "worker": worker_id, "error": repr(exc)})
        return
    conn.send({"op": "ready", "worker": worker_id})
    while True:
        request = conn.recv()
        op = request["op"]
        if op == "close":
            return
        if op == "snapshot":
            conn.send({"op": "snapshot", "worker": worker_id, "state": shard.snapshot()})
            continue
        if op == "peek":
            conn.send({"op": "peek", "worker": worker_id, **shard.observe()})
            continue
        if op == "restore":
            shard.restore(request["state"])
            conn.send({"op": "restored", "worker": worker_id})
            continue
        if op != "collect":
            raise RuntimeError(f"unknown collector worker op {op!r}")
        rows: list[dict[str, np.ndarray]] = []
        for _ in range(int(request["steps"])):
            obs = shard.observe()
            conn.send({"op": "obs", "worker": worker_id, **obs})
            reply = conn.recv()
            if reply.get("op") != "action":
                raise RuntimeError("collector expected hero actions")
            rows.append(shard.step(reply["action"]))
        conn.send({"op": "batch", "worker": worker_id, "rows": rows})


class PersistentCollector:
    """Persistent CPU workers with centralized GPU hero inference.

    The public ordering is sorted by ``(world_id, frame-within-rollout)``.  Worker
    scheduling therefore cannot alter PPO's sample order or recurrent reset flags.
    """

    def __init__(self, config: CollectorConfig) -> None:
        self.config = config
        self._ctx = mp.get_context("spawn")
        groups = np.array_split(np.arange(config.worlds, dtype=np.int64), config.processes)
        self._parents: list[Any] = []
        self._children: list[mp.Process] = []
        for worker, ids in enumerate(groups):
            parent, child = self._ctx.Pipe()
            process = self._ctx.Process(
                target=_worker_main, args=(child, config, worker, ids), daemon=True
            )
            process.start()
            child.close()
            self._parents.append(parent)
            self._children.append(process)
        for worker, conn in enumerate(self._parents):
            if not conn.poll(90):
                self.close()
                raise RuntimeError(
                    f"collector worker {worker} did not initialize within 90 seconds"
                )
            message = conn.recv()
            if message.get("op") != "ready":
                self.close()
                raise RuntimeError(
                    f"collector worker {worker} initialization failed: {message.get('error')}"
                )

    def __enter__(self) -> "PersistentCollector":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        for conn in self._parents:
            try:
                conn.send({"op": "close"})
            except (BrokenPipeError, EOFError):
                pass
        for process in self._children:
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
        self._parents.clear()
        self._children.clear()

    @staticmethod
    def _actions(policy: PolicyFn, message: Mapping[str, np.ndarray]) -> np.ndarray:
        mask = np.asarray(message["mask"], dtype=bool).copy()
        mask[~mask.any(axis=1), :3] = True
        obs = {key: message[key] for key in ("local", "global", "scalars", "world_id", "frame")}
        obs["mask"] = mask
        result = policy(obs, np.asarray(message["reset"], dtype=bool))
        actions = result[0] if isinstance(result, tuple) else result
        actions = np.asarray(actions, dtype=np.int64).reshape(len(message["world_id"]))
        if np.any((actions < 0) | (actions >= 6)):
            raise ValueError("hero policy returned action outside [0, 5]")
        if not mask[np.arange(len(actions)), actions].all():
            raise ValueError("sampled PPO action disagrees with its recorded action mask")
        return actions

    def collect(self, policy: PolicyFn, *, steps: int | None = None) -> RolloutBatch:
        if not self._parents:
            raise RuntimeError("collector is closed")
        steps = self.config.rollout_steps if steps is None else int(steps)
        started = time.perf_counter()
        for conn in self._parents:
            conn.send({"op": "collect", "steps": steps})
        # Barrier every frame: concatenate shards in worker/world order, invoke the
        # GPU policy once for all 128 heroes, then scatter the action slice back.
        # It is the recurrence boundary as well as the batching contract.
        for _ in range(steps):
            messages: dict[int, Mapping[str, np.ndarray]] = {}
            while len(messages) != len(self._parents):
                for conn in mp.connection.wait(self._parents):
                    message = conn.recv()
                    if message["op"] != "obs":
                        raise RuntimeError(f"expected observation, got {message['op']!r}")
                    messages[int(message["worker"])] = message
            ordered = [messages[index] for index in sorted(messages)]
            merged = {
                key: np.concatenate([item[key] for item in ordered], axis=0)
                for key in (
                    "local",
                    "global",
                    "scalars",
                    "mask",
                    "reset",
                    "world_id",
                    "frame",
                    "length",
                    "mix",
                    "episode",
                )
            }
            actions = self._actions(policy, merged)
            offset = 0
            for worker, message in zip(sorted(messages), ordered):
                count = len(message["world_id"])
                self._parents[worker].send(
                    {"op": "action", "action": actions[offset : offset + count]}
                )
                offset += count
        finished: dict[int, list[dict[str, np.ndarray]]] = {}
        while len(finished) != len(self._parents):
            for conn in mp.connection.wait(self._parents):
                message = conn.recv()
                if message["op"] != "batch":
                    raise RuntimeError(f"expected final batch, got {message['op']!r}")
                finished[int(message["worker"])] = message["rows"]
        rows = [item for worker in sorted(finished) for item in finished[worker]]
        # Worker emits one whole shard per frame.  Sort individual rows by their
        # global identity to make output independent of process completion order.
        flat = []
        for tick, row in enumerate(rows):
            for index, world in enumerate(row["world_id"]):
                flat.append((int(world), tick, row, index))
        flat.sort(key=lambda item: (item[0], item[1]))

        def stack(name: str) -> np.ndarray:
            return np.stack([item[2][name][item[3]] for item in flat])

        wall = time.perf_counter() - started
        active = len(flat)
        return RolloutBatch(
            local=stack("local"),
            global_=stack("global"),
            scalars=stack("scalars"),
            mask=stack("mask"),
            action=np.asarray([item[2]["action"][item[3]] for item in flat], dtype=np.int64),
            reward=np.asarray([item[2]["reward"][item[3]] for item in flat], dtype=np.float32),
            done=np.asarray([item[2]["done"][item[3]] for item in flat], dtype=bool),
            reset=np.asarray([item[2]["reset"][item[3]] for item in flat], dtype=bool),
            world_id=np.asarray([item[0] for item in flat], dtype=np.int64),
            frame=np.asarray([item[2]["frame"][item[3]] for item in flat], dtype=np.int32),
            length=np.asarray([item[2]["length"][item[3]] for item in flat], dtype=np.int32),
            mix=np.asarray([item[2]["mix"][item[3]] for item in flat], dtype="U8"),
            episode=np.asarray([item[2]["episode"][item[3]] for item in flat], dtype=np.int64),
            active_transitions=active,
            allocated_transitions=self.config.worlds * steps,
            wall_seconds=wall,
        )

    def ruler(
        self, policy: PolicyFn, *, warmup_steps: int = 8, measured_steps: int = 64
    ) -> dict[str, float]:
        self.collect(policy, steps=warmup_steps)
        batch = self.collect(policy, steps=measured_steps)
        return {
            "active_hero_transitions_per_second": batch.active_transitions / batch.wall_seconds,
            "active_hero_transitions": float(batch.active_transitions),
            "allocated_hero_transitions": float(batch.allocated_transitions),
            "wall_seconds": batch.wall_seconds,
            "worlds": float(self.config.worlds),
            "processes": float(self.config.processes),
        }

    def peek(self) -> dict[str, np.ndarray]:
        """Return the true next state for PPO's last-value bootstrap.

        This intentionally performs no policy call and does not reuse the last
        stored rollout observation; a terminal reset has already happened inside
        its worker before this method observes the state.
        """
        for conn in self._parents:
            conn.send({"op": "peek"})
        replies = [conn.recv() for conn in self._parents]
        if any(row.get("op") != "peek" for row in replies):
            raise RuntimeError("collector peek acknowledgement missing")
        replies.sort(key=lambda row: int(row["worker"]))
        merged = {
            key: np.concatenate([row[key] for row in replies], axis=0)
            for key in (
                "local",
                "global",
                "scalars",
                "mask",
                "reset",
                "world_id",
                "frame",
                "length",
                "mix",
                "episode",
            )
        }
        order = np.argsort(merged["world_id"], kind="stable")
        return {key: value[order] for key, value in merged.items()}

    def snapshot_state(self) -> list[dict[str, Any]]:
        for conn in self._parents:
            conn.send({"op": "snapshot"})
        return [conn.recv()["state"] for conn in self._parents]

    def restore_state(self, states: Sequence[Mapping[str, Any]]) -> None:
        if len(states) != len(self._parents):
            raise ValueError("snapshot worker count does not match collector")
        for conn, state in zip(self._parents, states):
            conn.send({"op": "restore", "state": dict(state)})
        for conn in self._parents:
            if conn.recv().get("op") != "restored":
                raise RuntimeError("collector restore acknowledgement missing")
