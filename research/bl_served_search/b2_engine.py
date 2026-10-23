"""B2 natural Watch6 driver and full-state continuation snapshots.

CPU-only science. No fixture injection, teacher/controller changes, provider calls,
or outcome summaries here. Pickles are internal snapshots from verified B2 exports;
never deserialize arbitrary external files.
"""

from __future__ import annotations

import base64
import concurrent.futures
import copy
import hashlib
import importlib
import multiprocessing
import os
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from research.bl_k1.native import RowStats, branch_scope, configure
from research.bl_served_search import ceiling
from research.bl_served_search.b2_units import root_hash
from research.bl_served_search.exports import digest

FRP3 = "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723"
CAUSES = {0: "none", 1: "wall", 2: "self", 3: "head", 4: "body"}


@dataclass(frozen=True)
class PreparedDecision:
    """One immutable B8 decision made for one prepared simulator state."""

    sim: Any
    decision: int
    prepare_generation: int
    observation: Mapping[str, np.ndarray]
    q_bytes: bytes
    legal: tuple[int, ...]
    bare: int

    def q_values(self) -> np.ndarray:
        """Return a fresh float32 view of the exact forward bytes."""
        return np.frombuffer(self.q_bytes, dtype=np.float32).copy()


class WatchDriver:
    """Bare B8 s3 and five independently carried FRP3-s12+v8 policies."""

    def __init__(self, seed: int, b8: Path, frp3: Path) -> None:
        from research.bl_ppo.collector import _FrozenRoster, _sim_config
        from research.t1_commitment_20261015.replay import configure_cpu
        from src.simd_env.ego2s_policy import Ego2sSimdPolicy
        from src.simd_env.grid_sim import GridBatchSim

        if digest(b8) != ceiling.B8_SHA256 or digest(frp3) != FRP3:
            raise ValueError("B2 checkpoint identity differs")
        configure_cpu()
        self.sim = GridBatchSim(_sim_config(1), seeds=[seed], train_mode=False, allow_respawn=True)
        self.frozen = _FrozenRoster(self.sim, [seed], [[("checkpoint", str(frp3))] * 5])
        self.hero = Ego2sSimdPolicy(str(b8), device="cpu", forward="rowwise", mask_mode="advisory")
        self.stats = RowStats()
        self.eligible_opponent_rows = 0
        self.decision, self.life = 0, 0
        self.awaiting_respawn = False
        # This is intentionally not snapshotted: packets are process-local and
        # must never survive a restore into a different simulator object.
        self._prepare_generation = 0

    def prepare(self) -> None:
        """The real maintenance/respawn seam; exactly one runtime prepare per tick."""
        from research.bl_ppo.collector import _prepare_sim

        _prepare_sim(self.sim)
        self._prepare_generation += 1
        if self.awaiting_respawn and bool(self.sim.alive[0, 0]):
            self.life += 1
            self.awaiting_respawn = False

    def prepared_decision(self) -> PreparedDecision:
        """Build exactly one observation and rowwise B8 forward for this decision."""
        from src.simd_env.ego2s_policy import ego_config_for, forward_q
        from src.simd_env.ego_raster import build_ego_raster

        if not getattr(self.sim, "_policy_phase_prepared", False):
            raise RuntimeError("B2 decision requires a prepared selector")
        if not bool(self.sim.alive[0, 0]):
            raise RuntimeError("B2 decision requires a living hero")
        value = build_ego_raster(
            self.sim, ego_config_for(self.hero.net), backend="numba", rows=np.array([[0, 0]])
        )
        observation = {}
        for key in ("local", "global", "scalars"):
            item = np.asarray(value[key][0]).copy()
            # Bytes-backed arrays stay read-only even if a caller tries to
            # re-enable writes, unlike an owned array with WRITEABLE cleared.
            item = np.frombuffer(item.tobytes(), dtype=item.dtype).reshape(item.shape)
            observation[key] = item
        q = np.asarray(forward_q(self.hero.net, value, "rowwise").numpy()[0], dtype=np.float32)
        q_bytes = q.tobytes()
        advisory = self.sim.get_advisory_action_mask()[0, 0]
        _mask, bare = ceiling._legal(q, advisory)
        # B2 explicitly labels all game-legal alternatives, including actions
        # excluded by the advisory safety mask. Bare retains its own convention.
        legal = tuple(np.flatnonzero(self.sim.get_legal_action_mask()[0, 0]).tolist())
        return PreparedDecision(
            self.sim,
            self.decision,
            self._prepare_generation,
            MappingProxyType(observation),
            q_bytes,
            legal,
            int(bare),
        )

    def legal_and_bare(self) -> tuple[list[int], int]:
        """Compatibility adapter; callers collecting labels use the packet directly."""
        packet = self.prepared_decision()
        self.last_root_q = packet.q_values()
        return list(packet.legal), packet.bare

    def _validate_packet(self, packet: PreparedDecision) -> None:
        if (
            packet.sim is not self.sim
            or packet.decision != self.decision
            or packet.prepare_generation != self._prepare_generation
        ):
            raise RuntimeError("stale B2 prepared decision")

    def shadow_action_and_q(self, packet: PreparedDecision) -> tuple[int, bytes]:
        """Run the legacy path for the sparse production equivalence assertion."""
        self._validate_packet(packet)
        rows = np.array([[0, 0]], dtype=np.int64)
        action = int(self.hero.actions(None, self.sim, rows)[0])
        # ``actions`` does not expose Q, so reproduce its pre-mask rowwise
        # forward to compare the complete float32 result as required by F0.
        from src.simd_env.ego2s_policy import student_q

        q = np.asarray(
            student_q(self.hero.net, self.sim, rows, "rowwise").numpy()[0], dtype=np.float32
        )
        return action, q.tobytes()

    def step(
        self, forced: int | None = None, packet: PreparedDecision | None = None
    ) -> dict[str, Any]:
        """Advance a prepared selector, then observe actual transition/death events."""
        if not getattr(self.sim, "_policy_phase_prepared", False):
            raise RuntimeError("B2 transition requires a prepared selector")
        if packet is not None:
            self._validate_packet(packet)
        actions = np.ones((1, 6), dtype=np.int64)
        if bool(self.sim.alive[0, 0]):
            if forced is not None:
                actions[0, 0] = forced
            elif packet is not None:
                actions[0, 0] = packet.bare
            else:
                rows = np.array([[0, 0]], dtype=np.int64)
                actions[0, 0] = int(self.hero.actions(None, self.sim, rows)[0])
        elif forced is not None:
            actions[0, 0] = forced
        # _FrozenRoster owns independent policy carry per slot. Its nested
        # S2 scope is routed to the K1 kernel; no stock fallback is accepted.
        self.eligible_opponent_rows += int(self.sim.alive[0, 1:].sum())
        with configure("kernel", self.stats), branch_scope("mixed"):
            self.frozen.actions(self.sim, actions)
        self.sim.step(actions)
        self.frozen.observe_step(self.sim)
        self.decision += 1
        events = self.sim.get_step_events()
        death = int(events["death_cause"][0, 0])
        if death:
            self.awaiting_respawn = True
        return {
            "decision": self.decision - 1,
            "life": self.life,
            "alive": bool(self.sim.alive[0, 0]),
            "death_cause": CAUSES[death],
            "all_death_causes": [CAUSES[int(c)] for c in events["death_cause"][0]],
            "hero_action": int(actions[0, 0]),
            "actions": actions[0].tolist(),
            "hero_length": int(self.sim.length[0, 0]) if bool(self.sim.alive[0, 0]) else 0,
            "frame": int(self.sim.frame[0]),
        }

    def snapshot(self) -> bytes:
        """Complete sim/RNG/veto/runtime state, including padded-grid alias metadata."""
        import torch

        from src.core.game_config import get_config

        sim = copy.copy(self.sim)
        kernel = getattr(sim, "_kernels", None)
        sim._kernels = None
        # No module handles or checkpoint lookup paths are needed on restore.
        return pickle.dumps(
            {
                "schema": "bl-b2-state/v2",
                "game_config": get_config(),
                "sim": sim,
                "kernel_module": getattr(kernel, "__name__", None),
                "frozen": self.frozen,
                "hero": self.hero,
                "decision": self.decision,
                "life": self.life,
                "awaiting_respawn": self.awaiting_respawn,
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
            },
            protocol=5,
        )

    @classmethod
    def restore(cls, payload: bytes) -> "WatchDriver":
        """Restore only trusted, SHA-verified B2 state; no file/path resolution."""
        import torch

        from src.core.game_config import AppConfig, initialize_config

        state = pickle.loads(payload)
        if state.get("schema") != "bl-b2-state/v2":
            raise ValueError("foreign B2 snapshot schema")
        if not isinstance(state.get("game_config"), AppConfig):
            raise ValueError("B2 snapshot is missing its immutable process game configuration")
        # Spawned workers begin with defaults; the live v8 rule guard reads
        # GameConfig as well as sim.cfg. Restore both before any branch action.
        initialize_config(state["game_config"])
        driver = cls.__new__(cls)
        for name in ("sim", "frozen", "hero", "decision", "life", "awaiting_respawn"):
            setattr(driver, name, state[name])
        sim = driver.sim
        pad, height, width = sim._grid_pad, sim.grid_h, sim.grid_w
        for name in ("_owner", "_slot", "_food"):
            view = getattr(sim, name + "_pad")[:, pad : pad + height, pad : pad + width]
            if not np.array_equal(getattr(sim, name), view):
                raise ValueError("B2 snapshot grid alias differs")
            setattr(sim, name, view)
        if state["kernel_module"]:
            sim._kernels = importlib.import_module(state["kernel_module"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        driver.stats = RowStats()
        driver.eligible_opponent_rows = 0
        driver._prepare_generation = 0
        return driver


def driver_stats(driver: Any) -> dict[str, Any]:
    value = driver.stats.as_dict()
    if hasattr(driver, "eligible_opponent_rows"):
        value["eligible_opponent_rows"] = driver.eligible_opponent_rows
    return value


def attest(stats: dict[str, Any]) -> None:
    """Executed native rows, never a flag, establish kernel-only coverage."""
    required = ("kernel_rows", "stock_rows", "fallback_rows")
    if any(type(stats.get(key)) is not int or stats[key] < 0 for key in required):
        raise ValueError("INVALID B2 native execution counters")
    if any(stats.get(key, 0) != 0 for key in ("stock_rows", "fallback_rows", "noncandidate_rows")):
        raise ValueError("INVALID B2 native execution counters")
    eligible = stats.get("eligible_opponent_rows")
    if eligible is not None:
        if type(eligible) is not int or eligible < 0 or stats["kernel_rows"] != eligible:
            raise ValueError("INVALID B2 native opportunity reconciliation")
    elif stats["kernel_rows"] <= 0:
        raise ValueError("INVALID B2 native execution counters")


def validate_snapshot(payload: bytes, root: dict) -> "WatchDriver":
    """Require a restorable selector state matching its natural-root identity."""
    from src.simd_env.grid_sim import GridBatchSim

    driver = WatchDriver.restore(payload)
    sim = driver.sim
    if (
        type(sim) is not GridBatchSim
        or sim.E != 1
        or sim.S != 6
        or not sim.allow_respawn
        or sim.train_mode
        or not bool(getattr(sim, "_policy_phase_prepared", False))
        or not bool(sim.alive[0, 0])
        or int(sim.length[0, 0]) != root["length"]
        or int(sim.frame[0]) != root["frame"]
        or driver.life != root["life"]
        or driver.decision != root["decision"]
    ):
        raise ValueError("INVALID B2 full snapshot/root identity mismatch")
    if len(driver.frozen.vector) != 5 or driver.hero.mask_mode != "advisory":
        raise ValueError("INVALID B2 restored policy roster")
    legal = np.flatnonzero(sim.get_legal_action_mask()[0, 0]).tolist()
    # Source Q bytes bind the Colab control without assuming Mac/x86 argmax identity.
    q = np.asarray(root["q_values"], dtype=np.float32)
    _mask, bare = ceiling._legal(q, sim.get_advisory_action_mask()[0, 0])
    if legal != root["legal_actions"] or bare != root["bare"]:
        raise ValueError("INVALID B2 snapshot legal/source-Q bare action identity")
    return driver


def continue_branch(
    snapshot: bytes,
    action: int,
    escape_depth: int,
    *,
    restore: Callable = WatchDriver.restore,
    pulse: Callable = lambda: None,
) -> dict[str, Any]:
    """Run H300, then continue the SAME surviving branch to its frozen endpoint."""
    driver = restore(snapshot)
    h300 = None
    trace = hashlib.sha256()
    result: dict[str, Any] = {}
    for offset in range(escape_depth):
        pulse()
        if offset:
            driver.prepare()
        event = driver.step(action if offset == 0 else None)
        trace.update(ceiling.canonical(event).encode())
        if offset == 299:
            h300 = {"survives": event["alive"], "frames_executed": 300}
        if not event["alive"]:
            result = {"survives": False, "frames_executed": offset + 1, **event}
            break
        if offset + 1 == escape_depth:
            result = {"survives": True, "frames_executed": escape_depth, **event}
    if h300 is None:
        h300 = {"survives": False, "frames_executed": result["frames_executed"]}
    stats = driver_stats(driver)
    attest(stats)
    return {
        "action": action,
        "H300": h300,
        "continuation": result if h300["survives"] else None,
        "death": result if not result["survives"] else None,
        "native_v8": stats,
        "action_trace_sha256": trace.hexdigest(),
    }


def _branch_task(task: tuple[bytes, int, int, float]) -> dict:
    """A single independent process owns all state/RNG for one alternative."""
    from research.t1_commitment_20261015.replay import configure_cpu

    configure_cpu()
    snapshot, action, endpoint, deadline = task

    def pulse() -> None:
        if time.monotonic() >= deadline:
            raise RuntimeError("INCOMPLETE B2 branch wall cap")

    result = continue_branch(snapshot, action, endpoint, pulse=pulse)
    result["worker_threads"] = ceiling.worker_threads()
    if result["worker_threads"]["torch"] != 1 or result["worker_threads"]["numba"] != 1:
        raise ValueError("INVALID B2 worker math threads")
    return result


def ordered_branches(
    tasks: list, workers: int, *, worker_fn: Callable = _branch_task, pulse: Callable = lambda: None
) -> list[dict]:
    """Spawn isolated one-thread workers; consume outcomes in frozen action order."""
    if workers == 1:
        result = []
        for task in tasks:
            pulse()
            result.append(worker_fn(task))
        return result
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [pool.submit(worker_fn, task) for task in tasks]
        result = []
        for future in futures:
            while True:
                pulse()
                try:
                    result.append(future.result(timeout=5))
                    break
                except concurrent.futures.TimeoutError:
                    continue
        return result


def collect(
    spec: dict,
    index: int,
    driver: Any,
    *,
    restore: Callable = WatchDriver.restore,
    progress: Callable[[dict], None] = lambda value: None,
    clock: Callable = time.monotonic,
) -> dict:
    """Select natural roots before running any alternatives, with bounded memory."""
    from research.bl_served_search.b2_units import unit_id

    started = last = clock()
    selected: list[tuple[str, dict, bytes]] = []
    deaths, trajectory, candidates = [], [], []

    def pulse() -> None:
        nonlocal last
        now = clock()
        if now - started >= spec["unit_timeout_seconds"]:
            raise RuntimeError("INCOMPLETE B2 unit wall cap")
        if now - last >= 110:
            progress({"done_frames": driver.decision, "total_frames": spec["trajectory_frames"]})
            last = now

    for decision in range(spec["trajectory_frames"]):
        pulse()
        driver.prepare()
        selector = {
            "decision": driver.decision,
            "life": driver.life,
            "frame": int(driver.sim.frame[0]),
            "alive": bool(driver.sim.alive[0, 0]),
            "length": int(driver.sim.length[0, 0]),
        }
        if (
            bool(driver.sim.alive[0, 0])
            and int(driver.sim.length[0, 0]) >= 800
            and decision % spec["root_stride"] == 0
        ):
            frame = int(driver.sim.frame[0])
            key = root_hash(spec["namespace"], index, driver.life, frame)
            candidates.append(
                {
                    "trajectory_index": index,
                    "life": driver.life,
                    "frame": frame,
                    "length": int(driver.sim.length[0, 0]),
                    "selection_hash": key,
                }
            )
            if len(selected) < 4 or key < selected[-1][0]:
                legal, bare = driver.legal_and_bare()
                metadata = {
                    "trajectory_index": index,
                    "life": driver.life,
                    "frame": frame,
                    "decision": decision,
                    "length": int(driver.sim.length[0, 0]),
                    "selection_hash": key,
                    "legal_actions": legal,
                    "bare": bare,
                    "q_values": getattr(
                        driver, "last_root_q", np.zeros(6, dtype=np.float32)
                    ).tolist(),
                    "mix": "mixed",
                }
                selected.append((key, metadata, driver.snapshot()))
                selected.sort(key=lambda row: row[0])
                selected = selected[:4]
        event = driver.step()
        trajectory.append({**event, "selector": selector})
        if any(cause != "none" for cause in event["all_death_causes"]):
            deaths.append(event)
    parent_stats = driver_stats(driver)
    attest(parent_stats)
    parent_seconds = clock() - started
    serial_replay_seconds = 0.0
    # No alternative outcome has been computed before final bottom-k selection.
    roots = []
    for _, metadata, snapshot in selected:
        actions = sorted(set([metadata["bare"], *metadata["legal_actions"]]))
        workers = spec.get("workers", 1)
        if workers > 1:
            deadline = started + spec["unit_timeout_seconds"]
            tasks = [
                (snapshot, action, spec["escape_total_frames"], deadline) for action in actions
            ]
            alternatives = ordered_branches(tasks, workers, pulse=pulse)
            # Excluded real family canary proves actual policy serial == parallel.
            if os.environ.get("BL_B2_CANARY") == "1":
                serial_started = clock()
                serial = ordered_branches(tasks, 1, pulse=pulse)
                serial_replay_seconds += clock() - serial_started
                if ceiling.canonical(serial) != ceiling.canonical(alternatives):
                    raise ValueError("INVALID B2 serial/parallel branch bytes differ")
        else:
            alternatives = [
                continue_branch(
                    snapshot, action, spec["escape_total_frames"], restore=restore, pulse=pulse
                )
                for action in actions
            ]
        roots.append(
            {
                **metadata,
                "snapshot_sha256": hashlib.sha256(snapshot).hexdigest(),
                "snapshot_b64": base64.b64encode(snapshot).decode(),
                "alternatives": alternatives,
            }
        )
    progress({"done_frames": spec["trajectory_frames"], "total_frames": spec["trajectory_frames"]})
    return {
        "schema": "bl-b2-trajectory/v1",
        "namespace": spec["namespace"],
        "allocation_digest": spec["allocation_digest"],
        "unit_id": unit_id(spec["namespace"], index),
        "trajectory_index": index,
        "seed": spec["seeds"][index],
        "trajectory_frames": spec["trajectory_frames"],
        "workers": spec.get("workers", 1),
        "serial_parallel_root_checks": (
            len(roots)
            if os.environ.get("BL_B2_CANARY") == "1" and spec.get("workers", 1) > 1
            else 0
        ),
        "native_v8": parent_stats,
        "phase_timings": {
            "parent_seconds": parent_seconds,
            "parent_frames": spec["trajectory_frames"],
            "serial_replay_seconds": serial_replay_seconds,
            "collect_seconds": clock() - started,
        },
        "deaths": deaths,
        "trajectory": trajectory,
        "candidate_roots": candidates,
        "roots": roots,
    }


def snapshot_sources() -> dict[str, str]:
    """Bind every serialized project class and its numeric implementation."""
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "src").rglob("*.py"))
    paths += [
        root / name
        for name in (
            "research/bl_served_search/b2_engine.py",
            "research/bl_ppo/collector.py",
            "research/bl_k1/native.py",
            "research/t34_speed_s2/native_v8.py",
        )
    ]
    return {str(path.relative_to(root)): digest(path) for path in sorted(paths)}
