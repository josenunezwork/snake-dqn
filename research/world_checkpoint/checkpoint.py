"""Snapshot, restore, atomic persistence, and deterministic hashing for paired-world state."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from research.t3_rollout_labels_20261015.trigger import DenseTrigger
from src.game.snake_geometry import clear_geometry_cache
from src.simd_env.grid_sim import FOOD_NONE, FOOD_OUTSIDE, GridBatchSim
from src.simd_env.vector61_policy import (
    Vector61Runtime,
    Vector61SimdPolicy,
    live_veto_hook,
)
from tools.ops.package_spec import publish_immutable, verify_immutable

WORLD_CHECKPOINT_SCHEMA = "watch6-world-checkpoint/v1"


def _fsync_dir(directory: Path) -> None:
    fd = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_atomic_replace(path: Path, data: bytes) -> None:
    tmp = path.parent / f".tmp-{path.name}-{os.getpid()}"
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def save_checkpoint(
    state: Mapping[str, Any],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Atomically write checkpoint payload, ``.sha256`` sidecar, and ``.complete`` marker."""
    target = Path(path)
    payload = dict(state)
    payload.setdefault("checkpoint_schema", WORLD_CHECKPOINT_SCHEMA)
    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    if not overwrite:
        meta = publish_immutable(target, raw)
        return {
            "path": str(target),
            "sha256": meta["sha256"],
            "size_bytes": int(meta["size"]),
        }

    target.parent.mkdir(parents=True, exist_ok=True)
    sha_path = Path(f"{target}.sha256")
    complete_path = Path(f"{target}.complete")
    if complete_path.exists():
        complete_path.unlink()
        _fsync_dir(target.parent)

    sha = hashlib.sha256(raw).hexdigest()
    _write_atomic_replace(target, raw)
    _write_atomic_replace(sha_path, f"{sha}  {target.name}\n".encode("utf-8"))
    _write_atomic_replace(complete_path, b"complete\n")
    _fsync_dir(target.parent)
    return {
        "path": str(target),
        "sha256": sha,
        "size_bytes": len(raw),
    }


def verify_checkpoint(path: str | Path) -> str:
    """Verify ``path``, ``path.sha256``, and ``path.complete`` and return the SHA-256 hex digest."""
    target = Path(path)
    complete_path = Path(f"{target}.complete")
    sha_path = Path(f"{target}.sha256")
    if not complete_path.is_file() or complete_path.is_symlink():
        raise ValueError(f"missing completion marker: {complete_path}")
    if not sha_path.is_file() or sha_path.is_symlink():
        raise ValueError(f"missing sha256 sidecar: {sha_path}")
    if not target.is_file() or target.is_symlink():
        raise ValueError(f"missing checkpoint file: {target}")
    try:
        meta = verify_immutable(target)
    except ValueError as exc:
        raise ValueError(
            f"checkpoint sha256 mismatch or invalid marker for {target}: {exc}"
        ) from exc
    return str(meta["sha256"])


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Verify ``.complete`` and ``.sha256`` sidecar, then load and validate checkpoint state."""
    target = Path(path)
    verify_checkpoint(target)
    raw = target.read_bytes()
    data = pickle.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"checkpoint payload must be a dict: {target}")
    if data.get("checkpoint_schema") != WORLD_CHECKPOINT_SCHEMA:
        raise ValueError(
            f"unexpected checkpoint schema {data.get('checkpoint_schema')!r} in {target}"
        )
    return data


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Return the newest verified checkpoint in ``checkpoint_dir``, ignoring incomplete files."""
    root = Path(checkpoint_dir)
    if not root.is_dir():
        return None
    candidates: list[tuple[int, str, Path]] = []
    for complete_file in root.glob("*.ckpt.complete"):
        ckpt_path = Path(str(complete_file)[: -len(".complete")])
        try:
            verify_checkpoint(ckpt_path)
            data = load_checkpoint(ckpt_path)
            seq = int(data.get("checkpoint_seq", 0))
            candidates.append((seq, ckpt_path.name, ckpt_path))
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][2]


def _snapshot_tensor(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    return {
        "array": tensor.detach().cpu().numpy().copy(),
        "device": str(tensor.device),
    }


def _restore_tensor(data: Mapping[str, Any] | None) -> torch.Tensor | None:
    if data is None:
        return None
    arr = np.asarray(data["array"]).copy()
    return torch.from_numpy(arr).to(str(data.get("device", "cpu")))


def _snapshot_counters_obj(obj: Any) -> dict[str, Any]:
    return {k: copy.deepcopy(v) for k, v in obj.__dict__.items()}


def _restore_counters_obj(obj: Any, data: Mapping[str, Any]) -> None:
    for k, v in data.items():
        setattr(obj, k, copy.deepcopy(v))


def _snapshot_veto(veto: Any) -> dict[str, Any] | None:
    if veto is None:
        return None
    snap: dict[str, Any] = {
        "class": type(veto).__name__,
        "counters": _snapshot_counters_obj(veto.counters),
    }
    if hasattr(veto, "v8"):
        snap["v8"] = _snapshot_counters_obj(veto.v8)
    if hasattr(veto, "v7"):
        v7 = veto.v7
        snap["v7"] = {
            "counters": _snapshot_counters_obj(v7.counters),
            "v7": _snapshot_counters_obj(v7.v7),
        }
    if hasattr(veto, "v5"):
        snap["v5"] = _snapshot_counters_obj(veto.v5)
    if getattr(veto, "_reference", None) is not None:
        ref = veto._reference
        snap["_reference"] = {
            "counters": _snapshot_counters_obj(ref.counters),
            "v7": _snapshot_counters_obj(ref.v7),
        }
    else:
        snap["_reference"] = None
    return snap


def _restore_veto(veto: Any, data: Mapping[str, Any] | None) -> None:
    if veto is None or data is None:
        return
    _restore_counters_obj(veto.counters, data["counters"])
    if "v8" in data and hasattr(veto, "v8"):
        _restore_counters_obj(veto.v8, data["v8"])
    if "v7" in data and hasattr(veto, "v7"):
        _restore_counters_obj(veto.v7.counters, data["v7"]["counters"])
        _restore_counters_obj(veto.v7.v7, data["v7"]["v7"])
    if "v5" in data and hasattr(veto, "v5"):
        _restore_counters_obj(veto.v5, data["v5"])
    if data.get("_reference") is not None and getattr(veto, "_reference", None) is not None:
        _restore_counters_obj(veto._reference.counters, data["_reference"]["counters"])
        _restore_counters_obj(veto._reference.v7, data["_reference"]["v7"])


def snapshot_rng_state(sim: GridBatchSim) -> dict[str, Any]:
    """Capture Python, NumPy, PyTorch, and per-env simulator RNG states."""
    return {
        "py_random": random.getstate(),
        "np_random": copy.deepcopy(np.random.get_state(legacy=False)),
        "torch_random": torch.get_rng_state().detach().cpu().numpy().copy(),
        "sim_rngs": [
            {
                "state": env_rng._rng.getstate(),
                "game_width": int(env_rng.game_width),
                "game_height": int(env_rng.game_height),
                "segment_size": int(env_rng.segment_size),
                "wall_thickness": int(env_rng.wall_thickness),
            }
            for env_rng in sim._rngs
        ],
    }


def restore_rng_state(sim: GridBatchSim, data: Mapping[str, Any]) -> None:
    """Restore Python, NumPy, PyTorch, and per-env simulator RNG states."""
    random.setstate(data["py_random"])
    np.random.set_state(copy.deepcopy(data["np_random"]))
    torch_state = torch.from_numpy(np.asarray(data["torch_random"], dtype=np.uint8).copy())
    torch.set_rng_state(torch_state)
    sim_rngs = data["sim_rngs"]
    if len(sim_rngs) != len(sim._rngs):
        raise ValueError(f"sim RNG count mismatch: {len(sim_rngs)} != {len(sim._rngs)}")
    for env_rng, saved_rng in zip(sim._rngs, sim_rngs):
        env_rng._rng.setstate(saved_rng["state"])
        env_rng.game_width = int(saved_rng["game_width"])
        env_rng.game_height = int(saved_rng["game_height"])
        env_rng.segment_size = int(saved_rng["segment_size"])
        env_rng.wall_thickness = int(saved_rng["wall_thickness"])


def snapshot_sim_state(sim: GridBatchSim) -> dict[str, Any]:
    """Capture complete GridBatchSim state using compact active-ring segments per snake."""
    E, S = int(sim.E), int(sim.S)
    active_segments: list[list[np.ndarray]] = []
    for e in range(E):
        env_segs: list[np.ndarray] = []
        for s in range(S):
            n = int(sim.seg_count[e, s])
            hp = int(sim.head_ptr[e, s])
            rings = (hp - np.arange(n, dtype=np.int64)) % sim.cap
            env_segs.append(sim.bodies[e, s, rings].astype(np.int16, copy=True))
        active_segments.append(env_segs)

    return {
        "E": E,
        "S": S,
        "cap": int(sim.cap),
        "train_mode": bool(sim.train_mode),
        "allow_respawn": bool(sim.allow_respawn),
        "_seeds": [int(x) for x in sim._seeds],
        "_constructor_food_draws_pending": bool(sim._constructor_food_draws_pending),
        "head_ptr": sim.head_ptr.copy(),
        "seg_count": sim.seg_count.copy(),
        "active_segments": active_segments,
        "length": sim.length.copy(),
        "alive": sim.alive.copy(),
        "direction": sim.direction.copy(),
        "boost_frames": sim.boost_frames.copy(),
        "frames_since_food": sim.frames_since_food.copy(),
        "respawn_timer": sim.respawn_timer.copy(),
        "_boosted_this_step": sim._boosted_this_step.copy(),
        "_reward_prev_length": sim._reward_prev_length.copy(),
        "_trav": sim._trav.copy(),
        "_trav_valid": sim._trav_valid.copy(),
        "food_cells": [[(int(c[0]), int(c[1])) for c in cells] for cells in sim.food_cells],
        "corpse_cells": [
            sorted((int(c[0]), int(c[1])) for c in cells) for cells in sim.corpse_cells
        ],
        "_food_version": sim._food_version.copy(),
        "frame": sim.frame.copy(),
        "_last_reward": sim._last_reward.copy(),
        "_last_mask": sim._last_mask.copy(),
        "_last_legal_mask": sim._last_legal_mask.copy(),
        "_last_resolved_mask": sim._last_resolved_mask.copy(),
        "_last_done": sim._last_done.copy(),
        "_last_transition_valid": sim._last_transition_valid.copy(),
        "_last_food_ate": sim._last_food_ate.copy(),
        "_last_ambient_food_ate": sim._last_ambient_food_ate.copy(),
        "_last_corpse_food_ate": sim._last_corpse_food_ate.copy(),
        "_step_ambient_food_ate": sim._step_ambient_food_ate.copy(),
        "_step_corpse_food_ate": sim._step_corpse_food_ate.copy(),
        "_last_death_cause": sim._last_death_cause.copy(),
        "_last_kills": sim._last_kills.copy(),
        "_last_kill_victim_len": [
            [list(sim._last_kill_victim_len[e, s]) for s in range(S)] for e in range(E)
        ],
    }


def restore_sim_state(sim: GridBatchSim, data: Mapping[str, Any]) -> None:
    """Restore complete GridBatchSim state and rebuild occupancy/food grids."""
    E, S = int(sim.E), int(sim.S)
    if (int(data["E"]), int(data["S"]), int(data["cap"])) != (E, S, int(sim.cap)):
        raise ValueError("GridBatchSim shape/capacity mismatch on restore")

    sim.train_mode = bool(data["train_mode"])
    sim.allow_respawn = bool(data["allow_respawn"])
    sim._seeds = [int(x) for x in data["_seeds"]]
    sim._constructor_food_draws_pending = bool(data["_constructor_food_draws_pending"])
    sim.head_ptr[...] = data["head_ptr"]
    sim.seg_count[...] = data["seg_count"]
    sim.bodies.fill(0)
    active_segments = data["active_segments"]
    for e in range(E):
        for s in range(S):
            n = int(sim.seg_count[e, s])
            hp = int(sim.head_ptr[e, s])
            rings = (hp - np.arange(n, dtype=np.int64)) % sim.cap
            segs = np.asarray(active_segments[e][s], dtype=np.int64).reshape(n, 2)
            sim.bodies[e, s, rings] = segs

    sim.length[...] = data["length"]
    sim.alive[...] = data["alive"]
    sim.direction[...] = data["direction"]
    sim.boost_frames[...] = data["boost_frames"]
    sim.frames_since_food[...] = data["frames_since_food"]
    sim.respawn_timer[...] = data["respawn_timer"]
    sim._boosted_this_step[...] = data["_boosted_this_step"]
    sim._reward_prev_length[...] = data["_reward_prev_length"]
    sim._trav[...] = data["_trav"]
    sim._trav_valid[...] = data["_trav_valid"]

    sim.food_cells = [[(int(c[0]), int(c[1])) for c in cells] for cells in data["food_cells"]]
    sim.corpse_cells = [{(int(c[0]), int(c[1])) for c in cells} for cells in data["corpse_cells"]]
    sim.food_set = [set(cells) for cells in sim.food_cells]

    sim.frame[...] = data["frame"]
    sim._last_reward[...] = data["_last_reward"]
    sim._last_mask[...] = data["_last_mask"]
    sim._last_legal_mask[...] = data["_last_legal_mask"]
    sim._last_resolved_mask[...] = data["_last_resolved_mask"]
    sim._last_done[...] = data["_last_done"]
    sim._last_transition_valid[...] = data["_last_transition_valid"]
    sim._last_food_ate[...] = data["_last_food_ate"]
    sim._last_ambient_food_ate[...] = data["_last_ambient_food_ate"]
    sim._last_corpse_food_ate[...] = data["_last_corpse_food_ate"]
    sim._step_ambient_food_ate[...] = data["_step_ambient_food_ate"]
    sim._step_corpse_food_ate[...] = data["_step_corpse_food_ate"]
    sim._last_death_cause[...] = data["_last_death_cause"]
    sim._last_kills[...] = data["_last_kills"]
    for e in range(E):
        for s in range(S):
            sim._last_kill_victim_len[e, s] = list(data["_last_kill_victim_len"][e][s])

    for attr in ("_policy_phase_prepared", "_policy_step_in_progress", "_active_env_mask"):
        if hasattr(sim, attr):
            delattr(sim, attr)

    sim._owner_pad.fill(-1)
    sim._slot_pad.fill(0)
    sim._food_pad.fill(FOOD_OUTSIDE)
    sim._food.fill(FOOD_NONE)
    for e in range(E):
        sim._rebuild_env_grids(e)
    sim.check_grid_invariants()
    sim._food_version[...] = data["_food_version"]


def snapshot_runtime_state(runtime: Vector61Runtime) -> dict[str, Any]:
    """Capture Vector61Runtime carry, enemy-trend memory, and selection buffers."""
    return {
        "controlled": runtime.controlled.copy(),
        "fresh_rows": int(runtime.fresh_rows),
        "carried_rows": int(runtime.carried_rows),
        "memory_prev_dist": runtime.memory.prev_dist.copy(),
        "memory_prev_id": runtime.memory.prev_id.copy(),
        "_carry_state": runtime._carry_state.copy(),
        "_carry_mask": runtime._carry_mask.copy(),
        "_carry_valid": runtime._carry_valid.copy(),
        "_alive_post": runtime._alive_post.copy(),
        "sel_state": runtime.sel_state.copy(),
        "sel_mask": runtime.sel_mask.copy(),
        "sel_ready": runtime.sel_ready.copy(),
        "sel_fresh": runtime.sel_fresh.copy(),
        "featurizer_fallbacks": int(getattr(runtime.featurizer, "fallbacks", 0)),
    }


def restore_runtime_state(runtime: Vector61Runtime, data: Mapping[str, Any]) -> None:
    """Restore Vector61Runtime carry, enemy-trend memory, and selection buffers."""
    runtime.controlled[...] = data["controlled"]
    runtime.fresh_rows = int(data["fresh_rows"])
    runtime.carried_rows = int(data["carried_rows"])
    runtime.memory.prev_dist[...] = data["memory_prev_dist"]
    runtime.memory.prev_id[...] = data["memory_prev_id"]
    runtime._carry_state[...] = data["_carry_state"]
    runtime._carry_mask[...] = data["_carry_mask"]
    runtime._carry_valid[...] = data["_carry_valid"]
    runtime._alive_post[...] = data["_alive_post"]
    runtime.sel_state[...] = data["sel_state"]
    runtime.sel_mask[...] = data["sel_mask"]
    runtime.sel_ready[...] = data["sel_ready"]
    runtime.sel_fresh[...] = data["sel_fresh"]
    featurizer = runtime.featurizer
    if hasattr(featurizer, "_food_sim_id"):
        featurizer._food_sim_id = None
        featurizer._food_lists = {}
        featurizer._food_buf = np.zeros((0, 0, 2), dtype=np.int64)
        featurizer._food_n = np.zeros(0, dtype=np.int64)
        featurizer.fallbacks = int(data.get("featurizer_fallbacks", 0))


def _snapshot_last_veto(last_veto: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if last_veto is None:
        return None
    out: dict[str, Any] = {}
    for k, v in last_veto.items():
        out[str(k)] = v.copy() if isinstance(v, np.ndarray) else copy.deepcopy(v)
    return out


def _restore_last_veto(data: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if data is None:
        return None
    out: dict[str, Any] = {}
    for k, v in data.items():
        out[str(k)] = v.copy() if isinstance(v, np.ndarray) else copy.deepcopy(v)
    return out


def snapshot_opponents_state(
    opponents: Sequence[Vector61SimdPolicy],
) -> list[dict[str, Any]]:
    """Capture Vector61SimdPolicy opponent veto counters, batch results, and live_vetoes."""
    out: list[dict[str, Any]] = []
    for opp in opponents:
        out.append(
            {
                "veto_slots": sorted(int(x) for x in opp.veto_slots),
                "veto_variant": str(opp.veto_variant),
                "veto_counters": {
                    int(k): _snapshot_counters_obj(v) for k, v in opp.veto_counters.items()
                },
                "boost_counters": {
                    int(k): _snapshot_counters_obj(v) for k, v in opp.boost_counters.items()
                },
                "last_veto": _snapshot_last_veto(opp.last_veto),
                "live_vetoes": {key: _snapshot_veto(veto) for key, veto in opp.live_vetoes.items()},
            }
        )
    return out


def restore_opponents_state(
    opponents: Sequence[Vector61SimdPolicy],
    data: Sequence[Mapping[str, Any]],
) -> None:
    """Restore Vector61SimdPolicy opponent veto counters, batch results, and live_vetoes."""
    if len(opponents) != len(data):
        raise ValueError(f"opponent count mismatch: {len(opponents)} != {len(data)}")
    from src.evaluation.safety_veto import SafetyVetoCounters
    from src.evaluation.safety_veto_v5 import BoostAwareCounters

    for opp, saved in zip(opponents, data):
        opp.veto_slots = frozenset(int(x) for x in saved["veto_slots"])
        opp.veto_variant = str(saved["veto_variant"])
        opp.veto_counters = {}
        for k, v in saved["veto_counters"].items():
            cnt = SafetyVetoCounters()
            _restore_counters_obj(cnt, v)
            opp.veto_counters[int(k)] = cnt
        opp.boost_counters = {}
        for k, v in saved["boost_counters"].items():
            bcnt = BoostAwareCounters()
            _restore_counters_obj(bcnt, v)
            opp.boost_counters[int(k)] = bcnt
        opp.last_veto = _restore_last_veto(saved["last_veto"])
        new_live_vetoes: dict[Any, Any] = {}
        for key, veto_snap in saved["live_vetoes"].items():
            existing = opp.live_vetoes.get(key)
            if existing is None:
                existing = live_veto_hook(opp.veto_spec)
            _restore_veto(existing, veto_snap)
            new_live_vetoes[key] = existing
        opp.live_vetoes = new_live_vetoes


def _snapshot_snake(snake: Any) -> dict[str, Any]:
    carried = getattr(snake, "_carried_selection", None)
    carried_snap = None
    if isinstance(carried, dict):
        carried_snap = {
            "frame": int(carried["frame"]),
            "state": _snapshot_tensor(carried["state"]),
            "safe_actions": [int(a) for a in carried["safe_actions"]],
        }
    snap: dict[str, Any] = {
        "id": int(snake.id),
        "color": tuple(snake.color) if isinstance(snake.color, (list, tuple)) else snake.color,
        "color_name": getattr(snake, "color_name", None),
        "segments": [(int(x), int(y)) for x, y in snake.segments],
        "direction": (int(snake.direction[0]), int(snake.direction[1])),
        "is_alive": bool(snake.is_alive),
        "length": int(snake.length),
        "segment_size": int(snake.segment_size),
        "game_width": int(snake.game_width),
        "game_height": int(snake.game_height),
        "food_capacity": int(snake.food_capacity),
        "respawn_timer": int(snake.respawn_timer),
        "frames_since_food": int(snake.frames_since_food),
        "_prev_nearest_enemy_dist": float(snake._prev_nearest_enemy_dist),
        "_prev_nearest_enemy_id": (
            None if snake._prev_nearest_enemy_id is None else int(snake._prev_nearest_enemy_id)
        ),
        "is_boosting": bool(snake.is_boosting),
        "boost_frames": int(snake.boost_frames),
        "last_move_positions": [(int(x), int(y)) for x, y in snake.last_move_positions],
        "pending_trail_pellets": [(int(x), int(y)) for x, y in snake.pending_trail_pellets],
        "auto_respawn": bool(getattr(snake, "auto_respawn", True)),
        "has_reward_prev_length": hasattr(snake, "_reward_prev_length"),
        "_reward_prev_length": int(getattr(snake, "_reward_prev_length", snake.length)),
        "last_reward_breakdown": copy.deepcopy(getattr(snake, "last_reward_breakdown", None)),
        "has_last_death_cause": hasattr(snake, "last_death_cause"),
        "last_death_cause": getattr(snake, "last_death_cause", None),
        "has_pre_collision_action": hasattr(snake, "_pre_collision_action"),
        "_pre_collision_action": getattr(snake, "_pre_collision_action", None),
        "has_pre_collision_epsilon": hasattr(snake, "_pre_collision_epsilon"),
        "_pre_collision_epsilon": getattr(snake, "_pre_collision_epsilon", None),
        "has_pre_collision_state": hasattr(snake, "_pre_collision_state"),
        "_pre_collision_state": _snapshot_tensor(getattr(snake, "_pre_collision_state", None)),
        "actor_id": None if snake.actor_id is None else int(snake.actor_id),
        "num_actors": None if snake.num_actors is None else int(snake.num_actors),
        "actor_epsilon_exponent": (
            None if snake.actor_epsilon_exponent is None else float(snake.actor_epsilon_exponent)
        ),
        "actor_epsilon": (None if snake.actor_epsilon is None else float(snake.actor_epsilon)),
        "last_state": _snapshot_tensor(snake.last_state),
        "last_action": None if snake.last_action is None else int(snake.last_action),
        "last_reward": None if snake.last_reward is None else float(snake.last_reward),
        "last_next_state": _snapshot_tensor(snake.last_next_state),
        "last_next_action_mask": (
            None
            if snake.last_next_action_mask is None
            else [bool(x) for x in snake.last_next_action_mask]
        ),
        "last_next_action_mask_semantics": snake.last_next_action_mask_semantics,
        "next_action_mask_semantics": getattr(snake, "next_action_mask_semantics", None),
        "current_action_mask": (
            None
            if snake.current_action_mask is None
            else [bool(x) for x in snake.current_action_mask]
        ),
        "current_action_mask_semantics": snake.current_action_mask_semantics,
        "current_legal_action_mask": (
            None
            if snake.current_legal_action_mask is None
            else [bool(x) for x in snake.current_legal_action_mask]
        ),
        "current_advisory_action_mask": (
            None
            if snake.current_advisory_action_mask is None
            else [bool(x) for x in snake.current_advisory_action_mask]
        ),
        "last_done": bool(snake.last_done),
        "last_transition_frame": (
            None if snake.last_transition_frame is None else int(snake.last_transition_frame)
        ),
        "_total_reward": float(snake._total_reward),
        "current_loss": (None if snake.current_loss is None else float(snake.current_loss)),
        "current_epsilon": (
            None if snake.current_epsilon is None else float(snake.current_epsilon)
        ),
        "boost_exploration_rate": (
            None if snake.boost_exploration_rate is None else float(snake.boost_exploration_rate)
        ),
        "danger_exploration_rate": (
            None if snake.danger_exploration_rate is None else float(snake.danger_exploration_rate)
        ),
        "action_history": copy.deepcopy(snake.action_history),
        "max_action_history": int(snake.max_action_history),
        "last_q_values": (
            None if snake.last_q_values is None else [float(x) for x in snake.last_q_values]
        ),
        "record_q_values": bool(snake.record_q_values),
        "carry_forward_selection": bool(snake.carry_forward_selection),
        "_carried_selection": carried_snap,
        "safety_veto": _snapshot_veto(getattr(snake, "safety_veto", None)),
    }
    return snap


def _restore_snake(snake: Any, data: Mapping[str, Any]) -> None:
    snake.id = int(data["id"])
    snake.color = data["color"]
    if data.get("color_name") is not None:
        snake.color_name = data["color_name"]
    snake.segments = [(int(x), int(y)) for x, y in data["segments"]]
    snake.direction = (int(data["direction"][0]), int(data["direction"][1]))
    snake.is_alive = bool(data["is_alive"])
    snake.length = int(data["length"])
    snake.segment_size = int(data["segment_size"])
    snake.game_width = int(data["game_width"])
    snake.game_height = int(data["game_height"])
    snake.food_capacity = int(data["food_capacity"])
    snake.respawn_timer = int(data["respawn_timer"])
    snake.frames_since_food = int(data["frames_since_food"])
    snake._prev_nearest_enemy_dist = float(data["_prev_nearest_enemy_dist"])
    snake._prev_nearest_enemy_id = data["_prev_nearest_enemy_id"]
    snake.is_boosting = bool(data["is_boosting"])
    snake.boost_frames = int(data["boost_frames"])
    snake.last_move_positions = [(int(x), int(y)) for x, y in data["last_move_positions"]]
    snake.pending_trail_pellets = [(int(x), int(y)) for x, y in data["pending_trail_pellets"]]
    snake.auto_respawn = bool(data["auto_respawn"])

    if data.get("has_reward_prev_length", True):
        snake._reward_prev_length = int(data["_reward_prev_length"])
    elif hasattr(snake, "_reward_prev_length"):
        delattr(snake, "_reward_prev_length")

    snake.last_reward_breakdown = copy.deepcopy(data["last_reward_breakdown"])

    if data.get("has_last_death_cause", False):
        snake.last_death_cause = data["last_death_cause"]
    elif hasattr(snake, "last_death_cause"):
        delattr(snake, "last_death_cause")

    if data.get("has_pre_collision_action", False):
        snake._pre_collision_action = data["_pre_collision_action"]
    elif hasattr(snake, "_pre_collision_action"):
        delattr(snake, "_pre_collision_action")

    if data.get("has_pre_collision_epsilon", False):
        snake._pre_collision_epsilon = data["_pre_collision_epsilon"]
    elif hasattr(snake, "_pre_collision_epsilon"):
        delattr(snake, "_pre_collision_epsilon")

    if data.get("has_pre_collision_state", False):
        snake._pre_collision_state = _restore_tensor(data["_pre_collision_state"])
    elif hasattr(snake, "_pre_collision_state"):
        delattr(snake, "_pre_collision_state")

    snake.actor_id = None if data["actor_id"] is None else int(data["actor_id"])
    snake.num_actors = None if data["num_actors"] is None else int(data["num_actors"])
    snake.actor_epsilon_exponent = (
        None if data["actor_epsilon_exponent"] is None else float(data["actor_epsilon_exponent"])
    )
    snake.actor_epsilon = None if data["actor_epsilon"] is None else float(data["actor_epsilon"])
    snake.last_state = _restore_tensor(data["last_state"])
    snake.last_action = data["last_action"]
    snake.last_reward = None if data["last_reward"] is None else float(data["last_reward"])
    snake.last_next_state = _restore_tensor(data["last_next_state"])
    snake.last_next_action_mask = (
        None
        if data["last_next_action_mask"] is None
        else [bool(x) for x in data["last_next_action_mask"]]
    )
    snake.last_next_action_mask_semantics = data["last_next_action_mask_semantics"]
    if "next_action_mask_semantics" in data and data["next_action_mask_semantics"] is not None:
        snake.next_action_mask_semantics = data["next_action_mask_semantics"]
    snake.current_action_mask = (
        None
        if data["current_action_mask"] is None
        else [bool(x) for x in data["current_action_mask"]]
    )
    snake.current_action_mask_semantics = data["current_action_mask_semantics"]
    snake.current_legal_action_mask = (
        None
        if data["current_legal_action_mask"] is None
        else [bool(x) for x in data["current_legal_action_mask"]]
    )
    snake.current_advisory_action_mask = (
        None
        if data["current_advisory_action_mask"] is None
        else [bool(x) for x in data["current_advisory_action_mask"]]
    )
    snake.last_done = bool(data["last_done"])
    snake.last_transition_frame = (
        None if data["last_transition_frame"] is None else int(data["last_transition_frame"])
    )
    snake._total_reward = float(data["_total_reward"])
    snake.current_loss = None if data["current_loss"] is None else float(data["current_loss"])
    snake.current_epsilon = (
        None if data["current_epsilon"] is None else float(data["current_epsilon"])
    )
    snake.boost_exploration_rate = (
        None if data["boost_exploration_rate"] is None else float(data["boost_exploration_rate"])
    )
    snake.danger_exploration_rate = (
        None if data["danger_exploration_rate"] is None else float(data["danger_exploration_rate"])
    )
    snake.action_history = copy.deepcopy(data["action_history"])
    snake.max_action_history = int(data["max_action_history"])
    snake.last_q_values = (
        None if data["last_q_values"] is None else [float(x) for x in data["last_q_values"]]
    )
    snake.record_q_values = bool(data["record_q_values"])
    snake.carry_forward_selection = bool(data["carry_forward_selection"])
    carried = data["_carried_selection"]
    if carried is None:
        snake._carried_selection = None
    else:
        snake._carried_selection = {
            "frame": int(carried["frame"]),
            "state": _restore_tensor(carried["state"]),
            "safe_actions": [int(a) for a in carried["safe_actions"]],
        }
    if hasattr(snake, "_free_space_cache"):
        delattr(snake, "_free_space_cache")
    _restore_veto(getattr(snake, "safety_veto", None), data["safety_veto"])


def snapshot_live_session_state(live: Any) -> dict[str, Any]:
    """Capture complete live GameSession, GameState, FoodManager, AISnake, and policy state."""
    game = live.game
    fm = game.food_manager
    policy = live.policy
    raw_obs = getattr(policy, "_obs", None)
    obs_snap = (
        None if raw_obs is None else {str(k): np.asarray(v).copy() for k, v in raw_obs.items()}
    )
    raw_q = getattr(policy, "_q", None)
    q_snap = None if raw_q is None else np.asarray(raw_q).copy()
    raw_dispatch = getattr(policy, "_dispatch", [])
    dispatch_snap = (
        [int(x) for x in raw_dispatch]
        if isinstance(raw_dispatch, (list, tuple))
        else int(raw_dispatch)
    )
    return {
        "playing": bool(live.playing),
        "speed": int(live.speed),
        "speed_ceiling": bool(live.speed_ceiling),
        "mode": str(live.mode),
        "hero_id": int(live.hero_id),
        "last_error": live.last_error,
        "_release_monitoring": bool(live._release_monitoring),
        "_release_latency_warmup_remaining": int(live._release_latency_warmup_remaining),
        "policy": {
            "total_reward": float(policy.total_reward),
            "_frame": policy._frame,
            "_q": q_snap,
            "_obs": obs_snap,
            "_id_to_row": {int(k): int(v) for k, v in policy._id_to_row.items()},
            "_dispatch": dispatch_snap,
        },
        "watch_opponent_policy": (
            None
            if getattr(live, "watch_opponent_policy", None) is None
            else {
                "epsilon": float(live.watch_opponent_policy.epsilon),
                "training": bool(live.watch_opponent_policy.training),
            }
        ),
        "game": {
            "frame": int(game.frame),
            "num_snakes": int(game.num_snakes),
            "alive_snakes": int(game.alive_snakes),
            "snake_id_counter": int(game.snake_id_counter),
            "best_reward": float(game.best_reward),
            "_train_mode": bool(game._train_mode),
            "_game_width": int(game._game_width),
            "_game_height": int(game._game_height),
            "_effective_initial_food": int(game._effective_initial_food),
            "_effective_max_food": int(game._effective_max_food),
            "frame_collisions": [
                (int(a), None if b is None else int(b), str(c)) for a, b, c in game.frame_collisions
            ],
            "frame_kills": {int(k): [int(v) for v in vs] for k, vs in game.frame_kills.items()},
            "frame_death_causes": {int(k): str(v) for k, v in game.frame_death_causes.items()},
            "frame_ate_food": sorted(int(x) for x in game.frame_ate_food),
            "episode_food_eaten": int(game.episode_food_eaten),
            "episode_deaths": int(game.episode_deaths),
            "episode_kills": int(game.episode_kills),
            "episode_best_length": int(game.episode_best_length),
            "episode_current_reward": float(game.episode_current_reward),
            "episode_best_reward": float(game.episode_best_reward),
            "episode_collision_counts": {
                str(k): int(v) for k, v in game.episode_collision_counts.items()
            },
            "food_manager": {
                "game_width": int(fm.game_width),
                "game_height": int(fm.game_height),
                "max_food": int(fm.max_food),
                "segment_size": int(fm.segment_size),
                "wall_thickness": int(fm.wall_thickness),
                "food": [(int(x), int(y)) for x, y in fm.food],
                "_corpse_positions": sorted((int(x), int(y)) for x, y in fm._corpse_positions),
            },
            "snakes": [_snapshot_snake(s) for s in game.snakes],
        },
    }


def restore_live_session_state(live: Any, data: Mapping[str, Any]) -> None:
    """Restore complete live GameSession, GameState, FoodManager, AISnake, and policy state."""
    live.playing = bool(data["playing"])
    live.speed = int(data["speed"])
    live.speed_ceiling = bool(data["speed_ceiling"])
    live.mode = str(data["mode"])
    live.hero_id = int(data["hero_id"])
    live.last_error = data["last_error"]
    live._release_monitoring = bool(data["_release_monitoring"])
    live._release_latency_warmup_remaining = int(data["_release_latency_warmup_remaining"])

    pol_data = data["policy"]
    policy = live.policy
    policy.total_reward = float(pol_data["total_reward"])
    policy._frame = pol_data["_frame"]
    policy._q = None if pol_data["_q"] is None else np.asarray(pol_data["_q"]).copy()
    policy._obs = (
        None
        if pol_data["_obs"] is None
        else {str(k): np.asarray(v).copy() for k, v in pol_data["_obs"].items()}
    )
    policy._id_to_row = {int(k): int(v) for k, v in pol_data["_id_to_row"].items()}
    raw_disp = pol_data["_dispatch"]
    policy._dispatch = (
        [int(x) for x in raw_disp] if isinstance(raw_disp, (list, tuple)) else int(raw_disp)
    )

    opp_pol_data = data.get("watch_opponent_policy")
    if opp_pol_data is not None and getattr(live, "watch_opponent_policy", None) is not None:
        live.watch_opponent_policy.epsilon = float(opp_pol_data["epsilon"])
        live.watch_opponent_policy.training = bool(opp_pol_data["training"])

    gdata = data["game"]
    game = live.game
    game.frame = int(gdata["frame"])
    game.num_snakes = int(gdata["num_snakes"])
    game.alive_snakes = int(gdata["alive_snakes"])
    game.snake_id_counter = int(gdata["snake_id_counter"])
    game.best_reward = float(gdata["best_reward"])
    game._train_mode = bool(gdata["_train_mode"])
    game._game_width = int(gdata["_game_width"])
    game._game_height = int(gdata["_game_height"])
    game._effective_initial_food = int(gdata["_effective_initial_food"])
    game._effective_max_food = int(gdata["_effective_max_food"])
    game.frame_collisions = [
        (int(a), None if b is None else int(b), str(c)) for a, b, c in gdata["frame_collisions"]
    ]
    game.frame_kills = {int(k): [int(v) for v in vs] for k, vs in gdata["frame_kills"].items()}
    game.frame_death_causes = {int(k): str(v) for k, v in gdata["frame_death_causes"].items()}
    game.frame_ate_food = {int(x) for x in gdata["frame_ate_food"]}
    game.episode_food_eaten = int(gdata["episode_food_eaten"])
    game.episode_deaths = int(gdata["episode_deaths"])
    game.episode_kills = int(gdata["episode_kills"])
    game.episode_best_length = int(gdata["episode_best_length"])
    game.episode_current_reward = float(gdata["episode_current_reward"])
    game.episode_best_reward = float(gdata["episode_best_reward"])
    game.episode_collision_counts = {
        str(k): int(v) for k, v in gdata["episode_collision_counts"].items()
    }

    fm_data = gdata["food_manager"]
    fm = game.food_manager
    fm.game_width = int(fm_data["game_width"])
    fm.game_height = int(fm_data["game_height"])
    fm.max_food = int(fm_data["max_food"])
    fm.segment_size = int(fm_data["segment_size"])
    fm.wall_thickness = int(fm_data["wall_thickness"])
    fm.food = [(int(x), int(y)) for x, y in fm_data["food"]]
    fm._corpse_positions = {(int(x), int(y)) for x, y in fm_data["_corpse_positions"]}

    snakes_data = gdata["snakes"]
    if len(game.snakes) != len(snakes_data):
        raise ValueError(f"snake count mismatch: {len(game.snakes)} != {len(snakes_data)}")
    for snake, sdata in zip(game.snakes, snakes_data):
        _restore_snake(snake, sdata)
        if (
            getattr(snake, "safety_veto", None) is not None
            and getattr(live, "safety_veto", None) is not None
        ):
            live.safety_veto.vetoes[int(snake.id)] = snake.safety_veto

    clear_geometry_cache()


def snapshot_run_session_state(session: Any) -> dict[str, Any]:
    """Capture the complete RunSession (live GameSession + CapturedWorld + RNGs + counters)."""
    world = session.world
    return {
        "rng": snapshot_rng_state(world.sim),
        "sim": snapshot_sim_state(world.sim),
        "runtime": snapshot_runtime_state(world.runtime),
        "opponents": snapshot_opponents_state(world.opponents),
        "live": snapshot_live_session_state(session.live),
        "run_session": {
            "arm": str(session.arm),
            "specs": [(str(a), str(b)) for a, b in session.specs],
            "life": int(session.life),
            "decisions": int(session.decisions),
            "boost_decisions": int(session.boost_decisions),
            "decisions_delta": int(session.decisions_delta),
            "boost_decisions_delta": int(session.boost_decisions_delta),
            "bare_action": session.bare_action,
            "executed_action": session.executed_action,
            "hero_death": bool(session.hero_death),
            "death_length": session.death_length,
            "last_update_ms": float(session.last_update_ms),
            "shadow_upkeep_ms": float(session.shadow_upkeep_ms),
            "first_result_latency_ms": session.first_result_latency_ms,
            "_shadow_cpu_seconds": float(session._shadow_cpu_seconds),
            "_latency_sequence": int(session._latency_sequence),
            "_live_lives": [int(x) for x in session._live_lives],
            "_grid_lives": [int(x) for x in session._grid_lives],
            "_prev_live": [bool(x) for x in session._prev_live],
            "_prev_grid": [bool(x) for x in session._prev_grid],
            "_prefix_initial": copy.deepcopy(session._prefix_initial),
            "_prefix_steps": copy.deepcopy(session._prefix_steps),
            "_prefix_intervened": bool(session._prefix_intervened),
            "_prefix_enabled": bool(session._prefix_enabled),
            "parity_status": str(getattr(session, "parity_status", "exact")),
            "identity_status": copy.deepcopy(getattr(session, "identity_status", {})),
        },
    }


def verify_session_parity(session: Any) -> None:
    """Assert physical, carry, and v8 parity between live GameState and GridBatchSim mirror."""
    from research.t34_shadow.identity import _carry, _physical, _value_digest, _veto

    pairs = {
        "physical": (
            _physical(
                session.live.game,
                session.world,
                session.specs,
                session._live_lives,
                live=True,
                rng=random.getstate(),
            ),
            _physical(None, session.world, session.specs, session._grid_lives, live=False),
        ),
        "carry": (
            _carry(session.live.game, session.world, session.specs, live=True),
            _carry(None, session.world, session.specs, live=False),
        ),
        "v8": (
            _veto(session.live.game, session.world, session.specs, live=True),
            _veto(None, session.world, session.specs, live=False),
        ),
    }
    for name, (actual, expected) in pairs.items():
        if _value_digest(actual) != _value_digest(expected):
            raise RuntimeError(f"Watch/Grid {name} identity failed at frame {session.frame}")


def restore_run_session_state(session: Any, data: Mapping[str, Any]) -> None:
    """Restore the complete RunSession and verify live/GridBatchSim parity."""
    world = session.world
    restore_sim_state(world.sim, data["sim"])
    restore_runtime_state(world.runtime, data["runtime"])
    restore_opponents_state(world.opponents, data["opponents"])
    restore_live_session_state(session.live, data["live"])

    rs = data["run_session"]
    session.arm = str(rs["arm"])
    session.specs = [(str(a), str(b)) for a, b in rs["specs"]]
    session.life = int(rs["life"])
    session.decisions = int(rs["decisions"])
    session.boost_decisions = int(rs["boost_decisions"])
    session.decisions_delta = int(rs["decisions_delta"])
    session.boost_decisions_delta = int(rs["boost_decisions_delta"])
    session.bare_action = rs["bare_action"]
    session.executed_action = rs["executed_action"]
    session.hero_death = bool(rs["hero_death"])
    session.death_length = rs["death_length"]
    session.last_update_ms = float(rs["last_update_ms"])
    session.shadow_upkeep_ms = float(rs["shadow_upkeep_ms"])
    session.first_result_latency_ms = rs["first_result_latency_ms"]
    session._shadow_cpu_seconds = float(rs["_shadow_cpu_seconds"])
    session._latency_sequence = int(rs["_latency_sequence"])
    session._live_lives = [int(x) for x in rs["_live_lives"]]
    session._grid_lives = [int(x) for x in rs["_grid_lives"]]
    session._prev_live = [bool(x) for x in rs["_prev_live"]]
    session._prev_grid = [bool(x) for x in rs["_prev_grid"]]
    session._prefix_initial = copy.deepcopy(rs["_prefix_initial"])
    session._prefix_steps = copy.deepcopy(rs["_prefix_steps"])
    session._prefix_intervened = bool(rs["_prefix_intervened"])
    session._prefix_enabled = bool(rs["_prefix_enabled"])
    session.parity_status = str(rs.get("parity_status", "exact"))
    session.identity_status = copy.deepcopy(rs.get("identity_status", {}))

    restore_rng_state(world.sim, data["rng"])
    verify_session_parity(session)


def snapshot_trigger_state(trigger: DenseTrigger | None) -> dict[str, Any] | None:
    """Capture DenseTrigger state."""
    if trigger is None:
        return None
    return {
        "min_length": int(trigger.min_length),
        "min_frame": int(trigger.min_frame),
        "interval": int(trigger.interval),
        "first_decision": (None if trigger.first_decision is None else int(trigger.first_decision)),
    }


def restore_trigger_state(data: Mapping[str, Any] | None) -> DenseTrigger | None:
    """Restore DenseTrigger state."""
    if data is None:
        return None
    trigger = DenseTrigger(
        min_length=int(data["min_length"]),
        min_frame=int(data.get("min_frame", 0)),
        interval=int(data["interval"]),
    )
    trigger.first_decision = (
        None if data.get("first_decision") is None else int(data["first_decision"])
    )
    return trigger


def _canonicalize_for_digest(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        }
    if isinstance(value, dict):
        return {
            str(k): _canonicalize_for_digest(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            if "seconds" not in str(k)
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_for_digest(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_canonicalize_for_digest(v) for v in value)
    if hasattr(value, "__dict__"):
        return _canonicalize_for_digest(dict(value.__dict__))
    return value


def state_digest(session: Any, arm_state: Mapping[str, Any] | None = None) -> str:
    """Return a deterministic SHA-256 digest of the complete paired-world frame state.

    Includes live GameState, GridBatchSim mirror, all four RNG streams, Vector61Runtime
    carry/memory, opponent v8 vetoes, DenseTrigger state, and deterministic arm counters
    (excluding wall-clock timing floats).
    """
    snap = snapshot_run_session_state(session)
    # Remove wall-clock latency fields from RunSession snapshot before hashing.
    rs = dict(snap["run_session"])
    for timing_key in (
        "last_update_ms",
        "shadow_upkeep_ms",
        "first_result_latency_ms",
        "_shadow_cpu_seconds",
        "_latency_sequence",
    ):
        rs.pop(timing_key, None)
    snap["run_session"] = rs

    arm_clean = None
    if arm_state is not None:
        arm_clean = {
            k: v
            for k, v in arm_state.items()
            if k
            not in {
                "main_seconds",
                "search_seconds",
                "search_seconds_lt600",
                "search_seconds_600_999",
                "search_seconds_ge1000",
            }
        }
    canonical = _canonicalize_for_digest({"session": snap, "arm_state": arm_clean})
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
