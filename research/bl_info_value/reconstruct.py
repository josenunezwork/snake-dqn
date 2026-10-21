"""Exact E3 T3-bank replay and D300 mass-return reconstruction.

This is deliberately a consumer of the T3 collector seam.  It never creates a
new evaluator, substitutes an endpoint label, or records an open-loop action
tape after a counterfactual first action.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from research.bl_info_value.io import sha256


class ReconstructionError(RuntimeError):
    """A saved T3 row could not be reproduced exactly."""


_OBS_FIELDS = ("local", "global", "scalars", "mask")


def _head(sim: Any, slot: int) -> tuple[int, int]:
    ring = int(sim.head_ptr[0, slot]) % int(sim.cap)
    cell = sim.bodies[0, slot, ring]
    return int(cell[0]), int(cell[1])


def _history_state(sim: Any, actions: np.ndarray | None = None) -> np.ndarray:
    """Encode one pre-decision six-seat state in fixed physical units.

    Coordinates are cell coordinates divided by the fixed 145-by-83 board,
    headings are cardinal unit vectors, logical mass is divided by 1000, and
    actions by five.  The final history channel is a record-valid flag.
    """
    from src.simd_env.batch_sim import CARDINAL

    out = np.zeros(43, dtype=np.float32)
    for slot in range(6):
        base = 7 * slot
        alive = bool(sim.alive[0, slot])
        if alive:
            x, y = _head(sim, slot)
            out[base] = x / float(sim.grid_w)
            out[base + 1] = y / float(sim.grid_h)
            dx, dy = CARDINAL[int(sim.direction[0, slot])]
            out[base + 2] = float(dx)
            out[base + 3] = float(dy)
            out[base + 4] = float(sim.length[0, slot]) / 1000.0
            out[base + 5] = 1.0
        if actions is not None:
            out[base + 6] = float(int(actions[0, slot])) / 5.0
    out[-1] = 1.0
    return out


def _history_window(rows: deque[np.ndarray]) -> np.ndarray:
    result = np.zeros((128, 43), dtype=np.float32)
    values = list(rows)[-128:]
    if values:
        result[-len(values) :] = np.stack(values)
    return result


def _geometry(sim: Any) -> np.ndarray:
    """Full board occupancy by owner plus unclipped tail-relative release order."""
    owner = np.asarray(sim._owner[0], dtype=np.int64)
    slot = np.asarray(sim._slot[0], dtype=np.int64)
    result = np.zeros((7, int(sim.grid_h), int(sim.grid_w)), dtype=np.float32)
    live = np.asarray(sim.alive[0], dtype=bool)
    for seat in range(6):
        occupied = (owner == seat) & live[seat]
        result[seat, occupied] = 1.0
        if occupied.any():
            offset = (int(sim.head_ptr[0, seat]) - slot[occupied]) % int(sim.cap)
            # This is tail-relative release order, rather than the B8 uint8 TTL.
            result[6, occupied] = (int(sim.seg_count[0, seat]) - offset) / 1000.0
    return result


def _exact(name: str, expected: np.ndarray, actual: np.ndarray, decision: int) -> None:
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        raise ReconstructionError(
            f"saved {name} shape/dtype mismatch at decision {decision}: "
            f"{expected.shape}/{expected.dtype} != {actual.shape}/{actual.dtype}"
        )
    if expected.tobytes(order="C") != actual.tobytes(order="C"):
        raise ReconstructionError(f"saved {name} identity mismatch at decision {decision}")


def _check_row(
    row: dict[str, np.ndarray], obs: dict[str, np.ndarray], action: int, decision: int, frame: int
) -> None:
    if "obs_hashes" in row:
        from research.bl_info_value.compact import array_binding

        for field in _OBS_FIELDS:
            if array_binding(np.asarray(obs[field][0])) != row["obs_hashes"].get(field):
                raise ReconstructionError(f"saved {field} identity mismatch at decision {decision}")
        if int(row["action"]) != action or int(row["frame"]) != frame:
            raise ReconstructionError(f"saved action/frame mismatch at decision {decision}")
        return
    for field in _OBS_FIELDS:
        if field not in row:
            raise ReconstructionError(f"source row lacks {field}")
        _exact(field, np.asarray(row[field]), np.asarray(obs[field][0]), decision)
    if int(np.asarray(row["action"]).item()) != action:
        raise ReconstructionError(f"saved action mismatch at decision {decision}")
    if "frame" in row and int(np.asarray(row["frame"]).item()) != frame:
        raise ReconstructionError(f"saved frame mismatch at decision {decision}")


def _mass_continuation(
    world: Any, forced: int, depth: int
) -> tuple[float, list[float], int | None]:
    """Run one reactive continuation, summing post-transition living mass."""
    from research.t3_rollout_labels_20261015.bank_engine import choose_and_step, step_prepared

    branch = world.clone()
    values: list[float] = []
    step_prepared(branch, forced)
    values.append(float(branch.sim.length[0, 0]) if bool(branch.sim.alive[0, 0]) else 0.0)
    death = 0 if not bool(branch.sim.alive[0, 0]) else None
    for offset in range(1, depth):
        if not bool(branch.sim.alive[0, 0]):
            values.append(0.0)
            continue
        choose_and_step(branch, capture_obs=False)
        values.append(float(branch.sim.length[0, 0]) if bool(branch.sim.alive[0, 0]) else 0.0)
        if death is None and not bool(branch.sim.alive[0, 0]):
            death = offset
    return float(sum(values) / depth), values, death


def reconstruct_world(task: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Reconstruct selected T3 roots and their exact reactive D300 targets.

    ``task`` supplies the persisted roster plan, explicit relocated B8 checkpoint,
    raw per-world ``source_npz`` and source-row ``indices``.  The return is
    ``(arrays, metrics)`` so timing never contaminates the deterministic payload.
    """
    from research.t3_rollout_labels_20261015.bank_engine import build_world, choose_and_step

    required = {"arm", "plan", "checkpoint", "indices", "depth", "split", "world"}
    missing = sorted(required.difference(task))
    if missing:
        raise ValueError(f"reconstruction task lacks {missing}")
    depth = int(task["depth"])
    if depth < 1 or depth > 300:
        raise ValueError("depth must be in [1, 300]")
    indices = [int(value) for value in task["indices"]]
    if len(indices) != len(set(indices)):
        raise ValueError("source indices must be unique")
    expected_count = task.get("selected_indices_count", len(indices))
    if int(expected_count) != len(indices):
        raise ReconstructionError("selected source-index count differs from task binding")
    if "row_bindings" in task:
        rows = {int(row["decision"]): row for row in task["row_bindings"]}
    else:
        source_path = Path(task["source_npz"])
        expected_source_sha = task.get("source_sha256")
        if expected_source_sha is not None and sha256(source_path) != str(expected_source_sha):
            raise ReconstructionError("source_npz SHA-256 differs from the task binding")
        with np.load(source_path, allow_pickle=False) as source:
            need = {"decision", "action", *_OBS_FIELDS}
            absent = need.difference(source.files)
            if absent:
                raise ReconstructionError(f"source shard lacks {sorted(absent)}")
            if any(index < 0 or index >= len(source["decision"]) for index in indices):
                raise IndexError("source row index outside shard")
            rows = {
                int(source["decision"][index]): {
                    name: np.array(source[name][index], copy=True) for name in source.files
                }
                for index in indices
            }
    if len(rows) != len(indices):
        raise ReconstructionError("selected source rows share a decision")

    started = time.perf_counter()
    world = build_world(
        str(task["arm"]), dict(task["plan"]), hero_checkpoint=Path(task["checkpoint"])
    )
    selected: dict[int, dict[str, Any]] = {}
    history: deque[np.ndarray] = deque(maxlen=128)
    active_controls: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    prepared_history: dict[int, np.ndarray] = {}
    max_decision = max(rows) if rows else -1
    original_prepare = world.runtime.prepare
    decision = 0
    last_progress = started

    def capture(sim: Any) -> None:
        original_prepare(sim)
        prepared_history[decision] = _history_state(sim)
        if decision not in rows:
            return
        # clone only after the original runtime prepared its carried vector state.
        selected[decision] = {
            "world": world.clone(),
            "history": _history_window(history),
            "geometry": _geometry(sim),
            "frame": int(sim.frame[0]),
            "resolved": np.asarray(sim.get_resolved_action_mask()[0, 0], dtype=bool).copy(),
        }

    world.runtime.prepare = capture
    try:
        # Continue through the latest selected root's D300 prefix.  The source T3
        # bank is terminal-hero, so death ends every outstanding comparison.
        while bool(world.sim.alive[0, 0]) and (decision <= max_decision or active_controls):
            now = time.perf_counter()
            if now - last_progress >= 60.0:
                print(
                    f"E3 reconstruct key={task.get('key', '')} decision={decision} "
                    f"roots={len(records)}/{len(indices)} elapsed_s={now - started:.1f}",
                    flush=True,
                )
                last_progress = now
            actions, _advisory, obs = choose_and_step(world, capture_obs=True)
            current_action = int(actions[0, 0])
            after_mass = float(world.sim.length[0, 0]) if bool(world.sim.alive[0, 0]) else 0.0
            for control in active_controls:
                offset = decision - control["decision"]
                if 0 <= offset < len(control["prefix"]):
                    expected = control["prefix"][offset]
                    if after_mass != expected:
                        raise ReconstructionError(
                            "B8 control mass mismatch at decision "
                            f"{decision}: {after_mass} != {expected}"
                        )
            active_controls = [c for c in active_controls if decision - c["decision"] + 1 < depth]
            before = prepared_history.pop(decision, _history_state(world.sim))
            before[6::7] = np.asarray(actions[0], dtype=np.float32) / 5.0
            history.append(before)
            if decision in rows:
                if obs is None or decision not in selected:
                    raise ReconstructionError(f"capture missing at source decision {decision}")
                _check_row(
                    rows[decision], obs, current_action, decision, int(selected[decision]["frame"])
                )
                snap = selected.pop(decision)
                legal = np.flatnonzero(snap["resolved"]).astype(np.int64)
                if not len(legal) or current_action not in legal:
                    raise ReconstructionError(
                        f"B8 action is not resolved-legal at decision {decision}"
                    )
                returns = np.zeros(6, dtype=np.float32)
                prefix = None
                death_offset = None
                for action in legal:
                    value, masses, dead = _mass_continuation(snap["world"], int(action), depth)
                    returns[int(action)] = value
                    if int(action) == current_action:
                        prefix, death_offset = masses, dead
                assert prefix is not None
                if prefix[0] != after_mass:
                    raise ReconstructionError(
                        "B8 forced-first mass mismatch at decision "
                        f"{decision}: {after_mass} != {prefix[0]}"
                    )
                if "outcome_death_frame" in rows[decision] and death_offset is not None:
                    saved_death = int(
                        np.asarray(rows[decision]["outcome_death_frame"])[current_action]
                    )
                    if saved_death >= 0 and saved_death != decision + death_offset:
                        raise ReconstructionError(
                            f"recorded B8 death mismatch at decision {decision}"
                        )
                active_controls.append({"decision": decision, "prefix": prefix})
                records.append(
                    {
                        "local": np.asarray(obs["local"][0], dtype=np.uint8).copy(),
                        "global": np.asarray(obs["global"][0], dtype=np.uint8).copy(),
                        "scalars": np.asarray(obs["scalars"][0], dtype=np.float32).copy(),
                        "geometry": snap["geometry"],
                        "history": snap["history"],
                        "legal": snap["resolved"],
                        "returns": returns,
                        "action": current_action,
                        "decisive": bool(np.asarray(rows[decision].get("decisive", False)).item()),
                        "decision": decision,
                        "frame": snap["frame"],
                    }
                )
            decision += 1
    finally:
        world.runtime.prepare = original_prepare
    if len(records) != len(indices):
        raise ReconstructionError(
            f"replayed {len(records)} selected roots, expected {len(indices)}"
        )
    task_death = task.get("death_decision")
    if task_death is not None:
        observed_death = decision - 1 if not bool(world.sim.alive[0, 0]) else None
        if observed_death != int(task_death):
            raise ReconstructionError(
                f"recorded world death mismatch: {observed_death} != {task_death}"
            )
    # A terminal-hero replay can end with active controls.  Their unvisited tail
    # is valid only when every B8 branch has the same zero-dead remainder.
    for control in active_controls:
        offset = decision - control["decision"]
        if any(value != 0.0 for value in control["prefix"][max(offset, 0) :]):
            raise ReconstructionError("B8 control survived after the recorded terminal death")
    records.sort(key=lambda item: int(item["decision"]))
    arrays = {
        name: np.stack([item[name] for item in records])
        for name in ("local", "global", "scalars", "geometry", "history", "legal", "returns")
    }
    arrays.update(
        action=np.asarray([item["action"] for item in records], dtype=np.int64),
        decisive=np.asarray([item["decisive"] for item in records], dtype=bool),
        world=np.full(len(records), int(task["world"]), dtype=np.int64),
        split=np.full(len(records), int(task["split"]), dtype=np.int64),
        decision=np.asarray([item["decision"] for item in records], dtype=np.int64),
        frame=np.asarray([item["frame"] for item in records], dtype=np.int64),
    )
    return arrays, {
        "key": str(task.get("key", "")),
        "roots": len(records),
        "depth": depth,
        "elapsed_seconds": time.perf_counter() - started,
        "replayed_decisions": decision,
    }
