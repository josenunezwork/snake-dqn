"""Bridge-free matched-state check for the frozen tick-14 FEATURE divergence.

JUDGMENT_IDENTITY_PATH_20261024 section 3 / operator step 2.  Every state comes
from the device tape of the excluded ``local-proof/v1`` frozen world (device
engine arm R on CPU torch, the arm of the original observation).  No RNG bridge
and no CPU-played trajectory is used:

(a) at each seam t in 4..14 the device seam is loaded into a fresh CPU world
    (``DeviceSim.copy_env_to_reference``), the CPU vector61 runtime is given the
    device's exact pre-prepare carry, and we compare
    (a1) stateless vector61 features (no memory update) for alive opponents,
    (a2) the full runtime state after ``prepare`` (selection state/mask, carry,
         trend memory), and
    (a3) the v8 opponent decision given identical Q, mask and seam;
(b) from the device's post-prepare carry and the matched seam, both sides
    step with the device's actions and ``observe_step``; the post-step state
    and the new carry are compared byte-for-byte.

A field whose dtypes cannot be mapped losslessly is NOT_COMPARABLE.  Arm I's
native routes have no CPU path at the pin, so arm I is UNCOVERED_ON_I.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.gpu_sim_spike import identity_device as device_mod
from research.gpu_sim_spike import identity_tape as tape

SCHEMA = "gpu-identity-frozen14-matched-state/v1"
SEAMS = tuple(range(4, 15))
WORLD_INDEX = 1  # local-proof/v1 frozen world (world-01-frozen)
REPLICAS = 2  # as in the original Mac probe
CARRY = (
    "prev_dist",
    "prev_id",
    "_carry_state",
    "_carry_mask",
    "_carry_valid",
    "_alive_post",
    "sel_state",
    "sel_mask",
    "sel_ready",
    "sel_fresh",
)
OBSERVE_FIELDS = (
    "prev_dist",
    "prev_id",
    "_carry_state",
    "_carry_mask",
    "_carry_valid",
    "_alive_post",
)
MEMORY = {"prev_dist", "prev_id"}
CHECKS = ("load", "a1", "a2", "a3", "b_post_step_state", "b", "b_prime")


def _device_carry(runtime: Any, env: int = 0) -> dict[str, np.ndarray]:
    return {name: getattr(runtime, name)[env].detach().cpu().numpy().copy() for name in CARRY}


def _cpu_get(runtime: Any, name: str) -> np.ndarray:
    return getattr(runtime.memory if name in MEMORY else runtime, name)


def _cpu_carry(runtime: Any) -> dict[str, np.ndarray]:
    return {name: np.asarray(_cpu_get(runtime, name))[0].copy() for name in CARRY}


def _set_cpu_carry(runtime: Any, carry: dict[str, np.ndarray]) -> list[str]:
    """Write device carry into the CPU runtime; returns NOT_COMPARABLE fields."""
    refused = []
    for name, value in carry.items():
        target = _cpu_get(runtime, name)
        if target.dtype != value.dtype and not (
            target.dtype.kind in "biu" and value.dtype.kind in "biu"
        ):
            refused.append(name)
            continue
        target[0] = value.astype(target.dtype)
    return refused


def compare_fields(cpu: dict[str, np.ndarray], dev: dict[str, np.ndarray]) -> dict[str, Any]:
    """Exact per-field comparison; integer/bool kinds are compared by value."""
    out: dict[str, Any] = {}
    for name in sorted(dev):
        a, b = np.asarray(cpu[name]), np.asarray(dev[name])
        if a.shape != b.shape:
            out[name] = {"status": "NOT_COMPARABLE", "reason": f"shape {a.shape} vs {b.shape}"}
            continue
        if a.dtype != b.dtype:
            if a.dtype.kind in "biu" and b.dtype.kind in "biu":
                equal = a.astype(np.int64) == b.astype(np.int64)
            else:
                out[name] = {"status": "NOT_COMPARABLE", "reason": f"{a.dtype} vs {b.dtype}"}
                continue
        else:
            bits = (a.view(np.uint8) == b.view(np.uint8)).reshape(a.shape + (-1,)).all(-1)
            equal = bits if a.dtype.kind == "f" else (a == b)
        if bool(np.all(equal)):
            out[name] = {"status": "EQUAL"}
            continue
        rows = np.argwhere(~equal)
        slots = sorted({int(row[0]) for row in rows}) if a.ndim else []
        detail: dict[str, Any] = {"status": "DIFFERENT", "slots": slots}
        if a.dtype.kind == "f":
            for slot in slots[:6]:
                columns = np.flatnonzero(~np.asarray(equal[slot]).reshape(-1)).tolist()
                detail.setdefault("columns", {})[str(slot)] = columns[:61]
                detail.setdefault("max_abs", {})[str(slot)] = float(
                    np.nanmax(np.abs(a[slot].astype(np.float64) - b[slot].astype(np.float64)))
                )
        out[name] = detail
    return out


def device_trajectory(row: dict[str, Any], checkpoint: Path, seams: int) -> list[dict[str, Any]]:
    """Device arm R closed loop with full seam/carry snapshots (no CPU involved)."""
    import torch

    world = tape.build_cpu_world(row, checkpoint)
    snaps: list[dict[str, Any]] = []
    with device_mod._arm("R"), torch.inference_mode():
        sim, policy = device_mod.build_case(world, "cpu", "R", REPLICAS)
        capture = device_mod.Capture(policy)
        runtime = policy.vector_runtime
        try:
            for t in range(seams):
                sim.prepare()
                snap: dict[str, Any] = {
                    "seam": sim.clone(),
                    "runtime_pre": copy.deepcopy(runtime),
                    "carry_pre": _device_carry(runtime),
                    "vector_prepared_pre": bool(policy.vector_prepared),
                }
                capture.reset()
                _q, _legal, hero = policy.evaluate(sim)
                actions = policy.actions(sim, hero)
                opponents = capture.opponents(int(sim.S))
                opponents.pop("_tensors")
                snap.update(
                    carry_post_prepare=_device_carry(runtime),
                    actions=actions[0].detach().cpu().numpy().astype(np.int64),
                    opponents=opponents,
                )
                sim.step(actions)
                snap["post_step"] = sim.clone()
                policy.after_step(sim)
                snap["carry_post_observe"] = _device_carry(runtime)
                snaps.append(snap)
        finally:
            capture.close()
    return snaps


def _events(snaps: list[dict[str, Any]], t: int) -> dict[str, Any]:
    """What happened between seam t-1 and seam t (device trajectory)."""
    prev = tape.device_seam_state(snaps[t - 1]["seam"], 0)
    cur = tape.device_seam_state(snaps[t]["seam"], 0)
    alive = tape.SNAKE_FIELDS.index("alive")
    ate = tape.SNAKE_FIELDS.index("_last_food_ate")
    return {
        "food_eaten_by": [int(s) for s in np.flatnonzero(cur["snake"][:, ate])],
        "respawned": [
            int(s)
            for s in np.flatnonzero((prev["snake"][:, alive] == 0) & (cur["snake"][:, alive] == 1))
        ],
        "died": [
            int(s)
            for s in np.flatnonzero((prev["snake"][:, alive] == 1) & (cur["snake"][:, alive] == 0))
        ],
        "food_count": [int(len(prev["food"])), int(len(cur["food"]))],
    }


def matched_seam(base_world: Any, snap: dict[str, Any], kinds: list[str]) -> dict[str, Any]:
    """Checks (a1)-(a3) and (b) at one seam, bridge-free."""
    import torch

    from research.t34_shadow.shadow import clone_world_exact

    world = clone_world_exact(base_world)  # fresh CPU world/runtime/v8 hooks
    sim, runtime = world.sim, world.runtime
    snap["seam"].copy_env_to_reference(sim, 0)
    loaded = tape.cpu_seam_state(sim)
    seam_map = tape.device_seam_state(snap["seam"], 0)
    load_ok = all(
        np.array_equal(loaded[k], seam_map[k]) for k in ("snake", "body", "frame", "food")
    )
    refused = _set_cpu_carry(runtime, snap["carry_pre"])
    alive = sim.alive[0].astype(bool)
    rows = np.asarray(
        [[0, s] for s in range(1, int(sim.S)) if alive[s] and runtime.controlled[0, s]],
        dtype=np.int64,
    )
    # (a1) stateless features: same memory, no update, both sides.
    memory = copy.deepcopy(runtime.memory)
    cpu_feat = np.asarray(
        runtime.featurizer.featurize(sim, rows, memory, update_memory=False), dtype=np.float32
    )
    dev_rt = copy.deepcopy(snap["runtime_pre"])
    with torch.inference_mode():
        dev_feat = (
            dev_rt.features(
                snap["seam"], torch.as_tensor(rows, dtype=torch.long), update_memory=False
            )
            .detach()
            .cpu()
            .numpy()
        )
    a1 = compare_fields(
        {"features": cpu_feat[np.argsort(rows[:, 1])]},
        {"features": dev_feat[np.argsort(rows[:, 1])]},
    )
    a1["features"]["row_slots"] = rows[:, 1].tolist()
    # (a2) prepare from the identical pre-prepare carry.
    if snap["vector_prepared_pre"]:
        a2: dict[str, Any] = {"status": "NOT_APPLICABLE", "reason": "seam pre-prepared at build"}
        _set_cpu_carry(runtime, snap["carry_post_prepare"])
    else:
        runtime.prepare(sim)
        a2 = compare_fields(_cpu_carry(runtime), snap["carry_post_prepare"])
    # (a3) v8 opponent decision with the device's Q at the same seam and carry.
    _set_cpu_carry(runtime, snap["carry_post_prepare"])
    a3 = {}
    for slot in rows[:, 1].tolist():
        if kinds[slot] != "network" or not snap["opponents"]["opp_recorded"][slot]:
            continue
        policy = world.opponents[slot - 1]
        q_dev = torch.as_tensor(snap["opponents"]["opp_q"][slot][None], dtype=torch.float32)
        policy.q_values = lambda _states, _q=q_dev: _q
        try:
            cpu_action = int(policy.actions(None, sim, np.asarray([[0, slot]]))[0])
        finally:
            del policy.q_values
        a3[str(slot)] = {
            "cpu": cpu_action,
            "device": int(snap["actions"][slot]),
            "status": "EQUAL" if cpu_action == int(snap["actions"][slot]) else "DIFFERENT",
            "note": "fresh CPU v8 live hook (stateless device v8_decide)",
        }
    # (b) transition from the device's post-prepare carry and matched seam.
    _set_cpu_carry(runtime, snap["carry_post_prepare"])
    sim.step(snap["actions"].reshape(1, -1))
    post_cpu = tape.cpu_seam_state(sim)
    post_dev = tape.device_seam_state(snap["post_step"], 0)
    post_equal = {
        k: bool(np.array_equal(post_cpu[k], post_dev[k]))
        for k in ("snake", "body", "frame", "food")
    }
    food_detail = {
        "cpu_post_step_food_count": int(len(post_cpu["food"])),
        "device_post_step_food_count": int(len(post_dev["food"])),
    }
    runtime.observe_step(sim)
    dev_obs = {k: v for k, v in snap["carry_post_observe"].items() if k in OBSERVE_FIELDS}
    b = compare_fields(
        {k: v for k, v in _cpu_carry(runtime).items() if k in OBSERVE_FIELDS}, dev_obs
    )
    # (b') same, but observe_step reads the device's own post-step state, which
    # isolates the vector61 carry arithmetic from post-step mechanics.
    world2 = clone_world_exact(base_world)
    snap["post_step"].copy_env_to_reference(world2.sim, 0)
    loaded_post = tape.cpu_seam_state(world2.sim)
    post_loaded = all(
        np.array_equal(loaded_post[k], post_dev[k]) for k in ("snake", "body", "frame", "food")
    )
    _set_cpu_carry(world2.runtime, snap["carry_post_prepare"])
    world2.runtime.observe_step(world2.sim)
    b_matched = compare_fields(
        {k: v for k, v in _cpu_carry(world2.runtime).items() if k in OBSERVE_FIELDS}, dev_obs
    )
    return {
        "seam_loaded_exactly": load_ok,
        "carry_not_comparable": refused,
        "a1_stateless_features": a1,
        "a2_prepare_from_matched_carry": a2,
        "a3_v8_decision_matched_q": a3,
        "b_post_step_state_equal": post_equal,
        "b_post_step_food": food_detail,
        "b_observe_carry": b,
        "b_prime_post_step_loaded_exactly": post_loaded,
        "b_prime_observe_carry_on_device_post_step": b_matched,
    }


def _different(block: Any) -> bool:
    if isinstance(block, dict):
        if block.get("status") in ("DIFFERENT", "NOT_COMPARABLE"):
            return True
        return any(_different(value) for value in block.values() if isinstance(value, dict))
    return False


def run(checkpoint: Path) -> dict[str, Any]:
    from research.gpu_sim_spike import identity_compare as compare

    started = time.monotonic()
    row = tape.world_rows("local")[WORLD_INDEX]
    if row["mix"] != "frozen":
        raise RuntimeError("local-proof world 1 is not the frozen world")
    kinds = compare.opponent_kinds(row)
    snaps = device_trajectory(row, checkpoint, max(SEAMS) + 1)
    base = tape.build_cpu_world(row, checkpoint)
    seams = {}
    first: dict[str, int | None] = {name: None for name in CHECKS}
    for t in SEAMS:
        result = matched_seam(base, snaps[t], kinds)
        result["events_since_previous_seam"] = _events(snaps, t)
        result["events_in_this_step"] = _events(snaps, t + 1) if t + 1 < len(snaps) else None
        seams[str(t)] = result
        flags = {
            "load": not result["seam_loaded_exactly"] or bool(result["carry_not_comparable"]),
            "a1": _different(result["a1_stateless_features"]),
            "a2": _different(result["a2_prepare_from_matched_carry"]),
            "a3": _different(result["a3_v8_decision_matched_q"]),
            "b_post_step_state": not all(result["b_post_step_state_equal"].values()),
            "b": _different(result["b_observe_carry"]),
            "b_prime": not result["b_prime_post_step_loaded_exactly"]
            or _different(result["b_prime_observe_carry_on_device_post_step"]),
        }
        result["mismatch_flags"] = flags
        for name, flag in flags.items():
            if flag and first[name] is None:
                first[name] = t
    return {
        "schema": SCHEMA,
        "world": {
            "namespace": tape.MODES["local"]["namespace"],
            "unit": "world-01-frozen",
            "mix": row["mix"],
            "scored": False,
        },
        "arm": "R (device engine on CPU torch; the arm of the original observation)",
        "arm_I": "UNCOVERED_ON_I",
        "seams": list(SEAMS),
        "first_mismatch_seam_by_check": first,
        "results": seams,
        "wall_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    args = parser.parse_args()
    if args.raw_output.exists():
        raise SystemExit(f"refusing to overwrite {args.raw_output}")
    result = run(args.checkpoint)
    result["checkpoint_sha256"] = tape.sha256_file(args.checkpoint)
    result["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    tape.publish_json(args.raw_output, result)
    print(json.dumps(result["first_mismatch_seam_by_check"]))


if __name__ == "__main__":
    main()
