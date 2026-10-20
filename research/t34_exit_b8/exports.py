"""Portable, create-only ExIt world exports.

The label job writes simulator facts rather than observations.  Consumers must
rebuild those observations through the normal prepared-policy seam and reject a
world wholesale when a physical-state digest differs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCHEMA = "t34-exit-world/v1"
_STATE_ARRAYS = (
    "head_ptr",
    "seg_count",
    "length",
    "alive",
    "direction",
    "boost_frames",
    "frames_since_food",
    "respawn_timer",
    "frame",
    "_reward_prev_length",
)
_STATE_ACCESSORS = (
    "get_action_mask",
    "get_legal_action_mask",
    "get_resolved_action_mask",
    "get_reward",
    "get_done",
    "get_death_cause",
    "get_kill_credit",
    "get_transition_valid",
    "get_boosted_this_step",
)


def _bytes(h: Any, value: Any) -> None:
    """Bind an array's type and shape as well as its deterministic bytes."""
    array = np.asarray(value)
    h.update(array.dtype.str.encode())
    h.update(repr(tuple(array.shape)).encode())
    h.update(np.ascontiguousarray(array).tobytes())


def state_hash(sim: Any, env: int = 0) -> str:
    """Return a canonical digest of one physical simulator world.

    This is deliberately based on the audited grid H5000 identity digest, with
    the fields needed to continue an exact episode added explicitly.  It never
    traverses ``vars(sim)``: caches, compiled dispatchers, object addresses, and
    timers must not decide replay identity.
    """
    if not 0 <= int(env) < int(sim.E):
        raise IndexError("environment is outside simulator batch")
    h = hashlib.blake2b(digest_size=20)
    h.update(b"t34-exit-state/v1\\0")
    for name in _STATE_ARRAYS:
        if hasattr(sim, name):
            h.update(name.encode() + b"\\0")
            _bytes(h, getattr(sim, name)[env])
    for name in _STATE_ACCESSORS:
        method = getattr(sim, name, None)
        if method is not None:
            h.update(name.encode() + b"\\0")
            _bytes(h, method()[env])
    cap = int(sim.cap)
    h.update(f"cap:{cap};S:{int(sim.S)}".encode())
    # Ring storage beyond seg_count is capacity scratch and is intentionally
    # excluded.  Ordered live bodies are game state and must be bound.
    for slot in range(int(sim.S)):
        count = int(sim.seg_count[env, slot])
        ring = (int(sim.head_ptr[env, slot]) - np.arange(count, dtype=np.int64)) % cap
        h.update(f"body:{slot}:{count}".encode())
        _bytes(h, sim.bodies[env, slot, ring])
        victims = sim.get_kill_victim_lengths(env, slot)
        h.update(repr(tuple(int(value) for value in victims)).encode())
    events = sim.get_step_events()
    for key in sorted(events):
        h.update(key.encode() + b"\\0")
        _bytes(h, events[key][env])
    h.update(repr(tuple(tuple(int(v) for v in cell) for cell in sim.food_cells[env])).encode())
    h.update(
        repr(tuple(sorted(tuple(int(v) for v in cell) for cell in sim.corpse_cells[env]))).encode()
    )
    # BatchSim's food, respawn, and collision draws share this exact stream.
    h.update(repr(sim._rngs[env]._rng.getstate()).encode())
    return h.hexdigest()


def _plain(value: Any) -> Any:
    """Convert engine diagnostics to finite JSON primitives without path leakage."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        raise ValueError("exports may not contain filesystem paths")
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("exports may not contain non-finite floats")
    if isinstance(value, str) and ("/Users/" in value or "\\\\Users\\\\" in value):
        raise ValueError("exports may not contain host filesystem paths")
    return value


def _fp16(value: Any) -> float:
    """Return a JSON number with IEEE binary16 quantization semantics."""
    value = float(value)
    encoded = np.float16(value)
    if not np.isfinite(encoded):
        raise ValueError("label mean_mass cannot be represented as finite float16")
    return float(encoded)


def _labels(raw: Any, horizon: int) -> list[dict[str, Any]]:
    """Normalize engine diagnostics while preserving only portable label facts."""
    out: list[dict[str, Any]] = []
    for item in raw:
        item = _plain(item)
        outcomes = item.get("outcomes", {})
        packed = []
        tested = []
        for action in range(6):
            source = outcomes.get(str(action), outcomes.get(action, {}))
            tested.append(bool(source))
            # ``mass`` is a branch integral in the search engine.  Export the
            # requested mean-horizon statistic so binary16 cannot overflow.
            mean_mass = float(source.get("mean_mass", source.get("mass", 0.0))) / (
                1 if "mean_mass" in source else max(1, horizon)
            )
            packed.append(
                {
                    "survive": bool(source.get("survives", source.get("survive", False))),
                    "alive_steps": int(source.get("alive_steps", 0)),
                    "mean_mass": _fp16(mean_mass),
                }
            )
        forbidden = item.get("forbidden", item.get("forbidden_set", [False] * 6))
        if isinstance(forbidden, Mapping):
            forbidden = [bool(forbidden.get(str(a), forbidden.get(a, False))) for a in range(6)]
        elif all(
            isinstance(value, (int, np.integer)) and not isinstance(value, bool)
            for value in forbidden
        ):
            forbidden = [action in set(int(value) for value in forbidden) for action in range(6)]
        if len(forbidden) != 6:
            raise ValueError("forbidden action set must have six entries")
        label = {
            "decision": int(item["decision"]),
            "action": int(item.get("action", item.get("recommended_action", 0))),
            "student_action": int(item["student_action"]),
            "executed_action": int(item.get("executed_action", item.get("action", 0))),
            "forbidden": [bool(value) for value in forbidden],
            "outcomes": packed,
            "tested": tested,
            "length": int(item.get("length", 0)),
            "reason": str(item.get("reason", "unknown")),
            "expanded": bool(item.get("expanded", False)),
        }
        # V2 adds provenance and label roles without changing the replay schema.
        # In particular KL anchors must not silently become decisive examples.
        for key in (
            "label_role",
            "shadow",
            "shadow_id",
            "shadow_frame",
            "shadow_confirmed",
            "lad",
            "t_star",
            "a_star",
            "F",
            "q",
            "search",
            "searched",
            "decisive",
            "kl_anchor_only",
            "legal",
            "per_action",
        ):
            if key in item:
                label[key] = _plain(item[key])
        out.append(label)
    return out


def normalize_world(result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and convert a collector result into the stable world schema."""
    raw = _plain(dict(result))
    if raw.get("schema", SCHEMA) != SCHEMA:
        raise ValueError("unexpected ExIt export schema")
    row = raw["row"]
    seed = int(raw["seed"])
    actions = raw["actions"]
    if any(len(frame) != 6 or any(not 0 <= int(a) < 6 for a in frame) for frame in actions):
        raise ValueError("each export frame must contain six actions in [0, 5]")
    hashes = {str(int(frame)): str(digest) for frame, digest in raw["hashes"].items()}
    if "0" not in hashes or str(len(actions)) not in hashes:
        raise ValueError("exports require initial and final state hashes")
    for frame in range(250, len(actions) + 1, 250):
        if str(frame) not in hashes:
            raise ValueError("exports require a state hash every 250 frames")
    horizon = int(raw.get("horizon", 300))
    normalized = {
        "schema": SCHEMA,
        "world_id": str(raw["world_id"]),
        "seed": seed,
        "mix": str(raw["mix"]),
        "row": row,
        "checkpoint_sha256": str(raw["checkpoint_sha256"]),
        "rollout_policy_sha256": str(raw.get("rollout_policy_sha256", raw["checkpoint_sha256"])),
        "actions": [[int(action) for action in frame] for frame in actions],
        "hashes": hashes,
        "rng_stream_ids": _plain(raw.get("rng_stream_ids", {"food": seed, "scripted": seed})),
        "labels": _labels(raw.get("labels", []), horizon),
        "rewinds": int(raw.get("rewinds", 0)),
        "complete": bool(raw.get("complete", True)),
        "frames": int(raw.get("frames", len(actions))),
        "horizon": horizon,
        "beta": float(raw.get("beta", 1.0)),
        # Timing belongs in a benchmark/chunk receipt, never in a per-world
        # identity export: wall clocks would defeat the A1 byte comparison.
        "profile": {
            key: int(raw.get("profile", {}).get(key, 0))
            for key in (
                "decisions",
                "verifications",
                "expansions",
                "active_row_steps",
                "resident_row_steps",
            )
        },
    }
    if "teacher_v2" in raw:
        normalized["teacher_v2"] = _plain(raw["teacher_v2"])
    return normalized


def write_world(path: str | Path, result: Mapping[str, Any]) -> Path:
    """Create a supervisor-discoverable payload, checksum, and completion marker.

    ``research.colab_cli.runtime`` accepts exactly ``payload`` +
    ``payload.sha256`` + ``payload.complete``.  The marker is created last, so
    incomplete uploads are invisible to its immutable export scanner.
    """
    path = Path(path)
    payload = normalize_world(result)
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    try:
        with path.with_name(path.name + ".sha256").open("x", encoding="utf-8") as stream:
            stream.write(f"{digest}  {path.name}\n")
        with path.with_name(path.name + ".complete").open("x", encoding="utf-8") as stream:
            stream.write("complete\n")
    except Exception:
        # The world remains a forensic partial export, but it is never accepted
        # by replay without its completion receipt.
        raise
    return path


def load_world(path: str | Path) -> dict[str, Any]:
    """Load only a complete, SHA-bound export sidecar pair."""
    path = Path(path)
    encoded = path.read_bytes()
    marker = path.with_name(path.name + ".complete")
    checksum = path.with_name(path.name + ".sha256")
    if not marker.is_file() or not checksum.is_file():
        raise ValueError(f"incomplete world export for {path}")
    digest = checksum.read_text(encoding="utf-8").strip().split(maxsplit=1)
    if (
        len(digest) != 2
        or digest[1] != path.name
        or digest[0] != hashlib.sha256(encoded).hexdigest()
    ):
        raise ValueError(f"world export digest mismatch for {path}")
    value = json.loads(encoded)
    if value.get("schema") != SCHEMA:
        raise ValueError(f"unknown export schema for {path}")
    return value
