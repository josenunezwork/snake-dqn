"""d.1 byte-identity hash-chain harness for the round-25 GC policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from research.bl_gc.common import attest, initialize, new_session, write_result

TIMING_GC_KEYS = {
    "update_ms",
    "latency_eligible",
    "gc_pause_ms",
    "gc_gen2_count",
    "gc_gen2_pause_ms",
    "gc_freeze_count",
    "gc_policy",
}


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def _scrub_dto(value: Any) -> Any:
    """Remove telemetry-only timing/GC fields recursively from a frame DTO."""
    if isinstance(value, dict):
        return {
            str(key): _scrub_dto(item)
            for key, item in value.items()
            if str(key) not in TIMING_GC_KEYS
            and "seconds" not in str(key)
            and "timing" not in str(key)
        }
    if isinstance(value, list):
        return [_scrub_dto(item) for item in value]
    return value


def _array_bytes(value: Any) -> bytes | None:
    if value is None:
        return None
    array = np.asarray(value)
    return b"|".join((str(array.dtype).encode(), repr(array.shape).encode(), array.tobytes()))


def _hero_observation_bytes(session: Any, hero: Any) -> bytes | None:
    """Capture raw B8 hero observation bytes, independent of display rounding."""
    policy = hero.policy
    ensure = getattr(policy, "_ensure_frame", None)
    if callable(ensure):
        ensure()
    obs = getattr(policy, "_obs", None)
    row_map = getattr(policy, "_id_to_row", {})
    row = row_map.get(int(hero.id)) if isinstance(row_map, dict) else None
    if not isinstance(obs, dict) or row is None:
        state = getattr(hero, "last_state", None)
        return _array_bytes(state)
    positions = np.flatnonzero(np.asarray(obs.get("rows", ())) == row)
    if not len(positions):
        return None
    position = int(positions[0])
    parts = []
    for key in sorted(obs):
        if key == "rows":
            continue
        parts.append(
            key.encode()
            + b"="
            + (_array_bytes(np.asarray(obs[key])[position : position + 1]) or b"")
        )
    return b"\n".join(parts)


def _tick_payload(session: Any, dto: dict[str, Any]) -> dict[str, Any]:
    game = session.game
    hero = next(snake for snake in game.snakes if int(snake.id) == int(session.hero_id))
    q_for = getattr(hero.policy, "q_for", None)
    q_values = q_for(hero.id) if callable(q_for) else getattr(hero, "get_q_values", lambda: None)()
    return {
        "frame": int(game.frame),
        "snakes": [
            {
                "id": int(snake.id),
                "is_alive": bool(snake.is_alive),
                "segments": [[int(x), int(y)] for x, y in snake.segments],
                "head": [int(snake.head[0]), int(snake.head[1])],
                "length": int(snake.length),
                "is_boosting": bool(snake.is_boosting),
                "respawn_timer": int(snake.respawn_timer),
            }
            for snake in game.snakes
        ],
        "food": [[int(x), int(y)] for x, y in game.food_manager.food],
        "corpse_positions": [list(item) for item in sorted(game.food_manager._corpse_positions)],
        "world_rng": pickle.dumps(__import__("random").getstate(), protocol=4).hex(),
        "hero_observation": (_hero_observation_bytes(session, hero) or b"").hex(),
        "hero_q": (_array_bytes(q_values) or b"").hex(),
        "chosen_actions": [
            {"id": int(snake.id), "action": getattr(snake, "last_action", None)}
            for snake in game.snakes
        ],
        "frame_death_causes": {
            str(key): str(value) for key, value in sorted(game.frame_death_causes.items())
        },
        "snapshot": _scrub_dto(dto),
    }


def run(policy: str, seed: int, ticks: int) -> dict[str, Any]:
    """Run an unpaced production Watch hash chain without retaining tick payloads."""
    receipt, initial = initialize(policy, "identity")
    session, roster = new_session(seed)
    chain = bytes(32)
    chain_rows: list[str] = []
    try:
        for _ in range(int(ticks)):
            session.step()
            dto = session.snapshot()
            chain = hashlib.sha256(chain + _json_bytes(_tick_payload(session, dto))).digest()
            chain_rows.append(chain.hex())
        return {
            "schema": "bl-gc-identity/v1",
            "policy": policy,
            "seed": int(seed),
            "ticks": int(ticks),
            "chain": chain_rows,
            "final_chain_sha256": chain.hex(),
            "attestation": {**initial, "roster": roster, "final": attest(session, receipt)},
        }
    finally:
        session.close(wait=True)


def compare(paths: list[Path]) -> dict[str, Any]:
    """Apply d.1's default/freeze/default final-chain admission bar."""
    if len(paths) != 3 or len({path.name for path in paths}) != 3:
        raise ValueError("comparison requires three uniquely named identity receipts")
    from research.bl_served_search.exports import verify_export

    for path in paths:
        verify_export(path)
    rows = [json.loads(path.read_text()) for path in paths]
    for row in rows:
        if (
            row.get("schema") != "bl-gc-identity/v1"
            or len(row.get("chain", [])) != row.get("ticks")
            or not row["chain"]
            or row["chain"][-1] != row.get("final_chain_sha256")
        ):
            raise ValueError("invalid identity chain receipt")
    for key in ("judgment_sha256", "interpreter", "version", "git_commit", "source_hashes"):
        if not rows[0]["attestation"].get(key) or any(
            row["attestation"].get(key) != rows[0]["attestation"][key] for row in rows
        ):
            raise ValueError(f"identity receipts differ in {key}")
    if [row.get("policy") for row in rows] != ["default", "freeze_startup", "default"]:
        raise ValueError("comparison order must be default, freeze_startup, default")
    if (
        len({int(row["seed"]) for row in rows}) != 1
        or len({int(row["ticks"]) for row in rows}) != 1
    ):
        raise ValueError("comparison receipts must use one common seed and tick count")
    hashes = [str(row["final_chain_sha256"]) for row in rows]
    return {
        "schema": "bl-gc-identity-comparison/v1",
        "inputs": [str(path) for path in paths],
        "seed": int(rows[0]["seed"]),
        "ticks": int(rows[0]["ticks"]),
        "final_chain_sha256s": hashes,
        "verdict": "PASS" if len(set(hashes)) == 1 else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("default", "freeze_startup"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ticks", type=int)
    parser.add_argument("--check", nargs=3, type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.check:
        result = compare(args.check)
        write_result(args.out, result)
        print(json.dumps({"verdict": result["verdict"], "out": str(args.out)}))
        if result["verdict"] != "PASS":
            raise SystemExit(2)
        return
    if args.policy is None or args.seed is None or args.ticks is None or args.ticks <= 0:
        raise ValueError("--ticks must be positive")
    result = run(args.policy, args.seed, args.ticks)
    write_result(args.out, result)
    print(json.dumps({"final_chain_sha256": result["final_chain_sha256"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
