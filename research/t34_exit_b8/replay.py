"""Streaming-safe regeneration of ExIt exports.

No observation leaves this module until the full action trace and every saved
physical-state hash have validated.  A caller may keep one world's rows in
memory, then hand the accepted result to a streaming fit loader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from research.t34_exit_b8.exports import SCHEMA, load_world, state_hash
from research.t34_exit_b8.trajectory import prepare_world


@dataclass(frozen=True)
class ReplayResult:
    world_id: str
    accepted: bool
    observations: tuple[dict[str, Any], ...]
    mismatch: str | None = None


def _default_builder(row: Mapping[str, Any], checkpoint: str | Path) -> Any:
    from research.t34_exit_b8.trajectory import build_world

    return build_world(dict(row), checkpoint)


def _default_observation(world: Any, label: Mapping[str, Any]) -> dict[str, Any]:
    """Capture B8's actual bv8 raster observation after ``prepare_world``."""
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

    obs = build_ego_raster(
        world.sim, EgoRasterConfig(version="bv8"), backend="numba", rows=np.asarray([[0, 0]])
    )
    return {
        "decision": int(label["decision"]),
        "local": np.asarray(obs["local"][0], dtype=np.uint8).copy(),
        "global": np.asarray(obs["global"][0], dtype=np.uint8).copy(),
        "scalars": np.asarray(obs["scalars"][0], dtype=np.float32).copy(),
        "mask": np.asarray(world.sim.get_advisory_action_mask()[0, 0], dtype=bool).copy(),
        "label": dict(label),
    }


def replay_world(
    exported: Mapping[str, Any],
    checkpoint: str | Path,
    *,
    build_world: Callable[[Mapping[str, Any], str | Path], Any] = _default_builder,
    observe: Callable[[Any, Mapping[str, Any]], dict[str, Any]] = _default_observation,
) -> ReplayResult:
    """Regenerate one world with its recorded six-action trace.

    Policy inference is deliberately absent.  ``prepare_world`` is still used
    because the Vector61 runtime's carried state is part of the official B8
    observation seam; the subsequent direct ``sim.step`` consumes that prepared
    seam exactly once.
    """
    if exported.get("schema") != SCHEMA:
        raise ValueError("unknown ExIt world export schema")
    expected_checkpoint = str(exported["checkpoint_sha256"])
    actual_checkpoint = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    if actual_checkpoint != expected_checkpoint:
        raise ValueError("checkpoint SHA-256 does not match world export")
    world = build_world(exported["row"], checkpoint)
    expected_hashes = {int(frame): str(digest) for frame, digest in exported["hashes"].items()}
    if 0 not in expected_hashes:
        raise ValueError("world export has no initial hash")
    actual = state_hash(world.sim)
    if actual != expected_hashes[0]:
        return ReplayResult(
            str(exported["world_id"]), False, (), f"frame 0: {actual} != {expected_hashes[0]}"
        )
    labels = {int(label["decision"]): label for label in exported.get("labels", [])}
    observations: list[dict[str, Any]] = []
    for decision, action_row in enumerate(exported["actions"]):
        prepare_world(world)
        label = labels.get(decision)
        if label is not None:
            observations.append(observe(world, label))
        actions = np.asarray(action_row, dtype=np.int64).reshape(1, 6)
        if actions.shape != (world.sim.E, world.sim.S):
            return ReplayResult(
                str(exported["world_id"]), False, (), "action trace shape differs from world"
            )
        world.sim.step(actions)
        world.runtime.observe_step(world.sim)
        frame = decision + 1
        if frame in expected_hashes:
            actual = state_hash(world.sim)
            if actual != expected_hashes[frame]:
                return ReplayResult(
                    str(exported["world_id"]),
                    False,
                    (),
                    f"frame {frame}: {actual} != {expected_hashes[frame]}",
                )
    final_frame = len(exported["actions"])
    if final_frame not in expected_hashes:
        raise ValueError("world export has no final hash")
    # Any declared checkpoint outside the trace is malformed rather than merely
    # ignored; this protects future schema extensions from silent downgrade.
    unknown = sorted(
        set(expected_hashes) - ({0, final_frame} | set(range(250, final_frame + 1, 250)))
    )
    if unknown:
        raise ValueError(f"unexpected hash checkpoints: {unknown}")
    return ReplayResult(str(exported["world_id"]), True, tuple(observations))


def replay_exports(
    export_dir: str | Path,
    checkpoint: str | Path,
    *,
    build_world: Callable[[Mapping[str, Any], str | Path], Any] = _default_builder,
    observe: Callable[[Any, Mapping[str, Any]], dict[str, Any]] = _default_observation,
) -> dict[str, Any]:
    """Replay all complete worlds, returning a report and accepted world rows."""
    paths = sorted(Path(export_dir).glob("world-*.json"))
    accepted, mismatches = 0, []
    for path in paths:
        try:
            result = replay_world(
                load_world(path), checkpoint, build_world=build_world, observe=observe
            )
        except Exception as exc:
            mismatches.append({"world": path.name, "mismatch": f"malformed: {exc}"})
            continue
        if result.accepted:
            accepted += 1
        else:
            mismatches.append({"world": result.world_id, "mismatch": result.mismatch})
    total = accepted + len(mismatches)
    drop_rate = len(mismatches) / total if total else 1.0
    return {
        "schema": "t34-exit-replay-report/v1",
        "worlds": total,
        "accepted": accepted,
        "dropped": len(mismatches),
        "drop_rate": drop_rate,
        "kill": total == 0 or drop_rate > 0.02,
        "mismatches": mismatches,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate and validate T34 ExIt world exports")
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True, help="create-only JSON replay report")
    args = parser.parse_args(argv)
    report = replay_exports(args.export_dir, args.checkpoint)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))
    # This CLI is the P3/A2 identity gate, whose threshold is ZERO mismatches.
    # The separate >2% ``kill`` field remains the fit-ingestion drop rule.
    return 2 if report["kill"] or report["dropped"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
