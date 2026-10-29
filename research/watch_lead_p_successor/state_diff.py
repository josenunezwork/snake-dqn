"""Prospective path-sensitivity reproducer; no historical/scientific payload reads."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

from research.bl_served_search.exports import export_json
from research.watch_lead_p_successor.units import CHECKPOINTS, sha


def canonical_world(world: Any) -> Any:
    """Exactly the digest's input; RNG is switched by the world's active context."""
    import numpy as np
    import torch

    from research.bl_served_search.served_tick_probe import _canonical
    from src.core.game_config import get_config

    with world.active():
        return _canonical(
            {
                "session": world.session,
                "game": world.session.game,
                "snake_policies": [getattr(s, "policy", None) for s in world.session.game.snakes],
                "python_random": random.getstate(),
                "numpy_random": np.random.get_state(),
                "torch_cpu_rng": torch.get_rng_state(),
                "runtime_config": get_config(),
            }
        )


def differences(left: Any, right: Any, path: str = "") -> list[dict]:
    if type(left) is not type(right):
        return [{"field": path, "left_sha256": sha(left), "right_sha256": sha(right)}]
    if isinstance(left, dict) and left.keys() == right.keys():
        return [d for k in left for d in differences(left[k], right[k], path + "/" + k)]
    if isinstance(left, list) and len(left) == len(right):
        return [
            d
            for i, (a, b) in enumerate(zip(left, right))
            for d in differences(a, b, path + f"/{i}")
        ]
    if left == right:
        return []
    return [{"field": path, "left_sha256": sha(left), "right_sha256": sha(right)}]


def reproduce(inputs: Path, seed: int, output: Path, *, frames: int = 4) -> dict:
    from research.bl_served_search.exports import digest
    from research.watch_lead_p.live import LiveWorld
    from research.watch_lead_p.uploaded import uploaded_checkpoints

    with tempfile.TemporaryDirectory(prefix="p-state-path-") as temporary:
        dirs = [Path(temporary) / label for label in ("allocation-a", "allocation-b")]
        worlds = []
        for directory in dirs:
            directory.mkdir()
            for name, expected in CHECKPOINTS.items():
                if digest(inputs / name) != expected:
                    raise ValueError("checkpoint bytes differ")
                os.link(inputs / name, directory / name)
            with uploaded_checkpoints(directory):
                worlds.append(LiveWorld(seed))
        trace_equal, checkpoints = True, []
        for frame in range(frames + 1):
            left, right = (canonical_world(w) for w in worlds)
            diff = differences(left, right)
            # Test causality on copies of diagnostic inputs only. Never change stored/full hashes.
            a, b = json.dumps(left, sort_keys=True), json.dumps(right, sort_keys=True)
            replaced_a, replaced_b = a.replace(str(dirs[0]), "CHECKPOINT_ROOT"), b.replace(
                str(dirs[1]), "CHECKPOINT_ROOT"
            )
            path_only = replaced_a == replaced_b
            checkpoints.append(
                {
                    "frame": frame,
                    "full_hashes": [w.digest() for w in worlds],
                    "differences": diff,
                    "checkpoint_root_only": path_only,
                }
            )
            if frame < frames:
                events = [w.step() for w in worlds]
                trace_equal &= events[0] == events[1]
        result = {
            "engineering_only": True,
            "seed": seed,
            "frames": frames,
            "checkpoint_sha256": CHECKPOINTS,
            "observations": checkpoints,
            "event_identity": trace_equal,
            "path_sensitivity_reproduced": trace_equal
            and all(
                c["differences"]
                and c["checkpoint_root_only"]
                and c["full_hashes"][0] != c["full_hashes"][1]
                for c in checkpoints
            ),
            "historical_payloads_read": False,
            "scope": "same-host prospective causal reproducer, not retrospective "
            "equality of unavailable historical states",
        }
        export_json(output, result)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reproduce(args.inputs, args.seed, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
