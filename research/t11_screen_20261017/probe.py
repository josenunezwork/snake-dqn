"""Registered CPU T11 LOCAL30 probe with frame-specific survivor lengths."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from research.t2_screen_20261015 import probe_t2

LENGTH_FRAMES = (1000, 2000, 3000, 4000, 5000)


class LengthCurveObserver(probe_t2.SurvivalObserver):
    """Record post-frame masses for the registered shared-survivor curve."""

    def __init__(self, seeds: list[int]) -> None:
        super().__init__(seeds)
        self.length_at_frames: dict[int, dict[int, int]] = {}

    def __call__(self, snapshot: Mapping[str, Any]) -> None:
        """Capture each registered frame's mass, using zero for post-frame deaths."""
        super().__call__(snapshot)
        frame = int(snapshot["frame"]) + 1
        if frame not in LENGTH_FRAMES:
            return
        post = snapshot["post"]
        self.length_at_frames[frame] = {
            seed: int(post["logical_mass"][index]) if post["alive"][index] else 0
            for index, seed in enumerate(snapshot["seeds"])
        }


def main() -> int:
    """Run a one-thread rowwise CPU probe and atomically write one result row."""
    import torch

    from src.core.device_manager import DeviceManager

    args = probe_t2.build_parser().parse_args()
    if args.frames != 5000:
        raise ValueError("T11 requires H5000")
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise ValueError("Set OMP_NUM_THREADS=1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    DeviceManager.override_device(torch.device("cpu"))
    observers: list[LengthCurveObserver] = []
    candidate = Path(args.student) if args.student else None
    before = hashlib.sha256(candidate.read_bytes()).hexdigest() if candidate else None

    def factory(seeds: list[int]) -> LengthCurveObserver:
        observer = LengthCurveObserver(seeds)
        observers.append(observer)
        return observer

    row = probe_t2.probe(args, observer_factory=factory)
    if candidate is not None:
        after = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if before != after or row["checkpoint_sha256"] != before:
            raise ValueError("candidate checkpoint changed during T11 probe")
    observer = observers[0]
    expected_seeds = set(observer.alive_frames)
    if set(observer.length_at_frames) != set(LENGTH_FRAMES) or any(
        set(values) != expected_seeds for values in observer.length_at_frames.values()
    ):
        raise ValueError("missing registered shared-survivor length observations")
    row["length_at_frames"] = observer.length_at_frames
    row["length_at_frames_convention"] = "post-frame logical mass; dead heroes zero"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(row, allow_nan=False) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"name": args.name, "mix": args.mix, "wall": row["wall"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
