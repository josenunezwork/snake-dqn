"""Registered T4c CPU probe, adding detached terminal-length diagnostics."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping

from research.t2_screen_20261015 import probe_t2


class TerminalLengthObserver(probe_t2.SurvivalObserver):
    """Collect post-frame logical length without influencing any policy action."""

    def __init__(self, seeds: list[int]) -> None:
        super().__init__(seeds)
        self.length_at_5000: dict[int, int] = {}

    def __call__(self, snapshot: Mapping[str, Any]) -> None:
        super().__call__(snapshot)
        if int(snapshot["frame"]) == 4999:
            for index, seed in enumerate(snapshot["seeds"]):
                post = snapshot["post"]
                self.length_at_5000[seed] = (
                    int(post["logical_mass"][index]) if post["alive"][index] else 0
                )


def main() -> int:
    """Run the unchanged T2 evaluator with an additional read-only observer."""
    import torch

    from src.core.device_manager import DeviceManager

    args = probe_t2.build_parser().parse_args()
    if args.frames != 5000:
        raise ValueError("T4c requires H5000")
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise ValueError("Set OMP_NUM_THREADS=1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    DeviceManager.override_device(torch.device("cpu"))
    observers: list[TerminalLengthObserver] = []

    def factory(seeds: list[int]) -> TerminalLengthObserver:
        observer = TerminalLengthObserver(seeds)
        observers.append(observer)
        return observer

    row = probe_t2.probe(args, observer_factory=factory)
    lengths = observers[0].length_at_5000
    if set(lengths) != set(observers[0].alive_frames):
        raise ValueError("Missing H5000 terminal-length observations")
    row["length_at_5000"] = lengths
    row["length_at_5000_convention"] = "post-frame 5000 logical mass; dead heroes zero"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(row, allow_nan=False) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"name": args.name, "mix": args.mix, "wall": row["wall"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
