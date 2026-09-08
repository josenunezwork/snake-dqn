"""Bounded headless serving check for a declared checkpoint contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from web.backend.session import GameSession


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Checkpoint to serve")
    parser.add_argument("--frames", type=int, default=10, help="Bounded Watch frames (1..100)")
    parser.add_argument("--output", required=True, help="Disposable JSON receipt path")
    args = parser.parse_args()
    if not 1 <= args.frames <= 100:
        parser.error("--frames must be in 1..100")

    session = GameSession(checkpoint=args.checkpoint)
    for _ in range(args.frames):
        session.step()
    frame = session.snapshot()
    receipt = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "frames_requested": args.frames,
        "frame": frame["frame"],
        "obs_spec": frame["obs_spec"],
        "serving_contract": frame.get("serving_contract"),
        "error": session.last_error,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
