"""Owned loopback UI server for the pinned T12 browser check; isolated score database."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from research.t12_b8_serving_20261017.serving_run import sha, utc, verify, write_new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8812)
    args = parser.parse_args()
    intent = verify(args.root)
    seed = intent["banks"]["ui"]["seeds"][0]
    auxiliary = args.root.parent / f"{args.root.name}-ui"
    auxiliary.mkdir(exist_ok=False)
    for key in (
        "SNAKE_SERVE_VETO_WATCH_HERO",
        "SNAKE_SERVE_VETO_PLAY_AI",
        "SNAKE_SERVE_VETO_VARIANT",
    ):
        os.environ.pop(key, None)
    os.environ["SNAKE_SERVE_CHECKPOINT"] = "b8-s3"
    os.environ["SNAKE_SCORES_DB"] = str(auxiliary / "scores.sqlite")
    import uvicorn

    from research.apex_veto_serving_20261001.serving_run import set_world_seed
    from web.backend import app

    original = app.GameSession

    def seeded_session():
        set_world_seed(seed)
        session = original()
        write_new(
            auxiliary / "backend.json",
            {
                "intent_sha256": sha(args.root / "intent.json"),
                "created_utc": utc(),
                "seed": seed,
                "served_checkpoint": session.served_checkpoint,
                "veto": session.safety_veto_state(),
                "obs_spec": session.obs_spec,
                "policy_class": type(session.policy).__name__,
                "score_db": os.environ["SNAKE_SCORES_DB"],
            },
        )
        return session

    app.GameSession = seeded_session
    uvicorn.run(app.app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
