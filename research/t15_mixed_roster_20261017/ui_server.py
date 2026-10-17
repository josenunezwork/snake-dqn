"""Loopback-only T15 mixed Watch UI server with an isolated score database."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from research.t15_mixed_roster_20261017.run import sha, utc, verify, write_new


def backend_evidence(session: object, intent_sha256: str, seed: int, score_db: str) -> dict:
    """Serialize the actual mixed-session identity for the browser evidence receipt."""
    return {
        "intent_sha256": intent_sha256,
        "created_utc": utc(),
        "seed": seed,
        "served_checkpoint": session.served_checkpoint,
        "veto": session.safety_veto_state(),
        "obs_spec": session.obs_spec,
        "policy_class": type(session.policy).__name__,
        "score_db": score_db,
        "watch_roster": session.watch_roster,
        "watch_roster_reason": session.watch_roster_reason,
        "opponent_checkpoint": session.watch_opponent_checkpoint,
        "speed_ceiling": session.speed_ceiling,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8815)
    args = parser.parse_args()
    intent = verify(args.root)
    auxiliary = args.root.parent / f"{args.root.name}-ui"
    auxiliary.mkdir(exist_ok=False)
    seed = int(intent["banks"]["ui"]["seeds"][0])
    for key in (
        "SNAKE_SERVE_VETO_VARIANT",
        "SNAKE_SERVE_VETO_WATCH_HERO",
        "SNAKE_SERVE_VETO_PLAY_AI",
    ):
        os.environ.pop(key, None)
    os.environ.pop("SNAKE_SERVE_CHECKPOINT", None)
    os.environ["SNAKE_SERVE_WATCH_ROSTER"] = "mixed"
    os.environ["SNAKE_SCORES_DB"] = str(auxiliary / "scores.sqlite")
    import uvicorn

    from research.apex_veto_serving_20261001.serving_run import set_world_seed
    from web.backend import app

    original = app.GameSession

    def seeded_session():
        set_world_seed(seed)
        session = original()
        session._release_monitoring = False
        write_new(
            auxiliary / "backend.json",
            backend_evidence(
                session, sha(args.root / "intent.json"), seed, os.environ["SNAKE_SCORES_DB"]
            ),
        )
        return session

    app.GameSession = seeded_session
    uvicorn.run(app.app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
