#!/usr/bin/env python3
"""Mac serving latency of the ego2s-b student in the web game (pre-M3 item 1).

Doc section 9 step 7 bar: <= 8 ms per frame for 12 snakes. Initializes the web serving
config through a real ``GameSession`` with the student, then drives a live ``GameState``
whose 12 (and 6) AI snakes are ALL served by the student, timing per frame (one CPU
torch thread):

* ``policy_ms``: the student's ``prepare_frame`` (live -> EgoGridView, ego2s-b numba
  featurizer for every living snake, one batched forward) on a fresh cache;
* ``game_update_ms``: the whole ``GameState.update`` (physics, every AI snake's own masks,
  the student's per-frame build, serialization not included).

Reports p50 / p95 / max over ``--frames`` frames after a warm-up (JIT compile excluded).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def measure(checkpoint: str, num_snakes: int, frames: int, warmup: int) -> dict:
    """A live ``GameState`` with ``num_snakes`` AI snakes all served by the student."""
    import numpy as np
    import torch

    from src.game.game_state import GameState
    from web.backend.ego2s_policy import Ego2sServingPolicy
    from web.backend.session import GameSession

    torch.set_num_threads(1)
    GameSession(checkpoint=checkpoint)  # initializes the serving config exactly as the web app
    pol = Ego2sServingPolicy(checkpoint)
    game = GameState(headless=True, num_snakes=num_snakes, shared_policy=pol, human_mode=False)
    pol.attach_game(game)
    pol_ms, step_ms, alive, longest = [], [], [], 0
    for i in range(warmup + frames):
        t0 = time.perf_counter()
        pol._invalidate()
        pol.prepare_frame()  # the decision-time build (fresh cache)
        t1 = time.perf_counter()
        game.update(train_mode=False, learn=False, allow_respawn=True)  # rebuilds it itself
        t2 = time.perf_counter()
        if i >= warmup:
            pol_ms.append(1e3 * (t1 - t0))
            step_ms.append(1e3 * (t2 - t1))
            alive.append(sum(bool(s.is_alive) for s in game.snakes))
            longest = max(longest, max(len(s.segments) for s in game.snakes))
    q = lambda xs, p: float(np.percentile(xs, p))  # noqa: E731
    return {
        "snakes": num_snakes,
        "mean_alive": float(np.mean(alive)),
        "max_length": int(longest),
        "policy_ms": {"p50": q(pol_ms, 50), "p95": q(pol_ms, 95), "max": max(pol_ms)},
        "game_update_ms": {"p50": q(step_ms, 50), "p95": q(step_ms, 95), "max": max(step_ms)},
        "frames": frames,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--frames", type=int, default=1500)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows = [
        measure(args.checkpoint, 12, args.frames, args.warmup),
        measure(args.checkpoint, 6, args.frames, args.warmup),
    ]
    rows.append({"loadavg": os.getloadavg(), "bar_ms_12_snakes": 8.0})
    text = json.dumps(rows, indent=1)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
