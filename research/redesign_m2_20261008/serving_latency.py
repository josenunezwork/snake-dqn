#!/usr/bin/env python3
"""Mac serving latency of the ego2s-b student in the web game (pre-M3 item 1).

Doc section 9 step 7 bar: <= 8 ms per frame for 12 snakes. Builds a real
``GameSession`` with the student checkpoint (Play mode with 11 AI opponents = 12 snakes,
and default Watch), steps it, and times per frame (one CPU torch thread):

* ``policy_ms``: the student's ``prepare_frame`` (live -> EgoGridView, ego2s-b numba
  featurizer for every living snake, one batched forward) on a fresh cache;
* ``step_ms``: the whole ``session.step()`` (game physics, the AI snakes' own masks, the
  student, serialization not included).

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


def measure(checkpoint: str, mode: str, opponents, frames: int, warmup: int) -> dict:
    import numpy as np
    import torch

    from web.backend.session import MODE_PLAY, MODE_WATCH, GameSession

    torch.set_num_threads(1)
    sess = GameSession(checkpoint=checkpoint)
    if mode == "play":
        sess.play_opponents = opponents
        sess.set_mode(MODE_PLAY)
    else:
        sess.set_mode(MODE_WATCH)
    pol = sess.policy
    pol_ms, step_ms, alive = [], [], []
    for i in range(warmup + frames):
        t0 = time.perf_counter()
        pol._invalidate()
        pol.prepare_frame()
        t1 = time.perf_counter()
        sess.step()
        t2 = time.perf_counter()
        if i >= warmup:
            pol_ms.append(1e3 * (t1 - t0))
            step_ms.append(1e3 * (t2 - t1))
            alive.append(sum(bool(s.is_alive) for s in sess.game.snakes))
    q = lambda xs, p: float(np.percentile(xs, p))  # noqa: E731
    return {
        "mode": mode,
        "snakes": len(sess.game.snakes),
        "mean_alive": float(np.mean(alive)),
        "max_length": int(max(len(s.segments) for s in sess.game.snakes)),
        "policy_ms": {"p50": q(pol_ms, 50), "p95": q(pol_ms, 95), "max": max(pol_ms)},
        "step_ms": {"p50": q(step_ms, 50), "p95": q(step_ms, 95), "max": max(step_ms)},
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
        measure(args.checkpoint, "play", 11, args.frames, args.warmup),
        measure(args.checkpoint, "watch", None, args.frames, args.warmup),
    ]
    rows.append({"loadavg": os.getloadavg(), "bar_ms_12_snakes": 8.0})
    text = json.dumps(rows, indent=1)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
