#!/usr/bin/env python3
"""Read-only: did the WORLD close the trap? (count-only PNR, self deaths, exact walks).

For each self death whose count-only walk is exact and whose taken action still had a
count-only escape at the PNR ``t``, the walk found no escaping action at ``t + 1``. This
re-runs the count-only search from the hero's actual state at ``t + 1`` in the frame-``t``
world (other snakes and food as at ``t``). An escape there means the route was closed by
the other snakes' (or food's) change between ``t`` and ``t + 1``, which no static search at
``t`` can see. It also reports the post-move state's own tail-aware count in both worlds
(a state exactly at ``need`` whose every continuation fails is a threshold case). Prints
JSON lines and a summary; writes nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.death_census_v8_20261003 import census as c  # noqa: E402
from research.trap_horizon_20261001 import diagnose as base  # noqa: E402


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else c.DEFAULT_ROOT
    rows = []
    for shard in sorted(root.glob("shard-*")):
        jobs = json.loads((shard / "jobs.json").read_text())
        meta = jobs[0]["job"][2] if jobs else None
        for path in sorted((shard / "deaths").glob("*.json")):
            death = json.loads(path.read_text())
            if death["cause"] != "self":
                continue
            walk = death["sensitivity"][base.COUNT_ONLY]["walk"]
            if walk["status"] != "exact" or walk["taken_status"] != base.ESCAPE:
                continue
            with np.load(shard / "windows" / death["window_file"]) as data:
                snaps = c.unpack_census_window({k: data[k] for k in data.files})
            t = int(walk["pnr_index"])
            world_t, _ = base.world_and_state(snaps[t], meta)
            world_1, state_1 = base.world_and_state(snaps[t + 1], meta)
            res = base.analyze_frame(
                world_t, state_1, depth=base.DEPTH - 1, criterion=base.COUNT_ONLY
            )
            need = base.need_for(state_1.length)
            row = {
                "death": path.stem,
                "frames_before_death": walk["frames_before_death"],
                "escape_in_frame_t_world": any(r.status == base.ESCAPE for r in res.values()),
                "unknown": any(r.status == base.UNKNOWN for r in res.values()),
                "post_move_count_t_world": base.tail_aware_count(world_t, state_1, need),
                "post_move_count_t1_world": base.tail_aware_count(world_1, state_1, need),
                "need": need,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    closed = sum(r["escape_in_frame_t_world"] for r in rows)
    threshold = sum(
        (not r["escape_in_frame_t_world"]) and r["post_move_count_t_world"] >= r["need"]
        for r in rows
    )
    print(json.dumps({"cases": len(rows), "closed_by_world": closed, "threshold": threshold}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
