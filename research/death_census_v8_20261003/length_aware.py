#!/usr/bin/env python3
"""Read-only what-if over a merged v8 census: would a LENGTH-AWARE free-space gate fire?

v8's hard gate calls a direction spacious when its tail-aware one-step count reaches
``need = min(length, cap)`` with ``cap = min(160, max(32, 2 * length))``: at most 160
cells, while the census's fatal lengths are 235-881. For every decision in each saved
death window (static world of that frame, exact own move), this computes each legal
action's post-move tail-aware count capped at the hero's LENGTH (not 160) and reports:

* ``opportunity``: the taken action's count was below ``length`` while some other
  masked-legal action's count reached ``length`` (a length-aware gate could have vetoed);
* the last such decision before death (frames before death) and whether v8's own
  160-capped view called the taken direction spacious there (v8 had no signal).

Static, one frame: it says nothing about whether the alternative then survives.
Writes ``<root>/merged/length_aware.json`` (create-only).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.death_census_v8_20261003 import census as c  # noqa: E402
from research.trap_horizon_20261001 import diagnose as base  # noqa: E402


def action_counts(snap: Dict[str, Any], meta: Dict[str, Any]) -> List[Optional[int]]:
    """Post-move tail-aware count (capped at length) per action; None = illegal/dies."""
    world, state = base.world_and_state(snap, meta)
    limit = max(1, int(state.length))
    out: List[Optional[int]] = []
    for action in range(base.NUM_ACTIONS):
        if not bool(snap["mask"][action]):
            out.append(None)
            continue
        child, _ = base.step_hero(world, state, action)
        out.append(None if child is None else base.tail_aware_count(world, child, limit))
    return out


def death_row(death: Dict[str, Any], snaps: List[Dict[str, Any]], meta) -> Dict[str, Any]:
    fatal = len(snaps) - 1
    last: Optional[Dict[str, Any]] = None
    opportunities = 0
    for t, snap in enumerate(snaps):
        counts = action_counts(snap, meta)
        length = int(snap["length"])
        taken = int(snap["final"])
        tc = counts[taken]
        alts = [a for a, n in enumerate(counts) if a != taken and n is not None and n >= length]
        if (tc is None or tc < length) and alts:
            opportunities += 1
            view = base._v2_view(snap)
            last = {
                "index": t,
                "frames_before_death": fatal - t,
                "taken_count": tc,
                "best_alternative_count": max(counts[a] for a in alts),
                "length": length,
                "v8_taken_direction_spacious": bool(view["taken_direction_spacious"]),
                "v8_reason": c.REASONS[int(snap["v8_reason"])],
            }
    return {
        "mix": death["mix"],
        "world_seed": death["world_seed"],
        "cause": death["cause"],
        "category": death.get("category"),
        "fatal_length": death["fatal_length"],
        "window": len(snaps),
        "opportunities": opportunities,
        "last_opportunity": last,
    }


def main(argv: List[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[0]) if argv else c.DEFAULT_ROOT
    out = root / "merged" / "length_aware.json"
    if out.exists():
        print(f"refusing: {out} exists", file=sys.stderr)
        return 2
    rows = []
    for shard in sorted(root.glob("shard-*")):
        jobs = json.loads((shard / "jobs.json").read_text())
        meta = jobs[0]["job"][2] if jobs else None
        for path in sorted((shard / "deaths").glob("*.json")):
            death = json.loads(path.read_text())
            with np.load(shard / "windows" / death["window_file"]) as data:
                snaps = c.unpack_census_window({k: data[k] for k in data.files})
            rows.append(death_row(death, snaps, meta))
            print(json.dumps({k: rows[-1][k] for k in ("cause", "opportunities")}), flush=True)
    with_opp = [r for r in rows if r["last_opportunity"]]
    summary = {
        "deaths": len(rows),
        "with_opportunity": len(with_opp),
        "with_opportunity_by_cause": base._tally(r["cause"] for r in with_opp),
        "deaths_by_cause": base._tally(r["cause"] for r in rows),
        "with_opportunity_by_category": base._tally(r["category"] for r in with_opp),
        "last_opportunity_frames_before_death": sorted(
            r["last_opportunity"]["frames_before_death"] for r in with_opp
        ),
        "last_opportunity_v8_taken_spacious": base._tally(
            r["last_opportunity"]["v8_taken_direction_spacious"] for r in with_opp
        ),
        "last_opportunity_v8_reason": base._tally(
            r["last_opportunity"]["v8_reason"] for r in with_opp
        ),
        "rows": rows,
    }
    base.write_new_json(out, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
