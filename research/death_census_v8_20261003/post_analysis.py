#!/usr/bin/env python3
"""Read-only post-analysis of a merged v8 census (no episodes, no searches).

For every death it reads the saved death JSON and decision window and reports how long
the hero had been in v8's ``no_spacious`` regime before dying (consecutive ``no_spacious``
decisions ending at the fatal one, capped by the 120-decision window), v8's view at the
count-only PNR (was the taken direction one-step spacious there: the veto's look-ahead
had no signal), and the window's last spacious decision. Writes
``<root>/merged/post_analysis.json`` (create-only).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.death_census_v8_20261003 import census as c  # noqa: E402
from research.trap_horizon_20261001 import diagnose as base  # noqa: E402

NO_SPACIOUS = c.REASONS.index("no_spacious")


def no_spacious_run(reasons: np.ndarray) -> int:
    run = 0
    for code in reasons[::-1]:
        if int(code) != NO_SPACIOUS:
            break
        run += 1
    return run


def death_row(death: Dict[str, Any], window: Dict[str, np.ndarray]) -> Dict[str, Any]:
    reasons = np.asarray(window["v8_reason"])
    run = no_spacious_run(reasons)
    row: Dict[str, Any] = {
        "mix": death["mix"],
        "world_seed": death["world_seed"],
        "cause": death["cause"],
        "fatal_length": death["fatal_length"],
        "fatal_frame": death["fatal_frame"],
        "no_spacious_run": run,
        "no_spacious_run_censored": run == len(reasons),
        "no_spacious_in_window": int((reasons == NO_SPACIOUS).sum()),
        "window": int(len(reasons)),
    }
    if death["cause"] == "self":
        cw = death["sensitivity"][base.COUNT_ONLY]["walk"]
        view = death.get("count_pnr_v8")
        row.update(
            {
                "category": death["category"],
                "pnr_status": death["walk"]["status"],
                "pnr_frames": death["walk"]["frames_before_death"],
                "count_status": cw["status"],
                "count_frames": cw["frames_before_death"],
                "count_frames_lower": cw.get("frames_before_lower"),
                "count_taken_status": cw["taken_status"],
                "count_taken_direction_spacious": (
                    None if view is None else view["taken_direction_spacious"]
                ),
                "count_pnr_v8_reason": None if view is None else view["v8_reason"],
                "count_pnr_max_count": None if view is None else view["max_count"],
                "need": None if view is None else view["need"],
                "pnr_escape_kinds": death["pnr_escape_kinds"],
                "enclosed_early": death["enclosed_early"],
                "deadline_hit": death["deadline_hit"],
            }
        )
    return row


def main(argv: List[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[0]) if argv else c.DEFAULT_ROOT
    out = root / "merged" / "post_analysis.json"
    if out.exists():
        print(f"refusing: {out} exists", file=sys.stderr)
        return 2
    rows = []
    for shard in sorted(root.glob("shard-*")):
        for path in sorted((shard / "deaths").glob("*.json")):
            death = json.loads(path.read_text())
            with np.load(shard / "windows" / death["window_file"]) as data:
                window = {k: data[k] for k in data.files}
            rows.append(death_row(death, window))
    self_rows = [r for r in rows if r["cause"] == "self"]

    def tally(values):
        return base._tally(values)

    runs = [r["no_spacious_run"] for r in rows]
    summary = {
        "deaths": len(rows),
        "no_spacious_run_by_cause": {
            cause: {
                "runs": sorted(r["no_spacious_run"] for r in rows if r["cause"] == cause),
                "censored_at_window": sum(
                    r["no_spacious_run_censored"] for r in rows if r["cause"] == cause
                ),
            }
            for cause in sorted({r["cause"] for r in rows})
        },
        "no_spacious_run_median": float(np.median(runs)) if runs else None,
        "self_count_pnr_taken_spacious": tally(
            r["count_taken_direction_spacious"] for r in self_rows
        ),
        "self_count_pnr_taken_spacious_exact_only": tally(
            r["count_taken_direction_spacious"] for r in self_rows if r["count_status"] == "exact"
        ),
        "self_count_status": tally(r["count_status"] for r in self_rows),
        "self_count_taken_status": tally(r["count_taken_status"] for r in self_rows),
        "rows": rows,
    }
    base.write_new_json(out, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
