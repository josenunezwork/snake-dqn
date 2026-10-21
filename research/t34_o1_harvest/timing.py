"""V3 excluded timing fixture; never used to select registered study roots."""

from __future__ import annotations

from typing import Any

TIMING_FIXTURE = {
    "revision": "O1_TIMING_V3",
    "rows": "excluded_timing_rows",
    "frames": 5000,
    "horizon": 500,
    "time_roots": [10, 20],
    "event_limit": 0,
}


def validate_timing(worlds: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    """Require the exact two excluded worlds and both fixed prepared roots."""
    if len(worlds) != 2 or [w["row"] for w in worlds] != rows:
        raise ValueError("excluded timing fixture world identities differ")
    for world in worlds:
        if (
            world["frames"] != TIMING_FIXTURE["frames"]
            or world["horizon"] != TIMING_FIXTURE["horizon"]
            or world["complete"] is not True
            or world["accounting"] is not True
            or world["cfg"]["mechanics_version"] != 2
            or [r["decision"] for r in world["roots"]] != TIMING_FIXTURE["time_roots"]
            or any(
                r["time_root"] is not True
                or r["event_root"] is not False
                or r["horizon"] != TIMING_FIXTURE["horizon"]
                for r in world["roots"]
            )
        ):
            raise ValueError("excluded timing fixture requires prepared roots 10/20 at H5000/H500")
