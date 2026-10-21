"""Create PLAN §2 slip receipts when a WP-1 deadline actually expires."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from research.bl_served_search.exports import export_json

PT = ZoneInfo("America/Los_Angeles")
GATES = {
    "code": datetime(2026, 10, 22, 10, tzinfo=PT),
    "first-chunks": datetime(2026, 10, 22, 12, tzinfo=PT),
    "latest-first-chunks": datetime(2026, 10, 22, 20, tzinfo=PT),
}


def record(
    gate: str,
    missing: list[str],
    out: Path,
    *,
    remaining_hours: float,
    earliest_finish: str,
    now: datetime | None = None,
) -> dict:
    """Missing evidence never silently extends a reservation or lowers a bar."""
    now = now or datetime.now(PT)
    if now < GATES[gate]:
        raise ValueError("gate has not expired; no premature slip receipt")
    result = {
        "builder": "Sol-S",
        "reviewer": "Codex-V",
        "gate": gate,
        "deadline_pt": GATES[gate].isoformat(),
        "observed_at": now.isoformat(),
        "missing_evidence": missing,
        "blocked_descendants": ["L2 chunks", "L2 pinned print"],
        "remaining_hours": remaining_hours,
        "earliest_finish": earliest_finish,
        "scientific_verdict_preserved": True,
        "readiness": "NOT_READY",
        "next_authorized_action": (
            "prepare fitting measured Y1-L4 forecast; " "Claude dispatches only after admission"
        ),
    }
    export_json(out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=GATES, required=True)
    parser.add_argument("--missing-evidence", action="append", required=True)
    parser.add_argument("--remaining-hours", type=float, required=True)
    parser.add_argument("--earliest-finish", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            record(
                args.gate,
                args.missing_evidence,
                args.out,
                remaining_hours=args.remaining_hours,
                earliest_finish=args.earliest_finish,
            )
        )
    )


if __name__ == "__main__":
    main()
