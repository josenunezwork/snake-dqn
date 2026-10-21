"""Outcome-blind selection of T3 dense all-die events (PREREGISTRATION.md, "Events").

An *all-die event* is one T34 shadow M2 case (one historical T3 action change with
a confirmed greedy doom) whose exact stride-one truth scan found no alternative
alive at the 300-step horizon anywhere in its registered window.  The M2 receipts
are immutable inputs; nothing here looks at two-deviation outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

M2_BANK = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-20261019/bank")
T3_CEILING = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/t3-rollout-labels-20261015/dense/ceiling"
)
EXPECTED_ALL_DIE = 33
PRIMARY_COUNT = 7

# Frozen by PREREGISTRATION.md Amendment 1 (decision unit = distinct world): the earliest
# all-die event of each of the 7 all-die worlds, worlds in name order.  Applied before data.
PRIMARY_EVENTS = (
    "b8_s0_21@3707",
    "b8_s0_8@4318",
    "b8_s3_19@4693",
    "b8_s3_26@3647",
    "b8_s3_6@4562",
    "b8_s3_8@4246",
    "b8_s3_9@4188",
)


def event_id(world: str, start: int) -> str:
    """Stable event identifier: ``<arm_env world>@<registered window start>``."""
    return f"{world}@{int(start)}"


def parse_event_id(value: str) -> tuple[str, int]:
    """Split ``world@start``."""
    world, start = value.split("@")
    return world, int(start)


def all_die_events(bank: Path = M2_BANK) -> list[dict[str, Any]]:
    """Every M2 case without a truth survivor, in (world name, window start) order."""
    events = []
    paths = sorted(bank.glob("b8_s?_*.json"))
    if len(paths) != 64:
        raise RuntimeError(f"M2 bank inventory drift: expected 64 receipts, found {len(paths)}")
    for path in paths:
        receipt = json.loads(path.read_text())
        if receipt.get("status") != "COMPLETE":
            raise RuntimeError(f"incomplete M2 receipt: {path}")
        for case in receipt["cases"]:
            if case["truth"]["t_star"] is not None:
                continue
            events.append(
                {
                    "id": event_id(receipt["world"], case["source_decision"]),
                    "world": receipt["world"],
                    "start": int(case["source_decision"]),
                    "doom": int(case["greedy_doom"]),
                    "historical_change": int(case["historical_action_change_decision"]),
                    "m2_receipt": str(path),
                }
            )
    events.sort(key=lambda e: (e["world"], e["start"]))
    if len(events) != EXPECTED_ALL_DIE:
        raise RuntimeError(f"all-die denominator drift: {len(events)} != {EXPECTED_ALL_DIE}")
    return events


def round_robin_order(events: list[dict[str, Any]]) -> list[str]:
    """Registered order: k-th earliest event of every world (sorted names), k = 1, 2, ...

    Spreading the sample over worlds avoids drawing half of it from one world's
    chain of successive T3 delays; it uses only world names and window starts.
    """
    by_world: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_world.setdefault(event["world"], []).append(event)
    for chain in by_world.values():
        chain.sort(key=lambda e: e["start"])
    order: list[str] = []
    depth = 0
    while len(order) < len(events):
        for world in sorted(by_world):
            if depth < len(by_world[world]):
                order.append(by_world[world][depth]["id"])
        depth += 1
    return order


def m2_truth_outcomes(event: dict[str, Any]) -> dict[str, Any]:
    """M2 truth-scan per-decision outcomes for one event (layer-1 identity control)."""
    receipt = json.loads(Path(event["m2_receipt"]).read_text())
    for case in receipt["cases"]:
        if int(case["source_decision"]) == event["start"]:
            return {
                str(probe["decision"]): {
                    "greedy": int(probe["greedy"]),
                    "outcomes": {
                        action: {
                            key: outcome.get(key)
                            for key in ("survives", "death_decision", "frames_executed", "horizon")
                        }
                        for action, outcome in probe["outcomes"].items()
                    },
                }
                for probe in case["truth"]["probes"]
            }
    raise KeyError(event["id"])


def m2_layer1_row_steps(event: dict[str, Any]) -> int:
    """M2 truth-scan alternative row-steps for the event window (outcome-blind cost proxy)."""
    receipt = json.loads(Path(event["m2_receipt"]).read_text())
    for case in receipt["cases"]:
        if int(case["source_decision"]) == event["start"]:
            return sum(
                int(outcome["frames_executed"])
                for probe in case["truth"]["probes"]
                for action, outcome in probe["outcomes"].items()
                if int(action) != int(probe["greedy"])
            )
    raise KeyError(event["id"])


def excluded_first_deviations(event: dict[str, Any]) -> list[tuple[int, int]]:
    """First deviations whose branch M2 already proved all-die (Amendment 1).

    T3's historical change ``(c, chosen)`` of this event, continued greedily, *is* the
    greedy shadow of the next M2 case of the same world when that case starts at
    ``c + 1``; its layer-2 window equals that case's window.  If that case is all-die the
    branch is a known negative and is excluded from the beam.
    """
    receipt = json.loads(Path(event["m2_receipt"]).read_text())
    change = int(event["historical_change"])
    following = [c for c in receipt["cases"] if int(c["source_decision"]) == change + 1]
    if not following or following[0]["truth"]["t_star"] is not None:
        return []
    t3 = json.loads((T3_CEILING / f"{event['world']}.json").read_text())
    rows = [t for t in t3["trigger_rows"] if int(t["decision"]) == change]
    if len(rows) != 1 or int(rows[0]["chosen"]) == int(rows[0]["greedy"]):
        raise RuntimeError(("historical change not found in the T3 receipt", event["id"]))
    return [(change, int(rows[0]["chosen"]))]


# Amendment 1 (A1.7): multi-deviation paths already visible in the immutable inputs.
# Reported separately by ``summarize``; never counted unless the registered beam finds them.
KNOWN_PATHS = {
    "b8_s3_19@4693": {"first": [4697, 2], "second": [4727, 4], "meets_600_bar": True},
    "b8_s3_9@4188": {"first": [4198, 0], "second_decision": 4204, "meets_600_bar": None},
}
