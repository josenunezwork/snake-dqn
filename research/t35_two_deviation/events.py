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
EXPECTED_ALL_DIE = 33
PRIMARY_COUNT = 10

# Frozen by PREREGISTRATION.md (selection rule applied once, before any T35 data).
PRIMARY_EVENTS = (
    "b8_s0_21@3707",
    "b8_s0_8@4318",
    "b8_s3_19@4693",
    "b8_s3_26@3647",
    "b8_s3_6@4562",
    "b8_s3_8@4246",
    "b8_s3_9@4188",
    "b8_s0_21@3737",
    "b8_s0_8@4353",
    "b8_s3_26@3675",
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
