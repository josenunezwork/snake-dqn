"""Independent Mac reducer for the fixed V8 split attribution decision."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import tempfile
from pathlib import Path
from typing import Any

from research.gpu_sim_spike import v8_split_units as units

ARMS = ("R", "G", "GV")
STAGES = {
    "vector61_non_flood": ("vector61_other",),
    "flood": ("vector61_flood_fill",),
    "b8_v8_bundle": ("b8_v8_features",),
    "opponent_v8_bundle": ("opponent_v8_bundle",),
    "v8_combined": ("b8_v8_features", "opponent_v8_bundle"),
    "movement_events": ("movement_collision_events",),
}


def _positive(v: Any) -> bool:
    return type(v) in (float, int) and math.isfinite(v) and v > 0


def validate_row(row: dict[str, Any], unit: str, *, canary: bool) -> str:
    """Recompute the decision without importing the producer's rule helpers."""
    _, mix, world = unit.split("-")
    if row.get("mix") != mix or row.get("worlds") != int(world[1:]):
        raise ValueError("split cell identity differs")
    if row.get("verdict") == "INVALID":
        return "INVALID"
    if not all(
        row.get(k) is True
        for k in (
            "feature_call_cpu_parity",
            "same_action_mechanics_byte_identical",
            "full_inventory_compared",
            "g_gv_byte_identical",
            "state_carry_byte_identical",
        )
    ):
        return "INVALID"
    repeats = 1 if canary else 3
    orders = [["R", "G", "GV"], ["G", "GV", "R"], ["GV", "R", "G"]]
    if (
        row.get("repeats") != repeats
        or row.get("warmups") != repeats
        or row.get("arm_orders") != orders[:repeats]
    ):
        raise ValueError("split repeat/warmup contract differs")
    for name in ("G", "GV"):
        oracle = row["cpu_oracle"][name]
        if (
            oracle.get("vector61", {}).get("match") is not True
            or len(oracle.get("v8", [])) != 6
            or not all(
                x.get("match") is True and x.get("feature_planes_match") is True
                for x in oracle["v8"]
            )
        ):
            return "INVALID"
    medians = {}
    for name in ARMS:
        arm = row[name]
        if any(len(arm[k]) != repeats for k in ("plain_ticks", "digests", "stages")):
            raise ValueError("split timing coverage differs")
        if not all(_positive(v) for v in arm["plain_ticks"]):
            raise ValueError("invalid plain wall")
        medians[name] = statistics.median(arm["plain_ticks"])
        for i, stage in enumerate(arm["stages"]):
            if stage["repeat"] != i or not _positive(stage["wall_seconds"]):
                raise ValueError("stage repeat/wall differs")
            clocks = stage["exclusive_cuda_elapsed_seconds"]
            if not clocks or any(
                type(v) not in (int, float) or not math.isfinite(v) or v < 0
                for v in clocks.values()
            ):
                raise ValueError("invalid stage wall")
    if row["G"]["digests"] != row["GV"]["digests"]:
        return "INVALID"
    for label, components in STAGES.items():
        derived = {
            a: statistics.median(
                sum(r["exclusive_cuda_elapsed_seconds"][k] for k in components)
                for r in row[a]["stages"]
            )
            for a in ARMS
        }
        if row["stage_seconds"][label] != derived:
            raise ValueError("split reported stage arithmetic differs")
    nonflood = row["stage_seconds"]["vector61_non_flood"]
    if not all(_positive(nonflood[a]) for a in ARMS):
        raise ValueError("invalid nonflood timing")
    outcome = "PASS_CELL"
    if (
        not canary
        and row["worlds"] == 1024
        and (nonflood["R"] / nonflood["G"] < 2 or medians["R"] / medians["G"] < 2)
    ):
        outcome = "STOP_SPLIT"
    if row.get("verdict") != outcome:
        raise ValueError("producer split verdict differs")
    return outcome


def aggregate(exports: list[Path], commit: str, *, canary: bool = False) -> dict[str, Any]:
    if platform.system() != "Darwin":
        raise ValueError("V8 split reducer is Mac-only")
    ns, cells = units.namespace(commit, canary=canary), units.allocation(canary=canary)
    found = {}
    terminal_invalid = False
    terminal_incomplete = False
    ingestion_errors = []
    for root in exports:
        native = root / "native-parity.json"
        try:
            units._verify(native)
            native_body = json.loads(native.read_text())
            if native_body.get("commit") != commit:
                raise ValueError("native source pin differs")
            terminal_invalid |= native_body.get("status") == "INVALID"
            terminal_incomplete |= native_body.get("status") not in {"PASS", "INVALID"}
            failure = root / "v8-split-failure.json"
            if failure.exists():
                units._verify(failure)
                terminal_invalid |= json.loads(failure.read_text()).get("status") == "INVALID"
        except (ValueError, OSError):
            terminal_incomplete = True
        with tempfile.TemporaryDirectory(prefix="v8-split-ledger-") as temporary:
            path = Path(temporary) / "verified-ledger.json"
            try:
                units.build_ledger(root, path, commit=commit, canary=canary)
                rows = units.load_ledger(path, ns, cells)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                terminal_incomplete = True
                ingestion_errors.append({"root": str(root), "error": str(exc)})
                continue
            if set(found) & set(rows):
                raise ValueError("duplicate split unit across attempts")
            found.update(rows)
    results = []
    outcomes = []
    for unit in cells:
        if unit not in found:
            continue
        try:
            outcome = validate_row(found[unit], unit, canary=canary)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            outcome = "INCOMPLETE"
            results.append({"unit": unit, "outcome": outcome, "reason": str(exc)})
        else:
            results.append({"unit": unit, "outcome": outcome})
        outcomes.append(outcome)
    missing = [u for u in cells if u not in found]
    stops = [r["unit"] for r in results if r["outcome"] == "STOP_SPLIT"]
    # Only the exact completed prefix ending at a registered E1024 failure may
    # retain STOP_SPLIT with later cells unrun. Holes/corrupt coverage take priority.
    prefix_stop = bool(stops) and len(stops) == 1 and list(found.keys()) != []
    if stops:
        index = cells.index(stops[0])
        prefix_stop = set(found) == set(cells[: index + 1])
    if terminal_invalid or "INVALID" in outcomes:
        verdict = "INVALID"
    elif terminal_incomplete or "INCOMPLETE" in outcomes or (missing and not prefix_stop):
        verdict = "INCOMPLETE"
    elif prefix_stop:
        verdict = "STOP_SPLIT"
    elif stops:
        verdict = "INCOMPLETE"
    elif canary:
        verdict = "PASS_CANARY"
    else:
        keep = True
        for unit in cells:
            row = found[unit]
            g, gv = row["G"]["plain_ticks"], row["GV"]["plain_ticks"]
            keep &= (
                all(v < x for v, x in zip(gv, g))
                if row["worlds"] == 1024
                else statistics.median(gv) <= statistics.median(g)
            )
        verdict = "SELECT_GV" if keep else "SELECT_G"
    return {
        "schema": "gpu-sim-spike-v8-split-analysis/v1",
        "commit": commit,
        "canary": canary,
        "verdict": verdict,
        "cells": results,
        "missing_units": missing,
        "ingestion_errors": ingestion_errors,
        "cpu_closed_loop_identity": False,
        "integrated_speed_pass": False,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--commit", required=True)
    p.add_argument("--exports", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--canary", action="store_true")
    a = p.parse_args()
    result = aggregate(a.exports, a.commit, canary=a.canary)
    units.publish_json(a.output, result)
    if a.canary and result["verdict"] != "PASS_CANARY":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
