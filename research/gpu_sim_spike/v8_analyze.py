"""Mac verifier/reducer for the frozen V8 A/B engineering matrix."""

from __future__ import annotations

import argparse
import math
import platform
import statistics
import tempfile
from pathlib import Path
from typing import Any

from research.gpu_sim_spike import v8_units as units
from research.gpu_sim_spike.v8_bench import decide


def aggregate(exports: list[Path], commit: str, *, canary: bool = False) -> dict[str, Any]:
    if platform.system() != "Darwin":
        raise ValueError("V8 reducer is Mac-only")
    ns, cells = units.namespace(commit, canary=canary), units.allocation(canary=canary)
    found: dict[str, Any] = {}
    for root in exports:
        with tempfile.TemporaryDirectory(prefix="v8-ledger-") as temp:
            ledger = Path(temp) / "verified-ledger.json"
            units.build_ledger(root, ledger, commit=commit, canary=canary)
            rows = units.load_ledger(ledger, ns, cells)
            if set(found) & set(rows):
                raise ValueError("duplicate V8 unit across attempts")
            found.update(rows)
    # A scientific stop at its registered cell boundary remains a stop even
    # though later cells were intentionally not run.
    invalid, stops, results = [], [], []
    for unit in cells:
        if unit not in found:
            continue
        row = found[unit]
        if row.get("verdict") == "INVALID":
            invalid.append(unit)
            results.append({"unit_id": unit, "outcome": "INVALID", "reason": row.get("reason")})
            continue
        if (
            row.get("mix") != unit.split("-")[1]
            or row.get("worlds") != int(unit.rsplit("e", 1)[1])
            or row.get("repeats") != (1 if canary else 3)
            or row.get("warmups") != (1 if canary else 3)
        ):
            raise ValueError("V8 cell identity/repeat contract differs")
        for arm in ("baseline", "candidate"):
            arm_data = row[arm]
            plain = arm_data["plain_ticks"]
            if (
                len(plain) != (1 if canary else 3)
                or len(arm_data["digests"]) != (1 if canary else 3)
                or len(arm_data["stages"]) != (1 if canary else 3)
                or any(
                    not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in plain
                )
            ):
                raise ValueError("V8 timing samples malformed")
        for name, components in {
            "vector61_non_flood": ("vector61_other",),
            "b8_v8_bundle": ("b8_v8_features",),
            "opponent_v8_bundle": ("opponent_v8_bundle",),
            "v8_combined": ("b8_v8_features", "opponent_v8_bundle"),
        }.items():
            values = {}
            for arm in ("baseline", "candidate"):
                values[arm] = statistics.median(
                    sum(r["exclusive_cuda_elapsed_seconds"][k] for k in components)
                    for r in row[arm]["stages"]
                )
            derived = {
                **values,
                "speedup": (
                    values["baseline"] / values["candidate"] if values["candidate"] > 0 else None
                ),
            }
            if row["stage_seconds"][name] != derived:
                raise ValueError("V8 reported stage arithmetic differs")
        result = decide(row)
        if result == "INVALID":
            invalid.append(unit)
        elif result == "STOP_ROUTE":
            stops.append(unit)
        results.append(
            {
                "unit_id": unit,
                "outcome": result,
                "stage_seconds": row["stage_seconds"],
                "cpu_oracle": row["cpu_oracle"],
                "full_tick_speedup": row["full_tick_speedup"],
            }
        )
    verdict = (
        "INVALID"
        if invalid
        else "STOP_ROUTE" if stops else "INCOMPLETE" if set(found) != set(cells) else "PASS_PACKET"
    )
    return {
        "schema": "gpu-sim-spike-v8-analysis/v1",
        "commit": commit,
        "canary": canary,
        "cells": results,
        "verdict": verdict,
        "invalid_cells": invalid,
        "kill_cells": stops,
        "missing_units": [u for u in cells if u not in found],
        "cpu_closed_loop_identity": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--exports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()
    units.publish_json(args.output, aggregate(args.exports, args.commit, canary=args.canary))


if __name__ == "__main__":
    main()
