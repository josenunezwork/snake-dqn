"""Independent Mac I0 reducer: coverage/identity precede measured speed."""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import statistics
from pathlib import Path
from typing import Any

from research.gpu_sim_spike import integration_units as units
from research.gpu_sim_spike.benchmark import source_hashes

HEX = re.compile(r"[0-9a-f]{64}")
ORDERS = [["R", "I"], ["I", "R"], ["R", "I"]]


def positive(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def native_valid(report: dict, commit: str, *, smoke: bool = False) -> bool:
    if report.get("commit") != commit or report.get("source_sha256") != source_hashes():
        return False
    if smoke:
        return report.get("status") == "PREFLIGHT" and report.get("native_cuda_executed") is False
    calls = report.get("geometry_native_calls", {})
    move = report.get("move2_events", {})
    rows = report.get("rows", [])
    return (
        report.get("status") == "PASS"
        and report.get("native_cuda_executed") is True
        and report.get("fallback") is False
        and len(rows) == 14
        and all(row.get("outcome") == "passed" for row in rows)
        and set(calls) == {"indexed_body_view", "indexed_features58"}
        and all(type(v) is int and v > 0 for v in calls.values())
        and move.get("status") == "PASS"
        and move.get("fallback") is False
        and bool(move.get("native_calls"))
        and all(type(v) is int and v > 0 for v in move["native_calls"].values())
        and set(report.get("compiled_binaries", {})) == {"geometry", "move", "events", "flood"}
    )


def continuous_valid(row: dict) -> bool:
    packet = row.get("continuous", {})
    if (
        packet.get("status") != "COMPLETE"
        or packet.get("orders") != ORDERS
        or packet.get("ticks") != 32
    ):
        return False
    for arm in ("R", "I"):
        repeats = packet.get("arms", {}).get(arm, {}).get("repeats", [])
        if len(repeats) != 3:
            return False
        digests = []
        for index, item in enumerate(repeats):
            wall, steps = item.get("wall_seconds"), item.get("completed_world_steps")
            rate, digest = item.get("complete_world_steps_per_second"), item.get("final_digest")
            ticks = item.get("tick_seconds", [])
            if (
                item.get("repeat") != index
                or not positive(wall)
                or steps != 32 * 1024
                or not positive(rate)
                or rate != steps / wall
                or not isinstance(digest, str)
                or not HEX.fullmatch(digest)
                or len(ticks) != 32
                or not all(positive(v) for v in ticks)
                or sum(ticks) > wall * 1.01
                or item.get("collector_positions") != list(range(32))
            ):
                return False
            digests.append(digest)
        if len(set(digests)) != 1:
            return False
    return True


def valid_cell(
    unit: str, row: dict[str, Any], *, canary: bool = False, smoke: bool = False
) -> bool:
    try:
        mix, worlds = unit.split("-")[1], int(unit.rsplit("e", 1)[1])
        n = 1 if canary or smoke else 3
        if (
            row.get("mix") != mix
            or row.get("worlds") != worlds
            or row.get("repeats") != n
            or row.get("warmups") != n
            or row.get("orders") != ORDERS[:n]
            or row.get("capacity") != 10400
            or row.get("lengths") != [800, 900, 1000, 1100, 1300, 1500]
        ):
            return False
        for arm in ("R", "I"):
            value = row.get("arms", {}).get(arm, {})
            samples, digests, stages = (
                value.get(key, []) for key in ("plain_seconds", "digests", "stages")
            )
            if (
                len(samples) != n
                or not all(positive(v) for v in samples)
                or len(digests) != n
                or any(not isinstance(v, str) or not HEX.fullmatch(v) for v in digests)
                or len(set(digests)) != 1
                or len(stages) != n
            ):
                return False
            for index, stage in enumerate(stages):
                outer = stage.get("wall_seconds")
                clock = (
                    stage.get("exclusive_host_seconds")
                    if smoke
                    else stage.get("exclusive_cuda_elapsed_seconds")
                )
                if (
                    stage.get("repeat") != index
                    or not positive(outer)
                    or not isinstance(clock, dict)
                    or not clock
                    or any(
                        type(v) not in (int, float) or not math.isfinite(v) or v < 0
                        for v in clock.values()
                    )
                    or abs(sum(clock.values()) - outer) > outer * 0.10
                ):
                    return False
        proof = row.get("combined_parity", {})
        features, move = proof.get("feature_calls", []), proof.get("frozen_move", {})
        if (
            proof.get("frozen_device_identity") is not True
            or proof.get("cpu_geometry_carry") is not True
            or set(move) != {"R", "I"}
            or any(v.get("match") is not True for v in move.values())
            or not features
            or not any(v.get("arm") == "I" for v in features)
            or any(v.get("cpu_match") is not True for v in features if v.get("arm") == "I")
            or set(proof.get("state_inventory", {})) != {"R", "I"}
            or proof.get("action_tape", {}).get("shape") != [worlds, 6]
        ):
            return False
        if worlds == 1024 and not (canary or smoke) and not continuous_valid(row):
            return False
        return (
            row.get("native_parity") == ("PREFLIGHT" if smoke else "PASS")
            and row.get("frozen_device_identity") is True
            and row.get("cpu_geometry_carry") is True
        )
    except (AttributeError, IndexError, TypeError, ValueError):
        return False


def aggregate(
    exports: list[Path], commit: str, *, canary: bool = False, smoke: bool = False
) -> dict[str, Any]:
    if platform.system() != "Darwin":
        raise ValueError("authoritative I0 aggregation is Mac-only")
    cells, ns = units.allocation(canary=canary, smoke=smoke), units.namespace(
        commit, canary=canary, smoke=smoke
    )
    seen, errors, invalid = {}, [], False
    native = []
    for root in map(Path, exports):
        try:
            path = root / "native-parity.json"
            units._verify(path)
            report = json.loads(path.read_text())
            invalid |= report.get("status") == "INVALID"
            if not native_valid(report, commit, smoke=smoke):
                raise ValueError("native report identity or executed proof differs")
            native.append(report["status"])
            register = root / "discrepancy-register.json"
            units._verify(register)
            body = json.loads(register.read_text())
            if (
                body.get("commit") != commit
                or body.get("cpu_component_register", {}).get("status") != "DIAGNOSTIC"
            ):
                raise ValueError("discrepancy classification missing")
            rows = units.load_ledger(root / "verified-ledger.json", ns, cells, exports=root)
            if set(seen) & set(rows):
                raise ValueError("duplicate I0 unit across attempts")
            for unit, row in rows.items():
                trace = row.get("trace", {})
                if trace.get("status") == "PASS":
                    relative = Path(trace.get("path", ""))
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("foreign trace path")
                    target = root / relative
                    if units._verify(target) != trace.get(
                        "sha256"
                    ) or target.stat().st_size != trace.get("size"):
                        raise ValueError("trace bytes differ")
            seen.update(rows)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(str(exc))
    missing = [unit for unit in cells if unit not in seen]
    malformed = [
        unit for unit, row in seen.items() if not valid_cell(unit, row, canary=canary, smoke=smoke)
    ]
    rates = {
        row["mix"]: 1024 / statistics.median(row["arms"]["I"]["plain_seconds"])
        for unit, row in seen.items()
        if row.get("worlds") == 1024 and valid_cell(unit, row, canary=canary, smoke=smoke)
    }
    continuous_rates = {
        row["mix"]: [
            v["complete_world_steps_per_second"] for v in row["continuous"]["arms"]["I"]["repeats"]
        ]
        for unit, row in seen.items()
        if row.get("worlds") == 1024 and unit not in malformed
    }
    if invalid:
        verdict = "INVALID"
    elif errors or missing or malformed or not native:
        verdict = "INCOMPLETE"
    elif smoke:
        verdict = "PREFLIGHT"
    elif canary:
        verdict = "PASS_CANARY"
    elif (
        len(rates) == 3
        and all(v >= 2000 for v in rates.values())
        and all(v >= 2000 for values in continuous_rates.values() for v in values)
    ):
        verdict = "PASS_SPEED"
    else:
        verdict = "STOP_SPEED"
    return {
        "schema": "gpu-sim-spike-i0-analysis/v1",
        "commit": commit,
        "verdict": verdict,
        "native_a100_parity": native,
        "missing_units": missing,
        "malformed_units": malformed,
        "ingestion_errors": errors,
        "e1024_complete_world_steps_per_second": rates,
        "continuous_complete_world_steps_per_second": continuous_rates,
        "bar_world_steps_per_second": 2000,
        "cells": [seen[k] for k in cells if k in seen],
        "cpu_engine_exactness": "UNPROVED",
        "cpu_closed_loop_identity": "UNPROVED",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exports", type=Path, action="append", required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--canary", action="store_true")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    units.publish_json(a.output, aggregate(a.exports, a.commit, canary=a.canary, smoke=a.smoke))


if __name__ == "__main__":
    main()
