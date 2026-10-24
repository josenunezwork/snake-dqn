"""Mac-only reducer that keeps benchmark identity, CPU diagnostics, and native status separate."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
from pathlib import Path
from typing import Any

from research.gpu_sim_spike import move2_pack, move2_units


def valid_cell(unit_id: str, row: dict[str, Any]) -> bool:
    """Validate the complete frozen ruler payload before interpreting its speed."""
    mix, worlds = unit_id.split("-")[1], int(unit_id.rsplit("e", 1)[1])
    if (
        row.get("mix") != mix
        or row.get("worlds") != worlds
        or row.get("warmed_interleaved_repeats") != 3
        or row.get("warmups") != 3
        or row.get("capacity") != 10400
        or row.get("lengths") != [800, 900, 1000, 1100, 1300, 1500]
    ):
        return False
    reference, indexed, proofs, inputs = (
        row.get("reference_seconds"),
        row.get("indexed_seconds"),
        row.get("frozen_device_identity"),
        row.get("recorded_inputs"),
    )
    if not all(
        isinstance(value, list) and len(value) == 3
        for value in (reference, indexed, proofs, inputs)
    ):
        return False
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
        for value in reference + indexed
    ):
        return False
    if [proof.get("repeat") if isinstance(proof, dict) else None for proof in proofs] != [0, 1, 2]:
        return False
    if [item.get("repeat") if isinstance(item, dict) else None for item in inputs] != [0, 1, 2]:
        return False
    if any(
        not isinstance(item.get("input_sha256"), str) or len(item["input_sha256"]) != 64
        for item in inputs
    ):
        return False
    if not all(
        proof.get("candidate_vs_reference", {}).get("match") is True
        and proof.get("candidate_vs_frozen", {}).get("match") is True
        for proof in proofs
    ):
        return False
    expected = statistics.median(reference) / statistics.median(indexed)
    return isinstance(row.get("speedup"), (int, float)) and row["speedup"] == expected


def aggregate(exports: list[Path], commit: str, *, canary: bool = False) -> dict[str, Any]:
    if platform.system() != "Darwin":
        raise ValueError("MOVE2 analysis is Mac-only")
    ns, cells, seen, diagnostics, native = (
        move2_units.namespace(commit, canary=canary),
        move2_units.allocation(canary=canary),
        {},
        [],
        [],
    )
    for root in exports:
        for filename, status in (
            ("cpu-discrepancies.json", "DIAGNOSTIC"),
            ("native-parity.json", "PASS"),
        ):
            path = root / filename
            move2_units._verify(path)
            body = json.loads(path.read_text())
            if body.get("commit") != commit:
                raise ValueError(f"foreign MOVE2 {filename}")
            bad_native = filename == "native-parity.json" and not move2_pack.native_report_valid(
                body
            )
            if body.get("status") != status or bad_native:
                incomplete = body.get("status") in {"NOT_RUN", "INCOMPLETE"}
                return {
                    "schema": "gpu-sim-spike-move2-analysis/v1",
                    "commit": commit,
                    "verdict": "INCOMPLETE" if incomplete else "INVALID",
                    "reason": f"{filename}:{body.get('status')}",
                    "optimization_identity": "INCOMPLETE",
                    "cpu_discrepancy_classification": (
                        body.get("status") if filename.startswith("cpu") else "DIAGNOSTIC"
                    ),
                    "native_a100_parity": body.get("status") if bad_native else "NOT_EXAMINED",
                    "proof_path": str(path),
                }
            (diagnostics if filename.startswith("cpu") else native).append(
                {"path": str(path), "sha256": move2_units._verify(path), "status": body["status"]}
            )
        rows = move2_units.load_ledger(root / "verified-ledger.json", ns, cells, exports=root)
        if set(seen) & set(rows):
            raise ValueError("duplicate MOVE2 unit across attempts")
        seen.update(rows)
    invalid, stops, output = [], [], []
    for unit in cells:
        if unit not in seen:
            continue
        row = seen[unit]
        identity = valid_cell(unit, row)
        if not identity:
            invalid.append(unit)
        elif row.get("worlds") == 1024 and row.get("speedup", 0) < 2:
            stops.append(unit)
        output.append(
            {
                "unit_id": unit,
                "identity": identity,
                "speedup": row.get("speedup"),
                "outcome": (
                    "INVALID" if unit in invalid else "STOP_ROUTE" if unit in stops else "PASS"
                ),
            }
        )
    # A missing cell prevents a timing verdict.  Invalid data still has priority.
    verdict = (
        "INVALID"
        if invalid
        else "INCOMPLETE" if set(seen) != set(cells) else "STOP_ROUTE" if stops else "PASS"
    )
    return {
        "schema": "gpu-sim-spike-move2-analysis/v1",
        "commit": commit,
        "namespace": ns,
        "canary": canary,
        "verdict": verdict,
        "optimization_identity": (
            "INVALID" if invalid else "PASS" if set(seen) == set(cells) else "INCOMPLETE"
        ),
        "cpu_discrepancy_classification": "DIAGNOSTIC" if diagnostics else "INCOMPLETE",
        "native_a100_parity": "PASS" if native else "INCOMPLETE",
        "cells": output,
        "missing_units": [u for u in cells if u not in seen],
        "invalid_cells": invalid,
        "stop_cells": stops,
        "diagnostic_reports": diagnostics,
        "native_reports": native,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exports", type=Path, action="append", required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--canary", action="store_true")
    a = p.parse_args()
    move2_units.publish_json(a.output, aggregate(a.exports, a.commit, canary=a.canary))


if __name__ == "__main__":
    main()
