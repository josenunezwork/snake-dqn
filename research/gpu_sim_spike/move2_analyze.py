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
    ns = move2_units.namespace(commit, canary=canary)
    cells = move2_units.allocation(canary=canary)
    seen: dict[str, dict[str, Any]] = {}
    diagnostics, native, failures = [], [], []
    for root in exports:
        for filename, expected in (
            ("cpu-discrepancies.json", "DIAGNOSTIC"),
            ("native-parity.json", "PASS"),
        ):
            path = root / filename
            if not all(
                Path(str(path) + suffix).is_file() for suffix in ("", ".sha256", ".complete")
            ):
                failures.append(
                    {"status": "INCOMPLETE", "path": str(path), "reason": "missing proof"}
                )
                continue
            digest = move2_units._verify(path)
            body = json.loads(path.read_text())
            if body.get("commit") != commit:
                raise ValueError(f"foreign MOVE2 {filename}")
            status = body.get("status")
            bad_native = filename == "native-parity.json" and not move2_pack.native_report_valid(
                body
            )
            if status != expected or bad_native:
                failures.append(
                    {
                        "status": (
                            "INCOMPLETE" if status in {"NOT_RUN", "INCOMPLETE"} else "INVALID"
                        ),
                        "path": str(path),
                        "reason": f"proof status {status}",
                    }
                )
            (diagnostics if filename.startswith("cpu") else native).append(
                {"path": str(path), "sha256": digest, "status": status}
            )
        ledger = root / "verified-ledger.json"
        if not ledger.is_file():
            failures.append(
                {"status": "INCOMPLETE", "path": str(ledger), "reason": "missing ledger"}
            )
            continue
        rows = move2_units.load_ledger(ledger, ns, cells, exports=root)
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
        elif row["worlds"] == 1024 and row["speedup"] < 2:
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
    # Inspect every supplied attempt before applying precedence. An early missing
    # native proof must never hide a later independently recorded mismatch.
    has_invalid = bool(invalid) or any(row["status"] == "INVALID" for row in failures)
    incomplete = bool(failures) or set(seen) != set(cells)
    verdict = (
        "INVALID"
        if has_invalid
        else "INCOMPLETE" if incomplete else "STOP_ROUTE" if stops else "PASS"
    )
    native_statuses = [row["status"] for row in native]
    native_status = (
        "INVALID"
        if any(row["status"] == "INVALID" and "native-parity" in row["path"] for row in failures)
        else (
            "PASS"
            if native_statuses and all(status == "PASS" for status in native_statuses)
            else (
                "NOT_RUN"
                if native_statuses and all(status == "NOT_RUN" for status in native_statuses)
                else "INCOMPLETE"
            )
        )
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
        "cpu_discrepancy_classification": (
            "INVALID"
            if any(row["status"] == "INVALID" for row in diagnostics)
            else (
                "DIAGNOSTIC"
                if diagnostics and all(row["status"] == "DIAGNOSTIC" for row in diagnostics)
                else "INCOMPLETE"
            )
        ),
        "native_a100_parity": native_status,
        "cells": output,
        "missing_units": [unit for unit in cells if unit not in seen],
        "invalid_cells": invalid,
        "stop_cells": stops,
        "proof_failures": failures,
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
