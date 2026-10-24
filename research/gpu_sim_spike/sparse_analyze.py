"""Mac-only final reducer for immutable sparse vector61 benchmark units."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import tempfile
from pathlib import Path
from typing import Any

from research.gpu_sim_spike import sparse_units


def aggregate(exports: list[Path], commit: str, *, canary: bool = False) -> dict[str, Any]:
    """Verify exactly one immutable unit per roster and take the registered look."""
    ns = sparse_units.namespace(commit, canary=canary)
    allocation = sparse_units.allocation(canary=canary)
    cells: dict[str, dict[str, Any]] = {}
    evidence: list[dict[str, Any]] = []
    for root in exports:
        with tempfile.TemporaryDirectory(prefix="sparse-ledger-") as temporary:
            ledger = Path(temporary) / "verified-ledger.json"
            sparse_units.build_ledger(root, ledger, commit=commit, canary=canary)
            verified = sparse_units.load_ledger(ledger, ns, allocation)
            if set(cells) & set(verified):
                raise ValueError("duplicate sparse units across attempts")
            for unit_id, value in verified.items():
                if unit_id != f"cell-{value.get('mix')}-e{value.get('worlds')}" or (
                    value.get("worlds") != 1024
                ):
                    raise ValueError("sparse cell differs from its frozen allocation")
                fixture = value.get("static", {}).get("feature_oracle", {}).get("recorded_fixture")
                if not isinstance(fixture, dict):
                    raise ValueError("static unit lacks a recorded-state fixture")
                expected = f"recorded-states/{value['mix']}-static-e{value['worlds']}.npz"
                if fixture.get("path") != expected:
                    raise ValueError("recorded-state fixture path is foreign")
                path = Path(root) / expected
                if (
                    not path.is_file()
                    or sparse_units._verify(path) != fixture.get("sha256")
                    or path.stat().st_size != fixture.get("size")
                ):
                    raise ValueError("recorded-state fixture hash differs")
                cells[unit_id] = value
            evidence.extend(json.loads(ledger.read_text())["units"])
    complete = set(cells) == set(allocation)
    verdict = "CANARY_DIAGNOSTIC" if canary else "PASS_PENDING_ENGINE_REVIEW"
    reasons = []
    for unit_id in allocation:
        if unit_id not in cells:
            continue
        cell = cells[unit_id]
        if cell.get("verdict") == "INVALID_FEATURE_MISMATCH" or not cell.get("static", {}).get(
            "feature_oracle", {}
        ).get("match"):
            verdict, reasons = "INVALID", [unit_id, "static full61 CPU byte mismatch"]
            break
        medians = {}
        for backend in ("dense", "static"):
            samples = cell.get(backend, {}).get("samples", [])
            if len(samples) != (1 if canary else 3):
                raise ValueError("sparse measurement repeat count differs")
            ticks = [row.get("tick_seconds") for row in samples]
            if any(
                isinstance(tick, bool)
                or not isinstance(tick, (int, float))
                or not math.isfinite(tick)
                or tick <= 0
                for tick in ticks
            ):
                raise ValueError("invalid sparse tick timing")
            medians[backend] = statistics.median(ticks)
        speedup = medians["dense"] / medians["static"]
        cell["mac_recomputed_speedup"] = speedup
        if not canary and speedup < 2.0:
            verdict, reasons = "STOP_ROUTE", [unit_id, "E1024 full-tick speedup below 2x"]
            break
    if not complete and verdict not in {"INVALID", "STOP_ROUTE"}:
        raise ValueError("sparse union is incomplete")
    return {
        "schema": "gpu-sim-spike-sparse-analysis/v1",
        "namespace": ns,
        "cells": [cells[item] for item in allocation if item in cells],
        "complete_allocation": complete,
        "verified_evidence": evidence,
        "verdict": verdict,
        "reasons": reasons,
        "claim": "Mac-side engineering reducer only; no gate, serving, or release authority",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, action="append", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Darwin":
        raise ValueError("sparse reducer is Mac-only")
    sparse_units.publish_json(args.output, aggregate(args.exports, args.commit, canary=args.canary))
    print(args.output)


if __name__ == "__main__":
    main()
