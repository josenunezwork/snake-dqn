"""Mac-side reducer for complete MOVE exports; no partial outcome look."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from research.gpu_sim_spike import move_units


def aggregate(exports: list[Path], commit: str, *, canary: bool = False) -> dict:
    if platform.system() != "Darwin":
        raise ValueError("MOVE analysis is Mac-only")
    ns, cells = move_units.namespace(commit, canary=canary), move_units.allocation(canary=canary)
    seen = {}
    for root in exports:
        cpu = root / "cpu-parity.json"
        move_units._verify(cpu)
        proof = json.loads(cpu.read_text())
        if proof.get("commit") != commit:
            raise ValueError("foreign CPU proof")
        if proof.get("status") != "PASS":
            return {"verdict": "INVALID", "reason": "MOVE_CPU_GRID_MISMATCH", "cpu_proof": str(cpu)}
        ledger = root / "verified-ledger.json"
        rows = move_units.load_ledger(ledger, ns, cells, exports=root)
        if set(seen) & set(rows):
            raise ValueError("duplicate MOVE units across attempts")
        seen.update(rows)
    if set(seen) != set(cells):
        raise ValueError("incomplete MOVE matrix")
    killed = []
    for unit, row in seen.items():
        if not all(p["match"] and p["frozen_oracle"]["match"] for p in row["parity"]):
            killed.append(f"{unit}: mismatch")
        if row["worlds"] == 1024 and row["stage_speedup"] < 2:
            killed.append(f"{unit}: below 2x")
    return {
        "schema": "gpu-sim-spike-move-analysis/v1",
        "namespace": ns,
        "cells": [seen[x] for x in cells],
        "verdict": "KILL" if killed else "PASS",
        "kill_reasons": killed,
        "cpu_parity_scope": "recorded GridBatchSim seam only; DeviceSim counter-RNG is divergent",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exports", type=Path, action="append", required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--canary", action="store_true")
    a = p.parse_args()
    move_units.publish_json(a.output, aggregate(a.exports, a.commit, canary=a.canary))


if __name__ == "__main__":
    main()
