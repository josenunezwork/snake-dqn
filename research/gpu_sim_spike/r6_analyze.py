"""Pre-registered R6 E-scaling reducer (frozen before data; engineering evidence only).

Bars, exactly as round-36 R6 states them (``R6_PROTOCOL_20261024.md``):

* ``KEEP_E4096`` only if, on **every** roster, at equal completed work:
  (1) complete world-step throughput gain of E4096 over E1024 is at least 25%:
      ``median_rate(E4096) / median_rate(E1024) - 1 >= 0.25``;
  (2) mapped outputs are exact (every E4096 world equals E1024 world j mod 1024
      on every proof tick, zero mismatches);
  (3) measured physical-memory headroom at E4096 is at least 20%:
      ``1 - (context overhead + peak reserved) / device total >= 0.20``.
  Otherwise ``DROP_E4096``.
* ``INVALID``: native parity failure, a producer ``r6-failure.json``
  (nondeterministic equal-work blocks), or a foreign/duplicate unit.
* ``INCOMPLETE``: any OOM or missing cell (never a smaller-capacity pass).

Descriptive only (no bar): E2048 gain/exactness/headroom, every block rate,
tick times, peak memory.  Larger E never replaces the E1024 bar.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research.gpu_sim_spike import r6_units as units

GAIN = 0.25
HEADROOM = 0.20


def _json(path: Path) -> dict[str, Any]:
    units._verify(path)
    return json.loads(path.read_text())


def aggregate(exports: Path, commit: str, *, canary: bool = False) -> dict[str, Any]:
    root = Path(exports)
    cells = units.allocation(canary=canary)
    ladder = units.ladder(canary=canary)
    errors: list[str] = []
    if (root / "r6-failure.json").exists():
        errors.append("producer reported INVALID (nondeterministic block or failure)")
    native = _json(root / "native-parity.json") if (root / "native-parity.json").exists() else {}
    if native.get("status") != "PASS" or native.get("native_cuda_executed") is not True:
        errors.append("native MOVE2/geometry parity did not PASS on CUDA")
    if native and native.get("commit") != commit:
        errors.append("native parity commit differs")
    rows = units.load_ledger(
        root / "verified-ledger.json", units.namespace(commit, canary=canary), cells, exports=root
    )
    result: dict[str, Any] = {
        "schema": "gpu-sim-spike-r6-analysis/v1",
        "commit": commit,
        "mode": "canary" if canary else "main",
        "ladder": list(ladder),
        "bars": {"min_throughput_gain": GAIN, "min_memory_headroom": HEADROOM},
        "errors": errors,
        "claim": "engineering evidence only; larger E never replaces the E1024 bar",
    }
    if errors:
        result["verdict"] = "INVALID"
        return result
    missing = [unit for unit in cells if unit not in rows]
    oom = [unit for unit, content in rows.items() if content.get("status") == "OOM"]
    table: dict[str, dict[str, Any]] = {}
    for unit, content in rows.items():
        if content.get("status") != "MEASURED":
            continue
        mix, worlds = units.parse(unit)
        table.setdefault(mix, {})[worlds] = {
            "median_rate": content["median_complete_world_steps_per_second"],
            "rates": [b["complete_world_steps_per_second"] for b in content["blocks"]],
            "exact": content["mapped_proof"]["exact"],
            "mismatches": content["mapped_proof"]["mismatched_world_ticks"],
            "headroom": content["memory"].get("headroom_fraction"),
            "peak_reserved_bytes": content["memory"].get("peak_reserved_bytes"),
        }
    base, double, top = ladder
    gains: dict[str, dict[str, float]] = {}
    for mix, sizes in table.items():
        if base in sizes:
            gains[mix] = {
                str(worlds): sizes[worlds]["median_rate"] / sizes[base]["median_rate"] - 1.0
                for worlds in (double, top)
                if worlds in sizes
            }
    result.update(
        {
            "missing_units": missing,
            "oom_units": oom,
            "cells": table,
            "throughput_gain_over_base": gains,
        }
    )
    if missing or oom:
        result["verdict"] = "INCOMPLETE"
        return result
    if canary:
        result["verdict"] = "PASS_CANARY"
        return result
    keep = {}
    for mix in units.MIXES:
        cell = table[mix][top]
        keep[mix] = {
            "gain": gains[mix][str(top)] >= GAIN,
            "exact": cell["exact"] is True,
            "headroom": cell["headroom"] is not None and cell["headroom"] >= HEADROOM,
        }
    result["keep_checks_e4096"] = keep
    result["verdict"] = (
        "KEEP_E4096" if all(all(v.values()) for v in keep.values()) else "DROP_E4096"
    )
    result["descriptive_e2048"] = {
        mix: {
            "gain": gains[mix][str(double)],
            "exact": table[mix][double]["exact"],
            "headroom": table[mix][double]["headroom"],
        }
        for mix in units.MIXES
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    value = aggregate(args.exports, args.commit, canary=args.canary)
    units.publish_json(args.output, value)
    print(json.dumps({"verdict": value["verdict"]}))


if __name__ == "__main__":
    main()
