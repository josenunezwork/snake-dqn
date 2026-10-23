"""Take A1's one Mac-side look after the entire frozen matrix is verified."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from research.gpu_sim_spike import a1_units, a100_ruler


def analyze(exports: Path, *, commit: str, hardware: str) -> dict[str, Any]:
    """Refuse missing/foreign units before reading any measured cell outcomes."""
    with tempfile.TemporaryDirectory(prefix="a1-mac-ledger-") as directory:
        ledger = Path(directory) / "verified-ledger.json"
        manifest = a1_units.build_ledger(
            exports, ledger, commit=commit, hardware=hardware, smoke=False
        )
        allocated = a1_units.allocation(False)
        if {row["unit_id"] for row in manifest["units"]} != set(allocated):
            raise ValueError("INCOMPLETE: no look until all eleven frozen A1 units are verified")
        values = a1_units.load_ledger(
            ledger, a1_units.namespace(commit, hardware, False), allocated
        )
    rows, traces = [], []
    for mix in a100_ruler.MIXES:
        for worlds in a100_ruler.SHAPES:
            key = f"cell-{mix.removeprefix('long-')}-e{worlds}"
            measured, trace = values[key]["measured"], values[key]["trace"]
            for cell in (measured, trace):
                if (
                    cell["mix"] != mix
                    or cell["requested_worlds"] != worlds
                    or cell["actual_concurrent_worlds"] != worlds
                    or cell["bodies"] != [800, 900, 1000, 1100, 1300, 1500]
                    or cell["repaired_event_path"] is not True
                ):
                    raise ValueError("A1 cell differs from its frozen allocation")
            if (measured["warmed_repeats"], measured["ticks_per_repeat"]) != (3, 8):
                raise ValueError("A1 measured cell lacks three repeats x eight complete ticks")
            if (trace["warmed_repeats"], trace["ticks_per_repeat"]) != (1, 1):
                raise ValueError("A1 cell lacks its independent instrumented pass")
            rows.append(measured)
            traces.append(trace)
    control = values["control-e6"]
    if (
        control["requested_worlds"] != 6
        or control["repaired_event_path"] is not False
        or control["mix"] != a100_ruler.MIXES[0]
    ):
        raise ValueError("A1 unrepaired E6 control is foreign")
    parity = values["parity-tape"]["parity"]
    return {
        "schema": "gpu-sim-spike-a1-mac-decision/v1",
        "namespace": a1_units.namespace(commit, hardware, False),
        "verified_units": manifest["units"],
        "source_commit": commit,
        "hardware": hardware,
        "kill_tier": a100_ruler.classify(rows, control, parity),
        "repaired_cells": rows,
        "instrumented_pass": traces,
        "unrepaired_e6_control": control,
        "parity_self_check": parity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--hardware", default="A100-40GB")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Mac decision already exists; never overwrite a look")
    result = analyze(args.exports, commit=args.commit, hardware=args.hardware)
    a1_units.publish_json(args.output, result)
    print(json.dumps({"output": str(args.output), "kill_tier": result["kill_tier"]}))


if __name__ == "__main__":
    main()
