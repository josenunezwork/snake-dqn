"""Compare immutable matched Mac shadow-batching receipts and export identities."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def compare(serial_path: Path, batched_path: Path) -> dict[str, Any]:
    """Compare only matched complete arms; never turn a partial run into parity."""
    serial, batched = [json.loads(path.read_text()) for path in (serial_path, batched_path)]
    if serial["implementation"] != "serial" or batched["implementation"] != "batched":
        raise ValueError("expected serial and batched profile receipts, in that order")
    if serial["shape"] != batched["shape"] or serial["threads"] != batched["threads"]:
        raise ValueError("profiles must have matched shape and thread limits")
    if serial["platform"] != batched["platform"]:
        raise ValueError("profiles must run on the same platform")
    if not serial["result"]["complete"] or not batched["result"]["complete"]:
        raise ValueError("partial profiles cannot establish matched parity or speedup")
    for arm in (serial, batched):
        if not arm.get("source_unchanged"):
            raise ValueError("profile sources must remain unchanged during measurement")
        if arm.get("warmup_worlds") != arm["shape"]["worlds"] or arm.get("warmup_mixes") != [
            "scripted",
            "frozen",
            "mixed",
        ]:
            raise ValueError("profiles require matched full-width, all-mix warm-up")
    replay = batched.get("replay", [])
    payloads = {key for key in batched["exports"] if key.endswith(".json")}
    if (
        not batched.get("replay_requested")
        or {row["world"] for row in replay} != payloads
        or len(replay) != len(payloads)
        or len(payloads) != batched["shape"]["worlds"]
    ):
        raise ValueError("candidate must replay every exported world exactly once")
    exports = (serial["exports"], batched["exports"])
    mismatches = sorted(
        key
        for key in set(exports[0]) | set(exports[1])
        if exports[0].get(key) != exports[1].get(key)
    )
    rates = [
        arm["result"]["active_row_steps"] / arm["elapsed_seconds"] for arm in (serial, batched)
    ]
    return {
        "schema": "t34-shadow-batching-comparison/v1",
        "receipts": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (serial_path, batched_path)
        },
        "shape": serial["shape"],
        "threads": serial["threads"],
        "platform": serial["platform"],
        "export_files": len(exports[0]),
        "export_byte_mismatches": mismatches,
        "serial_row_steps_per_second": rates[0],
        "batched_row_steps_per_second": rates[1],
        "speedup": serial["elapsed_seconds"] / batched["elapsed_seconds"],
        "same_work": serial["result"]["active_row_steps"] == batched["result"]["active_row_steps"],
        "serial_host_stage_seconds": serial["result"]["host_stage_seconds"],
        "batched_host_stage_seconds": batched["result"]["host_stage_seconds"],
        "replay_worlds": len(batched.get("replay", [])),
        "replay_mismatches": [r for r in batched.get("replay", []) if not r["accepted"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", type=Path, required=True)
    parser.add_argument("--batched", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.serial, args.batched)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["export_byte_mismatches"] or not result["same_work"] or result["replay_mismatches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
