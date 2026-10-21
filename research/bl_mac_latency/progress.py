"""Read-only occupancy snapshots during an already-authorized measurement."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from research.bl_mac_latency.host import snapshot
from research.bl_served_search.exports import export_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--measurement-pid", required=True, type=int)
    args = parser.parse_args()
    value = snapshot()
    command = subprocess.run(
        ["ps", "-p", str(args.measurement_pid), "-o", "command="],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    ).stdout.strip()
    verified = "research.bl_mac_latency." in command
    owned = {args.measurement_pid} if verified else set()
    parent_pairs = []
    for line in value["processes"].splitlines()[1:]:
        fields = line.split(None, 3)
        if len(fields) == 4:
            parent_pairs.append((int(fields[0]), int(fields[1])))
    while True:
        expanded = owned | {pid for pid, parent in parent_pairs if parent in owned}
        if expanded == owned:
            break
        owned = expanded
    value["verified_measurement_tree"] = sorted(owned)
    value["verified_measurement_pid"] = args.measurement_pid if verified else None
    value["measurement_command"] = command
    value["all_heavy_including_measurement"] = value["heavy_processes"]
    value["other_heavy_processes"] = [
        row for row in value["heavy_processes"] if row["pid"] not in owned
    ]
    value["quiet"] = not value["other_heavy_processes"]
    value["status"] = "QUIET_SAMPLED" if value["quiet"] else "CONTENDED"
    batches = list((args.root / "y1-mac-01" / "batches").glob("*.json"))
    value["completed_y1_batches"] = len(batches)
    export_json(args.root / "root-occupancy" / f"{value['monotonic_ns']}.json", value)
    print(
        json.dumps(
            {
                k: value[k]
                for k in (
                    "time",
                    "status",
                    "completed_y1_batches",
                    "load_average",
                    "other_heavy_processes",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
