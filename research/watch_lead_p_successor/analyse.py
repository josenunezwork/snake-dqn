"""Verify and summarize ONLY engineering profile exports and operational receipts."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from research.bl_served_search.exports import digest, verify_export
from research.watch_lead_p_successor.profile import concurrency


def analyse(output: Path) -> dict:
    """Refuse unsafe/partial files; never open scientific payloads."""
    profile = output / "exports/profile"
    files = sorted(profile.glob("*.json"))
    inventory = []
    for path in files:
        inventory.append(
            {"name": path.name, "size": path.stat().st_size, "sha256": verify_export(path)}
        )
    components = {p.name for p in profile.iterdir()}
    expected = {p.name + suffix for p in files for suffix in ("", ".sha256", ".complete")}
    if components != expected:
        raise ValueError("foreign or orphan profile component")
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    imported = {r["path"]: r for r in receipt["exports"]}
    for item in inventory:
        row = imported[f"profile/{item['name']}"]
        if row["sha256"] != item["sha256"] or row["size"] != item["size"]:
            raise ValueError("receipt/export mismatch")
    host = json.loads((profile / "host.json").read_text())
    cells = []
    identities = {}
    for label in ("arms", "contention"):
        for workers in (1, 4, 8, 11):
            path = profile / f"{label}-w{workers}.json"
            tasks = sorted(profile.glob(f"{label}-w{workers}-task*.json"))
            cell = {
                "label": label,
                "workers": workers,
                "complete": path.exists(),
                "task_exports": len(tasks),
            }
            if path.exists():
                value = json.loads(path.read_text())
                rows = value["rows"]
                for row in rows:
                    exported = json.loads(
                        (profile / f"{label}-w{workers}-task{row['task_index']}.json").read_text()
                    )
                    if {k: v for k, v in row.items() if k != "mean_concurrent_tasks"} != exported:
                        raise ValueError("sweep/task mismatch")
                check = [dict(r) for r in rows]
                concurrency(check)
                if any(
                    abs(a["mean_concurrent_tasks"] - b["mean_concurrent_tasks"]) > 1e-10
                    for a, b in zip(rows, check)
                ):
                    raise ValueError("incorrect concurrency")
                ids = [
                    (
                        r["seed"],
                        r["arm"],
                        r["frames"],
                        r["initial_state"],
                        r["final_state"],
                        r["trace_sha256"],
                    )
                    for r in rows
                ]
                if label in identities and identities[label] != ids:
                    raise ValueError("completed serial/parallel trajectory mismatch")
                identities[label] = ids
                samples = value["samples"]
                active = samples[-1]["child_cpu_seconds"] - samples[0]["child_cpu_seconds"]
                active /= samples[-1]["elapsed"] - samples[0]["elapsed"]
                cell.update(
                    wall_seconds=value["wall_seconds"],
                    pool_spawn_seconds=value["pool_spawn_seconds"],
                    active_cpu_cores=active,
                    overlap_mean=statistics.mean(r["mean_concurrent_tasks"] for r in rows),
                    arms={},
                )
                for arm in ("P0", "P48", "bare"):
                    chosen = [r for r in rows if r["arm"] == arm]
                    if not chosen:
                        continue
                    cell["arms"][arm] = {
                        "n": len(chosen),
                        "frames_each": sorted({r["frames"] for r in chosen}),
                        "seconds_per_frame_mean": statistics.mean(
                            r["seconds_per_frame"] for r in chosen
                        ),
                        "seconds_per_frame_max": max(r["seconds_per_frame"] for r in chosen),
                        "cpu_percent_mean": statistics.mean(
                            r["cpu_percent_one_core"] for r in chosen
                        ),
                        "search_calls": sum(r["search_calls"] for r in chosen),
                        "search_seconds": sum(r["search_seconds"] for r in chosen),
                    }
            else:
                cell["partial_rows"] = [json.loads(p.read_text()) for p in tasks]
            cells.append(cell)
    prefix = [json.loads(p.read_text()) for p in sorted(profile.glob("prefix-*.json"))]
    return {
        "output": str(output),
        "inventory": inventory,
        "json_exports": len(files),
        "receipt_sha256": digest(receipt_path),
        "receipt": {
            k: receipt[k]
            for k in (
                "status",
                "error",
                "stop_verified",
                "cleanup_error",
                "elapsed_seconds",
                "commit",
            )
        },
        "rate_units_per_hour": receipt["accounting"]["own_rate_units_per_hour"],
        "host": {
            **{k: host[k] for k in ("logical_cpus", "physical_cores", "affinity", "cgroup")},
            "model": sorted(set(re.findall(r"model name\s+: (.*)", host["cpuinfo"]))),
        },
        "cells": cells,
        "prefix": prefix,
        "completed_sweep_trajectory_identity": True,
        "envelope_qualified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a100", type=Path, required=True)
    parser.add_argument("--l4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"A100": analyse(args.a100), "L4": analyse(args.l4)}
    with args.output.open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                k: {"verified_json_exports": v["json_exports"], "envelope_qualified": False}
                for k, v in result.items()
            }
        )
    )


if __name__ == "__main__":
    main()
