"""Mac-only descriptive union of A1-PROF units; no outcome labels or gate evidence."""

from __future__ import annotations

import argparse
import platform
import statistics
import tempfile
from pathlib import Path
from typing import Any

from research.gpu_sim_spike import a1_prof_units as units


def aggregate(
    exports: list[Path], commit: str, smoke: bool = False, canary: bool = False
) -> dict[str, Any]:
    """Verify complete exported bytes; refuse foreign identities and duplicate cells."""
    namespace, allocation = units.namespace(commit, smoke, canary), units.allocation(smoke, canary)
    cells: dict[str, Any] = {}
    cell_roots: dict[str, Path] = {}
    evidence = []
    for root in exports:
        with tempfile.TemporaryDirectory(prefix="a1-prof-ledger-") as temporary:
            ledger = Path(temporary) / "verified-ledger.json"
            value = units.build_ledger(root, ledger, commit=commit, smoke=smoke, canary=canary)
            verified = units.load_ledger(ledger, namespace, allocation)
            if set(cells) & set(verified):
                raise ValueError("duplicate A1-PROF units across attempts")
            cells.update(verified)
            cell_roots.update({unit_id: root for unit_id in verified})
            evidence.extend(value["units"])
    if set(cells) != set(allocation):
        raise ValueError("A1-PROF union is incomplete; no partial measurement look")
    rows = []
    for unit_id in allocation:
        cell = cells[unit_id]
        traces = cell.get("chrome_traces", [])
        if not isinstance(traces, list) or len(traces) != 1:
            raise ValueError("A1-PROF unit lacks its single immutable Chrome trace")
        for trace in traces:
            if not isinstance(trace, dict) or not isinstance(trace.get("path"), str):
                raise ValueError("A1-PROF Chrome trace reference is malformed")
            expected = f"a1-prof-traces/{unit_id}-trace.json.gz"
            if trace["path"] != expected:
                raise ValueError("A1-PROF Chrome trace path is foreign")
            path = cell_roots[unit_id] / expected
            if not path.is_file():
                raise ValueError("A1-PROF Chrome trace is absent from its source export")
            try:
                digest = units._verify_sidecars(path)
            except ValueError as exc:
                raise ValueError("A1-PROF Chrome trace sidecars are invalid") from exc
            if digest != trace.get("sha256") or path.stat().st_size != trace.get("bytes"):
                raise ValueError("A1-PROF Chrome trace bytes differ from its unit reference")
        profile_means = {
            stage: {
                field: statistics.mean(
                    tick["stages"][stage][field] for tick in cell["profiler_ticks"]
                )
                for field in cell["profiler_ticks"][0]["stages"][stage]
            }
            for stage in cell["profiler_ticks"][0]["stages"]
        }
        rows.append(
            {
                "unit_id": unit_id,
                "mix": cell["mix"],
                "worlds": cell["worlds"],
                "metrics": cell["metrics"],
                "profiler_stage_means_per_tick": profile_means,
                "memory": cell["memory"],
                "peak_memory": cell["peak_memory"],
                "profiler_ticks": cell["profiler_ticks"],
                "native_launch_event_ticks": cell.get("native_launch_event_ticks", []),
                "chrome_traces": traces,
            }
        )
    return {
        "schema": "gpu-sim-spike-a1-prof-analysis/v1",
        "namespace": namespace,
        "claim": "Descriptive engineering timings only; no labels, gate or release evidence",
        "cells": rows,
        "verified_evidence": evidence,
        "round24_reference_seconds": 0.512,
        "measurement_notes": [
            (
                "World-steps/s uses uninstrumented synchronized full ticks, including "
                "resident reset/collection."
            ),
            (
                "Host stage timers are exclusive; CUDA events measure exclusive stream "
                "elapsed intervals."
            ),
            (
                "CUDA event elapsed includes starvation; kernel-busy time comes from a "
                "separate profiler pass."
            ),
            (
                "Do not add CPU launch/Python times to GPU time; these overlap on the "
                "host/device timeline."
            ),
            (
                "Flood share and remainder subtract CUDA event intervals from tick wall; "
                "instrumentation overhead ratio reported."
            ),
            (
                "Bodies are the same fixed long-body A1 ruler fixture, reset every tick; no "
                "natural-state claim."
            ),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, action="append", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Darwin":
        raise ValueError("A1-PROF measurement union is Mac-only")
    units.publish_json(args.output, aggregate(args.exports, args.commit, args.smoke, args.canary))
    print(str(args.output), flush=True)


if __name__ == "__main__":
    main()
