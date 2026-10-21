"""Matched, interleaved S0 whole-workload comparison for S2's D3 decision.

Excluded diagnostic on the frozen six S0 fixtures. Warm-up and coverage are excluded
from the three plain rounds per engine. One process, one CPU thread; scored roots
keep Python v8. An alarm bounds the entire invocation, including warm-up, to <= 1 h.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

from research.speed_s0_baseline import harness as h


def _deadline(signum: int, frame: Any) -> None:
    raise TimeoutError("S2 matched S0 comparison exceeded its wall deadline")


def benchmark(checkpoint: Path, output: Path, *, host_label: str) -> dict[str, Any]:
    """Run coverage then AB/BA/AB plain rounds, retaining all exports and hashes."""
    started = time.time()
    reference = h.load_reference(h.DEFAULT_REFERENCE)
    if h._sha256(checkpoint) != reference["checkpoint_sha256"]:
        raise ValueError("checkpoint differs from frozen S0 reference")
    if reference["fixture"] != h.fixture_identity():
        raise ValueError("fixture differs from frozen S0 reference")
    reports: dict[str, dict[str, Any]] = {engine: {"rounds": []} for engine in h.ENGINES}
    for engine in h.ENGINES:
        reports[engine]["warmup_seconds"] = h.warm_up(
            checkpoint, output / engine / "warmup", engine=engine
        )
        coverage = h.run_round(checkpoint, output / engine / "coverage", "coverage", engine=engine)
        reports[engine]["rounds"].append(coverage)
        h.write_completed(output / f"{engine}-coverage.json", coverage)
        print(f"coverage {engine}: {coverage['wall_seconds']:.3f}s", flush=True)
    order = []
    for repeat in range(3):
        engines = h.ENGINES if repeat % 2 == 0 else tuple(reversed(h.ENGINES))
        for engine in engines:
            load_before = list(os.getloadavg())
            result = h.run_round(
                checkpoint, output / engine / f"plain-{repeat}", "plain", engine=engine
            )
            result.update(load_before=load_before, load_after=list(os.getloadavg()))
            reports[engine]["rounds"].append(result)
            h.write_completed(output / f"{engine}-plain-{repeat}.json", result)
            order.append({"repeat": repeat, "engine": engine})
            print(f"plain {repeat} {engine}: {result['wall_seconds']:.3f}s", flush=True)
    summaries = {
        engine: h.summarize([report], reference["files"]) for engine, report in reports.items()
    }
    medians = {
        engine: summary["whole_workload"]["plain_round_wall_seconds_median"]
        for engine, summary in summaries.items()
    }
    identity = all(
        summary["identity"]["all_rounds_match_reference"]
        and summary["identity"]["all_rounds_self_identical"]
        for summary in summaries.values()
    )
    coverage_ok = all(
        all(summary["fixture_coverage"]["requirements_met"].values())
        for summary in summaries.values()
    )
    speedup = medians["stock"] / medians["native-v8"]
    return {
        "schema": "t34-speed-s2-matched-s0/v1",
        "authority": "excluded diagnostic; no study, gate, retention or serving qualification",
        "host_label": host_label,
        "source_commit": h._git_revision(),
        "argv": list(sys.argv),
        "hardware": h.runtime_hardware(),
        "started_unix": started,
        "elapsed_seconds": time.time() - started,
        "checkpoint_sha256": h._sha256(checkpoint),
        "reference_sha256": h._sha256(h.DEFAULT_REFERENCE),
        "harness_sha256": h._sha256(Path(h.__file__)),
        "benchmark_sha256": h._sha256(Path(__file__)),
        "engine_sources": {engine: h.engine_source_digest(engine=engine) for engine in h.ENGINES},
        "recipe": {
            "threads_per_process": 1,
            "processes": 1,
            "device": "cpu",
            "nice": os.getpriority(os.PRIO_PROCESS, 0),
            "round_order": order,
            "warmup_and_coverage_excluded": True,
            "roots": "Python",
        },
        "fixture": h.fixture_identity(),
        "reports": reports,
        "summaries": summaries,
        "median_wall_seconds": medians,
        "paired_speedups": [
            stock / native
            for stock, native in zip(
                summaries["stock"]["whole_workload"]["plain_round_wall_seconds"],
                summaries["native-v8"]["whole_workload"]["plain_round_wall_seconds"],
            )
        ],
        "whole_workload_speedup": speedup,
        "throughput_gain_fraction": speedup - 1.0,
        "wall_reduction_fraction": 1.0 - 1.0 / speedup,
        "all_exports_identical_to_stock_and_reference": identity,
        "fixture_coverage_met": coverage_ok,
        "d3_timing_and_s0_identity_pass": identity and coverage_ok and speedup >= 1.10,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--deadline-seconds", type=int, default=3600)
    args = parser.parse_args()
    if not 0 < args.deadline_seconds <= 3600:
        parser.error("deadline must be in (0, 3600]")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be fresh")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGALRM, _deadline)
    signal.alarm(args.deadline_seconds)
    try:
        h._cpu_ceiling()
        receipt = benchmark(args.checkpoint.resolve(), args.output_dir, host_label=args.host_label)
        h.write_completed(args.output_dir / "matched-s0.json", receipt)
        print(
            {
                key: receipt[key]
                for key in (
                    "median_wall_seconds",
                    "whole_workload_speedup",
                    "all_exports_identical_to_stock_and_reference",
                    "d3_timing_and_s0_identity_pass",
                )
            },
            flush=True,
        )
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
