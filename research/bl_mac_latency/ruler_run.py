"""Guarded Mac execution of the unchanged frozen Y1 operator."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from research.bl_mac_latency.host import GuardError, check_guard, process_screen, snapshot
from research.bl_served_search import ruler
from research.bl_served_search.exports import digest, export_json, verify_export
from research.bl_served_search.ruler_operator import ExactT3Operator


def distribution(samples: list[float]) -> dict[str, Any]:
    """Nearest-rank quantiles, preserving cold samples and the exact population."""
    values = sorted(samples)
    if not values:
        return {"n": 0, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "n": len(values),
        **{f"p{p}": values[math.ceil(p / 100 * len(values)) - 1] for p in (50, 95, 99)},
        "max": values[-1],
    }


class GuardedOperator(ExactT3Operator):
    """Check safety between dense batches without changing simulator/operator math."""

    def __init__(self, out: Path, deadline: float, require_quiet: bool = False) -> None:
        super().__init__()
        self.out = out
        self.deadline = deadline
        self.require_quiet = require_quiet
        self.guards: list[dict[str, Any]] = []
        self.completed = 0
        self.samples: dict[str, list[float]] = {}

    def batch(self, fixture: dict[str, Any]) -> dict[str, Any]:
        if time.monotonic() >= self.deadline:
            raise RuntimeError("Mac measurement budget exhausted")
        guard = check_guard()
        if self.require_quiet:
            guard["process_screen"] = process_screen()
            if guard["process_screen"]["other_heavy_processes"]:
                guard["reasons"] = ["foreign_cpu_load_not_quiet"]
                raise GuardError(guard)
        batch_started = time.monotonic()
        result = super().batch(fixture)
        elapsed = time.monotonic() - batch_started
        self.samples.setdefault(fixture["fixture_id"], []).append(elapsed)
        export_json(
            self.out / "batches" / f"batch-{self.completed:04d}.json",
            {
                "fixture_id": fixture["fixture_id"],
                "operator_wall_seconds": elapsed,
                "output": result,
                "guard": guard,
            },
        )
        self.guards.append(guard)
        self.completed += 1
        # Progress is deliberately printed on each batch, not only at 120s.
        print(
            json.dumps({"fixture": fixture["fixture_id"], "completed": self.completed}), flush=True
        )
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--budget-seconds", type=float, default=7200)
    parser.add_argument("--require-quiet", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    verify_export(args.fixtures)
    fixtures = json.loads(args.fixtures.read_text())
    if fixtures["source_identity"] != ruler.source_identity():
        raise RuntimeError("frozen fixture source drift")
    before = snapshot()
    export_json(args.out / "host-before.json", before)
    started = time.monotonic()
    operator = GuardedOperator(args.out, started + args.budget_seconds, args.require_quiet)
    try:
        receipt = ruler.run_ruler(
            fixtures, operator, platform_label="mac", workers=1, min_batches=30
        )
    except Exception as exc:
        export_json(
            args.out / "stopped.json",
            {
                "status": "STOPPED",
                "error": str(exc),
                "completed": operator.completed,
                "guards": operator.guards,
                "failed_observation": getattr(exc, "observation", None),
            },
        )
        raise
    receipt.update(
        operator_spec="research.bl_served_search.ruler_operator:create_operator",
        guard_wrapper="research.bl_mac_latency.ruler_run:GuardedOperator",
        guard_source_sha256=digest(Path(__file__)),
        host_source_sha256=digest(Path(__file__).with_name("host.py")),
        guard_checks=operator.guards,
        quiet_screen_each_batch=args.require_quiet,
        quantile_estimator="nearest_rank; all30 includes cold; warm29 excludes first",
        qualification="MAC_MICRO_TIMING_ONLY_PENDING_L4_SEMANTICS_AND_L1",
    )
    export_json(args.out / "mac-ruler-raw.json", receipt)
    try:
        after = snapshot()
    except Exception as exc:
        after = {"status": "STOPPED", "error": str(exc), "quiet": False}
    export_json(args.out / "host-after.json", after)
    receipt["host_quiet_endpoints_only"] = before["quiet"] and after["quiet"]
    receipt["entire_window_quiet_attested"] = False
    receipt["latency_seconds"] = {
        row["fixture_id"]: {
            "all30": distribution(operator.samples[row["fixture_id"]]),
            "warm29": distribution(operator.samples[row["fixture_id"]][1:]),
            "semantics_stable": len({ruler.canonical(b["semantic"]) for b in row["batches"]}) == 1,
        }
        for row in receipt["records"]
    }
    export_json(args.out / "mac-ruler.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "elapsed": receipt["elapsed_seconds"],
                "latency": receipt["latency_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
