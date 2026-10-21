"""Verify excluded tiny-run evidence and conservatively forecast O2 work."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from research.t34_o2_harvest.registration import require_registration


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def validate_preflight(receipt_path: Path, registration_path: Path) -> dict[str, Any]:
    """Check actual fixture identities and derive a cautious CPU extrapolation.

    The first world is a cold-start observation. Rates use the remaining two
    complete real-world triples, taking the slowest arm/component. The source
    fixture is tiny and does not estimate late-large-body throughput; a future
    full-horizon excluded Colab timing admission remains mandatory.
    """
    registration = require_registration(registration_path)
    receipt_path = receipt_path.resolve(strict=True)
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("schema") != "t34-o2-harvest-run/v1"
        or receipt.get("status") != "COMPLETE"
        or receipt.get("mode") != "preflight"
        or receipt.get("registration_sha256") != _sha(registration_path)
        or receipt.get("source_commit") != registration["source_commit"]
    ):
        raise ValueError("preflight must bind the frozen registration and complete tiny run")
    entries = receipt.get("worlds", [])
    expected = registration["bank"]["excluded_preflight_rows"]
    if len(entries) != 3 or len(expected) != 3:
        raise ValueError("preflight requires all three excluded paired-world triples")
    worlds, bindings = [], []
    for entry, row in zip(entries, expected):
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("preflight payload path escapes run directory")
        path = receipt_path.parent / relative
        if path.is_symlink() or not path.resolve().is_relative_to(receipt_path.parent):
            raise ValueError("preflight payload must be a regular contained file")
        if _sha(path) != entry["sha256"]:
            raise ValueError("preflight payload SHA differs")
        world = json.loads(path.read_text())
        if (
            world.get("world") != row
            or world.get("horizon") != 40
            or world.get("checkpoint_sha256") != registration["checkpoint_sha256"]
            or world.get("cfg", {}).get("mechanics_version") != 2
            or world.get("operational", {}).get("device") != "cpu"
            or world.get("operational", {}).get("threads") != 1
            or set(world.get("arms", {})) != {"bare", "d", "do"}
        ):
            raise ValueError("preflight world identity or CPU execution contract differs")
        for name, arm in world["arms"].items():
            roots = arm["controller"]["roots"]
            if name == "bare":
                if roots:
                    raise ValueError("bare fixture unexpectedly searched")
            elif [root["decision"] for root in roots] != [10, 20]:
                raise ValueError("tiny assisted fixture must exercise both early fixed roots")
            for root in roots:
                if root["root"]["horizon"] != 500 or root["root"]["source_preserved"] is not True:
                    raise ValueError("tiny branch horizon or source-preservation check differs")
            measured_rows = sum(
                branch["row_steps"]
                for root in roots
                for branch in root["root"]["branches"].values()
            )
            if measured_rows != arm["branch_row_steps"]:
                raise ValueError("measured branch counters do not reconcile")
            if arm["base_row_steps"] != arm["decisions"] or arm["decisions"] != 40:
                raise ValueError("tiny base-work counters do not reconcile")
        worlds.append(world)
        bindings.append({"path": str(path), "sha256": entry["sha256"]})
    base_rates, branch_rates = [], []
    for world in worlds[1:]:
        for arm in world["arms"].values():
            branch_seconds = sum(
                _positive(root["search_elapsed_seconds"], "whole search elapsed")
                for root in arm["controller"]["roots"]
            )
            base_seconds = _positive(arm["elapsed_seconds"] - branch_seconds, "base elapsed")
            base_rates.append(_positive(arm["base_row_steps"], "base rows") / base_seconds)
            if arm["branch_row_steps"]:
                branch_rates.append(
                    _positive(arm["branch_row_steps"], "branch rows")
                    / _positive(branch_seconds, "branch seconds")
                )
    if not branch_rates:
        raise ValueError("no real branch work was measured")
    base_rate, branch_rate = min(base_rates), min(branch_rates)
    # 32 paired triples per shard, 11 workers, at most 3 complete worker waves.
    world_base_rows, world_branch_rows = 15000, 48000
    # Charge one extra full-world wave for parallel timing and parity probes.
    waves, fixture_waves, slowdown, overhead = 3, 1, 3.0, 600.0
    seconds = (waves + fixture_waves) * slowdown * (
        world_base_rows / base_rate + world_branch_rows / branch_rate
    ) + overhead
    forecast = {
        "model": "three_study_plus_one_fixture_wave_times_slowest_component_times_3_plus_600s",
        "warm_fixture_indices": [1, 2],
        "base_rows_per_second_mac": base_rate,
        "branch_rows_per_second_mac": branch_rate,
        "world_base_row_bound": world_base_rows,
        "world_branch_row_bound": world_branch_rows,
        "all_study_row_steps_bound": 6048000,
        "workers": 11,
        "waves_per_chunk": waves,
        "excluded_fixture_waves_per_chunk": fixture_waves,
        "cpu_slowdown_factor": slowdown,
        "setup_export_seconds_per_chunk": overhead,
        "worst_chunk_seconds": seconds,
        "estimated_minutes": max(3, math.ceil(seconds / 60)),
        "aggregate_runtime_hours": seconds * 3 / 3600,
        "aggregate_units_at_2_per_hour": seconds * 3 / 1800,
        "fits_twofold_timeout_and_24_unit_envelope": seconds <= 7200,
        "l4_measured": False,
        "runtime_full_horizon_timing_admission_required": True,
        "limitations": "Tiny early-body CPU extrapolation; no late-body or L4 speed qualification",
    }
    return {
        "path": str(receipt_path),
        "sha256": _sha(receipt_path),
        "elapsed_seconds": _positive(receipt["elapsed_seconds"], "preflight elapsed"),
        "world_payloads": bindings,
        "forecast": forecast,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = validate_preflight(args.receipt, args.registration)
    with args.out.open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report["forecast"], sort_keys=True))


if __name__ == "__main__":
    main()
