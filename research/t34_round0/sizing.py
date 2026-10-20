"""Read saved A5/A6 evidence and prepare draft sizing without ratifying/dispatching."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from research.t34_round0.common import read_json, sha256, source_hashes, write_once
from research.t34_round0.make_manifest import size_round, validate_a5
from research.t34_round0.validate import validate


def build(benchmarks: list[Path], preflight: Path, setup_minutes: float) -> dict[str, Any]:
    """Bind real two-lane A5 receipts and separately timed excluded A6 costs."""
    verification = validate(preflight)
    chunk_path = preflight / "chunk.json"
    chunk = read_json(chunk_path)
    if len(benchmarks) != 2 or len(set(p.resolve() for p in benchmarks)) != 2:
        raise ValueError("sizing requires the two distinct saved A5 L4 lanes")
    reports = []
    for path in benchmarks:
        data = read_json(path)
        job = path.parent.parent
        receipt = read_json(job / "receipt.json")
        if (
            receipt.get("status") != "PASS"
            or receipt.get("stop_verified") is not True
            or not any(
                e.get("path") == "benchmark.json" and e.get("sha256") == sha256(path)
                for e in receipt.get("exports", [])
            )
            or data.get("schema") != "t34-a5-benchmark/v2"
            or data.get("invalid_reason") is not None
            or data.get("worlds_per_hour", 0) < 50
            or data.get("fanout", {}).get("nproc") != 12
        ):
            raise ValueError("A5 must have verified L4 PASS evidence and K12 fanout")
        workers = data["fanout"].get("worker_reports", [])
        if len(workers) != 12 or any(
            w.get("actual_numba_threads") != 1
            or w.get("actual_policy_device") != "cpu"
            or w.get("configured_root_capacity") != 1
            or w.get("steady_state_window_complete") is not True
            or w.get("cold_fill_complete") is not True
            or w.get("steady_state_window_seconds") != 1200
            for w in workers
        ):
            raise ValueError("A5 worker layout or steady-state measurement differs")
        reports.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "receipt_sha256": sha256(job / "receipt.json"),
                "worlds_per_hour": data["worlds_per_hour"],
                "active_row_steps_per_second": data["active_row_steps_per_second"],
                "export_bytes_per_world": data["export_bytes_per_world"],
                "observed_setup_export_cleanup_minutes": max(
                    0, (receipt["elapsed_seconds"] - data["elapsed_seconds"]) / 60
                ),
            }
        )
    if setup_minutes < max(r["observed_setup_export_cleanup_minutes"] for r in reports):
        raise ValueError("setup allowance is below measured A5 setup/export/cleanup overhead")
    pairs = [read_json(preflight / item["path"]) for item in chunk["pairs"]]
    # Scale two excluded H500 fixtures prospectively; retain raw times, horizon,
    # max-over-fixtures aggregation, and 3x platform assumption in the adapter.
    bare = max(pair["arms"]["bare"]["rollout_elapsed_seconds"] for pair in pairs) * 10
    metric = (
        max(
            sum(pair["arms"][arm]["metric_elapsed_seconds"] for arm in ("bare", "teacher"))
            for pair in pairs
        )
        * 10
    )
    evidence = {
        "a5_lanes": reports,
        "a6_chunk_sha256": sha256(chunk_path),
        "a6_verification": verification,
        "h500_to_h5000_factor": 10,
        "aggregation": "maximum single-fixture cost, then 10x horizon scaling",
        "cpu_slowdown_factor": 3.0,
        "setup_minutes": setup_minutes,
    }
    payload = {
        "schema": "t34-round0-a5/v1",
        "teacher_worlds_per_hour": min(r["worlds_per_hour"] for r in reports),
        "bare_seconds_per_world": bare,
        "metric_seconds_per_pair": metric,
        "cpu_slowdown_factor": 3.0,
        "a6_measurement_sha256": sha256(chunk_path),
        "a6_source_sha256": hashlib.sha256(
            json.dumps(source_hashes(), sort_keys=True).encode()
        ).hexdigest(),
        "steady_window_seconds": 1200,
        "cold_fill_excluded": True,
        "active_row_steps_per_second": min(r["active_row_steps_per_second"] for r in reports),
        "two_lane_rehearsal_complete": True,
        "measured_export_bytes_per_pair": max(r["export_bytes_per_world"] for r in reports),
        "evidence_sha256": hashlib.sha256(
            json.dumps(evidence, sort_keys=True).encode()
        ).hexdigest(),
        "evidence": evidence,
        "setup_minutes": setup_minutes,
    }
    validate_a5(payload)
    sizing = size_round(
        240,
        teacher_worlds_per_hour=payload["teacher_worlds_per_hour"],
        bare_seconds_per_world=bare,
        metric_seconds_per_pair=metric,
        cpu_slowdown_factor=3.0,
        setup_minutes=setup_minutes,
    )
    payload["draft_sizing"] = dict(
        sizing,
        initial_liability_units=sizing["unit_cap"] + sizing["reserve_units"],
        total_with_one_rerun_per_chunk_units=2 * (sizing["unit_cap"] + sizing["reserve_units"]),
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a5-benchmarks", nargs=2, type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--setup-minutes", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_once(args.out, build(args.a5_benchmarks, args.preflight_dir, args.setup_minutes))


if __name__ == "__main__":
    main()
