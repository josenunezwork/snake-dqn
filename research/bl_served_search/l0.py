"""L0' saved-smoke bookkeeping for Path B's deadline and demand evidence.

It reads immutable saved smoke receipts and fails closed when a requested
accounting field was not retained. Aggregate controller receipts are never
treated as per-batch dense-demand receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from research.bl_served_search.exports import export_bytes, export_json

SMOKE = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-pathb-smoke-20261021")
ARTIFACT_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l2")
FPS = 24.0
LATENCY_SECONDS = (1, 2, 3, 5, 8, 12, 18)
ROOT_INTERVAL_FRAMES = 5
# The historical publication's stated reachability counts use no additional
# controller margin. A future paced controller must freeze its own margin.
PUBLICATION_MARGIN_FRAMES = 0
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def digest(path: Path) -> str:
    """Return the SHA-256 of one immutable input."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Load one explicitly named JSON source object."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"object expected: {path}")
    return value


def is_number(value: object) -> bool:
    """Return whether a JSON value is numeric but not a boolean."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def t3_records(source: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Load the fixed 75-record T3 denominator and bind all smoke inputs."""
    input_paths = sorted(
        [*source.glob("lad-*.json"), *source.glob("n1-*.json"), *source.glob("n6-*.json")]
    )
    hashes = {str(path): digest(path) for path in input_paths}
    records = []
    for path in sorted(source.glob("n1-*.json")):
        record = load_json(path)
        if record.get("group") == "T3":
            record["_path"] = path.name
            records.append(record)
    if len(records) != 75:
        raise ValueError(f"expected fixed 75 T3 records, got {len(records)}")
    return sorted(records, key=lambda row: str(row["id"])), hashes


def first_improvement(record: dict[str, Any]) -> dict[str, Any] | None:
    """Return the earliest retained fully evaluated improvement, if complete."""
    values = record.get("fully_evaluated_improvements")
    if not isinstance(values, list) or not values:
        return None
    candidates = [
        item
        for item in values
        if isinstance(item, dict) and is_number(item.get("frame")) and is_number(item.get("wall"))
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: (float(item["wall"]), int(item["frame"])))


def own_deadline_table(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the retained own-target slack versus measured-compute table."""
    rows: list[dict[str, Any]] = []
    for record in records:
        detection = record.get("detections")
        first = first_improvement(record)
        compute = record.get("time_to_first_fully_evaluated_improving_action_seconds")
        detection_frame = record.get("detection_live_frame_at_24fps")
        first_detection = detection[0] if isinstance(detection, list) and detection else None
        valid = (
            isinstance(first_detection, dict)
            and is_number(first_detection.get("wall"))
            and is_number(detection_frame)
            and first is not None
            and is_number(compute)
            and math.isclose(
                float(compute), float(first["wall"]) - float(first_detection["wall"]), abs_tol=1e-9
            )
        )
        row: dict[str, Any] = {
            "id": record["id"],
            "rescue_capable": record.get("historical_survival_rescue") is True,
            "valid": valid,
        }
        if valid:
            slack_frames = int(first["frame"]) - int(detection_frame)
            row.update(
                {
                    "detection_live_frame": int(detection_frame),
                    "first_improving_frame": int(first["frame"]),
                    "own_target_slack_frames": slack_frames,
                    "own_target_slack_seconds_at_24fps": slack_frames / FPS,
                    "compute_seconds": float(compute),
                    "compute_within_own_target": float(compute) <= slack_frames / FPS,
                }
            )
        else:
            row["missing"] = [
                name
                for name, present in (
                    (
                        "detections[0].wall",
                        isinstance(first_detection, dict)
                        and is_number(first_detection.get("wall")),
                    ),
                    ("detection_live_frame_at_24fps", is_number(detection_frame)),
                    ("fully_evaluated_improvements.frame/wall", first is not None),
                    (
                        "validated_time_to_first_fully_evaluated_improving_action_seconds",
                        is_number(compute)
                        and first is not None
                        and isinstance(first_detection, dict)
                        and is_number(first_detection.get("wall"))
                        and math.isclose(
                            float(compute),
                            float(first["wall"]) - float(first_detection["wall"]),
                            abs_tol=1e-9,
                        ),
                    ),
                )
                if not present
            ]
        rows.append(row)
    return {
        "denominators": {"all_t3": 75, "rescue_capable": 34, "delay_only": 41, "m3_excluded": 5},
        "valid_rows": sum(row["valid"] for row in rows),
        "missing_rows": sum(not row["valid"] for row in rows),
        "rows": rows,
    }


def lad_frame(record: dict[str, Any]) -> int | None:
    """Return the exact-H500 LAD only from the retained LAD diagnostic object."""
    value = record.get("v2_lad")
    if not isinstance(value, dict) or not is_number(value.get("lad")):
        return None
    return int(value["lad"])


def reachability_curves(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute exact-H500 and rescue-only D300 LAD reachability at fixed S values."""
    exact_rows = [(record, lad_frame(record)) for record in records]
    rescue_rows = [record for record in records if record.get("historical_survival_rescue") is True]
    if any(
        lad is None or not is_number(record.get("detection_live_frame_at_24fps"))
        for record, lad in exact_rows
    ):
        raise ValueError("exact-H500 LAD/detection fields missing")
    if any(
        not is_number(record.get("historical_lad_d300"))
        or not is_number(record.get("detection_live_frame_at_24fps"))
        for record in rescue_rows
    ):
        raise ValueError("D300 LAD/detection fields missing for rescue-capable denominator")

    def curve(rows: list[tuple[int, int]], denominator: int) -> list[dict[str, Any]]:
        return [
            {
                "latency_seconds": seconds,
                "reachable": sum((doom - lad) / FPS >= seconds for doom, lad in rows),
                "denominator": denominator,
            }
            for seconds in LATENCY_SECONDS
        ]

    exact = [
        (int(lad), int(record["detection_live_frame_at_24fps"]))
        for record, lad in exact_rows
        if lad is not None
    ]
    d300 = [
        (int(record["historical_lad_d300"]), int(record["detection_live_frame_at_24fps"]))
        for record in rescue_rows
    ]
    return {
        "exact_h500": curve(exact, 75),
        "d300_survival_rescue_only": curve(d300, 34),
    }


def reachable_windows(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Publish D300-cut reachable roots for every rescue-capable doom and S."""
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("historical_survival_rescue") is not True:
            continue
        doom = record.get("doom")
        live = record.get("detection_live_frame_at_24fps")
        d300_lad = record.get("historical_lad_d300")
        if not is_number(doom) or not is_number(live) or not is_number(d300_lad):
            raise ValueError(f"missing/invalid D300 reachable window: {record['id']}")
        windows = []
        for seconds in LATENCY_SECONDS:
            start = int(live) + math.ceil(seconds * FPS) + PUBLICATION_MARGIN_FRAMES
            width = int(d300_lad) - start
            frames = max(0, width)
            root_count_upper_bound = (
                math.floor(width / ROOT_INTERVAL_FRAMES) + 1 if width >= 0 else 0
            )
            windows.append(
                {
                    "latency_seconds": seconds,
                    "reachable_window_start_frame": start,
                    "reachable_window_end_inclusive_frame": int(d300_lad),
                    "reachable_window_frames": frames,
                    "reachable_window_seconds_at_24fps": frames / FPS,
                    "periodic_root_count_upper_bound": root_count_upper_bound,
                }
            )
        rows.append(
            {
                "id": record["id"],
                "detection_live_frame": int(live),
                "d300_lad_frame": int(d300_lad),
                "doom_frame": int(doom),
                "periodic_root_interval_frames": ROOT_INTERVAL_FRAMES,
                "windows": windows,
            }
        )
    if len(rows) != 34:
        raise ValueError(f"expected 34 rescue-capable windows, got {len(rows)}")
    return {
        "root_count_definition": (
            "cadence-phase upper bound: max(0, floor((d300_lad - "
            "(live + ceil(S*24) + publication_margin)) / 5) + 1)"
        ),
        "publication_margin_frames": PUBLICATION_MARGIN_FRAMES,
        "interval": "[live + ceil(S*24) + publication_margin, d300_lad] (D300 LAD included)",
        "rows": rows,
    }


def dense_demand_census(source: Path) -> dict[str, Any]:
    """Require retained per-batch flags before publishing a dense-demand census."""
    n6 = sorted(source.glob("n6-*.json"))
    missing = []
    for path in n6:
        record = load_json(path)
        events = record.get("events")
        if (
            not isinstance(events, list)
            or not events
            or not all(
                isinstance(event, dict) and "bare_dies_within_h300" in event for event in events
            )
        ):
            missing.append(path.name)
    return {
        "status": "INCOMPLETE_MISSING_PER_BATCH_BARE_DIES_FLAGS",
        "required_fields": ["events[].bare_dies_within_h300", "events[].surviving_alternative"],
        "n6_receipts": len(n6),
        "receipts_without_complete_batch_flags": missing,
        "reason": (
            "aggregate/controller receipts cannot establish per-batch counts, runs, or alternatives"
        ),
    }


def inspect(source: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Assemble all L0' accounting components from the one authorized smoke."""
    records, hashes = t3_records(source)
    return {
        "own_deadline": own_deadline_table(records),
        "lad_reachability": reachability_curves(records),
        "per_doom_reachable_windows": reachable_windows(records),
        "dense_receipt_demand_census": dense_demand_census(source),
    }, hashes


def result_markdown(receipt: dict[str, Any]) -> str:
    """Render a compact handoff without upgrading missing evidence."""
    accounting = receipt["accounting"]
    own = accounting["own_deadline"]
    dense = accounting["dense_receipt_demand_census"]
    return "\n".join(
        [
            "# L0′ deadline and demand bookkeeping — 2026-10-21",
            "",
            (
                "**Verdict: INCOMPLETE for dense-demand sizing.** The own-deadline table, LAD "
                "curves, and 34 per-doom windows are reconstructed from retained T3 fields. The "
                "saved N=6 receipts do not retain per-batch bare-dies/alternative flags, so no "
                "demand count or roots/s admission bar is inferred."
            ),
            "",
            "- Fixed denominators: 75 T3 / 34 rescue-capable / 41 delay-only / 5 M3 excluded.",
            (
                f"- Own-deadline rows with all required fields: {own['valid_rows']}/75; "
                f"missing: {own['missing_rows']}/75."
            ),
            (
                "- LAD curves include LAD exactly at the latency boundary at 24 fps: exact H500 "
                "`v2_lad.lad` for all 75 and D300 LAD for the 34 rescue-capable records."
            ),
            (
                "- Per-doom windows are [live + ceil(S×24), D300 LAD], including the LAD endpoint. "
                "Root counts are cadence-phase upper bounds, not observed dense demand."
            ),
            (
                f"- Dense receipt census: {dense['status']}; "
                f"{len(dense['receipts_without_complete_batch_flags'])}/{dense['n6_receipts']} "
                "N=6 receipts lack complete per-batch flags."
            ),
            (
                "- No time-divided or k× replay was performed. This development-only receipt is "
                "excluded from registered verdicts and authorizes neither dispatch nor serving."
            ),
            "",
        ]
    )


def run(source: Path, artifact_root: Path) -> dict[str, Any]:
    """Publish a create-only L0' generation under a fresh artifact subdirectory."""
    started = time.monotonic()
    accounting, hashes = inspect(source)
    receipt = {
        "schema": "bl-served-search-l0-prime/v1",
        "development_only": True,
        "excluded_from_registered_verdicts": True,
        "status": "INCOMPLETE",
        "verdict": "INCOMPLETE_MISSING_PER_BATCH_DENSE_RECEIPTS",
        "accounting": accounting,
        "input_sha256": hashes,
        "execution": {
            "nice": os.nice(0),
            "max_processes": 1,
            "numerical_threads": 1,
            "elapsed_seconds": time.monotonic() - started,
        },
    }
    if artifact_root.exists():
        raise FileExistsError(f"L0' export directory already exists: {artifact_root}")
    export_json(artifact_root / "l0_prime_receipt.json", receipt)
    export_bytes(artifact_root / "RESULT_L0_PRIME.md", result_markdown(receipt).encode())
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SMOKE)
    parser.add_argument("--out", type=Path, default=ARTIFACT_ROOT / "amendment08-l0")
    args = parser.parse_args()
    if os.nice(0) < 10:
        parser.error("run under nice -n 10 or lower priority")
    if any(os.environ.get(name) != "1" for name in THREAD_VARS):
        parser.error("all numerical thread caps must explicitly equal one")
    receipt = run(args.source, args.out)
    print(json.dumps({"status": receipt["status"], "verdict": receipt["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
