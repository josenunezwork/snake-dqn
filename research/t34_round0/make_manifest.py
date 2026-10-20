"""Build bounded, validation-only T34 Round 0 Colab manifests.

This module intentionally has no allocation or dispatch operation.  It binds a
ratified registration, the audited bank, public input catalog, and the measured
A5 receipt before handing the result to the Colab CLI's local validator.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any

from research.colab_cli.supervisor import validate as validate_supervisor
from research.t34_exit_b8.qualification import QualificationRefused, require_qualified_transport
from research.t34_round0.bank import ordered_rows
from research.t34_round0.common import BINDING_DOCS
from research.t34_round0.registration import require_registration

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "research/t34_round0"
RATE_UNITS_PER_HOUR = 2.0
MAX_ESTIMATE_MINUTES = 180
TIMEOUT_GRACE_MINUTES = 25
DISPATCH_WINDOW_HOURS = 48
RESERVE_FRACTION = 0.20
MAX_EXPORT_BYTES = 2 * 1024**3
EXPORT_BYTES_PER_WORLD = 9 * 1024**2
EXPORT_SIZING_FACTOR = 1.25
LAYOUT = {"cpu_inference_processes": 12, "threads_per_process": 1, "worlds_per_process": 1}


class ManifestRefused(ValueError):
    """Round 0 has not supplied a complete, admissible registration package."""


def _positive_number(value: Any, name: str) -> float:
    """Return a finite positive numeric value, refusing booleans and zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestRefused(f"{name} must be finite and positive")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ManifestRefused(f"{name} must be finite and positive")
    return numeric


def size_chunk(
    pairs: int,
    *,
    teacher_worlds_per_hour: float,
    bare_seconds_per_world: float,
    metric_seconds_per_pair: float,
    cpu_slowdown_factor: float,
    cpu_inference_processes: int = 12,
    setup_minutes: float = 0.0,
) -> dict[str, float | int]:
    """Size one same-runtime paired chunk from its bound measured component costs.

    The teacher rate is an aggregate K12 rate.  A6 measures bare and metric
    CPU costs on one process, so their planning costs use the registered
    conservative platform factor divided across the same K12 process pool.
    Keeping them separate prevents a whole-pair preflight elapsed value from
    being misrepresented as a bare-controller measurement.
    """
    if not isinstance(pairs, int) or isinstance(pairs, bool) or pairs < 1:
        raise ManifestRefused("pairs must be a positive integer")
    teacher_rate = _positive_number(teacher_worlds_per_hour, "teacher_worlds_per_hour")
    bare_seconds = _positive_number(bare_seconds_per_world, "bare_seconds_per_world")
    metric_seconds = _positive_number(metric_seconds_per_pair, "metric_seconds_per_pair")
    slowdown = _positive_number(cpu_slowdown_factor, "cpu_slowdown_factor")
    if isinstance(setup_minutes, bool) or not isinstance(setup_minutes, (int, float)):
        raise ManifestRefused("setup_minutes must be finite and nonnegative")
    setup = float(setup_minutes)
    if not math.isfinite(setup) or setup < 0:
        raise ManifestRefused("setup_minutes must be finite and nonnegative")
    if (
        not isinstance(cpu_inference_processes, int)
        or isinstance(cpu_inference_processes, bool)
        or cpu_inference_processes != LAYOUT["cpu_inference_processes"]
    ):
        raise ManifestRefused("Round 0 requires exactly 12 CPU inference processes")
    if slowdown < 1:
        raise ManifestRefused("cpu_slowdown_factor must be at least one")
    teacher_seconds = pairs * 3600.0 / teacher_rate
    bare_total_seconds = pairs * bare_seconds * slowdown / cpu_inference_processes
    metric_total_seconds = pairs * metric_seconds * slowdown / cpu_inference_processes
    workload_minutes = setup + (teacher_seconds + bare_total_seconds + metric_total_seconds) / 60.0
    estimate_minutes = math.ceil(workload_minutes)
    timeout_minutes = max(
        math.ceil(workload_minutes + TIMEOUT_GRACE_MINUTES),
        2 * estimate_minutes,
    )
    unit_cap = timeout_minutes * RATE_UNITS_PER_HOUR / 60.0
    export_bytes = math.ceil(pairs * EXPORT_BYTES_PER_WORLD * EXPORT_SIZING_FACTOR)
    return {
        "pairs": pairs,
        "setup_minutes": setup,
        "teacher_seconds": teacher_seconds,
        "measured_bare_seconds": pairs * bare_seconds,
        "measured_metric_seconds": pairs * metric_seconds,
        "bare_seconds": bare_total_seconds,
        "metric_seconds": metric_total_seconds,
        "cpu_slowdown_factor": slowdown,
        "cpu_inference_processes": cpu_inference_processes,
        "workload_minutes": workload_minutes,
        "estimate_minutes": estimate_minutes,
        "timeout_minutes": timeout_minutes,
        "unit_cap": unit_cap,
        "reserve_units": unit_cap * RESERVE_FRACTION,
        "export_bytes": export_bytes,
    }


def maximum_chunk_pairs(
    *,
    teacher_worlds_per_hour: float,
    bare_seconds_per_world: float,
    metric_seconds_per_pair: float,
    cpu_slowdown_factor: float,
    setup_minutes: float = 0.0,
) -> int:
    """Return the largest chunk whose estimate and sized export are both bounded."""
    one_pair = size_chunk(
        1,
        teacher_worlds_per_hour=teacher_worlds_per_hour,
        bare_seconds_per_world=bare_seconds_per_world,
        metric_seconds_per_pair=metric_seconds_per_pair,
        cpu_slowdown_factor=cpu_slowdown_factor,
        setup_minutes=setup_minutes,
    )
    by_time = math.floor(
        (MAX_ESTIMATE_MINUTES - setup_minutes)
        / (float(one_pair["workload_minutes"]) - setup_minutes)
    )
    by_export = math.floor((MAX_EXPORT_BYTES - 1) / float(one_pair["export_bytes"]))
    maximum = min(by_time, by_export)
    # Every production chunk must end on a complete 4:3:3 stratum.  This
    # prevents a completed early chunk from revealing only one mix.
    maximum -= maximum % 10
    if maximum < 1:
        raise ManifestRefused("measured component costs leave no pair within Round 0 limits")
    return maximum


def size_round(
    pairs: int,
    *,
    teacher_worlds_per_hour: float,
    bare_seconds_per_world: float,
    metric_seconds_per_pair: float,
    cpu_slowdown_factor: float,
    setup_minutes: float = 0.0,
) -> dict[str, Any]:
    """Partition Round 0 before dispatch using only pinned sizing measurements."""
    if not isinstance(pairs, int) or isinstance(pairs, bool) or pairs < 1:
        raise ManifestRefused("pairs must be a positive integer")
    chunk_pairs = maximum_chunk_pairs(
        teacher_worlds_per_hour=teacher_worlds_per_hour,
        bare_seconds_per_world=bare_seconds_per_world,
        metric_seconds_per_pair=metric_seconds_per_pair,
        cpu_slowdown_factor=cpu_slowdown_factor,
        setup_minutes=setup_minutes,
    )
    chunks: list[dict[str, float | int]] = []
    for start in range(0, pairs, chunk_pairs):
        count = min(chunk_pairs, pairs - start)
        sizing = size_chunk(
            count,
            teacher_worlds_per_hour=teacher_worlds_per_hour,
            bare_seconds_per_world=bare_seconds_per_world,
            metric_seconds_per_pair=metric_seconds_per_pair,
            cpu_slowdown_factor=cpu_slowdown_factor,
            setup_minutes=setup_minutes,
        )
        if (
            sizing["estimate_minutes"] > MAX_ESTIMATE_MINUTES
            or sizing["export_bytes"] >= MAX_EXPORT_BYTES
        ):
            raise ManifestRefused("chunk exceeds a Round 0 estimate or export limit")
        chunks.append({"start": start, "count": count, **sizing})
    return {
        "chunk_pairs": chunk_pairs,
        "chunks": chunks,
        "unit_cap": sum(float(chunk["unit_cap"]) for chunk in chunks),
        "reserve_units": sum(float(chunk["reserve_units"]) for chunk in chunks),
    }


def sha256_file(path: Path) -> str:
    """Return a bounded-memory SHA-256 digest for a regular input file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestRefused(f"invalid {name}: {path}") from exc
    if not isinstance(value, dict):
        raise ManifestRefused(f"{name} must be a JSON object")
    return value


def _hex(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _regular(path: Path, name: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ManifestRefused(f"missing {name}: {path}")
    return path


def validate_a5(receipt: dict[str, Any]) -> dict[str, Any]:
    """Validate the immutable A5 acceptance fields required by Round 0."""
    required = {
        "schema": "t34-round0-a5/v1",
        "teacher_worlds_per_hour": None,
        "bare_seconds_per_world": None,
        "metric_seconds_per_pair": None,
        "cpu_slowdown_factor": None,
        "a6_measurement_sha256": None,
        "a6_source_sha256": None,
        "steady_window_seconds": 1200,
        "cold_fill_excluded": True,
        "active_row_steps_per_second": None,
        "two_lane_rehearsal_complete": True,
        "measured_export_bytes_per_pair": None,
        "evidence_sha256": None,
    }
    for key, expected in required.items():
        if (
            key not in receipt
            or (expected is not None and receipt[key] != expected)
            or (isinstance(expected, bool) and receipt[key] is not expected)
        ):
            raise ManifestRefused(f"A5 receipt lacks required {key}")
    for key in (
        "teacher_worlds_per_hour",
        "bare_seconds_per_world",
        "metric_seconds_per_pair",
        "cpu_slowdown_factor",
        "active_row_steps_per_second",
        "measured_export_bytes_per_pair",
    ):
        _positive_number(receipt[key], f"A5 receipt {key}")
    if receipt["teacher_worlds_per_hour"] < 50:
        raise ManifestRefused("A5 receipt misses the 50 completed worlds/hour per lane floor")
    if not _hex(receipt["a6_measurement_sha256"]):
        raise ManifestRefused("A5 receipt must bind the per-arm A6 measurement")
    if not _hex(receipt["a6_source_sha256"]):
        raise ManifestRefused("A5 receipt must bind the A6 measurement source")
    if not _hex(receipt["evidence_sha256"]):
        raise ManifestRefused("A5 receipt must hash-bound its evidence")
    if receipt["cpu_slowdown_factor"] < 1:
        raise ManifestRefused("cpu_slowdown_factor must be at least one")
    if receipt["measured_export_bytes_per_pair"] > EXPORT_BYTES_PER_WORLD:
        raise ManifestRefused(
            "measured exports exceed registered 9 MiB/world sizing; re-size before ratification"
        )
    return receipt


def _verify_a5(path: Path) -> dict[str, Any]:
    from research.t34_round0.sizing import verify_adapter

    payload = validate_a5(_load_json(path, "A5 receipt"))
    verify_adapter(payload)
    return payload


def _verify_bank(path: Path) -> list[dict[str, Any]]:
    bank = _load_json(path, "Round 0 bank")
    rows = bank.get("rows")
    if bank.get("schema") != "t34-round0-bank/v1" or not isinstance(rows, list) or len(rows) != 240:
        raise ManifestRefused("Round 0 bank must be t34-round0-bank/v1 with exactly 240 rows")
    mixes = [row.get("mix") for row in rows if isinstance(row, dict)]
    order = {"scripted": 0, "frozen": 1, "mixed": 2}
    if (
        len(mixes) != 240
        or any(mix not in order for mix in mixes)
        or mixes != sorted(mixes, key=order.__getitem__)
    ):
        raise ManifestRefused("Round 0 bank rows must be ordered scripted, frozen, mixed")
    if {mixes.count("scripted"), mixes.count("frozen"), mixes.count("mixed")} != {
        72,
        96,
    } or mixes.count("scripted") != 96:
        raise ManifestRefused("Round 0 bank mix counts must be 96 scripted / 72 frozen / 72 mixed")
    if any(
        not row.get("world_id") or row.get("world_seed") is None or not row.get("opponents")
        for row in rows
    ):
        raise ManifestRefused("Round 0 bank rows need world_id, world_seed, mix, and opponents")
    return rows


def _verify_ratification(
    path: Path, *, commit: str, registration: Path, bank: Path, audit: Path, a5: Path
) -> dict[str, Any]:
    ratification = _load_json(path, "ratification")
    if (
        ratification.get("schema") != "t34-round0-ratification/v1"
        or ratification.get("status") != "RATIFIED"
    ):
        raise ManifestRefused("ratification must be t34-round0-ratification/v1 and RATIFIED")
    expected = {
        "commit": commit,
        "registration_sha256": sha256_file(registration),
        "bank_sha256": sha256_file(bank),
        "audit_sha256": sha256_file(audit),
    }
    for key, value in expected.items():
        if ratification.get(key) != value:
            raise ManifestRefused(f"ratification {key} does not bind supplied bytes")
    if not isinstance(ratification.get("operator"), str) or not ratification["operator"].strip():
        raise ManifestRefused("ratification must name its operator")
    if (
        not isinstance(ratification.get("approved_at"), str)
        or not ratification["approved_at"].strip()
    ):
        raise ManifestRefused("ratification must record approved_at")
    a5_binding = ratification.get("a5")
    if not isinstance(a5_binding, dict) or a5_binding.get("sha256") != sha256_file(a5):
        raise ManifestRefused("ratification must bind the supplied A5 receipt")
    sources = ratification.get("source_hashes")
    if (
        not isinstance(sources, dict)
        or not sources
        or not all(_hex(value) for value in sources.values())
    ):
        raise ManifestRefused("ratification source_hashes must contain SHA-256 bindings")
    return ratification


def _input(path: Path, dest: str) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path), "dest": dest}


def dispatch_deadline(approved_at: str) -> float:
    """Return the registered UTC dispatch deadline from ratification time."""
    if not isinstance(approved_at, str):
        raise ManifestRefused("ratification approved_at must be an ISO-8601 timestamp")
    try:
        parsed = dt.datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ManifestRefused("ratification approved_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ManifestRefused("ratification approved_at must include a UTC offset")
    return (
        parsed.astimezone(dt.timezone.utc) + dt.timedelta(hours=DISPATCH_WINDOW_HOURS)
    ).timestamp()


def qualify_source(manifest: dict[str, Any]) -> None:
    """Exercise the exact bounded source transport locally before allocation."""
    from research.colab_cli.source import prepare

    with tempfile.TemporaryDirectory(prefix="t34-round0-source-") as temporary:
        prepare(ROOT, manifest, Path(temporary))


def make_manifests(
    *,
    a5_receipt: Path,
    ratification: Path,
    output_root: Path,
    out_dir: Path,
    commit: str,
    lanes: int,
    other_assignments: dict[str, float],
    setup_minutes: float,
    registration: Path = ROOT / PACKAGE / "PREREGISTRATION_ROUND0.md",
    bank: Path = ROOT / PACKAGE / "BANK_ROUND0.json",
    audit: Path = ROOT / PACKAGE / "SEED_AUDIT_ROUND0.json",
    public_inputs: Path = ROOT / "research/t34_exit_b8/PUBLIC_INPUTS.json",
) -> dict[str, Any]:
    """Create chunk manifests and a global forecast; never allocate a runtime."""
    if not isinstance(lanes, int) or isinstance(lanes, bool) or not 1 <= lanes <= 3:
        raise ManifestRefused("lanes must be an integer in [1, 3]")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or not all(c in "0123456789abcdef" for c in commit)
    ):
        raise ManifestRefused("commit must be a full lowercase 40-character SHA")
    if (
        isinstance(setup_minutes, bool)
        or not isinstance(setup_minutes, (int, float))
        or not math.isfinite(setup_minutes)
        or setup_minutes < 0
    ):
        raise ManifestRefused("setup_minutes must be a finite nonnegative measured value")
    if not isinstance(other_assignments, dict) or not all(
        isinstance(key, str)
        and key
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
        for key, value in other_assignments.items()
    ):
        raise ManifestRefused(
            "other_assignments must be an operator-reviewed endpoint-to-rate mapping"
        )
    if lanes + len(other_assignments) > 3:
        raise ManifestRefused(
            "lanes plus operator-reviewed other assignments exceeds the account cap of three"
        )
    registration, bank, audit, public_inputs = (
        _regular(registration, "registration"),
        _regular(bank, "bank"),
        _regular(audit, "audit"),
        _regular(public_inputs, "public inputs"),
    )
    a5_receipt, ratification = _regular(a5_receipt, "A5 receipt"), _regular(
        ratification, "ratification"
    )
    output_root, out_dir = output_root.expanduser(), out_dir.expanduser()
    if not output_root.is_absolute() or not out_dir.is_absolute():
        raise ManifestRefused(
            "output_root and fresh out_dir must be absolute, with out_dir below output_root"
        )
    output_root, out_dir = output_root.resolve(), out_dir.resolve()
    if out_dir.exists() or output_root == out_dir or output_root not in out_dir.parents:
        raise ManifestRefused(
            "output_root and fresh out_dir must be absolute, with out_dir below output_root"
        )
    receipt = _verify_a5(a5_receipt)
    rows = _verify_bank(bank)
    dispatch_rows = ordered_rows(rows)
    ratification_payload = _verify_ratification(
        ratification,
        commit=commit,
        registration=registration,
        bank=bank,
        audit=audit,
        a5=a5_receipt,
    )
    require_registration(ratification_payload, bank, audit, repo=ROOT)
    dispatch_by = dispatch_deadline(ratification_payload["approved_at"])
    try:
        require_qualified_transport(ROOT, commit)
    except QualificationRefused as exc:
        raise ManifestRefused(str(exc)) from exc

    teacher_rate = float(receipt["teacher_worlds_per_hour"])
    bare_seconds = float(receipt["bare_seconds_per_world"])
    metric_seconds = float(receipt["metric_seconds_per_pair"])
    cpu_slowdown_factor = float(receipt["cpu_slowdown_factor"])
    round_sizing = size_round(
        len(rows),
        teacher_worlds_per_hour=teacher_rate,
        bare_seconds_per_world=bare_seconds,
        metric_seconds_per_pair=metric_seconds,
        cpu_slowdown_factor=cpu_slowdown_factor,
        setup_minutes=float(setup_minutes),
    )
    frozen = ratification_payload.get("sizing", {})
    if any(frozen.get(key) != value for key, value in round_sizing.items()):
        raise ManifestRefused("manifest sizing differs from the ratified chunks and unit cap")
    chunk_pairs = int(round_sizing["chunk_pairs"])
    full_chunk = size_chunk(
        chunk_pairs,
        teacher_worlds_per_hour=teacher_rate,
        bare_seconds_per_world=bare_seconds,
        metric_seconds_per_pair=metric_seconds,
        cpu_slowdown_factor=cpu_slowdown_factor,
        setup_minutes=float(setup_minutes),
    )
    if full_chunk["estimate_minutes"] > MAX_ESTIMATE_MINUTES:
        raise ManifestRefused("chunk estimate exceeds the 180-minute cap")

    study_root_value = ratification_payload.get("study_root")
    if not isinstance(study_root_value, str) or not Path(study_root_value).is_absolute():
        raise ManifestRefused("ratification must bind an absolute study_root")
    study_root = Path(study_root_value).resolve()
    if output_root != study_root or out_dir != study_root / "manifests":
        raise ManifestRefused(
            "Round 0 paths must be the ratified study_root and study_root/manifests"
        )

    out_dir.mkdir(parents=True)
    inputs = [
        _input(public_inputs, "PUBLIC_INPUTS.json"),
        _input(bank, "BANK_ROUND0.json"),
        _input(audit, "SEED_AUDIT_ROUND0.json"),
        _input(registration, "PREREGISTRATION_ROUND0.md"),
        _input(ratification, "RATIFICATION.json"),
        _input(a5_receipt, "A5_RECEIPT.json"),
    ]
    manifests: list[dict[str, Any]] = []
    for number, chunk in enumerate(round_sizing["chunks"]):
        start, count = int(chunk["start"]), int(chunk["count"])
        world_ids = [row["world_id"] for row in dispatch_rows[start : start + count]]
        if len(world_ids) != count:
            raise ManifestRefused("interleaved bank slice does not match chunk count")
        lane = number % lanes + 1
        chunk_id = f"chunk-{number:03d}"
        attempt_id = "attempt-01"
        job_id = f"t34-round0-{chunk_id}-{attempt_id}-l{lane}"
        manifest = {
            "job_id": job_id,
            "commit": commit,
            "script": f"{PACKAGE}/run.sh",
            "claim": "DRAFT_NO_ALLOCATION_AUTHORIZED",
            "env": {
                "PYTHONUNBUFFERED": "1",
                "T34_INPUT_MODE": "public",
                "T34_DEVICE": "cpu",
                "T34_ROUND0_START": str(start),
                "T34_ROUND0_COUNT": str(count),
                "T34_ROUND0_BANK": "BANK_ROUND0.json",
                "T34_ROUND0_AUDIT": "SEED_AUDIT_ROUND0.json",
                "T34_ROUND0_REGISTRATION": "PREREGISTRATION_ROUND0.md",
                "T34_ROUND0_RATIFICATION": "RATIFICATION.json",
                "T34_ROUND0_A5_RECEIPT": "A5_RECEIPT.json",
                "T34_INPUT_CATALOG": "PUBLIC_INPUTS.json",
                "T34_TELEMETRY_SECONDS": "120",
                "T34_ROUND0_OUTPUT_DIR": "round0",
                "T34_B8_CHECKPOINT": "B8_s3_final.pth",
                "T34_ROUND0_RUNTIME_ID": job_id,
                "T34_ROUND0_COMMIT": commit,
                "T34_ROUND0_SECONDS": str((int(chunk["timeout_minutes"]) - 25) * 60),
                "T34_ROUND0_THREADS": "1",
                "T34_ROUND0_PROCESSES": "12",
                "T34_ROUND0_WORLDS_PER_PROCESS": "1",
            },
            "requirements": {
                "gpu": "L4",
                "allowed_gpus": ["L4"],
                "min_vcpu": 12,
                "est_minutes": chunk["estimate_minutes"],
            },
            "timeout_minutes": chunk["timeout_minutes"],
            "output_dir": str(study_root / "chunks" / chunk_id / attempt_id / "job"),
            "inputs": inputs,
            "budget": {
                "unit_cap": chunk["unit_cap"],
                "reserve_units": chunk["reserve_units"],
                "rate_units_per_hour": RATE_UNITS_PER_HOUR,
                "other_assignments": other_assignments,
            },
            "lease_seconds": 600,
            "max_export_bytes": chunk["export_bytes"],
            # Exact files avoid the transport's prohibited broad research path.
            "source_paths": sorted({*ratification_payload["source_hashes"], *BINDING_DOCS}),
            "round0": {
                "start": start,
                "count": count,
                "lane": lane,
                "chunk_id": chunk_id,
                "attempt": 1,
                "study_root": str(study_root),
                "a5_sha256": sha256_file(a5_receipt),
                "a6_measurement_sha256": receipt["a6_measurement_sha256"],
                "layout": LAYOUT,
                "expected_actual_minutes": chunk["workload_minutes"],
                "export_sizing": {
                    "bytes_per_world": EXPORT_BYTES_PER_WORLD,
                    "factor": EXPORT_SIZING_FACTOR,
                    "projected_bytes": chunk["export_bytes"],
                    "limit_bytes": MAX_EXPORT_BYTES,
                },
                "sizing": chunk,
                "world_ids": world_ids,
                "deadline": {
                    "dispatch_by_unix_seconds": dispatch_by,
                    "completion_within_seconds": int(chunk["timeout_minutes"]) * 60,
                },
            },
        }
        validate_supervisor(manifest, repo=ROOT)
        if not manifests:
            qualify_source(manifest)
        manifests.append(manifest)
        with (out_dir / f"{job_id}.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
    forecast = {
        "schema": "t34-round0-forecast/v1",
        "chunks": len(manifests),
        "lanes": lanes,
        "chunk_pairs": chunk_pairs,
        "study_root": str(study_root),
        "layout": LAYOUT,
        "expected_actual_minutes_per_full_chunk": full_chunk["workload_minutes"],
        "expected_units": sum(
            item["round0"]["expected_actual_minutes"] / 60.0 * RATE_UNITS_PER_HOUR
            for item in manifests
        ),
        "unit_cap": round_sizing["unit_cap"],
        "reserve_units": round_sizing["reserve_units"],
        "worst_case_units": sum(
            item["budget"]["unit_cap"] + item["budget"]["reserve_units"] for item in manifests
        ),
        "mechanical_rerun_liability_units": sum(
            item["budget"]["unit_cap"] + item["budget"]["reserve_units"] for item in manifests
        ),
        "registered_total_liability_units": 2
        * sum(item["budget"]["unit_cap"] + item["budget"]["reserve_units"] for item in manifests),
        "other_assignments": other_assignments,
    }
    with (out_dir / "FORECAST.json").open("x", encoding="utf-8") as stream:
        json.dump(forecast, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return {"manifests": manifests, "forecast": forecast, "out_dir": str(out_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a5", type=Path, required=True)
    parser.add_argument("--ratification", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--lanes", type=int, required=True)
    parser.add_argument("--other-assignments", type=json.loads, required=True)
    parser.add_argument("--setup-minutes", type=float, required=True)
    args = parser.parse_args()
    result = make_manifests(
        a5_receipt=args.a5,
        ratification=args.ratification,
        output_root=args.output_root,
        out_dir=args.out_dir,
        commit=args.commit,
        lanes=args.lanes,
        other_assignments=args.other_assignments,
        setup_minutes=args.setup_minutes,
    )
    print(
        json.dumps({"out_dir": result["out_dir"], "forecast": result["forecast"]}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
