"""Build bounded, validation-only T34 Round 0 Colab manifests.

This module intentionally has no allocation or dispatch operation.  It binds a
ratified registration, the audited bank, public input catalog, and the measured
A5 receipt before handing the result to the Colab CLI's local validator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any

from research.colab_cli.supervisor import validate as validate_supervisor
from research.t34_exit_b8.qualification import QualificationRefused, require_qualified_transport
from research.t34_round0.common import BINDING_DOCS
from research.t34_round0.registration import require_registration

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "research/t34_round0"
RATE_UNITS_PER_HOUR = 2.0
MAX_CHUNK_PAIRS = 240
MAX_ACTUAL_MINUTES = 180
TIMEOUT_MINUTES = 360
HARD_OPERATIONAL_UNITS = 12.0
RESERVE_FRACTION = 0.20
# Keep the published decimal caps stable in serialized manifest evidence.
RESERVE_UNITS = 2.4
UNIT_CAP = HARD_OPERATIONAL_UNITS + RESERVE_UNITS
MAX_EXPORT_BYTES = 2 * 1024**3
EXPORT_SIZING_FACTOR = 1.25
EXPORT_SIZING_RESERVE_BYTES = 16 * 1024**2


class ManifestRefused(ValueError):
    """Round 0 has not supplied a complete, admissible registration package."""


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
        "paired_worlds_per_hour": None,
        "completed_worlds_per_hour_per_lane": None,
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
        "paired_worlds_per_hour",
        "completed_worlds_per_hour_per_lane",
        "active_row_steps_per_second",
        "measured_export_bytes_per_pair",
    ):
        value = receipt[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ManifestRefused(f"A5 receipt {key} must be finite and positive")
    if receipt["completed_worlds_per_hour_per_lane"] < 50:
        raise ManifestRefused("A5 receipt misses the 50 completed worlds/hour per lane floor")
    if not _hex(receipt["evidence_sha256"]):
        raise ManifestRefused("A5 receipt must hash-bound its evidence")
    return receipt


def _verify_a5(path: Path) -> dict[str, Any]:
    return validate_a5(_load_json(path, "A5 receipt"))


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
    ratification_payload = _verify_ratification(
        ratification,
        commit=commit,
        registration=registration,
        bank=bank,
        audit=audit,
        a5=a5_receipt,
    )
    require_registration(ratification_payload, bank, audit, repo=ROOT)
    try:
        require_qualified_transport(ROOT, commit)
    except QualificationRefused as exc:
        raise ManifestRefused(str(exc)) from exc

    paired_per_hour = float(receipt["paired_worlds_per_hour"])
    by_rate = math.floor(2.5 * paired_per_hour)
    by_time = math.floor((MAX_ACTUAL_MINUTES - float(setup_minutes)) * paired_per_hour / 60.0)
    chunk_pairs = min(MAX_CHUNK_PAIRS, by_rate, by_time)
    if chunk_pairs < 1:
        raise ManifestRefused("A5 rate and measured setup leave no pair within the 180-minute cap")
    actual_minutes = float(setup_minutes) + 60.0 * chunk_pairs / paired_per_hour
    if actual_minutes > MAX_ACTUAL_MINUTES + 1e-9:
        raise ManifestRefused("chunk estimate exceeds the 180-minute actual-work cap")
    projected_export_bytes = (
        math.ceil(
            chunk_pairs * float(receipt["measured_export_bytes_per_pair"]) * EXPORT_SIZING_FACTOR
        )
        + EXPORT_SIZING_RESERVE_BYTES
    )
    if projected_export_bytes > MAX_EXPORT_BYTES:
        raise ManifestRefused("A5 measured export size exceeds the 2 GiB chunk export limit")

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
    for number, start in enumerate(range(0, len(rows), chunk_pairs)):
        count = min(chunk_pairs, len(rows) - start)
        chunk_actual_minutes = float(setup_minutes) + 60.0 * count / paired_per_hour
        chunk_export_bytes = (
            math.ceil(
                count * float(receipt["measured_export_bytes_per_pair"]) * EXPORT_SIZING_FACTOR
            )
            + EXPORT_SIZING_RESERVE_BYTES
        )
        lane = number % lanes + 1
        job_id = f"t34-round0-c{number:03d}-l{lane}"
        manifest = {
            "job_id": job_id,
            "commit": commit,
            "script": f"{PACKAGE}/run.sh",
            "claim": "DRAFT_NO_ALLOCATION_AUTHORIZED",
            "env": {
                "PYTHONUNBUFFERED": "1",
                "T34_INPUT_MODE": "public",
                "T34_DEVICE": "cuda",
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
                "T34_ROUND0_SECONDS": "21000",
                "T34_ROUND0_THREADS": "5",
            },
            "requirements": {
                "gpu": "L4",
                "allowed_gpus": ["L4"],
                "min_vcpu": 12,
                "est_minutes": math.ceil(chunk_actual_minutes),
            },
            "timeout_minutes": TIMEOUT_MINUTES,
            "output_dir": str(output_root / job_id),
            "inputs": inputs,
            "budget": {
                "unit_cap": UNIT_CAP,
                "reserve_units": RESERVE_UNITS,
                "rate_units_per_hour": RATE_UNITS_PER_HOUR,
                "other_assignments": other_assignments,
            },
            "lease_seconds": 600,
            "max_export_bytes": MAX_EXPORT_BYTES,
            # Exact files avoid the transport's prohibited broad research path.
            "source_paths": sorted({*ratification_payload["source_hashes"], *BINDING_DOCS}),
            "round0": {
                "start": start,
                "count": count,
                "lane": lane,
                "a5_sha256": sha256_file(a5_receipt),
                "expected_actual_minutes": chunk_actual_minutes,
                "export_sizing": {
                    "measured_export_bytes_per_pair": receipt["measured_export_bytes_per_pair"],
                    "factor": EXPORT_SIZING_FACTOR,
                    "reserve_bytes": EXPORT_SIZING_RESERVE_BYTES,
                    "projected_bytes": chunk_export_bytes,
                    "limit_bytes": MAX_EXPORT_BYTES,
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
        "setup_minutes": float(setup_minutes),
        "expected_actual_minutes_per_full_chunk": actual_minutes,
        "expected_units": sum(
            item["round0"]["expected_actual_minutes"] / 60.0 * RATE_UNITS_PER_HOUR
            for item in manifests
        ),
        "worst_case_units": len(manifests) * UNIT_CAP,
        "reserve_units": len(manifests) * RESERVE_UNITS,
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
