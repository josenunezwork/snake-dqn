"""Validation-only Round 0-D Colab manifests (chunks, A5-D, L4 replay); never allocates.

One builder serves both the operator and ``dispatch.py``'s closure check, so a
dispatched manifest must equal the manifest this module derives from the
ratification for that chunk.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Mapping

from research.colab_cli.supervisor import validate as validate_supervisor
from research.t34_exit_b8.qualification import require_qualified_transport
from research.t34_round0.make_manifest import (
    LAYOUT,
    RATE_UNITS_PER_HOUR,
    RESERVE_FRACTION,
    dispatch_deadline,
)
from research.t34_round0d.bank import ordered_rows
from research.t34_round0d.common import BINDING_DOCS, PACKAGE, ROOT, read_json, sha256

PACKAGE_REL = "research/t34_round0d"
PUBLIC_INPUTS = ROOT / "research/t34_exit_b8/PUBLIC_INPUTS.json"
A5D_TIMEOUT_MINUTES = 300
A5D_ESTIMATE_MINUTES = 150
A5D_WORK_SECONDS = 16200


class ManifestRefused(ValueError):
    """Round 0-D has not supplied an admissible package."""


def _input(path: Path, dest: str) -> dict[str, str]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ManifestRefused(f"missing input {path}")
    return {"path": str(path), "sha256": sha256(path), "dest": dest}


def _source_paths(registration: Mapping[str, Any]) -> list[str]:
    return sorted({*registration["source_hashes"], *BINDING_DOCS})


def chunk_manifest(
    registration: Mapping[str, Any],
    registration_path: Path,
    a5d_path: Path,
    number: int,
    *,
    lane: int,
    other_assignments: Mapping[str, float],
) -> dict[str, Any]:
    """Derive the exact manifest for chunk ``number`` from the frozen ratification."""
    chunks = registration["sizing"]["chunks"]
    chunk = chunks[number]
    study_root = Path(registration["study_root"])
    rows = ordered_rows(read_json(PACKAGE / "BANK_ROUND0D.json")["rows"])
    start, count = int(chunk["start"]), int(chunk["count"])
    chunk_id = f"chunk-{number:03d}"
    job_id = f"t34-round0d-{chunk_id}-attempt-01-l{lane}"
    timeout = int(chunk["timeout_minutes"])
    return {
        "job_id": job_id,
        "commit": registration["commit"],
        "script": f"{PACKAGE_REL}/run.sh",
        "claim": "DRAFT_NO_ALLOCATION_AUTHORIZED",
        "env": {
            "PYTHONUNBUFFERED": "1",
            "T34_INPUT_MODE": "public",
            "T34_DEVICE": "cpu",
            "T34_INPUT_CATALOG": "PUBLIC_INPUTS.json",
            "T34_TELEMETRY_SECONDS": "120",
            "T34_B8_CHECKPOINT": "B8_s3_final.pth",
            "T34_ROUND0D_START": str(start),
            "T34_ROUND0D_COUNT": str(count),
            "T34_ROUND0D_BANK": "BANK_ROUND0D.json",
            "T34_ROUND0D_AUDIT": "SEED_AUDIT_ROUND0D.json",
            "T34_ROUND0D_REGISTRATION": "PREREGISTRATION_ROUND0D.md",
            "T34_ROUND0D_RATIFICATION": "RATIFICATION.json",
            "T34_ROUND0D_A5D_RECEIPT": "A5D_RECEIPT.json",
            "T34_ROUND0D_RUNTIME_ID": job_id,
            "T34_ROUND0D_COMMIT": registration["commit"],
            "T34_ROUND0D_SECONDS": str((timeout - 25) * 60),
            "T34_ROUND0D_THREADS": "1",
            "T34_ROUND0D_PROCESSES": "12",
        },
        "requirements": {
            "gpu": "L4",
            "allowed_gpus": ["L4"],
            "min_vcpu": 12,
            "est_minutes": chunk["estimate_minutes"],
        },
        "timeout_minutes": timeout,
        "output_dir": str(study_root / "chunks" / chunk_id / "attempt-01" / "job"),
        "inputs": [
            _input(PUBLIC_INPUTS, "PUBLIC_INPUTS.json"),
            _input(PACKAGE / "BANK_ROUND0D.json", "BANK_ROUND0D.json"),
            _input(PACKAGE / "SEED_AUDIT_ROUND0D.json", "SEED_AUDIT_ROUND0D.json"),
            _input(PACKAGE / "PREREGISTRATION_ROUND0D.md", "PREREGISTRATION_ROUND0D.md"),
            _input(registration_path, "RATIFICATION.json"),
            _input(a5d_path, "A5D_RECEIPT.json"),
        ],
        "budget": {
            "unit_cap": chunk["unit_cap"],
            "reserve_units": chunk["reserve_units"],
            "rate_units_per_hour": RATE_UNITS_PER_HOUR,
            "other_assignments": dict(other_assignments),
        },
        "lease_seconds": 600,
        "max_export_bytes": chunk["export_bytes"],
        "source_paths": _source_paths(registration),
        "round0d": {
            "start": start,
            "count": count,
            "lane": lane,
            "chunk_id": chunk_id,
            "attempt": 1,
            "study_root": str(study_root),
            "layout": LAYOUT,
            "sizing": dict(chunk),
            "world_ids": [row["world_id"] for row in rows[start : start + count]],
            "deadline": {
                "dispatch_by_unix_seconds": dispatch_deadline(registration["approved_at"]),
                "completion_within_seconds": timeout * 60,
            },
        },
    }


def make_chunk_manifests(
    *,
    ratification: Path,
    a5d: Path,
    out_dir: Path,
    lanes: int,
    other_assignments: dict[str, float],
) -> dict[str, Any]:
    """Write every chunk manifest and a forecast for the operator's review."""
    from research.t34_round0d.registration import AUDIT, BANK, require_registration

    if not 1 <= lanes <= 2 or lanes + len(other_assignments) > 3:
        raise ManifestRefused("Round 0-D uses at most two lanes within the account cap of three")
    registration = read_json(ratification)
    require_registration(registration, BANK, AUDIT, repo=ROOT, mac_side=True)
    require_qualified_transport(ROOT, registration["commit"])
    study_root = Path(registration["study_root"])
    if out_dir.resolve() != study_root / "manifests" or out_dir.exists():
        raise ManifestRefused("manifests must be written to a fresh study_root/manifests")
    # Build the forecast from fields the re-scoped plan always emits, before any
    # create-only write, so a missing field can never leave partial manifests behind.
    sizing = registration["sizing"]
    forecast = {
        "schema": "t34-round0d-forecast/v2",
        "chunks": len(sizing["chunks"]),
        "chunk_pairs": sizing["chunk_pairs"],
        "lanes": lanes,
        "max_in_flight": sizing["max_in_flight"],
        "expected_units": sizing["expected_units"],
        "derived_ceiling": sizing["derived_ceiling"],
        "study_unit_ceiling": sizing["study_unit_ceiling"],
        "priced_chunk_caps_units": sizing["unit_cap"],
        "held_reserve_units": sizing["reserve_units"],
        "replay": sizing["replay"],
        "lane_hours_expected": sizing["lane_hours_expected"],
        "wall_hours_forecast": sizing["wall_hours_forecast"],
    }
    out_dir.mkdir(parents=True)
    manifests = []
    for number in range(len(registration["sizing"]["chunks"])):
        manifest = chunk_manifest(
            registration,
            ratification,
            a5d,
            number,
            lane=number % lanes + 1,
            other_assignments=other_assignments,
        )
        validate_supervisor(manifest, repo=ROOT)
        if not manifests:
            qualify_source(manifest)
        manifests.append(manifest)
        with (out_dir / f"{manifest['job_id']}.json").open("x") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
    with (out_dir / "FORECAST.json").open("x") as stream:
        json.dump(forecast, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return {"manifests": manifests, "forecast": forecast}


def qualify_source(manifest: dict[str, Any]) -> None:
    """Exercise the exact bounded source transport locally before allocation."""
    from research.colab_cli.source import prepare

    with tempfile.TemporaryDirectory(prefix="t34-round0d-source-") as temporary:
        prepare(ROOT, manifest, Path(temporary))


def a5d_manifest(*, commit: str, fidelity: Path, output_dir: Path) -> dict[str, Any]:
    """The single excluded A5-D lane job (5 h timeout, 10 u cap, K=12; may run pre-verdict)."""
    from research.t34_round0d.common import source_hashes

    hashes = source_hashes()
    unit_cap = A5D_TIMEOUT_MINUTES * RATE_UNITS_PER_HOUR / 60.0
    manifest = {
        "job_id": "t34-round0d-a5d-01",
        "commit": commit,
        "script": f"{PACKAGE_REL}/a5d.sh",
        "claim": "EXCLUDED_MEASUREMENT_ONLY",
        "env": {
            "PYTHONUNBUFFERED": "1",
            "T34_INPUT_MODE": "public",
            "T34_INPUT_CATALOG": "PUBLIC_INPUTS.json",
            "T34_B8_CHECKPOINT": "B8_s3_final.pth",
            "T34_ROUND0D_BANK": "BANK_ROUND0D.json",
            "T34_ROUND0D_FIDELITY": "FIDELITY_T3.json",
            "T34_ROUND0D_A5D_SECONDS": str(A5D_WORK_SECONDS),
            "T34_ROUND0D_COMMIT": commit,
        },
        "requirements": {
            "gpu": "L4",
            "allowed_gpus": ["L4"],
            "min_vcpu": 12,
            "est_minutes": A5D_ESTIMATE_MINUTES,
        },
        "timeout_minutes": A5D_TIMEOUT_MINUTES,
        "output_dir": str(output_dir),
        "inputs": [
            _input(PUBLIC_INPUTS, "PUBLIC_INPUTS.json"),
            _input(PACKAGE / "BANK_ROUND0D.json", "BANK_ROUND0D.json"),
            _input(fidelity, "FIDELITY_T3.json"),
        ],
        "budget": {
            "unit_cap": unit_cap,
            "reserve_units": unit_cap * RESERVE_FRACTION,
            "rate_units_per_hour": RATE_UNITS_PER_HOUR,
            "other_assignments": {},
        },
        "lease_seconds": 600,
        "max_export_bytes": 512 * 1024**2,
        "source_paths": sorted({*hashes, *BINDING_DOCS}),
    }
    if not math.isclose(unit_cap, 10.0):
        raise ManifestRefused("A5-D must stay within its 10 u cap")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    chunks = sub.add_parser("chunks")
    chunks.add_argument("--ratification", type=Path, required=True)
    chunks.add_argument("--a5d", type=Path, required=True)
    chunks.add_argument("--out-dir", type=Path, required=True)
    chunks.add_argument("--lanes", type=int, required=True)
    chunks.add_argument("--other-assignments", type=json.loads, required=True)
    a5d = sub.add_parser("a5d")
    a5d.add_argument("--commit", required=True)
    a5d.add_argument("--fidelity", type=Path, required=True)
    a5d.add_argument("--output-dir", type=Path, required=True)
    a5d.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "chunks":
        result = make_chunk_manifests(
            ratification=args.ratification,
            a5d=args.a5d,
            out_dir=args.out_dir,
            lanes=args.lanes,
            other_assignments=args.other_assignments,
        )
        print(json.dumps(result["forecast"], sort_keys=True))
    else:
        require_qualified_transport(ROOT, args.commit)
        manifest = a5d_manifest(
            commit=args.commit, fidelity=args.fidelity, output_dir=args.output_dir
        )
        validate_supervisor(manifest, repo=ROOT)
        qualify_source(manifest)
        with args.out.open("x") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(json.dumps({"job_id": manifest["job_id"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
