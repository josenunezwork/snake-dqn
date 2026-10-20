"""Rebuild a non-dispatchable Round 0 draft from saved measured costs, without gameplay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from research.colab_cli.supervisor import validate as validate_manifest
from research.t34_round0.bank import ordered_rows
from research.t34_round0.common import (
    PACKAGE,
    ROOT,
    head,
    read_json,
    sha256,
    source_hashes,
    write_once,
)
from research.t34_round0.make_manifest import LAYOUT, size_round, validate_a5
from research.t34_round0.registration import check_audit
from research.t34_round0.sizing import verify_adapter
from research.t34_round0.validate import validate


def build(a5_path: Path, out: Path, study_root: Path) -> dict[str, Any]:
    """Write full draft manifests whose DRAFT registration blocks production admission."""
    if out.exists() or not out.is_absolute() or study_root.resolve() != study_root:
        raise ValueError("draft output must be fresh and absolute; study root must be canonical")
    adapter = validate_a5(read_json(a5_path))
    verify_adapter(adapter)
    preparation = validate()
    bank = check_audit(PACKAGE / "BANK_ROUND0.json", PACKAGE / "SEED_AUDIT_ROUND0.json")
    rows = ordered_rows(bank["rows"])
    hashes = source_hashes()
    sizing = size_round(
        len(rows),
        **{
            key: adapter[key]
            for key in (
                "teacher_worlds_per_hour",
                "bare_seconds_per_world",
                "metric_seconds_per_pair",
                "cpu_slowdown_factor",
                "setup_minutes",
            )
        },
    )
    chunks = []
    for number, chunk in enumerate(sizing["chunks"]):
        group = rows[chunk["start"] : chunk["start"] + chunk["count"]]
        chunks.append(
            dict(
                chunk,
                chunk_id=f"chunk-{number:03d}",
                world_ids=[row["world_id"] for row in group],
                mix_counts={
                    mix: sum(r["mix"] == mix for r in group)
                    for mix in ("scripted", "frozen", "mixed")
                },
            )
        )
    current_digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    registration = {
        "schema": "t34-round0-ratification/v1",
        "status": "DRAFT_PENDING_OPERATOR_RATIFICATION",
        "commit": head(),
        "study_root": str(study_root),
        "history_ledger": {
            "repo": str(ROOT),
            "path": f"research/t34_round0/ledger/{study_root.name}.jsonl",
        },
        "registration_sha256": sha256(PACKAGE / "PREREGISTRATION_ROUND0.md"),
        "bank_sha256": sha256(PACKAGE / "BANK_ROUND0.json"),
        "audit_sha256": sha256(PACKAGE / "SEED_AUDIT_ROUND0.json"),
        "source_hashes": hashes,
        "sizing": sizing,
        "dispatch_window_hours": 48,
        "a5": {"path": str(a5_path), "sha256": sha256(a5_path)},
        "historical_a6_source_sha256": adapter["a6_source_sha256"],
        "current_a6_source_sha256": current_digest,
        "blockers": [
            "Source-current excluded A6 and independent ratification required",
            "Canonical registration JSON must be committed and pushed to main",
        ],
        "ratified": False,
        "dispatch_authorized": False,
        "release_authorized": False,
    }
    write_once(out / "REGISTRATION_DRAFT.json", registration)
    write_once(out / "CHUNKS.json", {"schema": "t34-round0-draft-chunks/v1", "chunks": chunks})
    inputs = [
        {"path": str(path), "sha256": sha256(path), "dest": dest}
        for path, dest in (
            (ROOT / "research/t34_exit_b8/PUBLIC_INPUTS.json", "PUBLIC_INPUTS.json"),
            (PACKAGE / "BANK_ROUND0.json", "BANK_ROUND0.json"),
            (PACKAGE / "SEED_AUDIT_ROUND0.json", "SEED_AUDIT_ROUND0.json"),
            (PACKAGE / "PREREGISTRATION_ROUND0.md", "PREREGISTRATION_ROUND0.md"),
            (out / "REGISTRATION_DRAFT.json", "RATIFICATION.json"),
            (a5_path, "A5_RECEIPT.json"),
        )
    ]
    manifests = []
    for chunk in chunks:
        job_id = f"t34-round0-{chunk['chunk_id']}-attempt-01-l1"
        env = {
            "PYTHONUNBUFFERED": "1",
            "T34_INPUT_MODE": "public",
            "T34_DEVICE": "cpu",
            "T34_ROUND0_START": str(chunk["start"]),
            "T34_ROUND0_COUNT": str(chunk["count"]),
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
            "T34_ROUND0_COMMIT": head(),
            "T34_ROUND0_SECONDS": str((chunk["timeout_minutes"] - 25) * 60),
            "T34_ROUND0_THREADS": "1",
            "T34_ROUND0_PROCESSES": "12",
            "T34_ROUND0_WORLDS_PER_PROCESS": "1",
        }
        manifest = {
            "job_id": job_id,
            "commit": head(),
            "script": "research/t34_round0/run.sh",
            "claim": "DRAFT_NO_ALLOCATION_AUTHORIZED",
            "env": env,
            "requirements": {
                "gpu": "L4",
                "allowed_gpus": ["L4"],
                "min_vcpu": 12,
                "est_minutes": chunk["estimate_minutes"],
            },
            "timeout_minutes": chunk["timeout_minutes"],
            "output_dir": str(study_root / "chunks" / chunk["chunk_id"] / "attempt-01/job"),
            "inputs": inputs,
            "budget": {
                "unit_cap": chunk["unit_cap"],
                "reserve_units": chunk["reserve_units"],
                "rate_units_per_hour": 2.0,
                "other_assignments": {},
            },
            "lease_seconds": 600,
            "max_export_bytes": chunk["export_bytes"],
            "source_paths": sorted(hashes),
            "round0": {
                "start": chunk["start"],
                "count": chunk["count"],
                "lane": 1,
                "chunk_id": chunk["chunk_id"],
                "attempt": 1,
                "study_root": str(study_root),
                "layout": LAYOUT,
                "world_ids": chunk["world_ids"],
                "deadline": {
                    "dispatch_by_unix_seconds": None,
                    "completion_within_seconds": chunk["timeout_minutes"] * 60,
                },
                "sizing": sizing["chunks"][len(manifests)],
            },
        }
        validate_manifest(manifest, repo=ROOT)
        write_once(out / "manifests" / f"{job_id}.json", manifest)
        manifests.append(manifest)
    result = {
        "schema": "t34-round0-fixes1-draft-validation/v1",
        "status": "PASS",
        "commit": head(),
        "selection_worlds": len(rows),
        "chunks": len(chunks),
        "draft_manifests_validated": len(manifests),
        "initial_cap_plus_reserve_units": sizing["unit_cap"] + sizing["reserve_units"],
        "maximum_rerun_liability_units": 2 * (sizing["unit_cap"] + sizing["reserve_units"]),
        "bank_sha256": registration["bank_sha256"],
        "source_hashes": hashes,
        "package_validation": preparation["status"],
        "ratified": False,
        "dispatch_authorized": False,
        "gameplay_run": False,
        "source_current_preflight": adapter["a6_source_sha256"] == current_digest,
    }
    write_once(out / "VALIDATION.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a5", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--study-root", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.a5, args.out, args.study_root)
    print(json.dumps({key: value for key, value in result.items() if key != "source_hashes"}))


if __name__ == "__main__":
    main()
