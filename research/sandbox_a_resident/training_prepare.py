"""Seal TRAIN-01 against a published source pin; never allocate a runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from research.sandbox_a_resident.prepare import runtime_source_paths
from research.sandbox_a_resident.training_task import seed_allocations
from research.sandbox_a_resident.training_units import empty_ledger
from tools.sandbox.registry import (
    TRAINING_OPPONENTS,
    digest,
    load_registry,
    validate_input,
    verify_sources,
    world_seed,
)

ROOT = Path(__file__).resolve().parents[2]
STUDY = "sandbox-a-train01"
ARTIFACT_ROOT = ROOT.parent / "snake-dqn-artifacts/sandbox-a-resident/train01-attempt-01"


def write_once(path: Path, value: dict[str, Any]) -> None:
    """Create JSON without replacing prior evidence."""
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def source_paths(commit: str, *, repo_root: Path = ROOT) -> list[str]:
    """Retain A-3's runtime closure and add the isolated training modules."""
    tracked = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit, "src", "research/sandbox_a_resident"],
        cwd=repo_root,
        text=True,
    ).splitlines()
    additions = [
        name
        for name in tracked
        if (name.startswith("src/") and name.endswith(".py"))
        or (
            Path(name).parent.as_posix() == "research/sandbox_a_resident"
            and (
                Path(name).name
                in {"train.py", "train_job.sh", "stall_watchdog.py", "compact_allowlist.json"}
                or (
                    Path(name).name.startswith("training_")
                    and Path(name).name
                    not in {
                        "training_prepare.py",
                        "training_preflight.py",
                        "training_continuation.py",
                    }
                )
            )
        )
    ]
    return sorted(set(runtime_source_paths(commit, repo_root=repo_root)) | set(additions))


def audit_seeds(allocations: list[dict[str, Any]], registry_path: Path) -> dict[str, Any]:
    """Exhaustively audit low63 aliases and exclusions without Python object banks.

    The create-only audit is committed once during preparation. Admission binds
    it to its allocation and registry rather than repeating 114M SHA operations
    during every broker validation. No numerical training payload is read.
    """
    registry = load_registry(registry_path)
    verify_sources(registry)
    ordered = sorted(allocations, key=lambda row: row["offset"])
    for row in ordered:
        if type(row["offset"]) is not int or type(row["count"]) is not int:
            raise ValueError("integer seed allocations required")
        if row["offset"] < 0 or row["count"] < 1:
            raise ValueError("positive finite seed allocation required")
    if any(a["offset"] + a["count"] > b["offset"] for a, b in zip(ordered, ordered[1:])):
        raise ValueError("overlapping seed allocations")
    count = sum(row["count"] for row in ordered)
    prefix = hashlib.sha256(f"a100-dev/{STUDY}/".encode())
    seed_hash = hashlib.sha256()
    with tempfile.TemporaryDirectory(prefix="sandbox-train-seed-audit-") as directory:
        values = np.memmap(Path(directory) / "keys", mode="w+", dtype=np.uint64, shape=(count,))
        cursor = 0
        for row in ordered:
            print(f"AUDIT_SEEDS phase={row['phase']} count={row['count']}", flush=True)
            for index in range(row["offset"], row["offset"] + row["count"]):
                key = prefix.copy()
                key.update(str(index).encode())
                raw = key.digest()[:8]
                seed_hash.update(raw)
                values[cursor] = int.from_bytes(raw, "big") & 0x7FFF_FFFF_FFFF_FFFF
                cursor += 1
        values.sort()
        if values[0] == 0:
            raise ValueError("zero effective seed")
        for start in range(0, count - 1, 1_000_000):
            end = min(count - 1, start + 1_000_000)
            if np.any(values[start:end] == values[start + 1 : end + 1]):
                raise ValueError("effective seed alias")
        for seed in registry["excluded_seeds"]:
            effective = int(seed) & 0x7FFF_FFFF_FFFF_FFFF
            index = int(np.searchsorted(values, effective))
            if index < count and int(values[index]) == effective:
                raise ValueError("excluded seed intersection")
        del values
    return {
        "schema": "sandbox-a-train-seed-audit/v1",
        "status": "PASS",
        "study": STUDY,
        "allocations": allocations,
        "count": count,
        "canonical_uint64_sha256": seed_hash.hexdigest(),
        "registry_sha256": digest(registry_path),
        "effective_seed_bits": 63,
        "aliases": 0,
        "excluded_intersection": 0,
    }


def prepare(
    packet: Path,
    commit: str,
    checkpoints: dict[str, Path],
    registry: Path,
    *,
    seed_audit: Path | None = None,
    allocation_index: int = 0,
    train_seed_offset: int = 0,
    job_id: str = "sandbox-a-train01-20261027-01",
    output_dir: Path | None = None,
    est_minutes: int = 480,
    timeout_minutes: int = 960,
    unit_cap: float = 129.0,
    sandbox_source_delivery: str | None = None,
    stall_timeout_seconds: int | None = None,
) -> None:
    """Create a finite training packet and ten-line adopted sandbox manifest."""
    from research.sandbox_a_resident.train import TrainControl

    if (
        type(est_minutes) is not int
        or est_minutes < 1
        or type(timeout_minutes) is not int
        or timeout_minutes != 2 * est_minutes
    ):
        raise ValueError("timeout_minutes must equal 2x integer est_minutes")
    base_seconds = est_minutes * 60
    effective_seconds = timeout_minutes * 60
    resolved_output = ARTIFACT_ROOT / "output" if output_dir is None else output_dir
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"], cwd=ROOT, text=True
    ).strip()
    control = asdict(
        TrainControl(
            study=STUDY,
            source_commit=commit,
            allocation_index=allocation_index,
            train_seed_offset=train_seed_offset,
        )
    )
    for name, path in checkpoints.items():
        validate_input(
            {
                "kind": "frozen-checkpoint" if name == "b8-s3" else "frozen-opponent",
                "path": str(path),
                "sha256": digest(path),
            }
        )
    if {digest(p) for name, p in checkpoints.items() if name != "b8-s3"} != TRAINING_OPPONENTS:
        raise ValueError("both canonical opponent checkpoints required")
    allocations = seed_allocations(control)
    check = (
        audit_seeds(allocations, registry)
        if seed_audit is None
        else json.loads(seed_audit.read_text())
    )
    if (
        check.get("status") != "PASS"
        or check.get("allocations") != allocations
        or check.get("registry_sha256") != digest(registry)
        or check.get("aliases") != 0
        or check.get("excluded_intersection") != 0
    ):
        raise ValueError("seed audit does not bind this exact allocation and registry")
    packet.mkdir(parents=True, exist_ok=False)
    write_once(packet / "control.json", control)
    write_once(packet / "registry-check.json", check)
    write_once(packet / "verified-ledger.json", empty_ledger(control))
    uploads = [
        {"path": str(path.resolve()), "dest": name + ".pth", "sha256": digest(path)}
        for name, path in checkpoints.items()
    ] + [
        {"path": str(registry.resolve()), "dest": "registry.json", "sha256": digest(registry)},
        {
            "path": str(packet / "verified-ledger.json"),
            "dest": "verified-ledger.json",
            "sha256": digest(packet / "verified-ledger.json"),
        },
    ]
    env = {
        "SANDBOX_A_CONTROL_JSON": json.dumps(control, sort_keys=True),
        "SANDBOX_A_SOURCE_COMMIT": commit,
        "SNAKE_DQN_DEVICE": "cuda",
        "MAX_JOBS": "1",
        "COLAB_CLI_REQUIREMENTS": "research/sandbox_a_resident/requirements.txt",
    }
    if stall_timeout_seconds is not None:
        if type(stall_timeout_seconds) is not int or not 0 < stall_timeout_seconds <= 900:
            raise ValueError("stall_timeout_seconds must be an integer in 1..900")
        env["SANDBOX_A_STALL_TIMEOUT_SECONDS"] = str(stall_timeout_seconds)
    manifest: dict[str, Any] = {
        "job_id": job_id,
        "commit": commit,
        "script": "research/sandbox_a_resident/train_job.sh",
        "device": "cuda",
        "source_paths": source_paths(commit),
        "requirements": {
            "gpu": "A100",
            "allowed_gpus": ["A100"],
            "min_vcpu": 2,
            "est_minutes": est_minutes,
        },
        "timeout_minutes": timeout_minutes,
        "output_dir": str(resolved_output),
        "max_export_bytes": 2 * 1024**3,
        "inputs": uploads,
        "env": env,
        "required_exports": ["training/summary.json"],
        "budget": {
            "rate_units_per_hour": 8.0,
            "unit_cap": float(unit_cap),
            "reserve_units": 0,
            "other_assignments": {},
        },
        "resume": {
            "unit_kind": "training-full-trajectory-snapshot",
            "ledger_path": "verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "sandbox_admission": {
            "schema": "standing-a100-development-sandbox-admission/v1",
            "manifest_path": str(packet / "sandbox.json"),
        },
        "sandbox_execution": "finite simulator-D calibration, critic fit and TRAIN-01; no crossing",
        "requirements_file": "research/sandbox_a_resident/requirements.txt",
        "sandbox_a": {
            "seed_allocations": allocations,
            "seed_audit_sha256": digest(packet / "registry-check.json"),
        },
        "technical_envelope": {
            "base_seconds": base_seconds,
            "effective_seconds": effective_seconds,
            "endpoint": "100663296 actor rows plus bounded preparation and proxy",
        },
    }
    if sandbox_source_delivery is not None:
        if sandbox_source_delivery != "background-sparse-v1":
            raise ValueError("sandbox_source_delivery must be background-sparse-v1")
        manifest["sandbox_source_delivery"] = sandbox_source_delivery
    write_once(packet / "dispatch-manifest.json", manifest)
    roots = [row["offset"] for row in allocations]
    inputs = [
        {
            "kind": "worlds",
            "study": STUDY,
            "indices": roots,
            "seeds": [world_seed(STUDY, i) for i in roots],
        }
    ]
    for row in uploads:
        dest = row["dest"]
        kind = (
            "frozen-checkpoint"
            if dest == "b8-s3.pth"
            else (
                "registry"
                if dest == "registry.json"
                else "verified-ledger" if dest == "verified-ledger.json" else "frozen-opponent"
            )
        )
        inputs.append({"kind": kind, "path": row["path"], "sha256": row["sha256"]})
    sandbox = {
        "schema": "standing-a100-development-sandbox/v1",
        "study": STUDY,
        "inputs": inputs,
        "salt": f"a100-dev/{STUDY}/",
        "endpoint": {"kind": "steps", "value": 100_663_296},
        "timeout": {
            "base_seconds": base_seconds,
            "seconds": effective_seconds,
            "purpose": "runaway-guard",
        },
        "excluded_studies": [
            "F5",
            "F4-fold-0",
            "29-census-parents",
            "R30",
            "strict",
            "probe",
            "serving",
            "D4096",
            "t34-round0-study-01",
        ],
        "owner_rule": "docs/research/governance_a100_sandbox_20261026/ADOPTED.md",
        "outputs": {
            "label": "simulator-D",
            "paths": [manifest["output_dir"]],
            "broker_manifest": {
                "path": str(packet / "dispatch-manifest.json"),
                "sha256": digest(packet / "dispatch-manifest.json"),
            },
        },
        "crossing": {"max_frozen_checkpoint_shas": 1},
    }
    lines = [
        ("{" if i == 0 else " ") + json.dumps(k) + ": " + json.dumps(v) + ("," if i < 9 else "}")
        for i, (k, v) in enumerate(sandbox.items())
    ]
    with (packet / "sandbox.json").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    print(f"PREPARED_TRAIN01 pin={commit} seeds={check['count']} allocations=0")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--seed-audit", type=Path)
    parser.add_argument("--allocation-index", type=int, default=0)
    parser.add_argument("--train-seed-offset", type=int, default=0)
    parser.add_argument("--job-id", default="sandbox-a-train01-20261027-01")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--est-minutes", type=int, default=480)
    parser.add_argument("--timeout-minutes", type=int, default=960)
    parser.add_argument("--unit-cap", type=float, default=129.0)
    parser.add_argument("--sandbox-source-delivery")
    parser.add_argument("--stall-timeout-seconds", type=int)
    args = parser.parse_args()
    prepare(
        args.packet.resolve(),
        args.commit,
        {
            "b8-s3": args.checkpoint_dir / "b8_s3_t7_20261016.pth",
            "frp3-s12": args.checkpoint_dir / "frp3_m3_s12_u60000_20261005.pth",
            "champion": args.checkpoint_dir / "champion_a5_freespace_20260621.pth",
        },
        ROOT / "tools/sandbox/SANDBOX_INPUT_REGISTRY.json",
        seed_audit=args.seed_audit,
        allocation_index=args.allocation_index,
        train_seed_offset=args.train_seed_offset,
        job_id=args.job_id,
        output_dir=None if args.output_dir is None else args.output_dir.resolve(),
        est_minutes=args.est_minutes,
        timeout_minutes=args.timeout_minutes,
        unit_cap=args.unit_cap,
        sandbox_source_delivery=args.sandbox_source_delivery,
        stall_timeout_seconds=args.stall_timeout_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
