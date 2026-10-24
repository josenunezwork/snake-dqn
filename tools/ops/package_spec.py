"""Small, fail-closed package constructor for resumed Colab verification work.

This module deliberately does not allocate, dispatch, authenticate, or implement
another transport.  It materializes input names exactly as the runtime will see
them, records their sidecars, and produces a manifest consumed by the existing
``research.colab_cli`` supervisor.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

SHA256 = re.compile(r"[0-9a-f]{64}")
SAFE_RELATIVE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")
LEDGER_SCHEMA = "r33-verified-ledger/v1"
VOLATILE_FIELDS = ("elapsed_seconds", "host_elapsed_seconds", "generated_at")


class PackageRefused(ValueError):
    """A package cannot safely be prepared."""


def digest(path: Path) -> str:
    """Return a SHA-256 without interpreting the file's contents."""
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _relative(value: str, label: str) -> str:
    if not isinstance(value, str) or not SAFE_RELATIVE.fullmatch(value):
        raise PackageRefused(f"{label} must be a normalized relative path")
    parts = value.split("/")
    if value.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        raise PackageRefused(f"{label} must be a normalized relative path")
    return value


def _safe_id(value: str, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise PackageRefused(f"{label} must be a safe nonempty identifier")
    return value


def _publish(path: Path, value: bytes) -> None:
    """Create immutable bytes; callers publish completion markers last."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    fd, temporary = tempfile.mkstemp(prefix=".r33-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def publish_immutable(path: Path, payload: bytes) -> dict[str, Any]:
    """Publish payload, destination-bound sidecar, then completion marker."""
    _publish(path, payload)
    checksum = hashlib.sha256(payload).hexdigest()
    _publish(Path(f"{path}.sha256"), f"{checksum}  {path.name}\n".encode())
    _publish(Path(f"{path}.complete"), b"complete\n")
    return {"path": str(path), "sha256": checksum, "size": len(payload)}


def publish_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Publish canonical JSON as an immutable three-file export."""
    return publish_immutable(
        path, (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    )


def verify_immutable(path: Path) -> dict[str, Any]:
    """Require payload, matching destination-sidecar, and final marker."""
    path = Path(path)
    if any(
        Path(f"{path}{suffix}").is_symlink() or not Path(f"{path}{suffix}").is_file()
        for suffix in ("", ".sha256", ".complete")
    ):
        raise PackageRefused(f"missing or unsafe immutable export: {path}")
    if Path(f"{path}.complete").read_bytes() != b"complete\n":
        raise PackageRefused(f"invalid completion marker: {path}")
    observed = digest(path)
    if Path(f"{path}.sha256").read_text().split() != [observed, path.name]:
        raise PackageRefused(f"destination-sidecar mismatch: {path}")
    return {"path": str(path), "sha256": observed, "size": path.stat().st_size}


def frozen_identity(*, pin: str, environment_sha256: str, packet: str) -> dict[str, str]:
    """Validate the identity which a ledger must preserve across attempts."""
    if (
        re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", pin) is None
        or SHA256.fullmatch(environment_sha256) is None
    ):
        raise PackageRefused("pin and environment identity must be SHA-256 values")
    return {
        "pin": pin,
        "environment_sha256": environment_sha256,
        "packet": _safe_id(packet, "packet"),
    }


def empty_ledger(
    namespace: str, allocation: str, identity: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Create the uploaded first-attempt ledger without reading outcomes."""
    return {
        "schema": LEDGER_SCHEMA,
        "namespace": _safe_id(namespace, "namespace"),
        "allocation": _safe_id(allocation, "allocation"),
        "identity": dict(identity) if identity is not None else None,
        "units": [],
    }


def unit_id(namespace: str, allocation: str, index: int) -> str:
    """Derive a deterministic completed-unit identity from the frozen index."""
    _safe_id(namespace, "namespace")
    _safe_id(allocation, "allocation")
    if type(index) is not int or index < 0:
        raise PackageRefused("allocation index must be a nonnegative integer")
    return f"{namespace}/{allocation}/{index:06d}"


def read_ledger(
    path: Path, namespace: str, allocation: str, identity: Mapping[str, str] | None = None
) -> list[dict[str, Any]]:
    """Validate a Mac-verified ledger and return only its identity metadata."""
    verify_immutable(path)
    value = json.loads(Path(path).read_text())
    if (
        not isinstance(value, dict)
        or value.get("schema") != LEDGER_SCHEMA
        or value.get("namespace") != namespace
        or value.get("allocation") != allocation
        or not isinstance(value.get("units"), list)
        or (identity is not None and value.get("identity") != dict(identity))
    ):
        raise PackageRefused("foreign or malformed verified ledger")
    checked: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for row in value["units"]:
        if not isinstance(row, dict) or type(row.get("index")) is not int:
            raise PackageRefused("malformed verified-ledger unit")
        expected = unit_id(namespace, allocation, row["index"])
        if (
            row.get("unit_id") != expected
            or not isinstance(row.get("sha256"), str)
            or SHA256.fullmatch(row["sha256"]) is None
            or type(row.get("size")) is not int
            or row["size"] < 0
            or expected in identifiers
        ):
            raise PackageRefused("foreign, duplicate, or malformed verified-ledger unit")
        identifiers.add(expected)
        checked.append(
            {
                "unit_id": expected,
                "index": row["index"],
                "sha256": row["sha256"],
                "size": row["size"],
            }
        )
    return sorted(checked, key=lambda row: row["index"])


@dataclasses.dataclass(frozen=True)
class Input:
    """A local input whose destination bytes must be bound before upload."""

    source: Path
    destination: str


def stage_inputs(staging: Path, inputs: Iterable[Input]) -> list[dict[str, Any]]:
    """Copy sources under runtime destinations and bind destination-sidecars."""
    rows: list[dict[str, Any]] = []
    destinations: set[str] = set()
    staging = Path(staging)
    for item in inputs:
        destination = _relative(item.destination, "input destination")
        if destination in destinations:
            raise PackageRefused("duplicate input destination")
        destinations.add(destination)
        source = Path(item.source).resolve(strict=True)
        if not source.is_file():
            raise PackageRefused(f"input is not a file: {source}")
        target = staging / destination
        if any(Path(f"{target}{suffix}").exists() for suffix in ("", ".sha256", ".complete")):
            raise FileExistsError(f"immutable staged input already exists: {target}")
        if not target.resolve().is_relative_to(staging.resolve()):
            raise PackageRefused("staged destination escapes through symlink")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = source.read_bytes()
        _publish(target, payload)
        checksum = hashlib.sha256(payload).hexdigest()
        _publish(Path(f"{target}.sha256"), f"{checksum}  {target.name}\n".encode())
        _publish(Path(f"{target}.complete"), b"complete\n")
        verified = verify_immutable(target)
        rows.append(
            {
                "path": verified["path"],
                "dest": destination,
                "sha256": verified["sha256"],
                "size_bytes": verified["size"],
                "sidecars": [f"{destination}.sha256", f"{destination}.complete"],
            }
        )
        for suffix in (".sha256", ".complete"):
            sidecar = Path(f"{target}{suffix}")
            rows.append(
                {
                    "path": str(sidecar),
                    "dest": f"{destination}{suffix}",
                    "sha256": digest(sidecar),
                    "size_bytes": sidecar.stat().st_size,
                    "sidecars": [],
                }
            )
    return rows


def canonical_identity(
    value: Mapping[str, Any], *, volatile: Iterable[str] = VOLATILE_FIELDS
) -> dict[str, Any]:
    """Return fields used to compare interrupted and resumed synthetic units."""
    excluded = tuple(volatile)
    if excluded != VOLATILE_FIELDS:
        raise PackageRefused("volatile allowlist is not frozen")
    return {key: value[key] for key in sorted(value) if key not in set(excluded)}


def require_published_commit(repo: Path, commit: str) -> str:
    """Refuse unpublished pins; manifests are never written against a local SHA."""
    repo = Path(repo).resolve()
    resolved = subprocess.check_output(
        ["git", "rev-parse", f"{commit}^{{commit}}"], cwd=repo, text=True
    ).strip()
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", resolved, "origin/main"], cwd=repo
    ).returncode:
        raise PackageRefused("manifest pin is not published on origin/main")
    return resolved


def build_cpu_manifest(
    *,
    repo: Path,
    commit: str,
    output_dir: Path,
    staged_inputs: list[dict[str, Any]],
    script: str,
    namespace: str,
    allocation: str,
    authority_receipts: list[dict[str, str]],
    job_id: str = "r33-cpu-pilot-01",
) -> dict[str, Any]:
    """Construct the row-2/3 CPU pilot manifest with the frozen effective envelope.

    The manifest is prepared only after the exact pin is published.  Dispatch remains
    an operator action through ``tools/ops/dispatch.sh``.
    """
    from research.r33_verification.contract import SOURCE_PATHS

    commit = require_published_commit(repo, commit)
    _relative(script, "script")
    ledger = next((row for row in staged_inputs if row["dest"] == "verified-ledger.json"), None)
    if ledger is None:
        raise PackageRefused("uploaded verified-ledger input is required")
    if ledger["path"].endswith("verified-ledger.json") is False:
        raise PackageRefused("uploaded ledger must retain its declared runtime destination")
    if not authority_receipts:
        raise PackageRefused("required authority receipts must be declared")
    return {
        "job_id": job_id,
        "commit": commit,
        "script": script,
        "source_paths": SOURCE_PATHS,
        "requirements": {"gpu": "cpu", "allowed_gpus": ["cpu"], "min_vcpu": 1, "est_minutes": 120},
        "timeout_minutes": 240,
        "output_dir": str(Path(output_dir).resolve()),
        "inputs": [{key: row[key] for key in ("path", "dest", "sha256")} for row in staged_inputs],
        "required_exports": ["terminal-receipt.json"],
        "env": {
            "COLAB_CLI_REQUIREMENTS": "research/r33_verification/requirements.txt",
            "R33_NAMESPACE": namespace,
            "R33_ALLOCATION": allocation,
        },
        "budget": {
            "unit_cap": 0.320,
            "rate_units_per_hour": 0.08,
            "reserve_units": 0.053333333,
            "other_assignments": {},
        },
        "resume": {
            "unit_kind": "verification-shard",
            "ledger_path": "inputs/verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "max_export_bytes": 16 * 1024 * 1024,
        "r33": {
            "schema": "r33-cpu-pilot/v1",
            "namespace": namespace,
            "allocation": allocation,
            "forecast": {"allocated_minutes": 120, "units": 0.16, "rate_units_per_hour": 0.08},
            "base_envelope": {
                "minutes": 144,
                "units": 0.192,
                "reserve_minutes": 24,
                "reserve_units": 0.032,
            },
            "effective_envelope": {
                "minutes": 288,
                "units": 0.384,
                "reserve_minutes": 48,
                "reserve_units": 0.064,
            },
            "attempts": {"original": 1, "max_mechanical_completions": 2, "operator_owned": True},
            "authority_receipts": authority_receipts,
            "dispatch": "PREPARED_NOT_DISPATCHED",
            "native_pending": ["Darwin numerical/ownership check", "CUDA-dependent acceptance"],
        },
    }
