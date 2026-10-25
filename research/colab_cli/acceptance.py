#!/usr/bin/env python3
"""Prepare, and only after review execute, the bounded Colab CLI acceptance plan.

The default is intentionally dry: it writes pinned, fresh manifests and a
review record but performs no Colab command.  ``--execute-after-review`` is an
operator action after the acceptance plan has been reviewed; it still has no
automatic retry, purchase, or fallback allocation path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from research.colab_cli import supervisor
from research.colab_cli.acceptance_faults import AcceptanceObserver

MAX_UNITS = 4.0
RESERVE_UNITS = 0.8
USABLE_UNITS = 3.2
L4_RATE_UNITS_PER_HOUR = 2.0
CPU_RATE_UNITS_PER_HOUR = 0.5
L4_MAX_MINUTES = 30
CPU_MAX_MINUTES = 20
PROFILES = ("final-only", "full-state")
FINAL_ARTIFACT_BYTES = 64 * 1024 * 1024
FULL_STATE_GATES = {"busy-full-snapshot", "exact-resume", "checkpoint-overhead"}
REVIEW_TOKEN = "COLAB_CLI_ACCEPTANCE_REVIEWED"
QUALIFIED_RECORD_SCHEMA = "snake-dqn-colab-cli-qualified/v1"
D1024_SYNTHETIC_INVENTORY_SCHEMA = "f4-d1024-synthetic-input-inventory/v1"
D1024_TRANSFER_BAR_SECONDS = 150.0
D1024_STAGING_CAP_SECONDS = 300.0
# These are size/count/transport-role facts from the failed D1024 canary's
# immutable manifest plus metadata-only stat results.  They intentionally do
# not contain a fit, label, or ledger payload digest.  The public B8 input
# remains a public-input exercise; the other three are deterministic synthetic
# local files with the same names and byte counts as the production inventory.
D1024_SYNTHETIC_INVENTORY = {
    "schema": D1024_SYNTHETIC_INVENTORY_SCHEMA,
    "inputs": [
        {
            "dest": "b8.pth",
            "size_bytes": 6_562_005,
            "transport": "https",
            "url": "https://github.com/josenunezwork/snake-dqn/releases/download/"
            "s1-inputs-20261019/B8_s3_final.pth",
            "sha256": "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8",
        },
        {
            "dest": "factory-verified-ledger.json",
            "size_bytes": 2_405_362,
            "transport": "upload",
        },
        {"dest": "train-1024.npz", "size_bytes": 45_397_643, "transport": "upload"},
        {"dest": "metadata.tar", "size_bytes": 2_211_840, "transport": "upload"},
    ],
}
BASELINE = {
    "supplied_at": "2026-10-18 18:38 PT",
    "balance_units": 1174.37,
    "aggregate_usage_rate_units_per_hour": 0.0,
    "active_notebook_assignments": 0,
    "note": (
        "Supplied context, not a new live measurement. Re-read usage and sessions before dispatch."
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: dict[str, Any]) -> str:
    """Hash a JSON contract independently of whitespace or dictionary order."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def d1024_synthetic_inventory() -> dict[str, Any]:
    """Return the payload-free D1024 inventory contract for requalification."""
    inventory = json.loads(json.dumps(D1024_SYNTHETIC_INVENTORY))
    inventory["sha256"] = canonical_sha256(inventory)
    return inventory


D1024_F4_INVENTORY_SHA256 = canonical_sha256(D1024_SYNTHETIC_INVENTORY)


def _write_synthetic_file(path: Path, size_bytes: int) -> str:
    """Create deterministic sparse acceptance bytes without copying study inputs."""
    if path.exists():
        raise ValueError(f"synthetic acceptance input already exists: {path}")
    with path.open("xb") as stream:
        stream.truncate(size_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_file(path)


def build_d1024_synthetic_inputs(directory: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize only synthetic local inputs while retaining the public B8 role."""
    directory.mkdir(parents=True, exist_ok=False)
    inventory = d1024_synthetic_inventory()
    inputs: list[dict[str, Any]] = []
    for row in inventory["inputs"]:
        if row["transport"] == "https":
            inputs.append(
                {
                    "dest": row["dest"],
                    "url": row["url"],
                    "size_bytes": row["size_bytes"],
                    "sha256": row["sha256"],
                }
            )
            continue
        path = directory / row["dest"]
        inputs.append(
            {
                "dest": row["dest"],
                "path": str(path),
                "size_bytes": row["size_bytes"],
                "sha256": _write_synthetic_file(path, row["size_bytes"]),
            }
        )
    return inputs, inventory


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def create_json(path: Path, value: dict[str, Any]) -> None:
    """Create an immutable execution receipt without replacing plan evidence."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def fresh_dir(path: Path) -> Path:
    if path.exists():
        raise ValueError(f"fresh output required; path already exists: {path}")
    path.mkdir(parents=True)
    return path


def manifest(
    *,
    job_id: str,
    commit: str,
    probe: Path,
    wrapper: Path,
    output: Path,
    gpu: str,
    minutes: int,
    priced_rate: float,
    other_assignments: dict[str, float],
    profile: str = "final-only",
    synthetic_inputs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "commit": commit,
        "script": "research/colab_cli/acceptance_probe.sh",
        "env": {
            "PYTHONUNBUFFERED": "1",
            "COLAB_ACCEPTANCE": "1",
            "COLAB_ACCEPTANCE_PROFILE": profile,
            "COLAB_ACCEPTANCE_ARM": gpu,
        },
        "requirements": {"gpu": gpu, "min_vcpu": 2, "est_minutes": minutes / 2},
        "timeout_minutes": minutes,
        "output_dir": str(output),
        "inputs": [
            {
                "path": str(probe),
                "sha256": sha256_file(probe),
                "dest": "acceptance_probe.py",
            },
            {"path": str(wrapper), "sha256": sha256_file(wrapper), "dest": "acceptance_probe.sh"},
        ]
        + (synthetic_inputs or []),
        "budget": {
            # This is deliberately a conservative price ceiling.  It controls
            # admission, the liability reservation, and the session cap even
            # when the provider later reports a lower actual rate.
            "unit_cap": priced_rate * minutes / 60.0,
            "rate_units_per_hour": priced_rate,
            "reserve_units": RESERVE_UNITS,
            # Every assignment already present at admission must be priced so
            # an account-wide balance delta cannot silently consume this arm's
            # small unit cap.  The broker rejects an unmapped endpoint.
            "other_assignments": other_assignments,
        },
        "lease_seconds": 300 if gpu == "cpu" else 180,
    }


def review_gates(profile: str = "final-only") -> list[dict[str, Any]]:
    """Every advice gate has a named required artifact; absence is INCOMPLETE."""
    if profile not in PROFILES:
        raise ValueError(f"unknown acceptance profile: {profile}")
    gates = [
        {
            "id": "account-assignment-spend",
            "evidence": "all pre-existing endpoints and their rates are recorded at admission",
            "owner": "Mac supervisor",
        },
        {
            "id": "fresh-oauth-refresh",
            "evidence": "fresh terminal whoami/usage after refresh; no browser prompt",
            "owner": "Mac supervisor",
        },
        {
            "id": "cold-kernel-metadata-recovery",
            "evidence": (
                "for CPU and L4: known kernel ID deliberately cleared locally, guard recovered the "
                "same unique idle kernel without changing tracked metadata, then the first upload "
                "completed in the supervisor transfer ledger"
            ),
            "owner": "Mac supervisor + cli guard",
        },
        {
            "id": "busy-full-snapshot",
            "evidence": "remote snapshot.complete.json, downloaded hash, atomic local file",
            "owner": "probe + supervisor",
        },
        {
            "id": "silence-timeout",
            "evidence": "over-60s gap and explicit supervisor timeout record",
            "owner": "probe + supervisor",
        },
        {
            "id": "url-no-second-allocation",
            "evidence": "sessions before/after url show same single assignment",
            "owner": "Mac supervisor",
        },
        {
            "id": "client-loss-reconnect",
            "evidence": "generation fence and one remote execution identity after reconnect",
            "owner": "Mac supervisor",
        },
        {
            "id": "failure-and-corrupt-export",
            "evidence": "intentional child exit recorded; corrupt download rejected",
            "owner": "probe + supervisor",
        },
        {
            "id": "exact-resume",
            "evidence": "uninterrupted and resumed full-state hashes match",
            "owner": "probe + supervisor",
        },
        {
            "id": "mac-loss-watchdog-unassign",
            "evidence": "lease expiry plus backend assignment disappearance",
            "owner": "runtime watchdog + supervisor",
        },
        {
            "id": "checkpoint-overhead",
            "evidence": "timed checkpoint overhead <= 2 percent",
            "owner": "supervisor",
        },
        {
            "id": "backend-stop-disappearance",
            "evidence": "stop issued then sessions/status proves assignment gone",
            "owner": "Mac supervisor",
        },
    ]

    gates.extend(
        [
            {
                "id": "rate-calibration",
                "evidence": "bounded positive provider calibration for CPU and L4",
                "owner": "Mac supervisor",
            },
            {
                "id": "final-artifact-download",
                "evidence": (
                    "exactly 64 MiB L4-to-Mac export, SHA256 verified, "
                    "end-to-end bytes/s lower bound"
                ),
                "owner": "probe + supervisor",
            },
        ]
    )
    for gate in gates:
        gate["required"] = (
            gate["id"] not in FULL_STATE_GATES
            if profile == "final-only"
            else gate["id"] != "final-artifact-download"
        )
    return gates


def source_manifest(root: Path) -> dict[str, str]:
    """Bind acceptance to every transport implementation and maintained launch file."""
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted((root / "research/colab_cli").iterdir())
        if path.is_file() and path.suffix in {".py", ".sh"}
    }


def transport_tree(root: Path, ref: str = "HEAD") -> str:
    """Resolve the committed CLI subtree used by an acceptance source manifest."""
    value = subprocess.run(
        ["git", "rev-parse", "--verify", f"{ref}:research/colab_cli"],
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
        timeout=15,
    )
    if value.returncode:
        raise ValueError(f"cannot resolve Colab CLI tree for {ref}")
    tree = value.stdout.strip()
    kind = subprocess.run(
        ["git", "cat-file", "-t", tree],
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
        timeout=15,
    )
    if kind.returncode or kind.stdout.strip() != "tree":
        raise ValueError("Colab CLI identity is not a Git tree")
    return tree


def _require_hex(value: Any, label: str, length: int = 40) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a {length}-character lowercase hex digest")
    return value


def _source_manifest_at_commit(repo: Path, commit: str, expected: dict[str, str]) -> None:
    """Ensure a record's per-file hashes came from its stated committed source."""
    if not isinstance(expected, dict) or not expected:
        raise ValueError("qualified acceptance record has no source manifest")
    for relative, digest in expected.items():
        _require_hex(digest, f"source manifest digest for {relative}", 64)
        if not isinstance(relative, str) or not relative.startswith("research/colab_cli/"):
            raise ValueError("qualified source manifest contains a non-transport path")
        pinned = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=repo,
            capture_output=True,
            check=False,
            timeout=15,
        )
        if pinned.returncode or hashlib.sha256(pinned.stdout).hexdigest() != digest:
            raise ValueError(f"qualified source manifest differs from commit: {relative}")


def verify_qualified_transport_record(
    repo: Path,
    record_path: Path,
    *,
    required_inventory_sha256: str = D1024_F4_INVENTORY_SHA256,
    reject_tree: str | None = None,
) -> dict[str, Any]:
    """Fail closed unless an operator-created PASS binds this exact CLI tree.

    This deliberately accepts a science commit that differs outside
    ``research/colab_cli``.  It refuses a copied PASS, an old tree, altered
    acceptance receipt, or an acceptance that omitted the D1024-size synthetic
    transport inventory.
    """
    repo = repo.resolve()
    registry = json.loads(record_path.read_text())
    if not isinstance(registry, dict) or registry.get("schema") != QUALIFIED_RECORD_SCHEMA:
        raise ValueError("invalid qualified transport registry")
    tree = _require_hex(registry.get("tree_hash"), "qualified tree")
    if reject_tree is not None and tree == _require_hex(reject_tree, "rejected tree"):
        raise ValueError("qualified transport registry still names the rejected tree")
    if tree != transport_tree(repo):
        raise ValueError("current Colab CLI tree does not match qualified transport registry")
    result_path = Path(registry.get("acceptance_result", ""))
    if not result_path.is_file():
        raise ValueError("qualified transport acceptance result is absent")
    if sha256_file(result_path) != _require_hex(
        registry.get("acceptance_result_sha256"), "acceptance result SHA-256", 64
    ):
        raise ValueError("qualified transport acceptance result hash differs from registry")
    result = json.loads(result_path.read_text())
    if (
        not isinstance(result, dict)
        or result.get("status") != "PASS"
        or result.get("qualified") is not True
        or result.get("return_code") != 0
        or result.get("profile") != "final-only"
    ):
        raise ValueError("qualified transport acceptance result is not a final-only PASS")
    commit = _require_hex(result.get("qualified_commit"), "qualified commit")
    if commit != _require_hex(registry.get("qualified_commit"), "registry qualified commit"):
        raise ValueError("qualified commit differs between registry and acceptance result")
    result_tree = _require_hex(result.get("qualified_tree_hash"), "acceptance result tree")
    if result_tree != tree or result_tree != transport_tree(repo, commit):
        raise ValueError("qualified acceptance tree differs from the accepted source commit")
    plan = result.get("final_plan")
    if not isinstance(plan, dict) or plan.get("commit") != commit or plan.get("tree_hash") != tree:
        raise ValueError("qualified acceptance plan identity differs from acceptance result")
    if plan.get("source_manifest") != result.get("source_manifest"):
        raise ValueError("qualified acceptance plan/source manifest mismatch")
    _source_manifest_at_commit(repo, commit, result["source_manifest"])
    if source_manifest(repo) != result["source_manifest"]:
        raise ValueError("current Colab CLI source manifest is not the qualified source")
    # Reuse the common final-only admission proof so this D1024-specific
    # consumer cannot mistake a copied PASS header for a qualified acceptance.
    from research.colab_cli import admission

    admission.require_final_only(result, repo)
    inventory = plan.get("d1024_synthetic_inventory")
    if not isinstance(inventory, dict) or inventory.get("sha256") != required_inventory_sha256:
        raise ValueError("qualified acceptance lacks the required D1024 synthetic inventory")
    unsigned_inventory = dict(inventory)
    supplied_digest = unsigned_inventory.pop("sha256", None)
    if supplied_digest != canonical_sha256(unsigned_inventory):
        raise ValueError("D1024 synthetic inventory digest is invalid")
    if result.get("d1024_synthetic_inventory_sha256") != supplied_digest:
        raise ValueError("acceptance result does not bind the D1024 synthetic inventory")
    gates = plan.get("gates")
    if not isinstance(gates, list):
        raise ValueError("qualified acceptance plan has no gate records")
    staging = next(
        (gate for gate in gates if gate.get("id") == "d1024-synthetic-full-inventory-staging"), None
    )
    if not isinstance(staging, dict) or staging.get("status") != "OBSERVED":
        raise ValueError("qualified acceptance did not observe synthetic D1024 staging")
    evidence = staging.get("evidence_record")
    if (
        not isinstance(evidence, dict)
        or evidence.get("status") != "PASS"
        or evidence.get("inventory_sha256") != supplied_digest
        or not isinstance(evidence.get("public_input_preflight"), str)
    ):
        raise ValueError("qualified acceptance synthetic D1024 staging evidence is incomplete")
    executions = {row.get("job"): row for row in plan.get("execution", []) if isinstance(row, dict)}
    if set(executions) != {"cpu", "l4"}:
        raise ValueError("qualified acceptance lacks exactly one CPU and L4 execution receipt")
    receipts: dict[str, tuple[Path, dict[str, Any]]] = {}
    for arm in ("cpu", "l4"):
        execution = executions[arm]
        receipt_path = Path(execution.get("receipt", ""))
        receipt_digest = _require_hex(
            execution.get("receipt_sha256"), f"{arm} acceptance receipt hash", 64
        )
        if not receipt_path.is_file() or sha256_file(receipt_path) != receipt_digest:
            raise ValueError(f"qualified acceptance {arm} receipt changed")
        receipt = json.loads(receipt_path.read_text())
        if not isinstance(receipt, dict) or receipt.get("commit") != commit:
            raise ValueError(f"qualified acceptance {arm} receipt is not from the qualified commit")
        receipts[arm] = (receipt_path.resolve(), receipt)
    l4_receipt_path, l4_receipt = receipts["l4"]
    measurement = evidence.get("upload_measurement")
    if not isinstance(measurement, dict):
        raise ValueError("qualified acceptance synthetic D1024 timing evidence is absent")
    measurement_path = Path(measurement.get("path", ""))
    if measurement_path.resolve() != (l4_receipt_path.parent / "upload-measurement.json").resolve():
        raise ValueError("qualified acceptance synthetic D1024 timing is not from the L4 receipt")
    if not measurement_path.is_file() or sha256_file(measurement_path) != _require_hex(
        measurement.get("sha256"), "synthetic staging measurement hash", 64
    ):
        raise ValueError("qualified acceptance synthetic D1024 timing evidence changed")
    recomputed = _d1024_staging_measurement(measurement_path)
    if any(measurement.get(name) != value for name, value in recomputed.items()):
        raise ValueError("qualified acceptance synthetic D1024 timing evidence differs")
    staging_receipt = evidence.get("receipt")
    if not isinstance(staging_receipt, dict):
        raise ValueError("qualified acceptance synthetic D1024 receipt evidence is absent")
    receipt_path = Path(staging_receipt.get("path", ""))
    receipt_digest = _require_hex(
        staging_receipt.get("sha256"), "synthetic staging receipt hash", 64
    )
    if not receipt_path.is_file() or sha256_file(receipt_path) != receipt_digest:
        raise ValueError("qualified acceptance synthetic D1024 receipt changed")
    l4_execution = executions["l4"]
    if (
        receipt_path.resolve() != l4_receipt_path
        or l4_execution.get("receipt_sha256") != receipt_digest
    ):
        raise ValueError(
            "qualified acceptance synthetic D1024 receipt is not the L4 acceptance receipt"
        )
    evidence_rows = evidence.get("inputs")
    if not isinstance(evidence_rows, list):
        raise ValueError("qualified acceptance synthetic D1024 input evidence is absent")
    rows_by_dest: dict[str, list[dict[str, Any]]] = {}
    for observed in evidence_rows:
        if not isinstance(observed, dict) or not isinstance(observed.get("dest"), str):
            raise ValueError("qualified acceptance synthetic D1024 input evidence is malformed")
        rows_by_dest.setdefault(observed["dest"], []).append(observed)
    expected_destinations = {row["dest"] for row in inventory["inputs"]}
    if set(rows_by_dest) != expected_destinations:
        raise ValueError("qualified acceptance synthetic D1024 input evidence destinations differ")
    uploads_by_dest: dict[str, list[dict[str, Any]]] = {}
    for transfer in l4_receipt.get("transfers", []):
        if (
            isinstance(transfer, dict)
            and transfer.get("direction") == "upload"
            and transfer.get("status") == "verified"
            and isinstance(transfer.get("remote"), str)
            and "/inputs/" in transfer["remote"]
        ):
            destination = transfer["remote"].split("/inputs/", 1)[1]
            uploads_by_dest.setdefault(destination, []).append(transfer)
    for row in inventory["inputs"]:
        observed_rows = rows_by_dest.get(row["dest"], [])
        if len(observed_rows) != 1:
            raise ValueError(
                "qualified acceptance has duplicate or missing synthetic input evidence "
                f"for {row['dest']}"
            )
        observed = observed_rows[0]
        if observed.get("transport") != row["transport"]:
            raise ValueError(
                f"qualified acceptance omitted synthetic input evidence for {row['dest']}"
            )
        if row["transport"] == "upload":
            if (
                observed.get("size_bytes") != row["size_bytes"]
                or observed.get("status") != "verified"
            ):
                raise ValueError(f"qualified acceptance synthetic upload differs for {row['dest']}")
            _require_hex(observed.get("sha256"), f"synthetic upload hash for {row['dest']}", 64)
            transfers = uploads_by_dest.get(row["dest"], [])
            if len(transfers) != 1:
                raise ValueError(
                    "qualified acceptance has duplicate or missing L4 upload receipt "
                    f"for {row['dest']}"
                )
            transfer = transfers[0]
            if any(
                transfer.get(field) != observed.get(field)
                for field in ("remote", "size_bytes", "sha256")
            ):
                raise ValueError(
                    "qualified acceptance synthetic upload evidence conflicts with L4 receipt "
                    f"for {row['dest']}"
                )
    return {
        "qualified_commit": commit,
        "tree_hash": tree,
        "acceptance_result": str(result_path),
        "acceptance_result_sha256": registry["acceptance_result_sha256"],
        "d1024_synthetic_inventory_sha256": supplied_digest,
    }


def d1024_synthetic_staging_evidence(
    plan: dict[str, Any], job: dict[str, Any], receipt: dict[str, Any], output: Path
) -> dict[str, Any]:
    """Require actual L4 staging evidence for every synthetic D1024 input role."""
    inventory = plan.get("d1024_synthetic_inventory")
    if not isinstance(inventory, dict):
        return {"status": "NOT_REQUIRED"}
    manifest = json.loads(Path(job["manifest"]).read_text())
    requested = {row["dest"]: row for row in inventory.get("inputs", [])}
    actual = {row.get("dest"): row for row in manifest.get("inputs", [])}
    if not set(requested).issubset(actual):
        raise supervisor.Refused("synthetic D1024 inventory is absent from acceptance manifest")
    uploads = {
        row["remote"].split("/inputs/", 1)[1]: row
        for row in receipt.get("transfers", [])
        if isinstance(row, dict)
        and row.get("direction") == "upload"
        and row.get("status") == "verified"
        and isinstance(row.get("remote"), str)
        and "/inputs/" in row["remote"]
    }
    observed: list[dict[str, Any]] = []
    for dest, expected in requested.items():
        row = actual[dest]
        if row.get("size_bytes") != expected["size_bytes"]:
            raise supervisor.Refused(f"synthetic D1024 size mismatch for {dest}")
        if expected["transport"] == "https":
            if row.get("url") != expected.get("url") or row.get("sha256") != expected.get("sha256"):
                raise supervisor.Refused("synthetic D1024 public input role differs")
            observed.append({"dest": dest, "transport": "https", "status": "manifest-bound"})
            continue
        path = Path(row.get("path", ""))
        if not path.is_file() or path.stat().st_size != expected["size_bytes"]:
            raise supervisor.Refused(f"synthetic D1024 local file differs for {dest}")
        digest = sha256_file(path)
        transfer = uploads.get(dest, {})
        if (
            row.get("sha256") != digest
            or transfer.get("size_bytes") != expected["size_bytes"]
            or transfer.get("sha256") != digest
        ):
            raise supervisor.Refused(f"synthetic D1024 verified upload is absent for {dest}")
        observed.append(
            {
                "dest": dest,
                "transport": "upload",
                "size_bytes": expected["size_bytes"],
                "sha256": digest,
                "remote": transfer["remote"],
                "status": "verified",
            }
        )
    public_preflight = output / "public-input-preflight.json"
    if not public_preflight.is_file():
        raise supervisor.Refused("synthetic D1024 public-input preflight is absent")
    measurement_path = output / "upload-measurement.json"
    measurement = _d1024_staging_measurement(measurement_path)
    receipt_path = output / "receipt.json"
    if not receipt_path.is_file() or json.loads(receipt_path.read_text()) != receipt:
        raise supervisor.Refused("synthetic D1024 staging receipt is absent")
    return {
        "status": "PASS",
        "inventory_sha256": inventory.get("sha256"),
        "inputs": observed,
        "public_input_preflight": str(public_preflight),
        "upload_measurement": {
            "path": str(measurement_path),
            "sha256": sha256_file(measurement_path),
            **measurement,
        },
        "receipt": {"path": str(receipt_path), "sha256": sha256_file(receipt_path)},
    }


def _d1024_staging_measurement(path: Path) -> dict[str, float]:
    """Read the persisted shared-clock measurement without relaxing either bar."""
    if not path.is_file():
        raise supervisor.Refused("synthetic D1024 upload measurement is absent")
    raw = json.loads(path.read_text())
    measured: dict[str, float] = {}
    for name in ("elapsed_seconds", "transfer_seconds", "source_delivery_seconds"):
        value = raw.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise supervisor.Refused(f"synthetic D1024 {name} is not finite")
        measured[name] = float(value)
    if measured["elapsed_seconds"] <= 0 or measured["elapsed_seconds"] > D1024_STAGING_CAP_SECONDS:
        raise supervisor.Refused("synthetic D1024 staging exceeds the 300-second cap")
    if (
        measured["transfer_seconds"] <= 0
        or measured["transfer_seconds"] > D1024_TRANSFER_BAR_SECONDS
    ):
        raise supervisor.Refused("synthetic D1024 transfer exceeds the 150-second bar")
    if measured["source_delivery_seconds"] < 0 or (
        measured["transfer_seconds"] + measured["source_delivery_seconds"]
        > measured["elapsed_seconds"] + 1e-6
    ):
        raise supervisor.Refused("synthetic D1024 staging components exceed elapsed time")
    return measured


def verify_plan_sources(plan: dict[str, Any], repo: Path) -> None:
    """Refuse dirty or stale implementation bytes before a paid allocation."""
    current = source_manifest(repo)
    if plan["source_manifest"] != current:
        raise supervisor.Refused("acceptance plan source changed before execution")
    for relative, digest in current.items():
        pinned = subprocess.run(
            ["git", "show", f"{plan['commit']}:{relative}"],
            cwd=repo,
            capture_output=True,
            timeout=15,
            check=False,
        )
        if pinned.returncode or hashlib.sha256(pinned.stdout).hexdigest() != digest:
            raise supervisor.Refused(f"acceptance source differs from pinned commit: {relative}")


def build_plan(
    root: Path,
    state_dir: Path,
    cpu_rate: float | None = CPU_RATE_UNITS_PER_HOUR,
    other_assignments: dict[str, float] | None = None,
    l4_rate: float = L4_RATE_UNITS_PER_HOUR,
    profile: str = "final-only",
    d1024_inventory: bool = False,
) -> dict[str, Any]:
    gates = review_gates(profile)
    if cpu_rate is None:
        cpu_rate = CPU_RATE_UNITS_PER_HOUR
    if not math.isfinite(cpu_rate) or cpu_rate <= 0:
        raise ValueError("--cpu-rate must be finite and positive")
    if not math.isfinite(l4_rate) or l4_rate <= 0:
        raise ValueError("--l4-rate must be finite and positive")
    other_assignments = other_assignments or {}
    if not all(
        isinstance(endpoint, str)
        and endpoint
        and isinstance(rate, (int, float))
        and not isinstance(rate, bool)
        and rate > 0
        for endpoint, rate in other_assignments.items()
    ):
        raise ValueError("other assignment rates must map nonempty endpoints to positive numbers")
    cpu_expected = cpu_rate * CPU_MAX_MINUTES / 60.0
    l4_expected = l4_rate * L4_MAX_MINUTES / 60.0
    if cpu_expected + l4_expected > USABLE_UNITS:
        raise ValueError(
            f"expected {cpu_expected + l4_expected:.3f} units exceeds usable {USABLE_UNITS:.1f}"
        )
    state_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"acceptance-{int(time.time())}"
    run_dir = fresh_dir(state_dir.resolve() / run_id)
    output_root = fresh_dir(run_dir / "outputs")
    manifests = fresh_dir(run_dir / "manifests")
    synthetic_inputs: list[dict[str, Any]] = []
    synthetic_inventory: dict[str, Any] | None = None
    if d1024_inventory:
        synthetic_inputs, synthetic_inventory = build_d1024_synthetic_inputs(
            run_dir / "d1024-synthetic-inputs"
        )
    probe = root / "research/colab_cli/acceptance_probe.py"
    wrapper = root / "research/colab_cli/acceptance_probe.sh"
    if not probe.is_file() or not wrapper.is_file():
        raise ValueError("missing acceptance probe or wrapper")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    rows = [
        ("cpu", "cpu", CPU_MAX_MINUTES, cpu_rate),
        ("l4", "L4", L4_MAX_MINUTES, l4_rate),
    ]
    jobs: list[dict[str, Any]] = []
    for name, gpu, minutes, rate in rows:
        # The broker refuses an existing job output directory, so this is a
        # planned path only.  It creates it after manifest validation.
        output = output_root / name
        payload = manifest(
            job_id=f"{run_id}-{name}",
            commit=commit,
            probe=probe,
            wrapper=wrapper,
            output=output,
            gpu=gpu,
            minutes=minutes,
            priced_rate=rate,
            other_assignments={key: float(value) for key, value in other_assignments.items()},
            profile=profile,
            synthetic_inputs=synthetic_inputs,
        )
        target = manifests / f"{name}.json"
        atomic_json(target, payload)
        jobs.append(
            {
                "name": name,
                "manifest": str(target),
                "priced_rate_units_per_hour": rate,
                "expected_units": payload["budget"]["unit_cap"],
            }
        )
    plan_gates = [
        {**gate, "status": "INCOMPLETE" if gate["required"] else "NOT_REQUIRED"} for gate in gates
    ]
    if synthetic_inventory is not None:
        plan_gates.append(
            {
                "id": "d1024-synthetic-full-inventory-staging",
                "evidence": (
                    "L4 receipt verified every local synthetic input with the D1024 sizes and "
                    "hashes; the B8 HTTPS role has its preflight evidence"
                ),
                "owner": "Mac supervisor + acceptance verifier",
                "required": True,
                "status": "INCOMPLETE",
            }
        )
    plan = {
        "schema": "snake-dqn-colab-cli-acceptance/v1",
        "status": "NOT_RUN",
        "profile": profile,
        "commit": commit,
        "tree_hash": transport_tree(root),
        "source_manifest": source_manifest(root),
        "baseline_context": BASELINE,
        "limits": {
            "max_units_including_reserve": MAX_UNITS,
            "reserve_units": RESERVE_UNITS,
            "usable_units": USABLE_UNITS,
            "first_cap_wins": True,
        },
        "expected_costs": {
            "cpu_20_minutes_at_priced_rate": cpu_expected,
            "l4_30_minutes_at_priced_rate": l4_expected,
            "total": cpu_expected + l4_expected,
        },
        "jobs": jobs,
        "gates": plan_gates,
        "qualification": "INCOMPLETE until every gate has independently verified evidence",
    }
    if synthetic_inventory is not None:
        plan["d1024_synthetic_inventory"] = synthetic_inventory
    atomic_json(run_dir / "acceptance-plan.json", plan)
    return {"run_dir": str(run_dir), "plan": plan}


def load_other_assignments(path: Path | None) -> dict[str, float]:
    """Load the reviewed endpoint-to-rate map without ever discovering endpoints by guesswork."""
    if path is None:
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("--other-assignments must contain a JSON object")
    result: dict[str, float] = {}
    for endpoint, rate in value.items():
        if not isinstance(endpoint, str) or not endpoint or isinstance(rate, bool):
            raise ValueError("--other-assignments has an invalid endpoint/rate")
        try:
            numeric = float(rate)
        except (TypeError, ValueError) as exc:
            raise ValueError("--other-assignments has a nonnumeric rate") from exc
        if numeric <= 0:
            raise ValueError("--other-assignments rates must be positive")
        result[endpoint] = numeric
    return result


def _record_d1024_synthetic_staging(
    observer: AcceptanceObserver,
    plan: dict[str, Any],
    job: dict[str, Any],
    receipt: dict[str, Any],
    output: Path,
) -> dict[str, str] | None:
    """Record a synthetic-staging refusal without hiding clean release evidence.

    The intentional L4 watchdog ends the broker call with a refusal after the
    runtime has released its assignment.  Staging verification is a separate
    qualification gate: its failure must not be recast as missing cleanup or
    provider calibration evidence.
    """
    if job["name"] != "l4" or "d1024_synthetic_inventory" not in plan:
        return None
    try:
        evidence = d1024_synthetic_staging_evidence(plan, job, receipt, output)
    except (OSError, ValueError, supervisor.Refused) as exc:
        failure = {
            "status": "INCOMPLETE",
            "error": str(exc),
            "exception_type": type(exc).__name__,
        }
        observer._record("d1024-synthetic-full-inventory-staging", **failure)
        return failure
    observer._record("d1024-synthetic-full-inventory-staging", **evidence)
    return None


def execute(
    plan: dict[str, Any], *, state_dir: Path, colab: str, review_token: str, oauth_python: str
) -> int:
    if review_token != REVIEW_TOKEN:
        raise ValueError("execution requires the exact reviewed acceptance token")
    profile = plan["profile"]
    verify_plan_sources(plan, supervisor.REPO)
    observer = AcceptanceObserver(oauth_python, profile=profile)
    results: list[dict[str, Any]] = []
    run_dir = Path(plan["jobs"][0]["manifest"]).parent.parent
    for job in plan["jobs"]:
        manifest_path = Path(job["manifest"])
        payload = json.loads(manifest_path.read_text())
        output = Path(payload["output_dir"])
        try:
            supervisor.Broker(state_dir.resolve() / "lanes", colab, observer=observer).run(payload)
            receipt = _clean_release_receipt(output)
            measurement = _calibration_evidence(receipt)
            staging_error = _record_d1024_synthetic_staging(observer, plan, job, receipt, output)
            observer._record("backend-stop-disappearance", receipt=str(output / "receipt.json"))
            results.append(
                {
                    "job": job["name"],
                    "status": (
                        "PASS"
                        if measurement["status"] == "PASS" and staging_error is None
                        else "INCOMPLETE"
                    ),
                    "receipt": str(output / "receipt.json"),
                    "receipt_sha256": sha256_file(output / "receipt.json"),
                    "priced_rate_units_per_hour": measurement["priced_rate_units_per_hour"],
                    "measured_rate_units_per_hour": measurement["observed_rate_units_per_hour"],
                    "rate_calibration": measurement,
                    "staging_error": staging_error,
                }
            )
        except (supervisor.Refused, OSError, subprocess.SubprocessError) as exc:
            # The L4 arm intentionally stops renewals.  It only passes when the
            # runtime-side release was observed within the stated bound *before*
            # broker cleanup, and the broker receipt reports a clean release.
            watchdog = observer.records.get("mac-loss-watchdog-unassign", {})
            released = bool(watchdog.get("assignment_disappeared_within_bound"))
            try:
                receipt = _clean_release_receipt(output, require_success=False)
                measurement = _calibration_evidence(receipt)
                clean_cleanup = True
            except (OSError, ValueError, supervisor.Refused):
                clean_cleanup = False
                measurement = {
                    "status": "INCOMPLETE",
                    "reason": "missing clean release receipt or provider rate calibration",
                    "priced_rate_units_per_hour": None,
                    "observed_rate_units_per_hour": None,
                }
            staging_error = (
                _record_d1024_synthetic_staging(observer, plan, job, receipt, output)
                if clean_cleanup
                else None
            )
            results.append(
                {
                    "job": job["name"],
                    "status": (
                        "PASS"
                        if (
                            job["name"] == "l4"
                            and released
                            and clean_cleanup
                            and measurement["status"] == "PASS"
                            and staging_error is None
                        )
                        else "INCOMPLETE"
                    ),
                    "error": str(exc),
                    "receipt": str(output / "receipt.json"),
                    "receipt_sha256": (
                        sha256_file(output / "receipt.json")
                        if (output / "receipt.json").is_file()
                        else None
                    ),
                    "priced_rate_units_per_hour": measurement["priced_rate_units_per_hour"],
                    "measured_rate_units_per_hour": measurement["observed_rate_units_per_hour"],
                    "rate_calibration": measurement,
                    "staging_error": staging_error,
                }
            )
        finally:
            if output.exists():
                observer.write(output)
        # A CPU admission/transport failure is not a reason to spend an L4
        # allocation. The watchdog fault is expected only after CPU passed.
        if job["name"] == "cpu" and results[-1]["status"] != "PASS":
            break
    observer._record(
        "rate-calibration",
        status=(
            "PASS"
            if len(results) == 2
            and all(row["rate_calibration"]["status"] == "PASS" for row in results)
            else "INCOMPLETE"
        ),
        arms=[{"arm": row["job"], **row["rate_calibration"]} for row in results],
    )
    gate_status = {gate["id"]: gate for gate in plan["gates"]}
    for gate, evidence in observer.records.items():
        if gate in gate_status and gate_status[gate]["required"]:
            incomplete = str(evidence.get("status", "")).startswith("INCOMPLETE")
            gate_status[gate]["status"] = "INCOMPLETE" if incomplete else "OBSERVED"
            gate_status[gate]["evidence_record"] = evidence
    plan["execution"] = results
    plan["status"] = "OBSERVED_NOT_QUALIFIED"
    plan["qualification"] = (
        "INCOMPLETE pending independent review of every observer/receipt artifact"
    )
    qualified = (
        all(gate["status"] == "OBSERVED" for gate in plan["gates"] if gate["required"])
        and len(results) == 2
        and all(result["status"] == "PASS" for result in results)
    )
    if plan["source_manifest"] != source_manifest(supervisor.REPO):
        qualified = False
    if qualified:
        plan["status"] = "PASS"
        plan["qualification"] = (
            "PASS for this profile and source; independent review required for S1"
        )
    return_code = 0 if qualified else 1
    result = {
        "schema": "snake-dqn-colab-cli-acceptance-result/v1",
        "return_code": return_code,
        "qualified": qualified,
        "status": "PASS" if qualified else "INCOMPLETE",
        "profile": profile,
        # These source identities are recorded even for an incomplete
        # execution so a later registry cannot silently substitute them.
        # The verifier separately requires PASS/true/0.
        "qualified_commit": plan["commit"],
        "qualified_tree_hash": plan["tree_hash"],
        "source_manifest": plan["source_manifest"],
        "final_artifact_transfer": observer.records.get("final-artifact-download"),
        "final_plan": plan,
    }
    if "d1024_synthetic_inventory" in plan:
        result["d1024_synthetic_inventory_sha256"] = plan["d1024_synthetic_inventory"]["sha256"]
    create_json(run_dir / "acceptance-result.json", result)
    return return_code


def _clean_release_receipt(output: Path, *, require_success: bool = True) -> dict[str, Any]:
    """Require broker cleanup evidence even when the watchdog ended execution.

    A release observation made by the L4 fault arm is not a waiver for a
    cleanup error.  In particular this prevents an earlier `stop` failure from
    being converted into a PASS merely because an observer record exists.
    """
    receipt_path = output / "receipt.json"
    if not receipt_path.is_file():
        raise supervisor.Refused("missing broker receipt after acceptance release")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("cleanup_error"):
        raise supervisor.Refused("acceptance cleanup error: " + str(receipt["cleanup_error"]))
    if not receipt.get("stop_verified"):
        raise supervisor.Refused("acceptance release was not backend-verified")
    if require_success and receipt.get("status") not in {"PASS", "COMPLETED_SESSION_RETAINED"}:
        raise supervisor.Refused("acceptance receipt is incomplete")
    return receipt


def _calibration_evidence(receipt: dict[str, Any]) -> dict[str, Any]:
    """Require the bounded post-allocation rate calibration in every acceptance result.

    The priced rate remains the cap and reservation input.  This record is the
    provider-observed rate used by later polling drift checks, so an absent,
    incomplete, or malformed calibration cannot be represented as a pass.
    """
    accounting = receipt.get("accounting")
    calibration = accounting.get("calibration") if isinstance(accounting, dict) else None
    if not isinstance(calibration, dict):
        return {
            "status": "INCOMPLETE",
            "reason": "missing provider rate calibration in broker receipt",
            "priced_rate_units_per_hour": None,
            "observed_rate_units_per_hour": None,
        }
    priced = calibration.get("priced_rate_units_per_hour")
    observed = calibration.get("observed_rate_units_per_hour")
    if (
        calibration.get("status") != "PASS"
        or isinstance(priced, bool)
        or not isinstance(priced, (int, float))
        or isinstance(observed, bool)
        or not isinstance(observed, (int, float))
        or not math.isfinite(float(priced))
        or float(priced) <= 0
        or not math.isfinite(float(observed))
        or float(observed) <= 0
        or float(observed) > float(priced) + 0.05
    ):
        return {
            "status": "INCOMPLETE",
            "reason": str(calibration.get("reason", "provider rate calibration incomplete")),
            "priced_rate_units_per_hour": priced,
            "observed_rate_units_per_hour": observed,
            "samples": calibration.get("samples", []),
        }
    return {
        "status": "PASS",
        "priced_rate_units_per_hour": float(priced),
        "observed_rate_units_per_hour": float(observed),
        "samples": calibration.get("samples", []),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", choices=PROFILES, default="final-only")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--verify-qualified-record",
        type=Path,
        help="fail closed unless this registry binds the current source to a reviewed PASS",
    )
    parser.add_argument(
        "--required-inventory-sha256",
        default=D1024_F4_INVENTORY_SHA256,
        help="D1024 synthetic input-inventory contract required by the caller",
    )
    parser.add_argument(
        "--reject-tree",
        help="known superseded CLI tree that this acceptance must not admit",
    )
    parser.add_argument(
        "--d1024-inventory",
        action="store_true",
        help="stage the prospective synthetic D1024 input inventory in this acceptance only",
    )
    parser.add_argument("--state-dir", type=Path, help="absolute supervisor state parent")
    parser.add_argument(
        "--cpu-rate",
        "--cpu-rate-units-per-hour",
        dest="cpu_rate",
        type=float,
        default=CPU_RATE_UNITS_PER_HOUR,
        help="conservative CPU price ceiling (default: 0.5 units/hour)",
    )
    parser.add_argument(
        "--l4-rate",
        "--l4-rate-units-per-hour",
        dest="l4_rate",
        type=float,
        default=L4_RATE_UNITS_PER_HOUR,
        help="conservative L4 price ceiling (default: 2.0 units/hour)",
    )
    parser.add_argument(
        "--other-assignments",
        type=Path,
        help="reviewed JSON endpoint-to-rate map for every pre-existing assignment",
    )
    parser.add_argument(
        "--colab", default="colab", help="official CLI executable, used only with execution"
    )
    parser.add_argument(
        "--oauth-python",
        default=str(Path.home() / ".venvs" / "colab-cli" / "bin" / "python"),
        help="official Colab CLI venv interpreter, used only after review",
    )
    parser.add_argument("--execute-after-review", action="store_true")
    parser.add_argument("--review-token", default="")
    args = parser.parse_args(argv)
    if args.verify_qualified_record is not None:
        try:
            verified = verify_qualified_transport_record(
                args.repo,
                args.verify_qualified_record,
                required_inventory_sha256=args.required_inventory_sha256,
                reject_tree=args.reject_tree,
            )
            print(json.dumps(verified, indent=2, sort_keys=True))
            return 0
        except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
            parser.exit(1, f"REFUSE: {exc}\n")
    if args.state_dir is None or not args.state_dir.is_absolute():
        parser.error("--state-dir must be absolute")
    try:
        record = build_plan(
            args.repo.resolve(),
            args.state_dir,
            args.cpu_rate,
            load_other_assignments(args.other_assignments),
            args.l4_rate,
            profile=args.profile,
            d1024_inventory=args.d1024_inventory,
        )
        print(json.dumps(record, indent=2, sort_keys=True))
        if args.execute_after_review:
            return execute(
                record["plan"],
                state_dir=args.state_dir,
                colab=args.colab,
                review_token=args.review_token,
                oauth_python=args.oauth_python,
            )
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
