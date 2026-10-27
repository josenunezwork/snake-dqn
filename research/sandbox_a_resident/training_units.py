"""Immutable full-trajectory exports for the simulator-D PPO package.

The training process only writes a snapshot once.  A separate Mac-side process
may construct a verified ledger from the resulting descriptors; loading a
ledger deliberately never opens numerical payloads.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from research.sandbox_a_resident.units import digest, publish, verified_descriptor

NAMESPACE = "a100-dev/sandbox-a/train01"
SCHEMA = "sandbox-a-train-ledger/v1"


def allocation(control: dict[str, Any]) -> str:
    """Return the frozen control identity independently of an attempt path."""
    return hashlib.sha256(
        json.dumps(control, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def unit_id(iteration: int) -> str:
    """Return the stable identity for an immutable outer-iteration snapshot."""
    if type(iteration) is not int or iteration < 0:
        raise ValueError("invalid training iteration")
    return f"{NAMESPACE}/iteration-{iteration:04d}"


def empty_ledger(control: dict[str, Any]) -> dict[str, Any]:
    """Return the first-attempt ledger; it contains no outcome information."""
    return {
        "schema": SCHEMA,
        "namespace": NAMESPACE,
        "allocation": allocation(control),
        "units": [],
    }


def load_ledger(
    path: Path, control: dict[str, Any], *, input_root: Path | None = None
) -> dict[int, dict[str, Any]]:
    """Validate a Mac-verified immutable ledger and return iteration descriptors."""
    payload = json.loads(path.read_text())
    if (
        payload.get("schema") != SCHEMA
        or payload.get("namespace") != NAMESPACE
        or payload.get("allocation") != allocation(control)
        or not isinstance(payload.get("units"), list)
    ):
        raise ValueError("foreign ledger namespace/allocation")
    rows: dict[int, dict[str, Any]] = {}
    for row in payload["units"]:
        identity = row.get("id", "")
        prefix = f"{NAMESPACE}/iteration-"
        if not isinstance(identity, str) or not identity.startswith(prefix):
            raise ValueError("foreign verified unit")
        try:
            iteration = int(identity.removeprefix(prefix))
        except ValueError as error:
            raise ValueError("malformed verified unit") from error
        if unit_id(iteration) != identity or iteration in rows:
            raise ValueError("foreign/duplicate verified unit")
        if not isinstance(row.get("exports"), list) or not row["exports"]:
            raise ValueError("snapshot requires complete exports")
        for export in row["exports"]:
            if (
                not isinstance(export, dict)
                or type(export.get("size")) is not int
                or export["size"] < 1
                or len(export.get("sha256", "")) != 64
                or not isinstance(export.get("path"), str)
            ):
                raise ValueError("malformed verified unit")
            raw_path = Path(export["path"])
            if input_root is not None:
                if raw_path.is_absolute() or ".." in raw_path.parts:
                    raise ValueError("unsafe relative ledger export path")
                export_path = (input_root / raw_path).resolve()
                export["path"] = str(export_path)
            else:
                export_path = raw_path
            sidecar = Path(str(export_path) + ".sha256")
            if (
                not export_path.is_file()
                or not Path(str(export_path) + ".complete").is_file()
                or not sidecar.is_file()
                or sidecar.read_text().strip() != export["sha256"]
                or digest(export_path) != export["sha256"]
                or export_path.stat().st_size != export["size"]
            ):
                raise ValueError("incomplete/corrupt verified unit")
        headers = [
            Path(export["path"]) for export in row["exports"] if export["path"].endswith(".json")
        ]
        if len(headers) != 1:
            raise ValueError("snapshot requires one metadata header")
        header = json.loads(headers[0].read_text())
        if header.get("metadata", {}).get("allocation") != allocation(control):
            raise ValueError("snapshot metadata allocation differs from ledger control")
        rows[iteration] = row
    return rows


def publish_snapshot(
    output: Path,
    iteration: int,
    writer: Callable[[Path], None],
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Publish a full state payload and a completion-last metadata export.

    The payload's sidecar and complete marker are emitted by :func:`publish`.
    The metadata itself is also immutable, so a receiver can transfer and hash
    both objects without inferring completion from a partially copied payload.
    """
    identity = unit_id(iteration)
    payload = publish(output / f"iteration-{iteration:04d}.pt", writer)
    descriptor = verified_descriptor(Path(payload["path"]), identity)
    envelope = {
        "schema": "sandbox-a-train-snapshot/v1",
        "id": identity,
        "iteration": iteration,
        "snapshot": descriptor,
        "metadata": metadata,
    }
    info_path = output / f"iteration-{iteration:04d}.json"
    info = publish(
        info_path, lambda path: path.write_text(json.dumps(envelope, sort_keys=True) + "\n")
    )
    return {"id": identity, "iteration": iteration, "exports": [descriptor, info]}


def write_ledger(path: Path, control: dict[str, Any], units: list[dict[str, Any]]) -> None:
    """Write a create-only ledger after checking output descriptors by hash only."""
    if path.exists():
        raise FileExistsError(path)
    seen: set[str] = set()
    for row in units:
        identity = row.get("id")
        if identity in seen:
            raise ValueError("duplicate verified unit")
        seen.add(identity)
    path.write_text(json.dumps({**empty_ledger(control), "units": units}, indent=2) + "\n")
