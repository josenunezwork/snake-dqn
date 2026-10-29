"""Immutable full-trajectory exports for the simulator-D PPO package.

The training process only writes a snapshot once.  A separate Mac-side process
may construct a verified ledger from the resulting descriptors; loading a
ledger deliberately never opens numerical payloads.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping

from research.sandbox_a_resident.units import digest, publish, verified_descriptor

NAMESPACE = "a100-dev/sandbox-a/train01"
SCHEMA = "sandbox-a-train-ledger/v1"
SOURCE_MIGRATION_SCHEMA = "sandbox-a-train-source-migration/v1"
# Explicit frozen predecessor pins whose verified snapshots may be migrated to a
# source-only successor.  This is intentionally not a general source-trust list.
TRAIN01_07_SOURCE_COMMIT = "152af938acc32da6aeea9fcb5a300f470d96c53c"
TRAIN01_11_SOURCE_COMMIT = "e86e4b0bc8ea37550103b14652878d1b63f5b0e1"
TRAIN01_15_SOURCE_COMMIT = "5c0a101a85237e2ce563dcaa6429828126a37d04"
TRAIN01_16_SOURCE_COMMIT = "5cad044f3d9725a0ddbb967998b82c6eb9cb7739"
ALLOWED_MIGRATION_SOURCE_COMMITS = frozenset(
    {
        TRAIN01_07_SOURCE_COMMIT,
        TRAIN01_11_SOURCE_COMMIT,
        TRAIN01_15_SOURCE_COMMIT,
        TRAIN01_16_SOURCE_COMMIT,
    }
)

COMPACT_RESUME_FIELD = "compact_resume"
COMPACT_RESUME_SCHEMA = "sandbox-a-train-compact-resume/v1"
COMPACT_RESUME_SCHEMA_V2 = "sandbox-a-train-compact-resume/v2"
COMPACT_SNAPSHOT_EXPORT_SCHEMA = "sandbox-a-train-compact-export/v1"
COMPACT_ALLOWLIST_SCHEMA = "sandbox-a-train-compact-allowlist/v2"
COMPACT_ALLOWLIST_PATH = Path(__file__).resolve().with_name("compact_allowlist.json")
COMPACT_DERIVATIVE_SUFFIX = ".xz"
COMPACT_MANIFEST_SUFFIX = ".xz.manifest"
TRAIN01_COMPACT_SOURCE_PACKET_RE = re.compile(
    r"^packet-train01-(?:11|15|16|17|s2-0[12345])(?:-c[12](?:-[a-z0-9][a-z0-9_-]*)?)?$"
)
# Legacy v1 allowlist retained verbatim so frozen packet-train01-15 preflight
# and historical v1 ledger bindings remain byte-exact.  Periodic continuations
# of train01-15/16/17 use COMPACT_RESUME_SCHEMA_V2 keyed by
# (source_packet_id, iteration, original_sha256) in compact_allowlist.json.
COMPACT_RESUME_ALLOWLIST: dict[str, dict[str, dict[str, Any]]] = {
    f"{NAMESPACE}/iteration-0000": {
        "original": {
            "path": "resume/iteration-0000.pt",
            "size": 1_327_620_931,
            "sha256": "2b5478e895e7b2b68c02e3bd118612257ea4e97c0cfab727013d7e2143338ab6",
        },
        "derivative": {
            "path": "resume/iteration-0000.pt.xz",
            "size": 8_804_064,
            "sha256": "a25562d7e95e5dd486620dbe3eee3e7445396c8eb5f5f5f6f1f0f4fdda05fb7b",
        },
        "manifest": {
            "path": "resume/iteration-0000.pt.xz.manifest",
            "size": 69_385,
            "sha256": "44c0b5d30e48ef3d3be370e30cc478b8ad0855ee2da236bdfaab4f9b5838efd5",
        },
    }
}


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(char in "0123456789abcdef" for char in value)
    )


def validate_compact_source_packet(source_packet_id: str, iteration: int) -> None:
    """Validate that a compact-resume source packet and iteration belong to TRAIN-01."""
    if not isinstance(source_packet_id, str) or not TRAIN01_COMPACT_SOURCE_PACKET_RE.fullmatch(
        source_packet_id
    ):
        raise ValueError(f"unsupported compact resume source packet: {source_packet_id!r}")
    if type(iteration) is not int or not 0 <= iteration <= 384:
        raise ValueError("compact resume iteration must lie in 0..384")
    if (
        source_packet_id == "packet-train01-11" or source_packet_id.startswith("packet-train01-11-")
    ) and iteration != 0:
        raise ValueError("packet-train01-11 compact resume is restricted to iteration 0")


def _validate_compact_descriptor(descriptor: Any, *, expected_path: str, role: str) -> None:
    if (
        not isinstance(descriptor, dict)
        or set(descriptor) != {"path", "size", "sha256"}
        or descriptor.get("path") != expected_path
        or type(descriptor.get("size")) is not int
        or descriptor["size"] < 1
        or not _is_hex(descriptor.get("sha256"), 64)
    ):
        raise ValueError(f"malformed compact allowlist {role} descriptor")


def _validate_compact_allowlist_entry(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("malformed compact allowlist entry")
    required = {"source_packet_id", "iteration", "unit", "original", "derivative", "manifest"}
    optional = {"source_commit"}
    if not required.issubset(item) or set(item) - (required | optional):
        raise ValueError("malformed compact allowlist entry keys")
    source_packet_id = item["source_packet_id"]
    iteration = item["iteration"]
    validate_compact_source_packet(source_packet_id, iteration)
    expected_unit = unit_id(iteration)
    if item["unit"] != expected_unit:
        raise ValueError("compact allowlist unit does not match iteration")
    stem = f"resume/iteration-{iteration:04d}.pt"
    _validate_compact_descriptor(item["original"], expected_path=stem, role="original")
    _validate_compact_descriptor(
        item["derivative"],
        expected_path=stem + COMPACT_DERIVATIVE_SUFFIX,
        role="derivative",
    )
    _validate_compact_descriptor(
        item["manifest"],
        expected_path=stem + COMPACT_MANIFEST_SUFFIX,
        role="manifest",
    )
    if "source_commit" in item and not _is_hex(item["source_commit"], 40):
        raise ValueError("compact allowlist source_commit must be a 40-char lowercase SHA")
    return copy.deepcopy(item)


def load_compact_allowlist(
    path: Path | None = None,
) -> dict[tuple[str, int, str], dict[str, Any]]:
    """Load the committed v2 compact-resume allowlist keyed by (packet, iter, sha256)."""
    target = COMPACT_ALLOWLIST_PATH if path is None else path
    payload = json.loads(target.read_text())
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != COMPACT_ALLOWLIST_SCHEMA
        or not isinstance(payload.get("entries"), list)
    ):
        raise ValueError("malformed compact allowlist document")
    table: dict[tuple[str, int, str], dict[str, Any]] = {}
    seen_slots: set[tuple[str, int]] = set()
    for raw in payload["entries"]:
        item = _validate_compact_allowlist_entry(raw)
        slot = (item["source_packet_id"], item["iteration"])
        key = (item["source_packet_id"], item["iteration"], item["original"]["sha256"])
        if slot in seen_slots or key in table:
            raise ValueError("duplicate compact allowlist source_packet_id/iteration entry")
        seen_slots.add(slot)
        table[key] = item
    return table


def register_compact_allowlist_entry(
    entry: Mapping[str, Any], *, path: Path | None = None
) -> dict[str, Any]:
    """Idempotently register one validated v2 allowlist entry without mutating others."""
    validated = _validate_compact_allowlist_entry(dict(entry))
    target = COMPACT_ALLOWLIST_PATH if path is None else path
    current = load_compact_allowlist(target)
    slot = (validated["source_packet_id"], validated["iteration"])
    for existing_key, existing in current.items():
        if existing_key[:2] == slot:
            if existing == validated:
                return existing
            raise ValueError("conflicting compact allowlist entry for source packet and iteration")
    entries = sorted(
        [*current.values(), validated],
        key=lambda row: (row["source_packet_id"], row["iteration"], row["original"]["sha256"]),
    )
    document = {"schema": COMPACT_ALLOWLIST_SCHEMA, "entries": entries}
    partial = target.with_name(target.name + ".partial")
    partial.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    os.replace(partial, target)
    load_compact_allowlist(target)
    return validated


def allowed_migration_source_commits(*, allowlist_path: Path | None = None) -> frozenset[str]:
    """Return frozen predecessor pins plus pins bound in the committed v2 allowlist."""
    commits = set(ALLOWED_MIGRATION_SOURCE_COMMITS)
    target = COMPACT_ALLOWLIST_PATH if allowlist_path is None else allowlist_path
    if target.is_file():
        for entry in load_compact_allowlist(target).values():
            commit = entry.get("source_commit")
            if _is_hex(commit, 40):
                commits.add(commit)
    return frozenset(commits)


def allocation(control: dict[str, Any]) -> str:
    """Return the frozen control identity independently of an attempt path."""
    return hashlib.sha256(
        json.dumps(control, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def control_digest(control: dict[str, Any]) -> str:
    """Return the canonical digest used to bind a source-control migration."""
    return hashlib.sha256(
        json.dumps(control, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def source_migration(
    source_control: dict[str, Any],
    target_control: dict[str, Any],
    *,
    allowlist_path: Path | None = None,
) -> dict[str, Any]:
    """Bind the sole permitted successor change: its committed source pin."""
    if source_control == target_control:
        raise ValueError("source migration requires a distinct successor control")
    if (
        not isinstance(source_control.get("source_commit"), str)
        or not isinstance(target_control.get("source_commit"), str)
        or source_control["source_commit"]
        not in allowed_migration_source_commits(allowlist_path=allowlist_path)
        or {key: value for key, value in source_control.items() if key != "source_commit"}
        != {key: value for key, value in target_control.items() if key != "source_commit"}
    ):
        raise ValueError("source migration may change only source_commit")
    return {
        "schema": SOURCE_MIGRATION_SCHEMA,
        "from_control": source_control,
        "from_control_sha256": control_digest(source_control),
        "from_allocation": allocation(source_control),
        "from_source_commit": source_control["source_commit"],
        "to_control_sha256": control_digest(target_control),
        "to_allocation": allocation(target_control),
        "to_source_commit": target_control["source_commit"],
    }


def migration_source_control(
    payload: dict[str, Any],
    target_control: dict[str, Any],
    *,
    allowlist_path: Path | None = None,
) -> dict[str, Any]:
    """Validate a ledger migration and return the bound snapshot control."""
    migration = payload.get("source_migration")
    if migration is None:
        return target_control
    if not isinstance(migration, dict) or migration.get("schema") != SOURCE_MIGRATION_SCHEMA:
        raise ValueError("malformed source migration")
    source = migration.get("from_control")
    if not isinstance(source, dict):
        raise ValueError("source migration lacks predecessor control")
    if migration != source_migration(source, target_control, allowlist_path=allowlist_path):
        raise ValueError("source migration does not exactly bind predecessor and successor")
    return source


def unit_id(iteration: int) -> str:
    """Return the stable identity for an immutable outer-iteration snapshot."""
    if type(iteration) is not int or iteration < 0:
        raise ValueError("invalid training iteration")
    return f"{NAMESPACE}/iteration-{iteration:04d}"


def validate_compact_resume(
    payload: dict[str, Any],
    *,
    allowlist: dict[Any, dict[str, Any]] | None = None,
    allowlist_path: Path | None = None,
) -> dict[str, Any] | None:
    """Validate a ledger's compact-transport binding without opening any payload.

    Supports both legacy ``v1`` bindings (unit-keyed ``COMPACT_RESUME_ALLOWLIST``)
    and ``v2`` bindings keyed by ``(source_packet_id, iteration, original_sha256)``
    in ``compact_allowlist.json``.  Returns ``None`` when the ledger has no
    compact binding.
    """
    if COMPACT_RESUME_FIELD not in payload:
        return None
    entry = payload[COMPACT_RESUME_FIELD]
    if not isinstance(entry, dict):
        raise ValueError("malformed compact resume binding")
    schema = entry.get("schema")
    v1_keys = {"schema", "unit", "original", "derivative", "manifest"}
    v2_keys = {
        "schema",
        "source_packet_id",
        "iteration",
        "unit",
        "original",
        "derivative",
        "manifest",
    }
    if schema == COMPACT_RESUME_SCHEMA:
        if set(entry) != v1_keys:
            raise ValueError("malformed compact resume binding")
        frozen_v1 = COMPACT_RESUME_ALLOWLIST if allowlist is None else allowlist
        expected = frozen_v1.get(entry.get("unit")) if isinstance(entry.get("unit"), str) else None
        if expected is None:
            raise ValueError("compact resume unit is not allowlisted")
        for role in ("original", "derivative", "manifest"):
            if entry[role] != expected[role]:
                raise ValueError(f"compact resume {role} differs from the frozen allowlist")
    elif schema == COMPACT_RESUME_SCHEMA_V2:
        if set(entry) != v2_keys:
            raise ValueError("malformed compact resume binding")
        source_packet_id = entry.get("source_packet_id")
        iteration = entry.get("iteration")
        original = entry.get("original")
        if (
            not isinstance(source_packet_id, str)
            or type(iteration) is not int
            or iteration < 0
            or not isinstance(entry.get("unit"), str)
            or not isinstance(original, dict)
            or not isinstance(original.get("sha256"), str)
        ):
            raise ValueError("malformed compact resume binding")
        if entry["unit"] != unit_id(iteration):
            raise ValueError("malformed compact resume binding")
        try:
            validate_compact_source_packet(source_packet_id, iteration)
        except ValueError as error:
            raise ValueError("compact resume unit is not allowlisted") from error
        frozen_v2 = load_compact_allowlist(allowlist_path) if allowlist is None else allowlist
        expected = frozen_v2.get((source_packet_id, iteration, original["sha256"]))
        if expected is None:
            if any(
                isinstance(key, tuple)
                and len(key) == 3
                and key[:2] == (source_packet_id, iteration)
                for key in frozen_v2
            ):
                raise ValueError("compact resume original differs from the frozen allowlist")
            raise ValueError("compact resume unit is not allowlisted")
        if entry["unit"] != expected.get("unit"):
            raise ValueError("compact resume unit differs from the frozen allowlist")
        for role in ("original", "derivative", "manifest"):
            if entry[role] != expected[role]:
                raise ValueError(f"compact resume {role} differs from the frozen allowlist")
        migration = payload.get("source_migration")
        if (
            isinstance(migration, dict)
            and "source_commit" in expected
            and migration.get("from_source_commit") != expected["source_commit"]
        ):
            raise ValueError("compact resume source_commit differs from ledger source migration")
    else:
        if set(entry) not in (v1_keys, v2_keys):
            raise ValueError("malformed compact resume binding")
        raise ValueError("compact resume unit is not allowlisted")
    units = payload.get("units")
    if (
        not isinstance(units, list)
        or len(units) != 1
        or not isinstance(units[0], dict)
        or units[0].get("id") != entry["unit"]
        or not isinstance(units[0].get("exports"), list)
        or not units[0]["exports"]
        or units[0]["exports"][0] != entry["original"]
    ):
        raise ValueError("compact resume must bind the ledger's sole full-state export")
    return entry


def empty_ledger(control: dict[str, Any]) -> dict[str, Any]:
    """Return the first-attempt ledger; it contains no outcome information."""
    return {
        "schema": SCHEMA,
        "namespace": NAMESPACE,
        "allocation": allocation(control),
        "units": [],
    }


def load_ledger(
    path: Path,
    control: dict[str, Any],
    *,
    input_root: Path | None = None,
    allowlist_path: Path | None = None,
) -> dict[int, dict[str, Any]]:
    """Validate a Mac-verified immutable ledger and return iteration descriptors."""
    payload = json.loads(path.read_text())
    source_control = migration_source_control(payload, control, allowlist_path=allowlist_path)
    # Validate the compact-transport binding against relative ledger paths
    # before export paths are resolved below.  The runtime materializes the
    # bound original before this call; the Mac checks its local original.
    validate_compact_resume(payload, allowlist_path=allowlist_path)
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
        metadata = header.get("metadata", {})
        if (
            metadata.get("allocation") != allocation(source_control)
            or (
                "source_commit" in metadata
                and metadata["source_commit"] != source_control.get("source_commit")
            )
            or (payload.get("source_migration") is not None and "source_commit" not in metadata)
        ):
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
