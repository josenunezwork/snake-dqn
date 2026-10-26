"""F5 selected-960 metadata and payload packaging boundary.

The status scan is lawful metadata, not a training table: it contains no death
labels and therefore cannot decide the F4 survivor/dier sampler.  This module
first freezes only selected corpus identity, exclusions, and source-row order.
The separately invoked adapter verifies an approved source hash before opening
the selected payload and rechecks every projected row against that metadata.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from research.bl_ppo.exports import export_bytes, verify_export
from research.bl_residual.data import tensor_cache, validate_rows

INVENTORY_SCHEMA = "bl-f5-status-input-inventory/v3"
SCAN_SCHEMA = "bl-f5-status-extractor/v5"
SELECTED_LIVES = 960
MIN_ELIGIBLE_PARENTS = 32
F4_COLUMNS = frozenset(
    {
        "local",
        "global",
        "scalars",
        "q",
        "legal",
        "action_mask",
        "bare",
        "died",
        "status",
        "mass",
        "ordinary",
        "natural_stream",
        "sampling_weight",
        "sampling_prob",
        "parent_world",
        "life_id",
        "row_id",
    }
)
PACKAGE_SCHEMA = "bl-f5-selected-training-package/v1"
PACKAGE_RECEIPT_SCHEMA = "bl-f5-selected-training-package-receipt/v1"
PRIVATE_BYTES_CAP = 45_397_643
TABLE_MEMBER_MAX_BYTES = 2 << 30
PROJECTION_MEMBER_MAX_BYTES = 64 << 20
RECEIPT_MEMBER_MAX_BYTES = 1 << 20
PACKAGE_PINS = {
    "inventory_sha256": "cc20e1e7f5c321de5a1f63b94945c92c15c852a8499e88ab56659d1d2d7c1615",
    "scan_sha256": "8b4910213475266038c8a6a65ad9aea00cd301eba08311f44bb59bc23c312ac4",
    "original_source_sha256": "6906bffc7241de568965c1a91f4faebe799895ee0b4650e59b20e43db1dac494",
}


@dataclass(frozen=True)
class SourcePin:
    """Identity that must be verified before a selected payload is opened."""

    sha256: str
    size: int


@dataclass(frozen=True)
class F5MetadataCache:
    """Selected, training-eligible row metadata in registered allocation order."""

    source: SourcePin
    rows: tuple[dict[str, Any], ...]
    selected_lives: tuple[str, ...]
    eligible_lives: tuple[str, ...]
    excluded_parents: frozenset[str]
    scan_sha256: str


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise ValueError(f"F5 {name} must be a lower-case SHA-256")
    return value


def _inventory_lives(inventory: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if inventory.get("schema") != INVENTORY_SCHEMA:
        raise ValueError("foreign F5 selected-960 inventory")
    if inventory.get("counts") != {"ordinary_only_siblings": 0, "selected": 960, "total": 960}:
        raise ValueError("selected-960 inventory count changed")
    records = inventory.get("original_life_sources")
    if not isinstance(records, list) or len(records) != SELECTED_LIVES:
        raise ValueError("selected-960 inventory requires exactly 960 records")
    result: dict[str, dict[str, Any]] = {}
    indexes: set[int] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("malformed selected life inventory")
        life, parent, index, order = (
            record.get("life_id"),
            record.get("parent_world"),
            record.get("selection_index"),
            record.get("allocation_order"),
        )
        if (
            not isinstance(life, str)
            or not isinstance(parent, str)
            or life.count("/") != 2
            or life.split("/")[1] != parent
            or type(index) is not int
            or type(order) is not int
            or index < 0
            or order < 0
            or index in indexes
            or life in result
            or record.get("ordinary_only") is not False
            or record.get("original_fold") not in (1, 2, 3, 4)
            or not isinstance(record.get("source_sha256"), str)
            or not isinstance(record.get("source_size"), int)
        ):
            raise ValueError("invalid selected life identity or ancestry")
        indexes.add(index)
        result[life] = record
    if indexes != set(range(SELECTED_LIVES)):
        raise ValueError("selected life indices are not the immutable 0..959 sequence")
    if (
        _require_hash(inventory.get("original_life_sources_sha256"), "selected inventory hash")
        != hashlib.sha256(_canonical(records)).hexdigest()
    ):
        raise ValueError("selected-960 inventory canonical hash mismatch")
    siblings = inventory.get("excluded_ordinary_only_sibling_sources")
    if not isinstance(siblings, list) or len(siblings) != 9:
        raise ValueError("F5 requires the nine immutable excluded siblings")
    if (
        _require_hash(
            inventory.get("excluded_ordinary_only_sibling_sources_sha256"), "sibling hash"
        )
        != hashlib.sha256(_canonical(siblings)).hexdigest()
    ):
        raise ValueError("excluded sibling inventory canonical hash mismatch")
    sibling_ids = {record.get("life_id") for record in siblings if isinstance(record, dict)}
    if (
        len(sibling_ids) != 9
        or sibling_ids & set(result)
        or any(record.get("ordinary_only") is not True for record in siblings)
    ):
        raise ValueError("excluded sibling identity changed")
    prefix = [
        result[life]
        for life in sorted(result, key=lambda item: result[item]["selection_index"])[:192]
    ]
    if any("mac-d256" not in str(record.get("source_namespace")) for record in prefix):
        raise ValueError("D256 prefix life identity changed")
    return result


def metadata_cache(scan: Mapping[str, Any], inventory: Mapping[str, Any]) -> F5MetadataCache:
    """Validate lawful metadata and return ordered selected, non-development rows.

    This function never follows a path or opens a payload.  The returned rows
    intentionally omit outcome fields, so callers cannot manufacture a batch
    plan before the authorized payload-packaging step.
    """
    lives = _inventory_lives(inventory)
    exclusions = inventory.get("training_exclusions")
    if (
        not isinstance(exclusions, list)
        or not exclusions
        or any(not isinstance(x, str) for x in exclusions)
    ):
        raise ValueError("F5 training exclusions are required")
    excluded = frozenset(exclusions)
    if scan.get("schema") != SCAN_SCHEMA:
        raise ValueError("foreign F5 status scan")
    binding = scan.get("source_binding")
    container = inventory.get("converted_container")
    if not isinstance(binding, dict) or not isinstance(container, dict):
        raise ValueError("F5 source binding missing")
    source_sha = _require_hash(binding.get("sha256"), "source hash")
    if source_sha != container.get("sha256") or binding.get("size") != container.get("size"):
        raise ValueError("scan source binding differs from selected-960 inventory")
    rows = scan.get("status_inventory")
    if not isinstance(rows, list) or not rows:
        raise ValueError("F5 status inventory missing")
    if scan.get("status_inventory_sha256") != hashlib.sha256(_canonical(rows)).hexdigest():
        raise ValueError("F5 status inventory hash mismatch")

    by_source: dict[int, list[dict[str, Any]]] = {}
    for record in rows:
        if not isinstance(record, dict):
            raise ValueError("malformed status inventory row")
        required = {
            "action",
            "allocation_order",
            "legal",
            "life_id",
            "parent_world",
            "row_id",
            "row_order",
            "served_selectable",
            "source_row_index",
            "status",
            "stratum",
        }
        if set(record) != required:
            raise ValueError("status inventory schema changed")
        life, parent, source_index = (
            record["life_id"],
            record["parent_world"],
            record["source_row_index"],
        )
        if (
            life not in lives
            or parent != lives[life]["parent_world"]
            or record["allocation_order"] != lives[life]["allocation_order"]
            or type(source_index) is not int
            or source_index < 0
            or type(record["row_order"]) is not int
            or record["row_order"] < 0
            or type(record["action"]) is not int
            or record["action"] not in range(6)
            or type(record["legal"]) is not bool
            or type(record["served_selectable"]) is not bool
            or record["status"] not in ("EXACT", "NONEXACT")
            or record["stratum"] not in ("ordinary", "doom")
        ):
            raise ValueError("status inventory identity or metadata changed")
        by_source.setdefault(source_index, []).append(record)

    ordered: list[dict[str, Any]] = []
    seen_rows: set[str] = set()
    for source_index, action_rows in by_source.items():
        action_rows.sort(key=lambda row: row["action"])
        first = action_rows[0]
        if (
            len(action_rows) != 6
            or [row["action"] for row in action_rows] != list(range(6))
            or any(
                (
                    row["life_id"],
                    row["parent_world"],
                    row["row_id"],
                    row["allocation_order"],
                    row["row_order"],
                    row["stratum"],
                )
                != (
                    first["life_id"],
                    first["parent_world"],
                    first["row_id"],
                    first["allocation_order"],
                    first["row_order"],
                    first["stratum"],
                )
                for row in action_rows
            )
            or first["row_id"] in seen_rows
        ):
            raise ValueError("status inventory action-unit disagreement")
        seen_rows.add(first["row_id"])
        if first["parent_world"] not in excluded:
            ordered.append(
                {
                    "source_row_index": source_index,
                    "life_id": first["life_id"],
                    "parent_world": first["parent_world"],
                    "row_id": first["row_id"],
                    "allocation_order": first["allocation_order"],
                    "row_order": first["row_order"],
                    "ordinary": first["stratum"] == "ordinary",
                    "status": tuple(row["status"] for row in action_rows),
                    "legal": tuple(row["legal"] for row in action_rows),
                    "action_mask": tuple(row["served_selectable"] for row in action_rows),
                }
            )
    ordered.sort(
        key=lambda row: (row["allocation_order"], row["row_order"], row["source_row_index"])
    )
    eligible_lives = tuple(
        sorted({row["life_id"] for row in ordered}, key=lambda life: lives[life]["selection_index"])
    )
    observed_lives = {
        record["life_id"] for action_rows in by_source.values() for record in action_rows
    }
    if observed_lives != set(lives):
        raise ValueError("selected-960 status scan does not observe every selected life")
    parents = {row["parent_world"] for row in ordered}
    if len(parents) < MIN_ELIGIBLE_PARENTS:
        raise ValueError("F5 needs at least 32 eligible parents after exclusions")
    if not any(row["ordinary"] for row in ordered) or not any(
        not row["ordinary"] for row in ordered
    ):
        raise ValueError("F5 selected training support requires both strata")
    return F5MetadataCache(
        source=SourcePin(source_sha, int(binding["size"])),
        rows=tuple(ordered),
        selected_lives=tuple(sorted(lives, key=lambda life: lives[life]["selection_index"])),
        eligible_lives=eligible_lives,
        excluded_parents=excluded,
        scan_sha256=hashlib.sha256(_canonical(rows)).hexdigest(),
    )


def selected_payload_table(source: Path, metadata: F5MetadataCache) -> dict[str, np.ndarray]:
    """Open a hash-pinned selected payload and convert it to F4's strict schema.

    This is the sole payload-reading API in this module.  It hashes before
    deserialization, filters by the already-frozen ``source_row_index`` set,
    and refuses changed ancestry, row order, masks, or status metadata.
    """
    if not source.is_file() or source.is_symlink() or source.stat().st_size != metadata.source.size:
        raise ValueError("F5 selected payload source identity mismatch")
    if _sha_file(source) != metadata.source.sha256:
        raise ValueError("F5 selected payload hash mismatch")
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != F4_COLUMNS:
            raise ValueError("F5 payload does not have the strict F4 table schema")
        source_rows = np.asarray([row["source_row_index"] for row in metadata.rows], dtype=np.int64)
        if source_rows.size != len(set(source_rows.tolist())):
            raise ValueError("F5 metadata duplicates source row identity")
        table = {name: np.asarray(archive[name])[source_rows].copy() for name in F4_COLUMNS}
    return _validate_selected_table(table, metadata, source_rows)


def _validate_selected_table(
    table: dict[str, np.ndarray], metadata: F5MetadataCache, source_rows: np.ndarray
) -> dict[str, np.ndarray]:
    """Apply the source-row projection checks to an already-loaded F4 table."""
    if set(table) != F4_COLUMNS:
        raise ValueError("F5 payload does not have the strict F4 table schema")
    n = len(metadata.rows)
    if any(value.shape[0] != n for value in table.values()):
        raise ValueError("F5 selected payload source index is outside a column")
    for index, row in enumerate(metadata.rows):
        if (
            str(table["life_id"][index]) != row["life_id"]
            or str(table["parent_world"][index]) != row["parent_world"]
            or str(table["row_id"][index]) != row["row_id"]
            or bool(table["ordinary"][index]) != row["ordinary"]
            or not np.array_equal(table["legal"][index], np.asarray(row["legal"], dtype=bool))
            or not np.array_equal(
                np.where(
                    table["action_mask"][index].any(),
                    table["action_mask"][index],
                    np.asarray([True, True, True, False, False, False]),
                ),
                np.asarray(row["action_mask"], dtype=bool),
            )
            or tuple(
                "EXACT" if value == "EXACT" else "NONEXACT" for value in table["status"][index]
            )
            != row["status"]
        ):
            raise ValueError("F5 payload row differs from frozen metadata")
    table["source_row_index"] = source_rows
    table["allocation_order"] = np.asarray(
        [row["allocation_order"] for row in metadata.rows], dtype=np.int64
    )
    table["row_order"] = np.asarray([row["row_order"] for row in metadata.rows], dtype=np.int64)
    validate_rows(table)
    return table


def _projection(metadata: F5MetadataCache) -> dict[str, Any]:
    """Serialize only selected metadata needed to validate the compact package."""
    return {
        "schema": PACKAGE_SCHEMA,
        "original_source": {"sha256": metadata.source.sha256, "size": metadata.source.size},
        "status_inventory_sha256": metadata.scan_sha256,
        "selected_lives": list(metadata.selected_lives),
        "eligible_lives": list(metadata.eligible_lives),
        "excluded_parents": sorted(metadata.excluded_parents),
        "rows": [
            {
                **row,
                "status": list(row["status"]),
                "legal": list(row["legal"]),
                "action_mask": list(row["action_mask"]),
            }
            for row in metadata.rows
        ],
    }


def selected_corpus_identity(metadata: F5MetadataCache, payload_sha256: str) -> str:
    """Return the runtime-recomputable selected corpus identity.

    The identity binds the original F4 source, every selected row's frozen
    metadata/order, and the compact selected payload.  A control file is never
    an authority for this value.
    """
    payload_sha256 = _require_hash(payload_sha256, "selected payload hash")
    projection_sha = hashlib.sha256(_canonical(_projection(metadata))).hexdigest()
    value = {
        "schema": "bl-f5-selected-corpus-identity/v1",
        "original_source_sha256": metadata.source.sha256,
        "status_inventory_sha256": metadata.scan_sha256,
        "projection_sha256": projection_sha,
        "selected_payload_sha256": payload_sha256,
    }
    return hashlib.sha256(_canonical(value)).hexdigest()


def _pins(value: Mapping[str, str] | None) -> dict[str, str]:
    pins = dict(PACKAGE_PINS if value is None else value)
    if set(pins) != set(PACKAGE_PINS) or any(
        _require_hash(item, key) != item for key, item in pins.items()
    ):
        raise ValueError("F5 package pins are malformed")
    return pins


def _npz_bytes(table: Mapping[str, np.ndarray]) -> bytes:
    """Create an uncompressed NPZ so outer xz compression sees all columns."""
    stream = io.BytesIO()
    np.savez(stream, **{name: table[name] for name in sorted(F4_COLUMNS)})
    return stream.getvalue()


def _tar_member(name: str, value: bytes) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = len(value)
    info.mode = 0o600
    return info


def _safe_file(path: Path) -> Path:
    """Refuse links and protected-study paths before any file access."""
    path = Path(path)
    if not path.is_absolute() or any(
        "d4096" in part.lower() or part == "t34-round0-study-01" for part in path.parts
    ):
        raise ValueError("F5 package path is not an allowed selected-source path")
    current = path
    while True:
        if current.is_symlink():
            raise ValueError("F5 package symlink path refused")
        if current.parent == current:
            break
        current = current.parent
    if not path.is_file():
        raise ValueError("F5 package input must be a regular file")
    return path


def _pinned_json(path: Path, expected: str, name: str) -> dict[str, Any]:
    """Hash an allowed raw metadata file before parsing it."""
    path = _safe_file(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"F5 {name} raw hash mismatch")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"F5 {name} must be a JSON object")
    return value


def package_selected(
    source: Path,
    scan: Mapping[str, Any],
    inventory: Mapping[str, Any],
    output: Path,
    *,
    pins: Mapping[str, str] | None = None,
    max_bytes: int = PRIVATE_BYTES_CAP,
) -> dict[str, Any]:
    """Create a bounded, create-only selected-training bundle on the Mac.

    The archive contains only the selected strict F4 table and its compact
    projection.  It deliberately omits the 214 MB scan and every unselected
    row.  A cap failure removes the just-created incomplete bundle and holds.
    """
    if output.exists() or output.is_symlink() or max_bytes <= 0:
        raise ValueError("F5 package output must be fresh with a positive byte cap")
    package_pins = _pins(pins)
    metadata = metadata_cache(scan, inventory)
    if metadata.source.sha256 != package_pins["original_source_sha256"]:
        raise ValueError("F5 package source pin differs from selected-source binding")
    table = selected_payload_table(source, metadata)
    table_bytes = _npz_bytes(table)
    payload_sha = hashlib.sha256(table_bytes).hexdigest()
    projection = _projection(metadata)
    projection_bytes = _canonical(projection)
    receipt = {
        "schema": PACKAGE_RECEIPT_SCHEMA,
        "pins": package_pins,
        "original_source": {"sha256": metadata.source.sha256, "size": metadata.source.size},
        "selected_payload_sha256": payload_sha,
        "projection_sha256": hashlib.sha256(projection_bytes).hexdigest(),
        "corpus_sha256": selected_corpus_identity(metadata, payload_sha),
        "row_count": len(metadata.rows),
        "selected_life_count": len(metadata.selected_lives),
        "eligible_life_count": len(metadata.eligible_lives),
    }
    receipt_bytes = _canonical(receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".f5-selected-", suffix=".tar.xz", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with tarfile.open(temporary, "w:xz", preset=9) as archive:
            for name, value in (
                ("selected-table.npz", table_bytes),
                ("projection.json", projection_bytes),
                ("receipt.json", receipt_bytes),
            ):
                archive.addfile(_tar_member(name, value), io.BytesIO(value))
        payload = temporary.read_bytes()
        sidecar_bytes = 64 + 2 + len(output.name.encode()) + 1 + len(b"complete\n")
        if len(payload) + sidecar_bytes > max_bytes:
            raise ValueError("HOLD: selected-training package exceeds private byte cap")
        publication = export_bytes(output, payload)
    except Exception:
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return {
        **receipt,
        "package_sha256": publication["sha256"],
        "package_bytes": publication["size"],
    }


def package_selected_files(
    source: Path, scan_path: Path, inventory_path: Path, output: Path
) -> dict[str, Any]:
    """Production package entry point with raw-file pins verified before parsing.

    Synthetic callers must use ``package_selected`` with explicit fixture pins.
    This wrapper is non-configurable so a production operator cannot substitute
    a different scan, selected inventory, or protected input path.
    """
    source = _safe_file(source)
    scan = _pinned_json(scan_path, PACKAGE_PINS["scan_sha256"], "scan")
    inventory = _pinned_json(inventory_path, PACKAGE_PINS["inventory_sha256"], "inventory")
    return package_selected(source, scan, inventory, output, pins=PACKAGE_PINS)


def _metadata_from_projection(value: Mapping[str, Any]) -> F5MetadataCache:
    expected = {
        "schema",
        "original_source",
        "status_inventory_sha256",
        "selected_lives",
        "eligible_lives",
        "excluded_parents",
        "rows",
    }
    if set(value) != expected or value.get("schema") != PACKAGE_SCHEMA:
        raise ValueError("F5 compact projection schema mismatch")
    source = value["original_source"]
    rows = value["rows"]
    if (
        not isinstance(source, dict)
        or set(source) != {"sha256", "size"}
        or not isinstance(rows, list)
        or not isinstance(value["selected_lives"], list)
        or not isinstance(value["eligible_lives"], list)
        or not isinstance(value["excluded_parents"], list)
    ):
        raise ValueError("F5 compact projection types mismatch")
    rebuilt = []
    required = {
        "source_row_index",
        "life_id",
        "parent_world",
        "row_id",
        "allocation_order",
        "row_order",
        "ordinary",
        "status",
        "legal",
        "action_mask",
    }
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != required or row["source_row_index"] in seen:
            raise ValueError("F5 compact projection row mismatch")
        seen.add(row["source_row_index"])
        if (
            type(row["source_row_index"]) is not int
            or type(row["ordinary"]) is not bool
            or any(
                not isinstance(row[key], list) or len(row[key]) != 6
                for key in ("status", "legal", "action_mask")
            )
        ):
            raise ValueError("F5 compact projection row type mismatch")
        rebuilt.append(
            {
                **row,
                "status": tuple(row["status"]),
                "legal": tuple(row["legal"]),
                "action_mask": tuple(row["action_mask"]),
            }
        )
    metadata = F5MetadataCache(
        source=SourcePin(_require_hash(source["sha256"], "compact source hash"), source["size"]),
        rows=tuple(rebuilt),
        selected_lives=tuple(value["selected_lives"]),
        eligible_lives=tuple(value["eligible_lives"]),
        excluded_parents=frozenset(value["excluded_parents"]),
        scan_sha256=_require_hash(value["status_inventory_sha256"], "compact scan hash"),
    )
    if not metadata.rows or len(set(metadata.selected_lives)) != SELECTED_LIVES:
        raise ValueError("F5 compact projection corpus mismatch")
    return metadata


def read_selected_package(
    package: Path, *, pins: Mapping[str, str] | None = None, max_bytes: int = PRIVATE_BYTES_CAP
) -> tuple[dict[str, np.ndarray], F5MetadataCache, dict[str, Any]]:
    """Read and verify a compact package without extracting archive members."""
    package_pins = _pins(pins)
    if not package.is_file() or package.is_symlink() or package.stat().st_size > max_bytes:
        raise ValueError("HOLD: selected-training package identity or byte cap failed")
    verify_export(package)
    with tarfile.open(package, "r:xz") as archive:
        members = archive.getmembers()
        limits = {
            "selected-table.npz": TABLE_MEMBER_MAX_BYTES,
            "projection.json": PROJECTION_MEMBER_MAX_BYTES,
            "receipt.json": RECEIPT_MEMBER_MAX_BYTES,
        }
        if (
            len(members) != 3
            or {item.name for item in members} != set(limits)
            or any(not item.isreg() or not 0 < item.size <= limits[item.name] for item in members)
            or sum(item.size for item in members)
            > TABLE_MEMBER_MAX_BYTES + PROJECTION_MEMBER_MAX_BYTES + RECEIPT_MEMBER_MAX_BYTES
        ):
            raise ValueError("F5 compact package member schema mismatch")
        contents = {item.name: archive.extractfile(item).read() for item in members}
    receipt = json.loads(contents["receipt.json"])
    projection = json.loads(contents["projection.json"])
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != PACKAGE_RECEIPT_SCHEMA
        or receipt.get("pins") != package_pins
    ):
        raise ValueError("F5 compact package receipt pin mismatch")
    metadata = _metadata_from_projection(projection)
    if metadata.source.sha256 != package_pins["original_source_sha256"] or hashlib.sha256(
        _canonical(projection)
    ).hexdigest() != receipt.get("projection_sha256"):
        raise ValueError("F5 compact package projection binding mismatch")
    payload_sha = hashlib.sha256(contents["selected-table.npz"]).hexdigest()
    if payload_sha != receipt.get("selected_payload_sha256") or selected_corpus_identity(
        metadata, payload_sha
    ) != receipt.get("corpus_sha256"):
        raise ValueError("F5 compact package corpus identity mismatch")
    with np.load(io.BytesIO(contents["selected-table.npz"]), allow_pickle=False) as archive:
        table = {name: np.asarray(archive[name]).copy() for name in archive.files}
    source_rows = np.asarray([row["source_row_index"] for row in metadata.rows], dtype=np.int64)
    return _validate_selected_table(table, metadata, source_rows), metadata, receipt


# The explicit export name is retained for the Mac packaging entry point; the
# shorter name is convenient for its focused synthetic contract tests.
export_selected_package = package_selected


def selected_tensor_cache(
    table: dict[str, np.ndarray], base: Any, *, batch_size: int = 128
) -> dict[str, torch.Tensor]:
    """Build the existing F4 feature cache only after ``selected_payload_table`` succeeds."""
    return tensor_cache(table, base, batch_size=batch_size)
