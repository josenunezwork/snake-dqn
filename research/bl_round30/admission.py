"""R30-0 metadata admission and immutable outcome-blind export primitives.

This module deliberately consumes JSON metadata only.  It has no label, feature,
or outcome-file opener: a caller must obtain a ``Codex-V`` approval bound to the
frozen inventory digest before it may hand the frozen package to a data consumer.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

INVENTORY_SCHEMA = "bl-round30-admission-inventory/v1"
APPROVAL_SCHEMA = "bl-round30-codex-v-approval/v1"
LEDGER_SCHEMA = "bl-round30-verified-ledger/v1"
UNIT_SCHEMA = "bl-round30-immutable-unit/v1"
SPLIT_ALGORITHM = "sha256-nul-parent-mod10-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PARENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}")


class AdmissionRefused(ValueError):
    """The corpus or resume evidence violates a pre-data R30 boundary."""


@dataclass(frozen=True)
class Admission:
    """Frozen metadata used by the later package builder, never label bytes."""

    inventory_sha256: str
    allocation_sha256: str
    salt: str
    train_parents: tuple[str, ...]
    validation_parents: tuple[str, ...]
    validation_rescue_parents: tuple[str, ...]
    r30_2_status: str
    inventory: dict[str, Any]


def canonical_bytes(value: Any) -> bytes:
    """Return the sole JSON encoding used for R30 hashes and immutable payloads."""
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
    except (TypeError, ValueError) as exc:
        raise AdmissionRefused("metadata is not canonical JSON") from exc


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_inventory_json(path: Path) -> dict[str, Any]:
    """Read one JSON inventory, without following any metadata path to label bytes."""
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AdmissionRefused("R30 inventory is not readable JSON metadata") from exc
    if not isinstance(value, dict):
        raise AdmissionRefused("R30 inventory must be a JSON object")
    return value


def parent_split(parent_id: str, salt: str) -> str:
    """Assign an ultimate parent once; the public rule has no sampling/redraw knob."""
    _require_parent(parent_id)
    if not isinstance(salt, str) or not salt:
        raise AdmissionRefused("R30 split salt must be a nonempty declared string")
    raw = hashlib.sha256((salt + "\0" + parent_id).encode("utf-8")).digest()
    return "validation" if int.from_bytes(raw, "big") % 10 < 2 else "train"


# Kept as the straightforward public spelling used by package preflight code.
split_parent = parent_split


def freeze_inventory(inventory: Mapping[str, Any]) -> Admission:
    """Validate and freeze R30 corpus metadata before any consumer opens labels.

    Every listed record is rooted at an ultimate parent.  Duplicates, protected
    provenance, stale pins, or data lacking explicit development reuse authority
    are access refusals, not rows to silently drop or replace.
    """
    value = _copy_json_object(inventory, "inventory")
    if value.get("schema") != INVENTORY_SCHEMA:
        raise AdmissionRefused("unrecognized R30 admission inventory schema")
    salt = value.get("split_salt")
    if not isinstance(salt, str) or not salt:
        raise AdmissionRefused("R30 inventory lacks a declared split_salt")
    _require_corpus_authority(value.get("corpus"))
    parents = value.get("parents")
    if not isinstance(parents, list) or not parents:
        raise AdmissionRefused("R30 inventory requires nonempty parent metadata")

    seen: set[str] = set()
    train: list[str] = []
    validation: list[str] = []
    rescue_validation: list[str] = []
    normalized: list[dict[str, Any]] = []
    for row in parents:
        parent = _require_parent_record(row)
        parent_id = parent["ultimate_parent_id"]
        if parent_id in seen:
            raise AdmissionRefused(f"duplicate ultimate parent: {parent_id}")
        seen.add(parent_id)
        assigned = parent_split(parent_id, salt)
        pinned = parent.get("split")
        if pinned is not None and pinned != assigned:
            raise AdmissionRefused(f"stale split pin for ultimate parent: {parent_id}")
        parent["split"] = assigned
        normalized.append(parent)
        if assigned == "validation":
            validation.append(parent_id)
            if parent["rescue_bearing"]:
                rescue_validation.append(parent_id)
        else:
            train.append(parent_id)

    # The exact frozen, normalized inventory is the only object whose hash can
    # be approved.  Existing saved assignments must match, so a caller cannot
    # substitute a salt or redraw after seeing an allocation.
    frozen_input = {**value, "parents": normalized}
    existing = value.get("frozen_split")
    expected_split = {parent["ultimate_parent_id"]: parent["split"] for parent in normalized}
    if existing is not None and existing != expected_split:
        raise AdmissionRefused("R30 frozen split differs from declared salt allocation")
    frozen_input["frozen_split"] = expected_split
    inventory_hash = sha256_json(frozen_input)
    allocation = {
        "schema": "bl-round30-parent-allocation/v1",
        "inventory_sha256": inventory_hash,
        "split_algorithm": SPLIT_ALGORITHM,
        "split_salt": salt,
        "train_parents": sorted(train),
        "validation_parents": sorted(validation),
    }
    allocation_hash = sha256_json(allocation)
    r30_2_status = _r30_2_archive_status(
        value.get("natural_ordinary_archive"), validation, normalized
    )
    return Admission(
        inventory_sha256=inventory_hash,
        allocation_sha256=allocation_hash,
        salt=salt,
        train_parents=tuple(sorted(train)),
        validation_parents=tuple(sorted(validation)),
        validation_rescue_parents=tuple(sorted(rescue_validation)),
        r30_2_status=r30_2_status,
        inventory=frozen_input,
    )


def require_codex_v_approval(admission: Admission, approval: Mapping[str, Any]) -> None:
    """Require an explicit approval pinned to this inventory and allocation bytes."""
    _verify_frozen_roundtrip(admission)
    value = _copy_json_object(approval, "Codex-V approval")
    if value.get("schema") != APPROVAL_SCHEMA or value.get("approved") is not True:
        raise AdmissionRefused("Codex-V approval is absent or not affirmative")
    if value.get("inventory_sha256") != admission.inventory_sha256:
        raise AdmissionRefused("Codex-V approval inventory hash is stale or foreign")
    if value.get("allocation_sha256") != admission.allocation_sha256:
        raise AdmissionRefused("Codex-V approval allocation hash is stale or foreign")


def validate_frozen_roundtrip(inventory: Mapping[str, Any]) -> Admission:
    """Re-freeze persisted metadata and reject a changed parent/split/hash binding."""
    return freeze_inventory(inventory)


def admission_status(admission: Admission) -> str:
    """Classify R30-0 support before a later package considers a data opener."""
    if len(admission.validation_parents) < 32 or len(admission.validation_rescue_parents) < 16:
        return "INSUFFICIENT"
    return "READY_R30_1" if admission.r30_2_status == "READY_R30_2" else "HOLD_R30_2"


def partition_parents(admission: Admission, partition: str) -> tuple[str, ...]:
    """Expose frozen IDs for a metadata-only preflight, never by rehashing rows."""
    if partition == "train":
        return admission.train_parents
    if partition == "validation":
        return admission.validation_parents
    raise AdmissionRefused("R30 partition must be train or validation")


def admission_binding(admission: Admission) -> dict[str, Any]:
    """Return the two identity pins a control/manifest must carry before an opener."""
    return {
        "inventory_sha256": admission.inventory_sha256,
        "allocation_sha256": admission.allocation_sha256,
        "split_algorithm": SPLIT_ALGORITHM,
        "split_salt": admission.salt,
        "train_parent_count": len(admission.train_parents),
        "validation_parent_count": len(admission.validation_parents),
        "validation_rescue_parent_count": len(admission.validation_rescue_parents),
        "r30_2_status": admission.r30_2_status,
    }


def publish_unit(
    root: Path,
    *,
    namespace: str,
    allocation_sha256: str,
    allocation_index: int,
    content: Mapping[str, Any],
) -> Path:
    """Create one immutable unit payload; its completion marker is published last."""
    _require_namespace(namespace)
    _require_sha(allocation_sha256, "allocation hash")
    if type(allocation_index) is not int or allocation_index < 0:
        raise AdmissionRefused("allocation index must be a nonnegative integer")
    if not isinstance(content, Mapping):
        raise AdmissionRefused("immutable unit content must be an object")
    path = Path(root) / "units" / f"{allocation_index:06d}.json"
    _publish_json(
        path,
        {
            "schema": UNIT_SCHEMA,
            "namespace": namespace,
            "allocation_sha256": allocation_sha256,
            "allocation_index": allocation_index,
            "content": dict(content),
        },
    )
    return path


def empty_ledger(
    namespace: str, allocation_sha256: str, allocation_indices: Iterable[int]
) -> dict[str, Any]:
    """Create an empty first-attempt ledger bound to one frozen allocation."""
    _require_namespace(namespace)
    _require_sha(allocation_sha256, "allocation hash")
    allocation = _normal_allocation(allocation_indices)
    return {
        "schema": LEDGER_SCHEMA,
        "namespace": namespace,
        "allocation_sha256": allocation_sha256,
        "allocation_indices": allocation,
        "units": [],
    }


def build_ledger(
    exports: Path,
    output: Path,
    *,
    namespace: str,
    allocation_sha256: str,
    allocation_indices: Iterable[int],
) -> dict[str, Any]:
    """Publish a ledger that skips only complete, byte-verified allocated units."""
    result = empty_ledger(namespace, allocation_sha256, allocation_indices)
    allowed = set(result["allocation_indices"])
    units = Path(exports) / "units"
    if units.exists():
        for path in sorted(units.glob("*.json")):
            # A partial artifact is intentionally left unlisted; it is eligible
            # in a later fresh attempt, never considered complete.
            if not _all_sidecars(path):
                continue
            wrapper = _verified_unit(path, namespace, allocation_sha256, allowed)
            result["units"].append(_ledger_row(path, wrapper["allocation_index"]))
    if len({row["allocation_index"] for row in result["units"]}) != len(result["units"]):
        raise AdmissionRefused("duplicate allocation index in immutable exports")
    _publish_json(Path(output), result)
    return result


def load_verified_ledger(
    ledger_path: Path,
    *,
    namespace: str,
    allocation_sha256: str,
    allocation_indices: Iterable[int],
    exports: Path | None = None,
) -> dict[int, dict[str, Any]]:
    """Verify ledger, payload size/hash/marker, and return exactly skippable units."""
    expected = empty_ledger(namespace, allocation_sha256, allocation_indices)
    _verify_sidecars(Path(ledger_path))
    value = _read_json(Path(ledger_path), "ledger")
    if not isinstance(value, dict) or value.get("schema") != LEDGER_SCHEMA:
        raise AdmissionRefused("unrecognized R30 verified ledger")
    for key in ("namespace", "allocation_sha256", "allocation_indices"):
        if value.get(key) != expected[key]:
            raise AdmissionRefused("verified ledger namespace or allocation is foreign")
    rows = value.get("units")
    if not isinstance(rows, list):
        raise AdmissionRefused("verified ledger units must be a list")
    root = Path(exports) if exports is not None else Path(ledger_path).parent
    allowed = set(expected["allocation_indices"])
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise AdmissionRefused("verified ledger unit is malformed")
        index, relative, digest, size = (
            row.get("allocation_index"),
            row.get("path"),
            row.get("sha256"),
            row.get("size"),
        )
        if type(index) is not int or index not in allowed or index in result:
            raise AdmissionRefused("verified ledger has foreign or duplicate allocation index")
        if relative != f"units/{index:06d}.json" or type(size) is not int or size <= 0:
            raise AdmissionRefused("verified ledger unit path or size is malformed")
        _require_sha(digest, "ledger unit hash")
        path = root / relative
        actual = _verify_sidecars(path)
        if actual != digest or path.stat().st_size != size:
            raise AdmissionRefused("verified ledger unit bytes differ from metadata")
        wrapper = _verified_unit(path, namespace, allocation_sha256, allowed)
        if wrapper["allocation_index"] != index:
            raise AdmissionRefused("verified ledger allocation index differs from payload")
        result[index] = wrapper["content"]
    return result


def _require_corpus_authority(value: Any) -> None:
    if not isinstance(value, dict):
        raise AdmissionRefused("R30 inventory lacks corpus authority metadata")
    required = ("collection_authority", "source_sha256", "label_sha256", "checkpoint_sha256")
    if any(not isinstance(value.get(key), str) or not value[key] for key in required):
        raise AdmissionRefused("corpus authority and source/label/checkpoint hashes are required")
    for key in required[1:]:
        _require_sha(value[key], key)
    if (
        value.get("explicitly_reusable_development") is not True
        or value.get("already_unsealed") is not True
    ):
        raise AdmissionRefused(
            "corpus is not explicitly reusable already-unsealed development material"
        )
    forbidden = (
        "f4",
        "protected",
        "strict",
        "retention",
        "natural_archive",
        "probe",
        "gameplay",
        "stale",
    )
    if any(type(value.get(key)) is not bool for key in forbidden):
        raise AdmissionRefused("corpus protection status is incomplete")
    if any(value.get(key) is True for key in forbidden):
        raise AdmissionRefused("corpus is protected, foreign, F4-bound, or stale")
    if value.get("pins_current") is not True:
        raise AdmissionRefused("corpus source/label/checkpoint pins are stale or unverified")


def _require_parent_record(row: Any) -> dict[str, Any]:
    value = _copy_json_object(row, "parent record")
    parent_id = value.get("ultimate_parent_id")
    _require_parent(parent_id)
    if type(value.get("previously_inspected")) is not bool:
        raise AdmissionRefused(f"ultimate parent lacks inspection status: {parent_id}")
    if not isinstance(value.get("prior_uses"), list):
        raise AdmissionRefused(f"ultimate parent lacks prior_uses: {parent_id}")
    if type(value.get("rescue_bearing")) is not bool:
        raise AdmissionRefused(f"ultimate parent lacks rescue_bearing metadata: {parent_id}")
    status_keys = ("protected", "foreign", "f4", "stale")
    if any(type(value.get(key)) is not bool for key in status_keys):
        raise AdmissionRefused(f"ultimate parent lacks complete protection status: {parent_id}")
    if any(value.get(key) is True for key in status_keys):
        raise AdmissionRefused(f"protected, foreign, F4, or stale parent: {parent_id}")
    return value


def _verify_frozen_roundtrip(admission: Admission) -> None:
    replayed = freeze_inventory(admission.inventory)
    if (
        replayed.inventory_sha256 != admission.inventory_sha256
        or replayed.allocation_sha256 != admission.allocation_sha256
        or replayed.train_parents != admission.train_parents
        or replayed.validation_parents != admission.validation_parents
    ):
        raise AdmissionRefused(
            "persisted frozen inventory does not round-trip to its approval binding"
        )


def _r30_2_archive_status(
    value: Any, validation: list[str], parent_rows: list[dict[str, Any]]
) -> str:
    if not isinstance(value, dict):
        return "HOLD_R30_2"
    parents = value.get("ultimate_parent_ids")
    probabilities = value.get("sampling_probabilities")
    if not isinstance(parents, list) or not parents or not isinstance(probabilities, dict):
        return "HOLD_R30_2"
    if len(set(parents)) != len(parents) or any(not isinstance(item, str) for item in parents):
        raise AdmissionRefused("ordinary archive parents are malformed or duplicate")
    if not set(parents).issubset(set(validation)):
        raise AdmissionRefused("ordinary archive includes non-validation or foreign parent")
    if set(probabilities) != set(parents):
        return "HOLD_R30_2"
    if any(
        type(probabilities[parent]) not in (int, float) or probabilities[parent] <= 0
        for parent in parents
    ):
        return "HOLD_R30_2"
    if (
        value.get("explicitly_permitted") is not True
        or value.get("includes_non_rescue") is not True
    ):
        return "HOLD_R30_2"
    by_id = {row["ultimate_parent_id"]: row for row in parent_rows}
    if not any(not by_id[parent]["rescue_bearing"] for parent in parents):
        return "HOLD_R30_2"
    return "READY_R30_2"


def _copy_json_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AdmissionRefused(f"{label} must be an object")
    copied = json.loads(canonical_bytes(dict(value)))
    assert isinstance(copied, dict)
    return copied


def _require_sha(value: Any, label: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise AdmissionRefused(f"{label} must be a lowercase sha256")


def _require_parent(value: Any) -> None:
    if not isinstance(value, str) or not _PARENT.fullmatch(value):
        raise AdmissionRefused("ultimate parent id is malformed")


def _require_namespace(value: Any) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"bl-r30-[a-z0-9-]{1,96}", value):
        raise AdmissionRefused("R30 immutable namespace is malformed")


def _normal_allocation(indices: Iterable[int]) -> list[int]:
    result = list(indices)
    if (
        not result
        or any(type(item) is not int or item < 0 for item in result)
        or len(set(result)) != len(result)
    ):
        raise AdmissionRefused("frozen allocation indices must be unique nonnegative integers")
    return sorted(result)


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AdmissionRefused(f"R30 {label} is not valid JSON") from exc


def _all_sidecars(path: Path) -> bool:
    return all(
        item.is_file()
        for item in (path, Path(str(path) + ".sha256"), Path(str(path) + ".complete"))
    )


def _verify_sidecars(path: Path) -> str:
    sidecar, complete = Path(str(path) + ".sha256"), Path(str(path) + ".complete")
    if not _all_sidecars(path) or complete.read_text() != "complete\n":
        raise AdmissionRefused("immutable export lacks payload, hash sidecar, or completion marker")
    match = re.fullmatch(r"([0-9a-f]{64})  ([^/]+)\n", sidecar.read_text())
    if not match or match.group(2) != path.name:
        raise AdmissionRefused("immutable export hash sidecar is malformed")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != match.group(1):
        raise AdmissionRefused("immutable export hash mismatch")
    return digest


def _verified_unit(
    path: Path, namespace: str, allocation_sha256: str, allowed: set[int]
) -> dict[str, Any]:
    value = _read_json(path, "unit")
    if (
        not isinstance(value, dict)
        or value.get("schema") != UNIT_SCHEMA
        or value.get("namespace") != namespace
        or value.get("allocation_sha256") != allocation_sha256
        or type(value.get("allocation_index")) is not int
        or value["allocation_index"] not in allowed
        or not isinstance(value.get("content"), dict)
    ):
        raise AdmissionRefused(
            "immutable unit has foreign namespace/allocation or malformed content"
        )
    return value


def _ledger_row(path: Path, allocation_index: int) -> dict[str, Any]:
    return {
        "allocation_index": allocation_index,
        "path": f"units/{path.name}",
        "sha256": _verify_sidecars(path),
        "size": path.stat().st_size,
    }


def _publish_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_bytes(dict(value))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for candidate in (path, Path(str(path) + ".sha256"), Path(str(path) + ".complete")):
        if candidate.exists():
            raise FileExistsError(f"immutable R30 export already exists: {candidate}")
    digest = hashlib.sha256(payload).hexdigest()
    _atomic_create(path, payload)
    _atomic_create(Path(str(path) + ".sha256"), f"{digest}  {path.name}\n".encode())
    _atomic_create(Path(str(path) + ".complete"), b"complete\n")


def _atomic_create(path: Path, payload: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".r30-", delete=False) as stream:
        temp = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temp, path)
    except FileExistsError:
        raise FileExistsError(f"immutable R30 export already exists: {path}") from None
    finally:
        temp.unlink(missing_ok=True)
