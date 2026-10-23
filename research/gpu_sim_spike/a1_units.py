"""Immutable, outcome-blind resume units for the A1 full-tick ruler.

The runtime publishes one JSON payload for each independently repeatable A1
cell.  A Mac-side ledger builder deliberately reads only filenames, sizes and
checksum sidecars; it never deserializes measurement content when deciding what
may be resumed.  Completion markers are always written last.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

SCHEMA = "gpu-sim-spike-a1-units/v1"
LEDGER_SCHEMA = "gpu-sim-spike-a1-verified-ledger/v1"
UNIT_DIRECTORY = "a1-units"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_HARDWARE = re.compile(r"[A-Za-z0-9_.-]+")
_UNIT = re.compile(r"(?:cell-(?:scripted|frozen|mixed)-e(?:6|256|1024)|control-e6|parity-tape)")


def namespace(commit: str, hardware: str, smoke: bool) -> str:
    """Return the namespace frozen by source identity, target hardware and mode."""
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        raise ValueError("A1 namespace requires a full lowercase commit SHA")
    if not isinstance(hardware, str) or not _HARDWARE.fullmatch(hardware):
        raise ValueError("A1 hardware must be a safe nonempty identifier")
    return f"a1-ruler-{commit}-{hardware}-{'smoke' if smoke else 'full'}"


def allocation(smoke: bool) -> tuple[str, ...]:
    """Return the frozen unit allocation, ordered independently of outcomes."""
    cells = tuple(
        f"cell-{mix}-e{worlds}"
        for mix in ("scripted", "frozen", "mixed")
        for worlds in (6, 256, 1024)
    )
    # The transport canary intentionally exercises E=1024 for every mix: a
    # tiny-only canary would not detect the allocation failure that stopped A1.
    smoke_cells = tuple(cell for cell in cells if cell.endswith(("-e6", "-e1024")))
    return (smoke_cells if smoke else cells) + ("control-e6", "parity-tape")


def empty_ledger(unit_namespace: str, unit_allocation: tuple[str, ...]) -> dict[str, Any]:
    """Make a first-attempt empty ledger, without inspecting any outcome."""
    _validate_contract(unit_namespace, unit_allocation)
    return {
        "schema": LEDGER_SCHEMA,
        "namespace": unit_namespace,
        "allocation": list(unit_allocation),
        "units": [],
        "published_last": "units",
    }


def payload_path(root: Path, unit_id: str) -> Path:
    """Return the portable immutable payload path for a frozen unit."""
    _validate_unit_id(unit_id)
    return Path(root) / UNIT_DIRECTORY / f"{unit_id}.json"


def publish_unit(
    output_root: Path,
    unit_namespace: str,
    unit_allocation: tuple[str, ...],
    unit_id: str,
    content: dict[str, Any],
) -> Path:
    """Atomically publish an immutable unit; its ``.complete`` marker is last."""
    _validate_contract(unit_namespace, unit_allocation)
    _validate_unit_id(unit_id)
    if unit_id not in unit_allocation:
        raise ValueError("A1 unit is outside the frozen allocation")
    if not isinstance(content, dict):
        raise ValueError("A1 unit content must be an object")
    path = payload_path(output_root, unit_id)
    _publish_json(
        path,
        {
            "schema": SCHEMA,
            "namespace": unit_namespace,
            "allocation": list(unit_allocation),
            "unit_id": unit_id,
            "content": content,
        },
    )
    return path


def publish_json(path: Path, value: dict[str, Any]) -> None:
    """Publish an arbitrary immutable A1 JSON control artifact with sidecars.

    Pack preparation uses this for an empty verified ledger.  It shares the
    same create-only and completion-marker-last contract as cell payloads.
    """
    if not isinstance(value, dict):
        raise ValueError("A1 JSON export must be an object")
    _publish_json(Path(path), value)


def load_ledger(
    path: Path,
    unit_namespace: str,
    unit_allocation: tuple[str, ...],
    *,
    exports: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Validate a verified ledger and its immutable payload bytes before skipping.

    ``exports`` defaults to the ledger's parent, making a copied ledger and its
    ``a1-units/`` directory portable as one input bundle.
    """
    _validate_contract(unit_namespace, unit_allocation)
    _verify_sidecars(path)
    value = _read_json(path, "ledger")
    if not isinstance(value, dict) or value.get("schema") != LEDGER_SCHEMA:
        raise ValueError("unrecognized A1 verified ledger")
    if value.get("namespace") != unit_namespace or value.get("allocation") != list(unit_allocation):
        raise ValueError("A1 ledger namespace or allocation is foreign")
    rows = value.get("units")
    if not isinstance(rows, list):
        raise ValueError("A1 ledger units must be a list")
    root = Path(exports) if exports is not None else Path(path).parent
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("A1 ledger unit is malformed")
        unit_id, relative, digest, size = (
            row.get("unit_id"),
            row.get("path"),
            row.get("sha256"),
            row.get("size"),
        )
        _validate_unit_id(unit_id)
        if unit_id not in unit_allocation:
            raise ValueError("A1 ledger contains a foreign allocation unit")
        if unit_id in result:
            raise ValueError("A1 ledger has duplicate unit ids")
        expected = f"{UNIT_DIRECTORY}/{unit_id}.json"
        if (
            relative != expected
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError("A1 ledger unit path or hash is malformed")
        if type(size) is not int or size <= 0:
            raise ValueError("A1 ledger unit size is malformed")
        payload = root / relative
        actual = _verify_sidecars(payload)
        if payload.stat().st_size != size or actual != digest:
            raise ValueError("A1 ledger payload bytes differ from verified metadata")
        wrapper = _read_json(payload, "unit")
        if (
            not isinstance(wrapper, dict)
            or wrapper.get("schema") != SCHEMA
            or wrapper.get("namespace") != unit_namespace
            or wrapper.get("allocation") != list(unit_allocation)
            or wrapper.get("unit_id") != unit_id
            or not isinstance(wrapper.get("content"), dict)
        ):
            raise ValueError("A1 ledger payload wrapper is foreign or malformed")
        # The runner can pass this direct content to publish_unit when it
        # materializes a skipped unit into a fresh output tree.
        result[unit_id] = wrapper["content"]
    return result


def build_ledger(
    exports: Path, output: Path, *, commit: str, hardware: str, smoke: bool
) -> dict[str, Any]:
    """Build a Mac ledger using only complete payload paths, sizes and hashes."""
    unit_namespace = namespace(commit, hardware, smoke)
    unit_allocation = allocation(smoke)
    root = Path(exports)
    rows: list[dict[str, Any]] = []
    directory = root / UNIT_DIRECTORY
    if directory.exists():
        for payload in sorted(directory.glob("*.json")):
            unit_id = payload.stem
            try:
                _validate_unit_id(unit_id)
            except ValueError as exc:
                raise ValueError("refusing foreign A1 export unit") from exc
            if unit_id not in unit_allocation:
                raise ValueError("refusing foreign A1 export unit")
            if _partial_export(payload):
                # An interrupted unit is deliberately absent from the ledger:
                # its allocation remains eligible in the next fresh attempt.
                continue
            digest = _verify_sidecars(payload)
            _copy_verified_export(payload, Path(output).parent)
            rows.append(
                {
                    "unit_id": unit_id,
                    "path": f"{UNIT_DIRECTORY}/{payload.name}",
                    "sha256": digest,
                    "size": payload.stat().st_size,
                }
            )
    if len({row["unit_id"] for row in rows}) != len(rows):
        raise ValueError("refusing duplicate A1 export unit ids")
    result = empty_ledger(unit_namespace, unit_allocation)
    result["units"] = rows
    _publish_json(Path(output), result)
    return result


def _validate_contract(unit_namespace: str, unit_allocation: tuple[str, ...]) -> None:
    if not isinstance(unit_namespace, str) or not unit_namespace.startswith("a1-ruler-"):
        raise ValueError("A1 namespace is malformed")
    if not isinstance(unit_allocation, tuple) or not unit_allocation:
        raise ValueError("A1 allocation must be a nonempty frozen tuple")
    if len(set(unit_allocation)) != len(unit_allocation):
        raise ValueError("A1 allocation has duplicate ids")
    for unit_id in unit_allocation:
        _validate_unit_id(unit_id)


def _validate_unit_id(unit_id: Any) -> None:
    if not isinstance(unit_id, str) or not _UNIT.fullmatch(unit_id):
        raise ValueError("A1 unit id is malformed")


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"A1 {label} is not valid JSON") from exc


def _verify_sidecars(path: Path) -> str:
    complete = path.with_name(path.name + ".complete")
    sidecar = path.with_name(path.name + ".sha256")
    if (
        not path.is_file()
        or not complete.is_file()
        or complete.read_text() != "complete\n"
        or not sidecar.is_file()
    ):
        raise ValueError("A1 export lacks a complete payload and sidecars")
    match = re.fullmatch(r"([0-9a-f]{64})  ([^/]+)\n", sidecar.read_text())
    if not match or match.group(2) != path.name:
        raise ValueError("A1 export checksum sidecar is malformed")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != match.group(1):
        raise ValueError("A1 export checksum mismatch")
    return digest


def _partial_export(path: Path) -> bool:
    """Whether a payload was interrupted before all three immutable files existed."""
    files = (path, path.with_name(path.name + ".sha256"), path.with_name(path.name + ".complete"))
    return any(item.exists() for item in files) and not all(item.is_file() for item in files)


def _copy_verified_export(payload: Path, destination_root: Path) -> None:
    """Copy byte-verified files beside a recovery ledger without inspecting JSON."""
    target = payload_path(destination_root, payload.stem)
    if target.resolve() == payload.resolve():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    for source, destination in (
        (payload, target),
        (payload.with_name(payload.name + ".sha256"), target.with_name(target.name + ".sha256")),
        (
            payload.with_name(payload.name + ".complete"),
            target.with_name(target.name + ".complete"),
        ),
    ):
        if destination.exists():
            if destination.read_bytes() != source.read_bytes():
                raise FileExistsError(
                    "A1 recovery export destination has different immutable bytes"
                )
            continue
        shutil.copyfile(source, destination)


def _publish_json(path: Path, value: dict[str, Any]) -> None:
    """Create payload and sidecars without exposing a partial complete marker."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for candidate in (
        path,
        path.with_name(path.name + ".sha256"),
        path.with_name(path.name + ".complete"),
    ):
        if candidate.exists():
            raise FileExistsError(f"A1 immutable export already exists: {candidate}")
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    _atomic_create(path, payload)
    _atomic_create(path.with_name(path.name + ".sha256"), f"{digest}  {path.name}\n".encode())
    _atomic_create(path.with_name(path.name + ".complete"), b"complete\n")


def _atomic_create(path: Path, payload: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".a1-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"A1 immutable export already exists: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    value = build_ledger(
        args.exports, args.output, commit=args.commit, hardware=args.hardware, smoke=args.smoke
    )
    print(
        json.dumps(
            {"ledger": str(args.output), "verified_units": len(value["units"])}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
