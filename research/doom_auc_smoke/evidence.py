"""Create-only units, receipt-bound outcome-blind ledgers and full-state resume."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import torch

NAMESPACE = "t34-doom-auc-20261030/v1"
ARMS = ("spatial", "b8")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic(path: Path, content: bytes) -> None:
    """Hard-link publication refuses overwrites, including concurrent writers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".partial-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def json_once(path: Path, value: Any) -> None:
    atomic(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def empty_ledger(identity: str) -> dict[str, Any]:
    return dict(namespace=NAMESPACE, allocation_index=0, identity=identity, units=[])


def unit_path(unit: str) -> str:
    if not re.fullmatch(r"(spatial|b8)-(fit-\d{4}|prediction)", unit):
        raise ValueError("foreign unit ID")
    return f"units/a0000-{unit}.pt"


def validate_ledger(ledger: dict[str, Any], identity: str) -> None:
    expected = empty_ledger(identity)
    if any(ledger.get(k) != expected[k] for k in ("namespace", "allocation_index", "identity")):
        raise ValueError("foreign namespace/allocation/identity")
    if type(ledger["allocation_index"]) is not int or not isinstance(ledger.get("units"), list):
        raise ValueError("malformed ledger")
    seen = set()
    for row in ledger["units"]:
        unit = row["unit"]
        if unit in seen or row["path"] != unit_path(unit):
            raise ValueError("duplicate or foreign unit")
        seen.add(unit)
        if (
            not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
            or type(row["size"]) is not int
            or not 0 < row["size"] < 32 * 1024**2
        ):
            raise ValueError("invalid unit hash/size")
        if "-fit-" in unit and (int(unit[-4:]) < 1 or int(unit[-4:]) > 3000):
            raise ValueError("ineligible trajectory snapshot")


def publish(root: Path, unit: str, identity: str, payload: dict[str, Any]) -> dict[str, Any]:
    path = root / unit_path(unit)
    buffer = io.BytesIO()
    torch.save(
        dict(namespace=NAMESPACE, allocation_index=0, identity=identity, unit=unit, **payload),
        buffer,
    )
    if buffer.tell() >= 32 * 1024**2:
        raise ValueError("unit exceeds frozen 32MiB bound")
    atomic(path, buffer.getvalue())
    descriptor = dict(
        unit=unit, path=unit_path(unit), size=path.stat().st_size, sha256=digest(path)
    )
    atomic(Path(str(path) + ".sha256"), (descriptor["sha256"] + "\n").encode())
    json_once(
        Path(str(path) + ".complete"),
        dict(namespace=NAMESPACE, allocation_index=0, identity=identity, **descriptor),
    )
    return descriptor


def verified(root: Path, row: dict[str, Any], identity: str) -> Path:
    validate_ledger(dict(empty_ledger(identity), units=[row]), identity)
    path = root / row["path"]
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        # macOS /tmp resolves through /private; callers pass resolved roots.
        raise ValueError("symlink evidence path")
    marker_path, sidecar = Path(str(path) + ".complete"), Path(str(path) + ".sha256")
    if (
        marker_path.is_symlink()
        or sidecar.is_symlink()
        or marker_path.stat().st_size > 4096
        or sidecar.stat().st_size > 1024
    ):
        raise ValueError("unsafe unit publication sidecars")
    marker = json.loads(marker_path.read_text())
    if marker != dict(namespace=NAMESPACE, allocation_index=0, identity=identity, **row):
        raise ValueError("foreign completion marker")
    if (
        path.stat().st_size != row["size"]
        or digest(path) != row["sha256"]
        or Path(str(path) + ".sha256").read_text().strip() != row["sha256"]
    ):
        raise ValueError("unit bytes differ")
    return path


def load_unit(root: Path, row: dict[str, Any], identity: str) -> dict[str, Any]:
    result = torch.load(verified(root, row, identity), map_location="cpu", weights_only=True)
    for key, expected in dict(
        namespace=NAMESPACE, allocation_index=0, identity=identity, unit=row["unit"]
    ).items():
        if result.get(key) != expected:
            raise ValueError("foreign unit payload")
    return result


def build_ledger(outputs: list[Path], identity: str) -> dict[str, Any]:
    """No torch.load: receipt/marker/path/size/hash checks only, including stop proof."""
    result = empty_ledger(identity)
    seen = set()
    for output in outputs:
        receipt = json.loads((output / "receipt.json").read_text())
        if receipt.get("stop_verified") is not True:
            raise ValueError("verified stop required")
        exports = {r["path"]: r for r in receipt["exports"]}
        if len(exports) != len(receipt["exports"]):
            raise ValueError("duplicate receipt export paths")
        root = output / "exports/doom"
        for marker in sorted(root.glob("units/*.complete")):
            meta = json.loads(marker.read_text())
            row = {k: meta[k] for k in ("unit", "path", "size", "sha256")}
            if row["unit"] in seen:
                raise ValueError("duplicate unit across attempts")
            seen.add(row["unit"])
            path = verified(root, row, identity)
            for file in (path, Path(str(path) + ".sha256"), marker):
                rel = "doom/" + str(file.relative_to(root))
                bound = exports.get(rel, {})
                if bound.get("sha256") != digest(file) or bound.get("size") != file.stat().st_size:
                    raise ValueError("unit/sidecars not supervisor-receipt-bound")
            result["units"].append(row)
    validate_ledger(result, identity)
    return result


def latest(ledger: dict[str, Any], arm: str) -> dict[str, Any] | None:
    rows = [r for r in ledger["units"] if r["unit"].startswith(arm + "-fit-")]
    return max(rows, key=lambda r: r["unit"]) if rows else None
