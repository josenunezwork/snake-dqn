"""Immutable, resume-safe timing cells for the MOVE A100 packet."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from research.gpu_sim_spike import a1_prof_units

SCHEMA = "gpu-sim-spike-move-unit/v1"
LEDGER_SCHEMA = "gpu-sim-spike-move-ledger/v1"
UNIT_DIRECTORY = "move-units"
_SHA = re.compile(r"[0-9a-f]{40}")
_UNIT = re.compile(r"cell-(?:scripted|frozen|mixed)-e(?:1|64|256|1024)")


def namespace(commit: str, *, smoke: bool = False, canary: bool = False) -> str:
    if not isinstance(commit, str) or not _SHA.fullmatch(commit):
        raise ValueError("MOVE requires a full lowercase commit SHA")
    if smoke and canary:
        raise ValueError("smoke and canary are mutually exclusive")
    return f"move-{commit}-A100-{'smoke' if smoke else 'canary' if canary else 'full'}"


def allocation(*, smoke: bool = False, canary: bool = False) -> tuple[str, ...]:
    if smoke and canary:
        raise ValueError("smoke and canary are mutually exclusive")
    worlds = (1,) if smoke else ((64,) if canary else (64, 256, 1024))
    return tuple(
        f"cell-{mix}-e{world}" for world in worlds for mix in ("scripted", "frozen", "mixed")
    )


def publish_json(path: Path, value: dict[str, Any]) -> None:
    a1_prof_units._publish_json(Path(path), value)


def publish_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    """Create an immutable payload and checksum, completion marker last."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if any(
        candidate.exists()
        for candidate in (path, Path(str(path) + ".sha256"), Path(str(path) + ".complete"))
    ):
        raise FileExistsError(path)
    digest = hashlib.sha256(payload).hexdigest()
    a1_prof_units._atomic_create(path, payload)
    a1_prof_units._atomic_create(Path(str(path) + ".sha256"), f"{digest}  {path.name}\n".encode())
    a1_prof_units._atomic_create(Path(str(path) + ".complete"), b"complete\n")
    return {"sha256": digest, "size": len(payload)}


def empty_ledger(unit_namespace: str, unit_allocation: tuple[str, ...]) -> dict[str, Any]:
    return {
        "schema": LEDGER_SCHEMA,
        "namespace": unit_namespace,
        "allocation": list(unit_allocation),
        "units": [],
        "fixtures": [],
        "published_last": "units",
    }


def _verify(path: Path) -> str:
    return a1_prof_units._verify_sidecars(Path(path))


def publish_unit(
    root: Path,
    unit_namespace: str,
    unit_allocation: tuple[str, ...],
    unit_id: str,
    content: dict[str, Any],
) -> Path:
    if unit_id not in unit_allocation or not _UNIT.fullmatch(unit_id):
        raise ValueError("unit is outside MOVE frozen allocation")
    path = Path(root) / UNIT_DIRECTORY / f"{unit_id}.json"
    publish_json(
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


def load_ledger(
    path: Path,
    unit_namespace: str,
    unit_allocation: tuple[str, ...],
    *,
    exports: Path | None = None,
) -> dict[str, dict[str, Any]]:
    _verify(path)
    ledger = json.loads(path.read_text())
    if (
        ledger.get("schema") != LEDGER_SCHEMA
        or ledger.get("namespace") != unit_namespace
        or ledger.get("allocation") != list(unit_allocation)
    ):
        raise ValueError("foreign MOVE verified ledger")
    root, found = Path(exports) if exports else path.parent, {}
    if not isinstance(ledger.get("units"), list) or not isinstance(ledger.get("fixtures"), list):
        raise ValueError("malformed MOVE ledger inventories")
    fixture_paths = set()
    for row in ledger["fixtures"]:
        relative = row.get("path", "")
        if (
            not re.fullmatch(r"recorded-states/[a-z0-9-]+\.npz", relative)
            or relative in fixture_paths
        ):
            raise ValueError("foreign or duplicate MOVE fixture")
        payload = root / relative
        if _verify(payload) != row.get("sha256") or payload.stat().st_size != row.get("size"):
            raise ValueError("MOVE fixture bytes differ")
        fixture_paths.add(relative)
    for row in ledger.get("units", []):
        unit_id = row.get("unit_id") if isinstance(row, dict) else None
        payload = root / UNIT_DIRECTORY / f"{unit_id}.json"
        if (
            unit_id in found
            or unit_id not in unit_allocation
            or row.get("path") != f"{UNIT_DIRECTORY}/{unit_id}.json"
        ):
            raise ValueError("foreign or duplicate MOVE unit")
        digest = _verify(payload)
        if digest != row.get("sha256") or payload.stat().st_size != row.get("size"):
            raise ValueError("MOVE ledger payload differs")
        body = json.loads(payload.read_text())
        if (
            body.get("schema") != SCHEMA
            or body.get("namespace") != unit_namespace
            or body.get("allocation") != list(unit_allocation)
            or body.get("unit_id") != unit_id
            or not isinstance(body.get("content"), dict)
        ):
            raise ValueError("foreign MOVE unit payload")
        found[unit_id] = body["content"]
    return found


def build_ledger(
    exports: Path, output: Path, *, commit: str, smoke: bool = False, canary: bool = False
) -> dict[str, Any]:
    ns, cells = namespace(commit, smoke=smoke, canary=canary), allocation(
        smoke=smoke, canary=canary
    )
    rows = []
    for source in (
        sorted((Path(exports) / UNIT_DIRECTORY).glob("*.json"))
        if (Path(exports) / UNIT_DIRECTORY).exists()
        else ()
    ):
        unit_id = source.stem
        if unit_id not in cells:
            raise ValueError("foreign MOVE export unit")
        if (
            not Path(str(source) + ".complete").exists()
            or not Path(str(source) + ".sha256").exists()
        ):
            continue
        digest = _verify(source)
        target = Path(output).parent / UNIT_DIRECTORY / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        for src, dst in (
            (source, target),
            (Path(str(source) + ".sha256"), Path(str(target) + ".sha256")),
            (Path(str(source) + ".complete"), Path(str(target) + ".complete")),
        ):
            if not dst.exists():
                dst.write_bytes(src.read_bytes())
        if _verify(target) != digest:
            raise ValueError("copied MOVE unit differs")
        rows.append(
            {
                "unit_id": unit_id,
                "path": f"{UNIT_DIRECTORY}/{source.name}",
                "sha256": digest,
                "size": source.stat().st_size,
            }
        )
    fixtures = []
    for source in sorted((Path(exports) / "recorded-states").glob("*.npz")):
        if not Path(str(source) + ".complete").exists():
            continue
        digest = _verify(source)
        relative = source.relative_to(exports).as_posix()
        target = Path(output).parent / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("", ".sha256", ".complete"):
            src, dst = Path(str(source) + suffix), Path(str(target) + suffix)
            if src.resolve() != dst.resolve():
                if dst.exists() and dst.read_bytes() != src.read_bytes():
                    raise ValueError("immutable MOVE fixture copy differs")
                if not dst.exists():
                    shutil.copyfile(src, dst)
        fixtures.append({"path": relative, "size": source.stat().st_size, "sha256": digest})
    result = empty_ledger(ns, cells)
    result["fixtures"] = fixtures
    result["units"] = rows
    publish_json(output, result)
    return result
