"""Immutable, resume-safe units for the prospective MOVE2 A100 packet."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from research.gpu_sim_spike import a1_prof_units

SCHEMA = "gpu-sim-spike-move2-unit/v1"
LEDGER_SCHEMA = "gpu-sim-spike-move2-ledger/v1"
UNIT_DIRECTORY = "move2-units"
_SHA = re.compile(r"[0-9a-f]{40}")
_UNIT = re.compile(r"cell-(?:scripted|frozen|mixed)-e(?:1|64|256|1024)")


def namespace(commit: str, *, smoke: bool = False, canary: bool = False) -> str:
    if not isinstance(commit, str) or not _SHA.fullmatch(commit):
        raise ValueError("MOVE2 requires a full lowercase commit SHA")
    if smoke and canary:
        raise ValueError("smoke and canary are mutually exclusive")
    return (
        f"gpu-move-a100-20261023-02-{commit}-{'smoke' if smoke else 'canary' if canary else 'main'}"
    )


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


def _verify(path: Path) -> str:
    return a1_prof_units._verify_sidecars(Path(path))


def empty_ledger(unit_namespace: str, unit_allocation: tuple[str, ...]) -> dict[str, Any]:
    return {
        "schema": LEDGER_SCHEMA,
        "namespace": unit_namespace,
        "allocation": list(unit_allocation),
        "units": [],
        "fixtures": [],
        "published_last": "units",
    }


def publish_unit(
    root: Path,
    unit_namespace: str,
    unit_allocation: tuple[str, ...],
    unit_id: str,
    content: dict[str, Any],
) -> Path:
    if unit_id not in unit_allocation or not _UNIT.fullmatch(unit_id):
        raise ValueError("unit is outside MOVE2 frozen allocation")
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
        raise ValueError("foreign MOVE2 verified ledger")
    if not isinstance(ledger.get("units"), list) or not isinstance(ledger.get("fixtures"), list):
        raise ValueError("malformed MOVE2 ledger inventories")
    root, found = (Path(exports) if exports else path.parent), {}
    # MOVE2 exports compact deterministic descriptors within each unit.  Full
    # recorded-state traces are deliberately outside the slow Colab result path.
    if ledger["fixtures"]:
        raise ValueError("MOVE2 does not admit binary fixture exports")
    for row in ledger["units"]:
        unit_id = row.get("unit_id") if isinstance(row, dict) else None
        payload = root / UNIT_DIRECTORY / f"{unit_id}.json"
        if (
            unit_id in found
            or unit_id not in unit_allocation
            or row.get("path") != f"{UNIT_DIRECTORY}/{unit_id}.json"
        ):
            raise ValueError("foreign or duplicate MOVE2 unit")
        if _verify(payload) != row.get("sha256") or payload.stat().st_size != row.get("size"):
            raise ValueError("MOVE2 ledger payload differs")
        body = json.loads(payload.read_text())
        if (
            body.get("schema") != SCHEMA
            or body.get("namespace") != unit_namespace
            or body.get("allocation") != list(unit_allocation)
            or body.get("unit_id") != unit_id
            or not isinstance(body.get("content"), dict)
        ):
            raise ValueError("foreign MOVE2 unit payload")
        found[unit_id] = body["content"]
    return found


def build_ledger(
    exports: Path, output: Path, *, commit: str, smoke: bool = False, canary: bool = False
) -> dict[str, Any]:
    ns, cells = namespace(commit, smoke=smoke, canary=canary), allocation(
        smoke=smoke, canary=canary
    )
    result = empty_ledger(ns, cells)
    if (Path(exports) / "recorded-states").exists():
        raise ValueError("MOVE2 refuses recorded-state trace exports")
    for directory, key in ((UNIT_DIRECTORY, "units"),):
        for source in sorted((Path(exports) / directory).glob("*.json")):
            if (
                not Path(str(source) + ".complete").exists()
                or not Path(str(source) + ".sha256").exists()
            ):
                continue
            digest, relative = _verify(source), source.relative_to(exports).as_posix()
            target = output.parent / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            for suffix in ("", ".sha256", ".complete"):
                src, dst = Path(str(source) + suffix), Path(str(target) + suffix)
                if not dst.exists():
                    shutil.copyfile(src, dst)
                elif dst.read_bytes() != src.read_bytes():
                    raise ValueError("immutable MOVE2 export copy differs")
            if source.stem not in cells:
                raise ValueError("foreign MOVE2 export unit")
            body = json.loads(source.read_text())
            if (
                body.get("schema") != SCHEMA
                or body.get("namespace") != ns
                or body.get("allocation") != list(cells)
                or body.get("unit_id") != source.stem
                or not isinstance(body.get("content"), dict)
            ):
                raise ValueError("foreign MOVE2 unit payload")
            result[key].append(
                {
                    "unit_id": source.stem,
                    "path": relative,
                    "sha256": digest,
                    "size": source.stat().st_size,
                }
            )
    publish_json(output, result)
    return result
