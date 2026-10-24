"""Immutable, outcome-blind R6 E-scaling cell exports and resume ledger.

Same contract as :mod:`r2_units`, with an R6 namespace and the E ladder
(base, 2x base, 4x base) for each roster.  Main: E1024/E2048/E4096.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from research.gpu_sim_spike import a1_prof_units

SCHEMA = "gpu-sim-spike-r6-unit/v1"
LEDGER_SCHEMA = "gpu-sim-spike-r6-ledger/v1"
UNIT_DIRECTORY = "r6-units"
MIXES = ("frozen", "mixed", "scripted")
LADDERS = {"smoke": (1, 2, 4), "canary": (64, 128, 256), "main": (1024, 2048, 4096)}
_SHA = re.compile(r"[0-9a-f]{40}")
_UNIT = re.compile(r"cell-(?:scripted|frozen|mixed)-e(?:1|2|4|64|128|256|1024|2048|4096)")


def mode(*, smoke: bool = False, canary: bool = False) -> str:
    if smoke and canary:
        raise ValueError("smoke and canary are exclusive")
    return "smoke" if smoke else "canary" if canary else "main"


def namespace(commit: str, *, smoke: bool = False, canary: bool = False) -> str:
    if not isinstance(commit, str) or not _SHA.fullmatch(commit):
        raise ValueError("R6 namespace requires a full SHA")
    return f"r6-escaling-{commit}-A100-{mode(smoke=smoke, canary=canary)}"


def ladder(*, smoke: bool = False, canary: bool = False) -> tuple[int, int, int]:
    return LADDERS[mode(smoke=smoke, canary=canary)]


def allocation(*, smoke: bool = False, canary: bool = False) -> tuple[str, ...]:
    """Per roster, base E first, then 2x and 4x (memory grows monotonically)."""
    sizes = ladder(smoke=smoke, canary=canary)
    return tuple(f"cell-{mix}-e{world}" for mix in MIXES for world in sizes)


def parse(unit: str) -> tuple[str, int]:
    if not isinstance(unit, str) or not _UNIT.fullmatch(unit):
        raise ValueError("malformed R6 unit")
    _cell, mix, worlds = unit.split("-")
    return mix, int(worlds[1:])


def publish_json(path: Path, value: dict[str, Any]) -> None:
    a1_prof_units._publish_json(Path(path), value)


def _verify(path: Path) -> str:
    return a1_prof_units._verify_sidecars(Path(path))


def empty_ledger(ns: str, cells: tuple[str, ...]) -> dict[str, Any]:
    _validate(ns, cells)
    return {
        "schema": LEDGER_SCHEMA,
        "namespace": ns,
        "allocation": list(cells),
        "units": [],
        "published_last": "units",
    }


def payload_path(root: Path, unit: str) -> Path:
    parse(unit)
    return Path(root) / UNIT_DIRECTORY / f"{unit}.json"


def publish_unit(
    root: Path, ns: str, cells: tuple[str, ...], unit: str, content: dict[str, Any]
) -> Path:
    _validate(ns, cells)
    if unit not in cells or not isinstance(content, dict):
        raise ValueError("R6 unit is foreign or malformed")
    path = payload_path(root, unit)
    publish_json(
        path,
        {
            "schema": SCHEMA,
            "namespace": ns,
            "allocation": list(cells),
            "unit_id": unit,
            "content": content,
        },
    )
    return path


def load_ledger(
    path: Path, ns: str, cells: tuple[str, ...], *, exports: Path | None = None
) -> dict[str, dict[str, Any]]:
    _validate(ns, cells)
    _verify(path)
    ledger = json.loads(path.read_text())
    if (
        not isinstance(ledger, dict)
        or ledger.get("schema") != LEDGER_SCHEMA
        or ledger.get("namespace") != ns
        or ledger.get("allocation") != list(cells)
        or not isinstance(ledger.get("units"), list)
    ):
        raise ValueError("foreign R6 verified ledger")
    found: dict[str, dict[str, Any]] = {}
    root = Path(exports) if exports else path.parent
    for row in ledger["units"]:
        if not isinstance(row, dict):
            raise ValueError("malformed R6 ledger unit")
        unit = row.get("unit_id")
        if unit in found or unit not in cells or row.get("path") != f"{UNIT_DIRECTORY}/{unit}.json":
            raise ValueError("foreign or duplicate R6 unit")
        payload = root / str(row["path"])
        if _verify(payload) != row.get("sha256") or payload.stat().st_size != row.get("size"):
            raise ValueError("R6 ledger payload differs")
        body = json.loads(payload.read_text())
        if (
            body.get("schema") != SCHEMA
            or body.get("namespace") != ns
            or body.get("allocation") != list(cells)
            or body.get("unit_id") != unit
            or not isinstance(body.get("content"), dict)
        ):
            raise ValueError("foreign R6 unit payload")
        found[unit] = body["content"]
    return found


def build_ledger(
    exports: Path, output: Path, *, commit: str, smoke: bool = False, canary: bool = False
) -> dict[str, Any]:
    ns, cells = namespace(commit, smoke=smoke, canary=canary), allocation(
        smoke=smoke, canary=canary
    )
    result = empty_ledger(ns, cells)
    for payload in sorted((Path(exports) / UNIT_DIRECTORY).glob("*.json")):
        unit = payload.stem
        if unit not in cells:
            raise ValueError("foreign R6 export unit")
        if not all(
            Path(str(payload) + suffix).is_file() for suffix in ("", ".sha256", ".complete")
        ):
            continue
        digest = _verify(payload)
        body = json.loads(payload.read_text())
        if (
            body.get("schema") != SCHEMA
            or body.get("namespace") != ns
            or body.get("allocation") != list(cells)
            or body.get("unit_id") != unit
        ):
            raise ValueError("foreign R6 export payload")
        destination = output.parent / UNIT_DIRECTORY / payload.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("", ".sha256", ".complete"):
            src, dst = Path(str(payload) + suffix), Path(str(destination) + suffix)
            if dst.exists() and dst.read_bytes() != src.read_bytes():
                raise ValueError("immutable R6 recovery differs")
            if not dst.exists():
                shutil.copyfile(src, dst)
        result["units"].append(
            {
                "unit_id": unit,
                "path": f"{UNIT_DIRECTORY}/{payload.name}",
                "sha256": digest,
                "size": payload.stat().st_size,
            }
        )
    publish_json(output, result)
    return result


def _validate(ns: str, cells: tuple[str, ...]) -> None:
    if (
        not isinstance(ns, str)
        or not ns.startswith("r6-escaling-")
        or not isinstance(cells, tuple)
        or not cells
        or len(set(cells)) != len(cells)
        or any(not _UNIT.fullmatch(cell) for cell in cells)
    ):
        raise ValueError("malformed R6 allocation contract")
