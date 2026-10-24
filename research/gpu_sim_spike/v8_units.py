"""Immutable, outcome-blind V8 A/B benchmark cells and recovery ledgers."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from research.gpu_sim_spike import a1_prof_units

SCHEMA = "gpu-sim-spike-v8-units/v1"
LEDGER_SCHEMA = "gpu-sim-spike-v8-verified-ledger/v1"
UNIT_DIRECTORY = "v8-units"
_SHA = re.compile(r"[0-9a-f]{40}")
_UNIT = re.compile(r"cell-(?:scripted|frozen|mixed)-e(?:1|64|256|1024)")


def namespace(commit: str, *, smoke: bool = False, canary: bool = False) -> str:
    if not _SHA.fullmatch(commit) or smoke and canary:
        raise ValueError("V8 namespace requires a full SHA and one mode")
    return f"v8-indexed-{commit}-A100-{'smoke' if smoke else 'canary' if canary else 'full'}"


def allocation(*, smoke: bool = False, canary: bool = False) -> tuple[str, ...]:
    if smoke and canary:
        raise ValueError("smoke and canary are exclusive")
    worlds = (1,) if smoke else (64, 256, 1024)
    return tuple(
        f"cell-{mix}-e{world}" for mix in ("scripted", "frozen", "mixed") for world in worlds
    )


def empty_ledger(ns: str, cells: tuple[str, ...]) -> dict[str, Any]:
    _contract(ns, cells)
    return {
        "schema": LEDGER_SCHEMA,
        "namespace": ns,
        "allocation": list(cells),
        "units": [],
        "published_last": "units",
    }


def payload_path(root: Path, unit: str) -> Path:
    _unit(unit)
    return Path(root) / UNIT_DIRECTORY / f"{unit}.json"


def publish_json(path: Path, value: dict[str, Any]) -> None:
    """Use the established atomic payload/sidecar/marker-last primitive."""
    a1_prof_units._publish_json(Path(path), value)


def publish_unit(
    root: Path, ns: str, cells: tuple[str, ...], unit: str, content: dict[str, Any]
) -> None:
    _contract(ns, cells)
    if unit not in cells:
        raise ValueError("V8 unit is outside allocation")
    _content(content, Path(root))
    publish_json(
        payload_path(root, unit),
        {
            "schema": SCHEMA,
            "namespace": ns,
            "allocation": list(cells),
            "unit_id": unit,
            "content": content,
        },
    )


def _verify(path: Path) -> str:
    if not path.is_file() or path.with_name(path.name + ".complete").read_text() != "complete\n":
        raise ValueError("V8 export lacks completion marker")
    sidecar = path.with_name(path.name + ".sha256")
    match = re.fullmatch(
        r"([0-9a-f]{64})  ([^/]+)\n", sidecar.read_text() if sidecar.is_file() else ""
    )
    if not match or match.group(2) != path.name:
        raise ValueError("V8 export checksum sidecar malformed")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != match.group(1):
        raise ValueError("V8 export checksum mismatch")
    return actual


def load_ledger(
    path: Path, ns: str, cells: tuple[str, ...], *, exports: Path | None = None
) -> dict[str, dict[str, Any]]:
    _contract(ns, cells)
    _verify(path)
    value = json.loads(path.read_text())
    if (
        value.get("schema") != LEDGER_SCHEMA
        or value.get("namespace") != ns
        or value.get("allocation") != list(cells)
    ):
        raise ValueError("V8 ledger namespace or allocation is foreign")
    result: dict[str, dict[str, Any]] = {}
    root = exports or path.parent
    for row in value.get("units", []):
        unit = row.get("unit_id")
        _unit(unit)
        if (
            unit not in cells
            or unit in result
            or row.get("path") != f"{UNIT_DIRECTORY}/{unit}.json"
        ):
            raise ValueError("V8 ledger has foreign or duplicate unit")
        payload = root / row["path"]
        if _verify(payload) != row.get("sha256") or payload.stat().st_size != row.get("size"):
            raise ValueError("V8 ledger payload differs")
        wrapper = json.loads(payload.read_text())
        _wrapper(wrapper, ns, cells, unit, root)
        result[unit] = wrapper["content"]
    return result


def build_ledger(
    exports: Path, output: Path, *, commit: str, smoke: bool = False, canary: bool = False
) -> dict[str, Any]:
    ns, cells = namespace(commit, smoke=smoke, canary=canary), allocation(
        smoke=smoke, canary=canary
    )
    rows = []
    directory = Path(exports) / UNIT_DIRECTORY
    if directory.exists():
        for payload in sorted(directory.glob("*.json")):
            unit = payload.stem
            _unit(unit)
            if unit not in cells:
                raise ValueError("refusing foreign V8 export unit")
            trio = [
                payload,
                payload.with_name(payload.name + ".sha256"),
                payload.with_name(payload.name + ".complete"),
            ]
            if not all(item.is_file() for item in trio):
                continue
            digest = _verify(payload)
            wrapper = json.loads(payload.read_text())
            _wrapper(wrapper, ns, cells, unit, Path(exports))
            target = output.parent / UNIT_DIRECTORY / payload.name
            target.parent.mkdir(parents=True, exist_ok=True)
            for source, destination in zip(
                trio,
                [
                    target,
                    target.with_name(target.name + ".sha256"),
                    target.with_name(target.name + ".complete"),
                ],
            ):
                if destination.exists():
                    raise FileExistsError(destination)
                shutil.copyfile(source, destination)
            fixture = _fixture(wrapper["content"], Path(exports))
            if fixture is not None:
                relative = Path(fixture["path"])
                for source, destination in zip(
                    _fixture_trio(Path(exports) / relative), _fixture_trio(output.parent / relative)
                ):
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists():
                        raise FileExistsError(destination)
                    shutil.copyfile(source, destination)
            rows.append(
                {
                    "unit_id": unit,
                    "path": f"{UNIT_DIRECTORY}/{unit}.json",
                    "sha256": digest,
                    "size": payload.stat().st_size,
                }
            )
    if len({r["unit_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate V8 unit")
    result = empty_ledger(ns, cells)
    result["units"] = rows
    publish_json(output, result)
    return result


def _unit(value: Any) -> None:
    if not isinstance(value, str) or not _UNIT.fullmatch(value):
        raise ValueError("V8 unit id malformed")


def _contract(ns: str, cells: tuple[str, ...]) -> None:
    if (
        not isinstance(ns, str)
        or not ns.startswith("v8-indexed-")
        or not cells
        or len(set(cells)) != len(cells)
    ):
        raise ValueError("V8 contract malformed")
    for cell in cells:
        _unit(cell)


def _wrapper(value: Any, ns: str, cells: tuple[str, ...], unit: str, root: Path) -> None:
    if (
        not isinstance(value, dict)
        or value.get("schema") != SCHEMA
        or value.get("namespace") != ns
        or value.get("unit_id") != unit
        or value.get("allocation") != list(cells)
    ):
        raise ValueError("V8 payload is foreign or malformed")
    _content(value.get("content"), root)


def _content(value: Any, root: Path) -> None:
    if not isinstance(value, dict):
        raise ValueError("V8 payload content is malformed")
    _fixture(value, root)


def _fixture(content: dict[str, Any], root: Path) -> dict[str, Any] | None:
    value = content.get("recorded_fixture")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("V8 recorded fixture is malformed")
    relative, digest, size = value.get("path"), value.get("sha256"), value.get("size")
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or not relative.endswith(".npz")
        or not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
        or type(size) is not int
        or size <= 0
    ):
        raise ValueError("V8 recorded fixture binding is malformed")
    payload = Path(root) / relative
    if _verify(payload) != digest or payload.stat().st_size != size:
        raise ValueError("V8 recorded fixture bytes differ")
    return value


def _fixture_trio(payload: Path) -> tuple[Path, Path, Path]:
    return (
        payload,
        payload.with_name(payload.name + ".sha256"),
        payload.with_name(payload.name + ".complete"),
    )
