"""Outcome-blind, immutable whole-roster T4A units and verified resume ledgers."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

from research.gpu_sim_spike import a1_prof_units

SCHEMA = "gpu-sim-spike-t4a-unit/v1"
LEDGER_SCHEMA = "gpu-sim-spike-t4a-ledger/v1"
UNIT_DIRECTORY = "t4a-units"
MIXES = ("frozen", "mixed", "scripted")
_NS = re.compile(r"t4a-dev-01-[0-9a-f]{40}-A100-(?:smoke|canary|main)")


def mode(*, smoke: bool = False, canary: bool = False) -> str:
    if smoke and canary:
        raise ValueError("smoke and canary are exclusive")
    return "smoke" if smoke else "canary" if canary else "main"


def namespace(commit: str, *, smoke: bool = False, canary: bool = False) -> str:
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("T4A requires a full commit SHA")
    return f"t4a-dev-01-{commit}-A100-{mode(smoke=smoke, canary=canary)}"


def allocation(*, smoke: bool = False, canary: bool = False) -> tuple[str, ...]:
    mode(smoke=smoke, canary=canary)
    return tuple(f"roster-{mix}" for mix in MIXES)


def parse(unit: str) -> str:
    if unit not in allocation():
        raise ValueError("foreign T4A roster unit")
    return unit.removeprefix("roster-")


def _validate(ns: str, cells: tuple[str, ...]) -> None:
    if not isinstance(ns, str) or not _NS.fullmatch(ns) or cells != allocation():
        raise ValueError("foreign T4A namespace or allocation")


def publish_json(path: Path, value: dict[str, Any]) -> None:
    a1_prof_units.publish_json(Path(path), value)


def _verify(path: Path) -> str:
    return a1_prof_units._verify_sidecars(Path(path))


def empty_ledger(ns: str, cells: tuple[str, ...]) -> dict[str, Any]:
    _validate(ns, cells)
    return {"schema": LEDGER_SCHEMA, "namespace": ns, "allocation": list(cells), "units": []}


def payload_path(root: Path, unit: str) -> Path:
    parse(unit)
    return Path(root) / UNIT_DIRECTORY / f"{unit}.json"


def publish_unit(
    root: Path, ns: str, cells: tuple[str, ...], unit: str, content: dict[str, Any]
) -> Path:
    _validate(ns, cells)
    path = payload_path(root, unit)
    if not isinstance(content, dict):
        raise ValueError("malformed T4A content")
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


def _identity(body: Any, ns: str, cells: tuple[str, ...], unit: str) -> None:
    if (
        not isinstance(body, dict)
        or body.get("schema") != SCHEMA
        or body.get("namespace") != ns
        or body.get("allocation") != list(cells)
        or body.get("unit_id") != unit
        or not isinstance(body.get("content"), dict)
    ):
        raise ValueError("foreign T4A unit payload")


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
        raise ValueError("foreign T4A ledger")
    root = Path(exports) if exports is not None else path.parent
    found: dict[str, dict[str, Any]] = {}
    for row in ledger["units"]:
        if not isinstance(row, dict):
            raise ValueError("malformed T4A ledger row")
        unit = row.get("unit_id")
        if unit not in cells or unit in found or row.get("path") != f"{UNIT_DIRECTORY}/{unit}.json":
            raise ValueError("duplicate or foreign T4A ledger row")
        payload = payload_path(root, unit)
        if _verify(payload) != row.get("sha256") or payload.stat().st_size != row.get("size"):
            raise ValueError("T4A ledger bytes differ")
        body = json.loads(payload.read_text())
        _identity(body, ns, cells, unit)
        found[unit] = body["content"]
    return found


def build_ledger(
    exports: Path, output: Path, *, commit: str, smoke: bool = False, canary: bool = False
) -> dict[str, Any]:
    """Verify path/size/hash and identity only; never inspect comparison outcomes."""
    ns = namespace(commit, smoke=smoke, canary=canary)
    cells = allocation(smoke=smoke, canary=canary)
    result = empty_ledger(ns, cells)
    directory = Path(exports) / UNIT_DIRECTORY
    for payload in sorted(directory.glob("*.json")):
        unit = payload.stem
        parse(unit)
        if not all(Path(str(payload) + suffix).is_file() for suffix in (".sha256", ".complete")):
            continue
        digest = _verify(payload)
        _identity(json.loads(payload.read_text()), ns, cells, unit)
        target = payload_path(output.parent, unit)
        target.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("", ".sha256", ".complete"):
            src, dst = Path(str(payload) + suffix), Path(str(target) + suffix)
            if dst.exists():
                if dst.read_bytes() != src.read_bytes():
                    raise ValueError("immutable T4A recovery differs")
            else:
                shutil.copyfile(src, dst)
        result["units"].append(
            {
                "unit_id": unit,
                "path": f"{UNIT_DIRECTORY}/{unit}.json",
                "sha256": digest,
                "size": payload.stat().st_size,
            }
        )
    publish_json(output, result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exports", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--canary", action="store_true")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    result = build_ledger(a.exports, a.output, commit=a.commit, smoke=a.smoke, canary=a.canary)
    print(json.dumps({"verified_units": len(result["units"]), "ledger": str(a.output)}))


if __name__ == "__main__":
    main()
