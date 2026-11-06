"""Mac tools for multi-attempt Bank-v2 screen/confirmation runs (never dispatch).

  ledger  build the verified ledger (r33-verified-ledger/v1) from any number of prior
          attempt output dirs and/or merged dirs, for ``prepare_screen --ledger``;
  merge   union the verified unit results of several attempt dirs into one --out
          directory for ``analyze_screen`` (copies payload + sidecars, create-only).

Accepted source layouts: ``<attempt>/exports/screen`` (Colab output_dir),
``<attempt>/screen`` or a screen/merged dir itself (``units/`` + ``contract.json``).
Every unit must verify (payload, .sha256, .complete) and match the contract's
allocation identity, index, arm, world, seed and horizon; the same index appearing in
several sources must have the same SHA-256 (else refuse).  Unit outcome fields are
never read beyond these identity checks.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from research.watch_replay.screen_l1_os import ALLOCATION, allocation, identity
from tools.ops.package_spec import (
    LEDGER_SCHEMA,
    PackageRefused,
    publish_json,
    unit_id,
    verify_immutable,
)


def screen_root(source: Path) -> Path:
    """Resolve an attempt/merged directory to the directory holding units/."""
    for candidate in (source / "exports" / "screen", source / "screen", source):
        if (candidate / "units").is_dir() or (candidate / "contract.json").is_file():
            return candidate
    raise PackageRefused(f"no screen units under {source}")


def _check_contract(contract: dict) -> None:
    frozen = allocation(contract["namespace"], contract["count"], contract["horizon"])
    if {k: v for k, v in contract.items() if k != "commit"} != frozen:
        raise PackageRefused("contract differs from the frozen allocation")


def verified_units(contract: dict, sources: list[Path]) -> dict[int, dict[str, Any]]:
    """index -> {sha256, size, path} over all sources; refuse conflicting duplicates."""
    _check_contract(contract)
    by_index = {u["index"]: u for u in contract["units"]}
    found: dict[int, dict[str, Any]] = {}
    for source in sources:
        root = screen_root(source)
        src_contract = root / "contract.json"
        if src_contract.is_file():
            value = json.loads(src_contract.read_text())
            if {k: v for k, v in value.items() if k != "commit"} != {
                k: v for k, v in contract.items() if k != "commit"
            }:
                raise PackageRefused(f"source belongs to a different allocation: {root}")
        for path in sorted((root / "units").glob("[0-9]" * 6 + ".json")):
            try:
                meta = verify_immutable(path)
            except PackageRefused:
                continue  # incomplete publication: not a verified unit
            index = int(path.stem)
            unit = by_index.get(index)
            value = json.loads(path.read_text())
            if (
                unit is None
                or value.get("allocation") != identity(contract)
                or value.get("index") != index
                or value.get("arm") != unit["arm"]
                or value.get("world") != unit["world"]
                or value.get("seed") != unit["seed"]
                or value.get("horizon") != contract["horizon"]
                or value.get("unit_id") != unit_id(contract["namespace"], ALLOCATION, index)
            ):
                raise PackageRefused(f"foreign or mismatched unit result: {path}")
            row = {"sha256": meta["sha256"], "size": meta["size"], "path": str(path)}
            if index in found and found[index]["sha256"] != row["sha256"]:
                raise PackageRefused(
                    f"unit {index} differs across attempts: {found[index]['path']} vs {path}"
                )
            found.setdefault(index, row)
    return found


def build_ledger(contract: dict, sources: list[Path]) -> dict[str, Any]:
    units = verified_units(contract, sources)
    return {
        "schema": LEDGER_SCHEMA,
        "namespace": contract["namespace"],
        "allocation": ALLOCATION,
        "identity": identity(contract),
        "units": [
            {
                "unit_id": unit_id(contract["namespace"], ALLOCATION, index),
                "index": index,
                "sha256": row["sha256"],
                "size": row["size"],
            }
            for index, row in sorted(units.items())
        ],
    }


def merge(contract: dict, sources: list[Path], out: Path) -> dict[str, Any]:
    """Copy all verified units into out/units (create-only; identical re-merge is a no-op)."""
    units = verified_units(contract, sources)
    out.mkdir(parents=True, exist_ok=True)
    contract_path = out / "contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise PackageRefused("merge target belongs to a different contract")
    else:
        publish_json(contract_path, contract)
    copied, present = [], []
    for index, row in sorted(units.items()):
        target = out / "units" / f"{index:06d}.json"
        if target.exists():
            if verify_immutable(target)["sha256"] != row["sha256"]:
                raise PackageRefused(f"merge target unit {index} differs from source")
            present.append(index)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("", ".sha256"):
            shutil.copyfile(f"{row['path']}{suffix}", f"{target}{suffix}")
        Path(f"{target}.complete").write_bytes(b"complete\n")  # completion marker last
        verify_immutable(target)
        copied.append(index)
    missing = sorted(set(range(len(contract["units"]))) - set(units))
    return {"copied": copied, "already_present": present, "missing": missing}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("ledger", "merge"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--contract", type=Path, required=True)
        cmd.add_argument("--sources", type=Path, nargs="+", required=True)
        cmd.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    if args.cmd == "ledger":
        ledger = build_ledger(contract, args.sources)
        publish_json(args.out, ledger)
        print(json.dumps({"verified_units": len(ledger["units"]), "ledger": str(args.out)}))
    else:
        report = merge(contract, args.sources, args.out)
        print(json.dumps({k: len(v) for k, v in report.items()} | {"missing": report["missing"]}))


if __name__ == "__main__":
    main()
