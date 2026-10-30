"""Verify a complete allocation on the Mac, then publish its local playback bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.watch_replay.resume import build
from research.watch_replay.run import identity
from tools.ops.package_spec import publish_immutable, unit_id
from web.backend.watch_replay import Recording


def install(contract: dict, roots: list[Path], bank: Path) -> int:
    """No fixture, missing, duplicate, foreign or corrupt session enters a bank."""
    ledger = build(contract, roots)
    if {row["index"] for row in ledger["units"]} != set(range(contract["count"])):
        raise ValueError("complete frozen session union required")
    sources = []
    for root in roots:
        for path in sorted(root.glob("*.watch.gz")):
            record = Recording(path)
            index = int(path.name.split(".")[0])
            if (
                record.header.get("allocation") != identity(contract)
                or record.header.get("unit_id") != unit_id(contract["namespace"], "sessions", index)
                or record.header.get("source_commit") != contract["commit"]
                or record.header.get("controller") != contract["controller"]
                or record.header.get("seed") != contract["rows"][index]["seed"]
            ):
                raise ValueError("foreign recorded allocation")
            sources.append(path)
    bank.mkdir(parents=True, exist_ok=False)
    for path in sources:
        publish_immutable(bank / path.name, path.read_bytes())
    return len(sources)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--bank", type=Path, required=True)
    args = parser.parse_args()
    print(install(json.loads(args.contract.read_text()), args.roots, args.bank))


if __name__ == "__main__":
    main()
