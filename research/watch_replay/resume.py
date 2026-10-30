"""Mac-only, outcome-blind union ledger from immutable completed exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.watch_replay.run import identity
from tools.ops.package_spec import empty_ledger, publish_json, unit_id, verify_immutable


def build(contract: dict, roots: list[Path]) -> dict:
    """Read only frozen paths/sizes/digests, never frames or session stats."""
    value = empty_ledger(contract["namespace"], "sessions", identity(contract))
    seen = set()
    for root in roots:
        for path in sorted(root.glob("*.watch.gz")):
            index = int(path.name.split(".")[0])
            if index in seen or not 0 <= index < contract["count"]:
                raise ValueError("duplicate or foreign allocation index")
            # Every exported attempt must carry the exact frozen allocation.
            binding = json.loads((root / "contract.json").read_text())
            verify_immutable(root / "contract.json")
            if binding != contract:
                raise ValueError("foreign namespace/source/allocation")
            item = verify_immutable(path)
            value["units"].append(
                {
                    "index": index,
                    "unit_id": unit_id(contract["namespace"], "sessions", index),
                    "size": item["size"],
                    "sha256": item["sha256"],
                }
            )
            seen.add(index)
    value["units"].sort(key=lambda row: row["index"])
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--roots", required=True, nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    publish_json(args.out, build(json.loads(args.contract.read_text()), args.roots))


if __name__ == "__main__":
    main()
