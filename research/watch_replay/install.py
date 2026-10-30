"""Install the verified subset and retain operational exclusions on the Mac."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from research.watch_replay.run import identity
from tools.ops.package_spec import publish_immutable, publish_json, unit_id, verify_immutable
from web.backend.watch_replay import Recording


def install(contract: dict, roots: list[Path], bank: Path) -> int:
    """Exclude individual failures; refuse foreign contracts and duplicate unit IDs."""
    bank.mkdir(parents=True, exist_ok=False)
    installed, exclusions, seen = [], [], set()
    try:
        # Refuse foreign attempt bindings before installing any payload.
        for root in roots:
            binding = root / "contract.json"
            try:
                verify_immutable(binding)
                if json.loads(binding.read_text()) != contract:
                    raise ValueError("foreign namespace/source/allocation")
            except Exception as exc:
                exclusions.append({"path": str(binding), "error": str(exc)})
                raise
        for root in roots:
            for path in sorted(root.glob("*.watch.gz")):
                try:
                    stem = path.name.removesuffix(".watch.gz")
                    if len(stem) != 6 or not stem.isdigit():
                        raise ValueError("invalid allocation filename")
                    index = int(stem)
                    if not 0 <= index < contract["count"]:
                        raise ValueError("out-of-range allocation filename")
                    if index in seen:
                        raise ValueError("duplicate allocation index")
                    record = Recording(path)
                    if (
                        record.header.get("allocation") != identity(contract)
                        or record.header.get("unit_id")
                        != unit_id(contract["namespace"], "sessions", index)
                        or record.header.get("source_commit") != contract["commit"]
                        or record.header.get("controller") != contract["controller"]
                        or record.header.get("seed") != contract["rows"][index]["seed"]
                    ):
                        raise ValueError("foreign recorded allocation")
                    # Recheck copied bytes against the exact verified payload.
                    payload = path.read_bytes()
                    if hashlib.sha256(payload).hexdigest() != record.sha256:
                        raise ValueError("replay changed during install")
                    publish_immutable(bank / path.name, payload)
                    seen.add(index)
                    installed.append(
                        {"index": index, "sha256": record.sha256, "size": len(payload)}
                    )
                except Exception as exc:
                    exclusions.append({"path": str(path), "error": str(exc)})
    finally:
        publish_json(
            bank / "operational.json",
            {
                "allocation": identity(contract),
                "total": contract["count"],
                "completed": len(installed),
                "units": installed,
                "missing_indices": sorted(set(range(contract["count"])) - seen),
                "exclusions": exclusions,
            },
        )
    return len(installed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--bank", type=Path, required=True)
    args = parser.parse_args()
    print(install(json.loads(args.contract.read_text()), args.roots, args.bank))


if __name__ == "__main__":
    main()
