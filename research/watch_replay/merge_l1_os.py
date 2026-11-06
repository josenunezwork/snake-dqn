"""Merge complete L1-OS packet outputs into a fresh, renumbered install directory.

No serving setting is changed. Payload bytes and original packet/unit identities
are preserved; destination-bound sidecars are regenerated. --fixture explicitly
permits short smoke inputs and does not produce a product-qualified bank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

from research.watch_replay.run import allocation, identity
from research.watch_replay.seed_audit import (
    L1_OS_BANK_COUNT,
    L1_OS_BANK_NAMESPACES,
    L1_OS_SMOKE_NAMESPACE,
)
from tools.ops.package_spec import publish_immutable, publish_json, unit_id, verify_immutable
from web.backend.watch_replay import Recording

CONTROLLER = "exact-stop-l1-os"


def merge(roots: list[Path], bank: Path, *, fixture: bool = False) -> dict:
    """Require a complete, conflict-free union before publishing any destination files.

    Roots are explicit sessions directories, including previous completion attempts
    if needed. Duplicate units are refused even when their payloads are identical.
    Memory use is bounded to one compressed session during validation and copying.
    """
    if bank.exists() or bank.is_symlink():
        raise FileExistsError(f"destination bank already exists: {bank}")
    if not roots:
        raise ValueError("at least one sessions directory is required")
    contracts, verified, seen_seeds = {}, {}, set()
    for root in roots:
        if not root.is_dir() or root.is_symlink():
            raise ValueError(f"source must be a sessions directory: {root}")
        binding = root / "contract.json"
        verify_immutable(binding)
        contract = json.loads(binding.read_text())
        namespace = contract["namespace"]
        count, horizon = contract["count"], contract["horizon"]
        if (
            type(count) is not int
            or type(horizon) is not int
            or not re.fullmatch(r"[0-9a-f]{40}", contract.get("commit", ""))
            or contract
            != {
                **allocation(namespace, count, CONTROLLER, horizon),
                "commit": contract["commit"],
            }
        ):
            raise ValueError("malformed L1-OS packet contract")
        if fixture:
            if namespace not in (*L1_OS_BANK_NAMESPACES, L1_OS_SMOKE_NAMESPACE) or horizon > 300:
                raise ValueError("fixture merge requires a known short smoke/packet allocation")
        elif namespace not in L1_OS_BANK_NAMESPACES or count != L1_OS_BANK_COUNT or horizon != 6000:
            raise ValueError("product merge requires the three 44-session H6000 packets")
        if namespace in contracts and contracts[namespace] != contract:
            raise ValueError("conflicting packet contracts")
        contracts[namespace] = contract
        for path in sorted(root.glob("*.watch.gz")):
            match = re.fullmatch(r"(\d{6})\.watch\.gz", path.name)
            if not match or not 0 <= int(match[1]) < count:
                raise ValueError(f"foreign allocation filename: {path}")
            index = int(match[1])
            key = (namespace, index)
            if key in verified:
                raise ValueError(f"duplicate/conflicting completed unit: {namespace}/{index}")
            item = verify_immutable(path)
            record = Recording(path, product=not fixture)
            try:
                header = record.header
                if (
                    header.get("allocation") != identity(contract)
                    or header.get("unit_id") != unit_id(namespace, "sessions", index)
                    or header.get("index") != index
                    or header.get("seed") != contract["rows"][index]["seed"]
                    or header.get("source_commit") != contract["commit"]
                    or header.get("controller") != CONTROLLER
                    or header.get("horizon") != horizon
                    or header.get("fixture") is not fixture
                    or record.sha256 != item["sha256"]
                ):
                    raise ValueError(f"foreign recorded allocation: {path}")
                if header["seed"] in seen_seeds:
                    raise ValueError(f"duplicate seed: {header['seed']}")
                seen_seeds.add(header["seed"])
                verified[key] = {
                    "source": str(path),
                    "namespace": namespace,
                    "source_index": index,
                    "unit_id": header["unit_id"],
                    "seed": header["seed"],
                    "sha256": item["sha256"],
                    "size": item["size"],
                }
            finally:
                record.close()
    if not fixture and set(contracts) != set(L1_OS_BANK_NAMESPACES):
        raise ValueError("product merge requires all three packet namespaces")
    if len({c["commit"] for c in contracts.values()}) != 1:
        raise ValueError("conflicting packet source commits")
    expected = {(ns, i) for ns, c in contracts.items() for i in range(c["count"])}
    if set(verified) != expected:
        raise ValueError(f"incomplete packet union: {len(expected - set(verified))} missing units")

    bank.mkdir(parents=True, exist_ok=False)
    units = []
    try:
        for index, key in enumerate(sorted(verified)):
            item = verified[key]
            path = Path(item["source"])
            if verify_immutable(path)["sha256"] != item["sha256"]:
                raise ValueError(f"source changed during merge: {path}")
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != item["sha256"]:
                raise ValueError(f"source changed during read: {path}")
            target = bank / f"{index:06d}.watch.gz"
            publish_immutable(target, payload)
            verify_immutable(target)
            units.append({**item, "index": index, "file": target.name})
        report = {
            "controller": CONTROLLER,
            "fixture": fixture,
            "source_commit": next(iter(contracts.values()))["commit"],
            "bank": str(bank.resolve()),
            "packet_counts": {ns: c["count"] for ns, c in sorted(contracts.items())},
            "copied_count": len(units),
            "units": units,
        }
        publish_json(bank / "merge.json", report)
    except BaseException:
        shutil.rmtree(bank)
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True, help="Sessions directories")
    parser.add_argument("--bank", type=Path, required=True, help="Fresh destination directory")
    parser.add_argument(
        "--fixture", action="store_true", help="Allow short local smoke inputs only"
    )
    args = parser.parse_args()
    report = merge(args.roots, args.bank, fixture=args.fixture)
    print(json.dumps({k: v for k, v in report.items() if k != "units"}, sort_keys=True))


if __name__ == "__main__":
    main()
