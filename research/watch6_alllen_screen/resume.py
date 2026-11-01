"""Mac-only outcome-blind union ledger builder from immutable completed unit exports."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from research.watch6_alllen_screen import protocol as p
from research.watch6_alllen_screen.run import validate_contract
from tools.ops.package_spec import empty_ledger, publish_json, unit_id, verify_immutable

COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def _allocation_without_commit(contract: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in contract.items() if k != "commit"}


def build(
    contract: dict[str, Any],
    roots: list[Path],
    *,
    allowed_commits: set[str] | None = None,
) -> dict[str, Any]:
    """Build a verified resume ledger using only path, size, SHA256, and contract binding."""
    validate_contract(contract)
    commits = set(allowed_commits) if allowed_commits is not None else {contract["commit"]}
    commits.add(contract["commit"])
    if any(not isinstance(c, str) or COMMIT_RE.fullmatch(c) is None for c in commits):
        raise ValueError("allowed_commits must be 40-char lowercase hex SHAs")

    expected_alloc = _allocation_without_commit(contract)
    ledger = empty_ledger(contract["namespace"], p.UNIT_KIND, p.contract_identity(contract))
    seen: set[int] = set()
    for root in roots:
        if not root.is_dir():
            continue
        # Only units whose final `.complete` marker was published count as completed;
        # an interrupted partial transfer without `.complete` is left for recomputation.
        markers = sorted(root.glob("unit-*.json.complete"))
        if not markers:
            continue
        contract_path = root / "contract.json"
        verify_immutable(contract_path)
        binding = json.loads(contract_path.read_text(encoding="utf-8"))
        validate_contract(binding)
        if (
            _allocation_without_commit(binding) != expected_alloc
            or binding["commit"] not in commits
        ):
            raise ValueError("foreign namespace/source/allocation in export root")
        for marker in markers:
            path = root / marker.name[: -len(".complete")]
            stem = path.stem
            if not stem.startswith("unit-"):
                continue
            index = int(stem.split("-", 1)[1])
            if index in seen or not 0 <= index < contract["count"]:
                raise ValueError(f"duplicate or out-of-range allocation index: {index}")
            verified = verify_immutable(path)
            ledger["units"].append(
                {
                    "index": index,
                    "unit_id": unit_id(contract["namespace"], p.UNIT_KIND, index),
                    "size": verified["size"],
                    "sha256": verified["sha256"],
                }
            )
            seen.add(index)
    ledger["units"].sort(key=lambda row: row["index"])
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--roots", required=True, nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--allow-commit", action="append", default=[])
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    allowed = {contract["commit"], *args.allow_commit} if args.allow_commit else None
    publish_json(
        args.out.resolve(),
        build(contract, [r.resolve() for r in args.roots], allowed_commits=allowed),
    )


if __name__ == "__main__":
    main()
