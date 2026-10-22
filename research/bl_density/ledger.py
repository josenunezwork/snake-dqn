"""Build an outcome-blind ledger from completed density episode exports.

This module intentionally never opens an episode JSON export.  It establishes
resume eligibility from filenames, byte sizes, digest sidecars, completion
markers, and the supervisor's terminal export inventory only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from research.bl_ppo.exports import export_json, verify_export

SCHEMA = "bl-density-verified-ledger/v1"


def validate_ledger(
    ledger_value: Any, *, namespace: str, allocation: dict[str, dict[str, int]]
) -> list[dict[str, Any]]:
    """Validate ledger metadata against a fixed allocation without opening exports."""
    if not isinstance(ledger_value, dict):
        raise ValueError("ledger must be an object")
    if ledger_value.get("schema") != SCHEMA or ledger_value.get("namespace") != namespace:
        raise ValueError("ledger schema or namespace does not match")
    entries = ledger_value.get("entries")
    if not isinstance(entries, list):
        raise ValueError("ledger entries must be a list")
    allowed = {
        f"{namespace}/{arm}/{mix}/{index:03d}"
        for arm, by_mix in allocation.items()
        for mix, count in by_mix.items()
        for index in range(count)
    }
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("ledger entry is malformed")
        episode_id, digest, size = entry.get("episode_id"), entry.get("sha256"), entry.get("size")
        if not isinstance(episode_id, str) or not isinstance(digest, str) or type(size) is not int:
            raise ValueError("ledger entry lacks id/hash/size")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("ledger sha256 is invalid")
        if size < 1 or episode_id not in allowed or episode_id in seen:
            raise ValueError("ledger has duplicate or out-of-allocation episode")
        seen.add(episode_id)
    return sorted(entries, key=lambda entry: str(entry["episode_id"]))


def _episode_id_from_name(path: Path, namespace: str) -> str:
    """Recover the fixed episode identity from an export filename alone."""
    prefix = namespace + "_"
    if path.suffix != ".json" or not path.name.startswith(prefix):
        raise ValueError(f"not a density episode export for {namespace}: {path.name}")
    remainder = path.stem[len(prefix) :]
    arm, mix, index = remainder.rsplit("_", 2)
    if not index.isdecimal() or len(index) != 3:
        raise ValueError(f"invalid density episode filename: {path.name}")
    return f"{namespace}/{arm}/{mix}/{index}"


def _terminal_entries(path: Path) -> dict[str, tuple[str, int]]:
    """Read only path/hash/size fields from a supervisor terminal inventory."""
    payload = json.loads(path.read_text())
    exports = payload.get("exports")
    if not isinstance(exports, list):
        raise ValueError("terminal inventory has no exports list")
    entries: dict[str, tuple[str, int]] = {}
    for item in exports:
        if not isinstance(item, dict):
            raise ValueError("terminal inventory export entry is malformed")
        value = item.get("path")
        digest = item.get("sha256")
        size = item.get("size")
        if not isinstance(value, str) or not isinstance(digest, str) or type(size) is not int:
            raise ValueError("terminal inventory entry lacks path/hash/size")
        if value in entries:
            raise ValueError(f"duplicate terminal inventory path: {value}")
        entries[value] = (digest, size)
    return entries


def _inventory_matches(
    inventory: dict[str, tuple[str, int]], path: Path, digest: str, size: int
) -> bool:
    """Match an export by stable suffix because supervisor roots are transport-owned."""
    suffix = "/" + path.as_posix().lstrip("/")
    matches = [
        entry for name, entry in inventory.items() if name == suffix[1:] or name.endswith(suffix)
    ]
    if len(matches) != 1:
        return False
    return matches[0] == (digest, size)


def build_ledger(
    exports_root: Path,
    terminal_inventory: Path,
    *,
    namespace: str,
    expected_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Verify completed exports and return a deterministic metadata-only ledger."""
    exports_root = Path(exports_root)
    inventory = _terminal_entries(Path(terminal_inventory))
    expected = set(expected_ids) if expected_ids is not None else None
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(exports_root.glob("*.json")):
        episode_id = _episode_id_from_name(path, namespace)
        if episode_id in seen:
            raise ValueError(f"duplicate local episode id: {episode_id}")
        if expected is not None and episode_id not in expected:
            raise ValueError(f"episode outside frozen allocation: {episode_id}")
        digest = verify_export(path)
        size = path.stat().st_size
        relative = path.relative_to(exports_root.parent.parent)
        if not _inventory_matches(inventory, relative, digest, size):
            raise ValueError(f"terminal inventory does not verify {path.name}")
        seen.add(episode_id)
        entries.append({"episode_id": episode_id, "sha256": digest, "size": size})
    if not entries:
        raise ValueError("ledger has no verified episode exports")
    return {"schema": SCHEMA, "namespace": namespace, "entries": entries}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports-root", required=True, type=Path, action="append")
    parser.add_argument("--terminal-inventory", required=True, type=Path, action="append")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit("--out must be fresh")
    if len(args.exports_root) != len(args.terminal_inventory):
        parser.error("one --terminal-inventory is required per --exports-root")
    rows: dict[str, dict[str, Any]] = {}
    for root, inventory in zip(args.exports_root, args.terminal_inventory):
        if not any(root.glob("*.json")):
            continue  # a mechanically failed attempt can export zero completed units
        part = build_ledger(root, inventory, namespace=args.namespace)
        for entry in part["entries"]:
            if entry["episode_id"] in rows:
                raise ValueError("duplicate episode_id across ledger input trees")
            rows[entry["episode_id"]] = entry
    if not rows:
        raise ValueError("ledger union has no verified completed units")
    ledger = {
        "schema": SCHEMA,
        "namespace": args.namespace,
        "entries": [rows[key] for key in sorted(rows)],
    }
    export_json(args.out, ledger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
