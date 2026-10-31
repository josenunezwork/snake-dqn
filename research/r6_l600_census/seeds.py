"""Deterministic seed derivation and disjointness audit for R6 Watch6-v2 census."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from research.r6_l600_census import NAMESPACE

ROOT = Path(__file__).resolve().parents[2]

KNOWN_WATCH_REPLAY_NAMESPACES: tuple[str, ...] = (
    "t34-watch-replay-bank-01",
    "t34-watch-replay-bank-02",
    "t34-watch-replay-bank-03",
    "t34-watch-replay-bank-04",
    "t34-watch-replay-full-size-fixture-v1",
    "watch-replay-bank-01",
    "watch-replay-bank-02",
    "watch-replay-bank-03",
    "watch-replay-bank-04",
)


def derive_world_seeds(namespace: str = NAMESPACE, count: int = 64) -> list[int]:
    """Derive ``count`` deterministic unique 31-bit integer world seeds >= 10,000,000.

    Args:
        namespace: Domain-separated study namespace string.
        count: Number of world seeds to allocate.

    Returns:
        Ordered list of ``count`` unique positive 31-bit integers.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    seeds: list[int] = []
    seen: set[int] = set()
    for idx in range(count):
        nonce = 0
        while True:
            tag = f"{namespace}/world/{idx}" if nonce == 0 else f"{namespace}/world/{idx}/{nonce}"
            raw = int.from_bytes(hashlib.sha256(tag.encode("utf-8")).digest()[:4], "big")
            candidate = raw & 0x7FFF_FFFF
            if candidate >= 10_000_000 and candidate not in seen:
                seen.add(candidate)
                seeds.append(candidate)
                break
            nonce += 1
    return seeds


def _collect_json_ints(obj: Any, out: set[int], *, key_hint: str = "") -> None:
    """Recursively collect seed-like integers from registered JSON structures."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _collect_json_ints(v, out, key_hint=str(k).lower())
    elif isinstance(obj, list):
        if "seed" in key_hint:
            for item in obj:
                if type(item) is int:
                    out.add(item)
                else:
                    _collect_json_ints(item, out, key_hint=key_hint)
        else:
            for item in obj:
                _collect_json_ints(item, out, key_hint=key_hint)
    elif type(obj) is int and "seed" in key_hint:
        out.add(obj)


def collect_excluded_seeds(repo: Path = ROOT) -> tuple[set[int], dict[str, int]]:
    """Collect historical/registered seeds from repo registries without opening sealed exports.

    Never reads Watch6 V2 ``exports/`` or any ``pair-*.json``.

    Args:
        repo: Repository root path.

    Returns:
        Tuple ``(excluded_set, source_counts)`` summarizing each audited inventory.
    """
    excluded: set[int] = set()
    counts: dict[str, int] = {}

    registry_path = repo / "tools/sandbox/SANDBOX_INPUT_REGISTRY.json"
    if registry_path.is_file():
        reg = json.loads(registry_path.read_text())
        reg_seeds: set[int] = set()
        _collect_json_ints(reg, reg_seeds)
        excluded |= reg_seeds
        counts["sandbox_input_registry"] = len(reg_seeds)

    research_dir = repo / "research"
    bank_seeds: set[int] = set()
    if research_dir.is_dir():
        from research.t34_claim_a.seed_audit import _bank_seeds, namespace_seeds

        for pattern in ("*SEED_AUDIT*.json", "*BANK*.json"):
            for path in sorted(research_dir.rglob(pattern)):
                if "exports" in path.parts or path.name.startswith("pair-"):
                    continue
                try:
                    payload = json.loads(path.read_text())
                except (OSError, ValueError):
                    continue
                bank_seeds |= _bank_seeds(payload)
                _collect_json_ints(payload, bank_seeds)
                if isinstance(payload, dict) and payload.get("domain") == "t34-exit-b8-20261019/v1":
                    for ns, entry in payload.get("namespaces", {}).items():
                        if isinstance(entry, dict) and "count" in entry:
                            bank_seeds |= set(namespace_seeds(str(ns), int(entry["count"])))
    excluded |= bank_seeds
    counts["research_banks_and_audits"] = len(bank_seeds)

    replay_seeds: set[int] = set()
    for ns in KNOWN_WATCH_REPLAY_NAMESPACES:
        for i in range(128):
            s = int.from_bytes(hashlib.sha256(f"{ns}/{i}".encode()).digest()[:4], "big")
            replay_seeds.add(s)
    excluded |= replay_seeds
    counts["watch_replay_namespaces"] = len(replay_seeds)

    return excluded, counts


def audit_seeds(seeds: Sequence[int], repo: Path = ROOT) -> dict[str, Any]:
    """Verify 100% uniqueness and disjointness of ``seeds`` against audited inventories."""
    seed_list = [int(s) for s in seeds]
    if len(set(seed_list)) != len(seed_list):
        raise ValueError("duplicate seeds in R6 allocation")
    excluded, counts = collect_excluded_seeds(repo)
    collisions = sorted(set(seed_list) & excluded)
    if collisions:
        raise ValueError(f"R6 seed collision with excluded inventory: {collisions[:10]}")
    digest = hashlib.sha256(json.dumps(seed_list).encode("utf-8")).hexdigest()
    return {
        "status": "CLEAR",
        "num_seeds": len(seed_list),
        "seeds_sha256": digest,
        "excluded_total": len(excluded),
        "inventory_counts": counts,
        "collisions": 0,
    }
