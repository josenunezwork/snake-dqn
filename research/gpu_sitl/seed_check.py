"""Repository seed audit for namespace ``gpu-sitl-preflight-20261109-v1``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from research.devicesim_v2_parity.harness import derive_world_seeds
from research.fixed_search import fixed_string_hits
from research.t3_rollout_labels_20261015.bank_engine import fresh_seeds as t3_fresh_seeds
from research.watch_replay.run import ROOT as REPO_ROOT
from research.watch_replay.seed_audit import (
    BANK_V2_NAMESPACES,
    COMMITTED_BANK_FILES,
    KNOWN_FIXTURE_SEEDS,
    KNOWN_NAMESPACE_PREFIXES,
    L1_OS_NAMESPACES,
    LATER_STUDY_NAMESPACES,
    seed_for,
)

PREFLIGHT_NAMESPACE = "gpu-sitl-preflight-20261109-v1"
NATURAL_WORLD_INDICES: tuple[int, ...] = (0, 1, 2, 3)
LONG_HERO_WORLD_INDICES: tuple[int, ...] = (4, 5, 6, 7)
EXTENSION_MATURE_WORLD_INDICES: tuple[int, ...] = tuple(range(8, 40))
AUDIT_INDEX_RANGE = 512


def preflight_seeds(indices: Sequence[int] = NATURAL_WORLD_INDICES) -> list[int]:
    """Return deterministic world seeds for ``indices`` in :data:`PREFLIGHT_NAMESPACE`."""
    return [int(seed_for(PREFLIGHT_NAMESPACE, int(idx))) for idx in indices]


def collect_known_repository_seeds(index_horizon: int = AUDIT_INDEX_RANGE) -> dict[str, set[int]]:
    """Collect all seed sets tracked by the repository's seed audit registries."""
    by_source: dict[str, set[int]] = {}
    all_namespaces = (
        *KNOWN_NAMESPACE_PREFIXES,
        *LATER_STUDY_NAMESPACES,
        *BANK_V2_NAMESPACES,
        *L1_OS_NAMESPACES,
    )
    for ns, declared_limit in all_namespaces:
        horizon = max(int(declared_limit), int(index_horizon))
        by_source[f"namespace:{ns}"] = {seed_for(ns, i) for i in range(horizon)}

    by_source["devicesim_v2_parity"] = {
        int(s) for shard in range(16) for s in derive_world_seeds(shard, 64)
    }

    by_source["known_fixture_seeds"] = {int(s) for s in KNOWN_FIXTURE_SEEDS}
    by_source["t3_rollout_labels_dense"] = {int(s) for s in t3_fresh_seeds(512)}

    bank_seeds: set[int] = set()
    for rel_path in COMMITTED_BANK_FILES:
        full_path = REPO_ROOT / rel_path
        if full_path.is_file():
            payload = json.loads(full_path.read_text(encoding="utf-8"))
            for key in ("seeds", "dev_seeds", "test_seeds", "selected_seeds"):
                vals = payload.get(key)
                if isinstance(vals, list):
                    for item in vals:
                        if isinstance(item, int):
                            bank_seeds.add(int(item))
                        elif isinstance(item, dict) and "seed" in item:
                            bank_seeds.add(int(item["seed"]))
            for row in payload.get("sessions", []):
                if isinstance(row, dict) and "seed" in row:
                    bank_seeds.add(int(row["seed"]))
    by_source["committed_bank_files"] = bank_seeds
    return by_source


def audit_preflight_seeds(
    *,
    natural_indices: Sequence[int] = NATURAL_WORLD_INDICES,
    long_hero_indices: Sequence[int] = LONG_HERO_WORLD_INDICES,
    extension_indices: Sequence[int] = EXTENSION_MATURE_WORLD_INDICES,
    index_horizon: int = AUDIT_INDEX_RANGE,
    scan_code_roots: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Verify that ``gpu-sitl-preflight-20261109-v1`` collides with nothing in the repo."""
    natural_seeds = preflight_seeds(natural_indices)
    long_hero_seeds = preflight_seeds(long_hero_indices)
    extension_seeds = preflight_seeds(extension_indices)
    all_allocated = natural_seeds + long_hero_seeds + extension_seeds

    internal_256 = [seed_for(PREFLIGHT_NAMESPACE, i) for i in range(256)]
    internal_unique = len(internal_256) == len(set(internal_256))
    allocated_unique = len(all_allocated) == len(set(all_allocated))

    known_by_source = collect_known_repository_seeds(index_horizon=index_horizon)
    allocated_set = set(all_allocated)
    prefix_256_set = set(internal_256)

    collisions_allocated: dict[str, list[int]] = {}
    collisions_prefix_256: dict[str, list[int]] = {}
    for source, seeds in known_by_source.items():
        hit_alloc = sorted(allocated_set & seeds)
        if hit_alloc:
            collisions_allocated[source] = hit_alloc
        hit_256 = sorted(prefix_256_set & seeds)
        if hit_256:
            collisions_prefix_256[source] = hit_256

    if scan_code_roots is None:
        scan_code_roots = [
            REPO_ROOT / "src",
            REPO_ROOT / "web",
            REPO_ROOT / "configs",
            REPO_ROOT / "docs",
        ]
    tokens = {str(s): int(s) for s in all_allocated}
    literal_hits: dict[str, list[int]] = {}
    for root in scan_code_roots:
        if Path(root).is_dir():
            hits = fixed_string_hits(Path(root), tokens, word_regexp=True)
            for rel_file, matched_tokens in hits.items():
                literal_hits[f"{Path(root).name}/{rel_file}"] = sorted(
                    tokens[t] for t in matched_tokens
                )

    ok = (
        internal_unique
        and allocated_unique
        and not collisions_allocated
        and not collisions_prefix_256
        and not literal_hits
    )
    return {
        "namespace": PREFLIGHT_NAMESPACE,
        "ok": bool(ok),
        "natural_world_indices": [int(i) for i in natural_indices],
        "natural_world_seeds": natural_seeds,
        "long_hero_world_indices": [int(i) for i in long_hero_indices],
        "long_hero_world_seeds": long_hero_seeds,
        "extension_mature_world_indices": [int(i) for i in extension_indices],
        "extension_mature_world_seeds": extension_seeds,
        "all_allocated_seeds": all_allocated,
        "internal_256_unique": bool(internal_unique),
        "allocated_unique": bool(allocated_unique),
        "audited_namespace_count": len(known_by_source),
        "index_horizon_per_namespace": int(index_horizon),
        "total_known_unique_seeds_checked": len(set().union(*known_by_source.values())),
        "collisions_allocated": collisions_allocated,
        "collisions_prefix_256": collisions_prefix_256,
        "literal_code_hits": literal_hits,
    }
