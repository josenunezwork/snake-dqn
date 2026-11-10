"""Repository seed registration and disjointness audit for Step B1a and Step B1b namespaces."""

from __future__ import annotations

import json
import re
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
T34_PREFLIGHT_NAMESPACE = "t34-gpu-sitl-preflight-20261109-v1"
TRAIN_NAMESPACE = "t34-gpu-sitl-train-20261109-v1"
PROBE32_NAMESPACE = "t34-gpu-sitl-probe32-20261109-v1"
SELECTIVITY_NAMESPACE = "t34-gpu-sitl-selectivity-20261109-v1"
SCREEN_NAMESPACE = "t34-gpu-sitl-screen-20261109-v1"

TRAIN_WORLD_COUNT = 256
PROBE32_WORLD_COUNT = 32
SELECTIVITY_WORLD_COUNT = 48
SCREEN_WORLD_COUNT = 48

B1B_REGISTERED_NAMESPACES: tuple[tuple[str, int], ...] = (
    (T34_PREFLIGHT_NAMESPACE, 4),
    (TRAIN_NAMESPACE, TRAIN_WORLD_COUNT),
    (PROBE32_NAMESPACE, PROBE32_WORLD_COUNT),
    (SELECTIVITY_NAMESPACE, SELECTIVITY_WORLD_COUNT),
    (SCREEN_NAMESPACE, SCREEN_WORLD_COUNT),
)

CONTAMINATED_FORBIDDEN_DATA_NAMESPACES: tuple[str, ...] = (
    "t34-openstop-gate-20261104-v1",
    "t34-hero-probe-20261103-v1",
)

NATURAL_WORLD_INDICES: tuple[int, ...] = (0, 1, 2, 3)
LONG_HERO_WORLD_INDICES: tuple[int, ...] = (4, 5, 6, 7)
EXTENSION_MATURE_WORLD_INDICES: tuple[int, ...] = tuple(range(8, 40))
AUDIT_INDEX_RANGE = 512


def namespace_seeds(namespace: str, count_or_indices: int | Sequence[int]) -> list[int]:
    """Return deterministic world seeds for ``namespace``."""
    if isinstance(count_or_indices, int):
        if count_or_indices < 0:
            raise ValueError(f"count must be >= 0, got {count_or_indices}")
        indices: Sequence[int] = range(count_or_indices)
    else:
        indices = count_or_indices
    return [int(seed_for(namespace, int(idx))) for idx in indices]


def preflight_seeds(indices: Sequence[int] = NATURAL_WORLD_INDICES) -> list[int]:
    """Return deterministic world seeds for ``indices`` in :data:`PREFLIGHT_NAMESPACE`."""
    return namespace_seeds(PREFLIGHT_NAMESPACE, indices)


def train_seeds(count: int = TRAIN_WORLD_COUNT) -> list[int]:
    """Return deterministic online training world seeds for :data:`TRAIN_NAMESPACE`."""
    return namespace_seeds(TRAIN_NAMESPACE, count)


def probe32_seeds(count: int = PROBE32_WORLD_COUNT) -> list[int]:
    """Return deterministic probe-bank world seeds for :data:`PROBE32_NAMESPACE`."""
    return namespace_seeds(PROBE32_NAMESPACE, count)


def selectivity_seeds(count: int = SELECTIVITY_WORLD_COUNT) -> list[int]:
    """Return deterministic one-look selectivity world seeds for :data:`SELECTIVITY_NAMESPACE`."""
    return namespace_seeds(SELECTIVITY_NAMESPACE, count)


def screen_seeds(count: int = SCREEN_WORLD_COUNT) -> list[int]:
    """Return deterministic one-look closed-loop screen world seeds for :data:`SCREEN_NAMESPACE`."""
    return namespace_seeds(SCREEN_NAMESPACE, count)


def collect_known_repository_seeds(index_horizon: int = AUDIT_INDEX_RANGE) -> dict[str, set[int]]:
    """Collect all seed sets tracked by the repository's seed audit registries.

    Outcome-blind and contamination-safe: never opens ``t34-openstop-gate-20261104-v1`` or
    ``t34-hero-probe-20261103-v1`` data files (their seeds are derived algebraically via
    :func:`seed_for` from ``LATER_STUDY_NAMESPACES`` / ``L1_OS_NAMESPACES``).
    """
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

    by_source["known_fixture_seeds"] = {
        int(s) for s in KNOWN_FIXTURE_SEEDS
    } | {int(seed_for("t34-watch-replay-full-size-fixture-v1", 0))}
    by_source["t3_rollout_labels_dense"] = {int(s) for s in t3_fresh_seeds(512)}

    bank_seeds: set[int] = set()
    for rel_path in COMMITTED_BANK_FILES:
        full_path = REPO_ROOT / rel_path
        if full_path.is_file():
            text = full_path.read_text(encoding="utf-8")
            for match in re.findall(r'"(?:world_seed|seed)"\s*:\s*(\d+)', text):
                bank_seeds.add(int(match))
            payload = json.loads(text)
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

    exclusions_path = (
        REPO_ROOT / "docs" / "experiments" / "2026-10-08_frp_v5_screen" / "training_exclusions.json"
    )
    if exclusions_path.is_file():
        excl_text = exclusions_path.read_text(encoding="utf-8")
        excl_seeds = {int(x) for x in re.findall(r"\b(\d{5,12})\b", excl_text)}
        if excl_seeds:
            by_source["frp_v5_training_exclusions"] = excl_seeds
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


def audit_b1b_seeds(
    *,
    index_horizon: int = AUDIT_INDEX_RANGE,
    scan_code_roots: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Verify zero seed collisions for all Step B1a and Step B1b namespaces:

    - ``gpu-sitl-preflight-20261109-v1`` (`40` allocated, `256` prefix)
    - ``t34-gpu-sitl-preflight-20261109-v1`` (`4` allocated, `256` prefix)
    - ``t34-gpu-sitl-train-20261109-v1`` (`256` allocated, `512` prefix)
    - ``t34-gpu-sitl-probe32-20261109-v1`` (`32` allocated, `256` prefix)
    - ``t34-gpu-sitl-selectivity-20261109-v1`` (`48` allocated, `256` prefix)
    - ``t34-gpu-sitl-screen-20261109-v1`` (`48` allocated, `256` prefix)

    Never reads data files from ``t34-openstop-gate-20261104-v1`` or
    ``t34-hero-probe-20261103-v1``.
    """
    preflight_report = audit_preflight_seeds(
        index_horizon=index_horizon,
        scan_code_roots=[],
    )
    known_by_source = collect_known_repository_seeds(index_horizon=index_horizon)

    # Confirm both contaminated namespaces are present in the algebraic seed registry
    for forbidden_ns in CONTAMINATED_FORBIDDEN_DATA_NAMESPACES:
        if f"namespace:{forbidden_ns}" not in known_by_source:
            raise AssertionError(
                f"Contaminated namespace {forbidden_ns!r} missing from known seed registry"
            )

    all_audited_specs: tuple[tuple[str, int, int], ...] = (
        (PREFLIGHT_NAMESPACE, 40, 256),
        (T34_PREFLIGHT_NAMESPACE, 4, 256),
        (TRAIN_NAMESPACE, TRAIN_WORLD_COUNT, max(512, int(index_horizon))),
        (PROBE32_NAMESPACE, PROBE32_WORLD_COUNT, 256),
        (SELECTIVITY_NAMESPACE, SELECTIVITY_WORLD_COUNT, 256),
        (SCREEN_NAMESPACE, SCREEN_WORLD_COUNT, 256),
    )

    per_namespace: dict[str, dict[str, Any]] = {}
    allocated_by_ns: dict[str, set[int]] = {}
    prefix_by_ns: dict[str, set[int]] = {}
    all_allocated_union: list[int] = []

    for ns, alloc_count, prefix_len in all_audited_specs:
        alloc_list = namespace_seeds(ns, alloc_count)
        prefix_list = namespace_seeds(ns, prefix_len)
        alloc_set = set(alloc_list)
        prefix_set = set(prefix_list)
        alloc_unique = len(alloc_list) == len(alloc_set)
        prefix_unique = len(prefix_list) == len(prefix_set)
        allocated_by_ns[ns] = alloc_set
        prefix_by_ns[ns] = prefix_set
        all_allocated_union.extend(alloc_list)

        repo_collisions_alloc: dict[str, list[int]] = {}
        repo_collisions_prefix: dict[str, list[int]] = {}
        for source, seeds in known_by_source.items():
            hit_a = sorted(alloc_set & seeds)
            if hit_a:
                repo_collisions_alloc[source] = hit_a
            hit_p = sorted(prefix_set & seeds)
            if hit_p:
                repo_collisions_prefix[source] = hit_p

        per_namespace[ns] = {
            "allocated_count": int(alloc_count),
            "prefix_horizon": int(prefix_len),
            "allocated_seeds": alloc_list,
            "allocated_unique": bool(alloc_unique),
            "prefix_unique": bool(prefix_unique),
            "repo_collisions_allocated": repo_collisions_alloc,
            "repo_collisions_prefix": repo_collisions_prefix,
            "ok": bool(
                alloc_unique
                and prefix_unique
                and not repo_collisions_alloc
                and not repo_collisions_prefix
            ),
        }

    cross_collisions_allocated: dict[str, list[int]] = {}
    cross_collisions_prefix: dict[str, list[int]] = {}
    ns_names = [ns for ns, _, _ in all_audited_specs]
    for i, ns_a in enumerate(ns_names):
        for ns_b in ns_names[i + 1 :]:
            pair_key = f"{ns_a} vs {ns_b}"
            overlap_alloc = sorted(allocated_by_ns[ns_a] & allocated_by_ns[ns_b])
            if overlap_alloc:
                cross_collisions_allocated[pair_key] = overlap_alloc
            overlap_pref = sorted(prefix_by_ns[ns_a] & prefix_by_ns[ns_b])
            if overlap_pref:
                cross_collisions_prefix[pair_key] = overlap_pref

    if scan_code_roots is None:
        scan_code_roots = [
            REPO_ROOT / "src",
            REPO_ROOT / "web",
            REPO_ROOT / "configs",
            REPO_ROOT / "docs",
        ]
    tokens = {str(s): int(s) for s in set(all_allocated_union)}
    literal_hits: dict[str, list[int]] = {}
    for root in scan_code_roots:
        if Path(root).is_dir():
            hits = fixed_string_hits(Path(root), tokens, word_regexp=True)
            for rel_file, matched_tokens in hits.items():
                literal_hits[f"{Path(root).name}/{rel_file}"] = sorted(
                    tokens[t] for t in matched_tokens
                )

    all_ns_ok = all(info["ok"] for info in per_namespace.values())
    union_unique = len(all_allocated_union) == len(set(all_allocated_union))
    ok = (
        bool(preflight_report["ok"])
        and all_ns_ok
        and union_unique
        and not cross_collisions_allocated
        and not cross_collisions_prefix
        and not literal_hits
    )
    return {
        "ok": bool(ok),
        "status": "CLEAR" if ok else "COLLISION",
        "namespaces": per_namespace,
        "total_allocated_seeds": len(all_allocated_union),
        "union_allocated_unique": bool(union_unique),
        "cross_collisions_allocated": cross_collisions_allocated,
        "cross_collisions_prefix": cross_collisions_prefix,
        "audited_repository_source_count": len(known_by_source),
        "index_horizon_per_namespace": int(index_horizon),
        "total_known_unique_seeds_checked": len(set().union(*known_by_source.values())),
        "contaminated_data_files_opened": 0,
        "contaminated_namespaces_checked_algebraically": list(
            CONTAMINATED_FORBIDDEN_DATA_NAMESPACES
        ),
        "literal_code_hits": literal_hits,
    }


if __name__ == "__main__":
    print(json.dumps(audit_b1b_seeds(), indent=2, sort_keys=True))
