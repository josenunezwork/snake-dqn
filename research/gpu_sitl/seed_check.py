"""Repository seed registration and disjointness audit for Step B1a and Step B1b namespaces."""

from __future__ import annotations

import ast
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

# Dev-only namespaces for the L4 CUDA shake-out (NOT EVIDENCE; never overlap registered namespaces)
SHAKEOUT_SMOKE_NAMESPACE = "gpu-sitl-shakeout-smoke-dev-20261110-v1"
SHAKEOUT_PROBE_NAMESPACE = "gpu-sitl-shakeout-probe-dev-20261110-v1"
SHAKEOUT_TRAIN_NAMESPACE = "gpu-sitl-shakeout-train-dev-20261110-v1"

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

FORBIDDEN_TRAINING_NAMESPACES: frozenset[str] = frozenset(
    {
        SELECTIVITY_NAMESPACE,
        SCREEN_NAMESPACE,
        *CONTAMINATED_FORBIDDEN_DATA_NAMESPACES,
    }
)

DEFAULT_SCAN_CODE_ROOTS: tuple[Path, ...] = (
    REPO_ROOT / "src",
    REPO_ROOT / "web",
    REPO_ROOT / "configs",
    REPO_ROOT / "research" / "gpu_sitl",
)

NATURAL_WORLD_INDICES: tuple[int, ...] = (0, 1, 2, 3)
LONG_HERO_WORLD_INDICES: tuple[int, ...] = (4, 5, 6, 7)
EXTENSION_MATURE_WORLD_INDICES: tuple[int, ...] = tuple(range(8, 40))
AUDIT_INDEX_RANGE = 512


def assert_allowed_training_seeds(
    *,
    namespace: str,
    seeds: Sequence[int],
    context: str,
    index_horizon: int = AUDIT_INDEX_RANGE,
) -> None:
    """Fail closed if ``namespace`` or ``seeds`` touch held-out or contaminated namespaces."""
    ns_str = str(namespace)
    if ns_str in FORBIDDEN_TRAINING_NAMESPACES:
        raise ValueError(
            f"Forbidden held-out/contaminated namespace {ns_str!r} rejected in {context}"
        )
    seed_set = {int(s) for s in seeds}
    if not seed_set:
        return
    horizon = max(int(index_horizon), 512)
    for bad_ns in sorted(FORBIDDEN_TRAINING_NAMESPACES):
        bad_seeds = {int(seed_for(bad_ns, idx)) for idx in range(horizon)}
        overlap = sorted(seed_set & bad_seeds)
        if overlap:
            raise ValueError(
                f"Forbidden seed overlap in {context} with held-out/contaminated namespace "
                f"{bad_ns!r}: {overlap}"
            )


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


def shakeout_smoke_seeds(count: int = 8) -> list[int]:
    """Return dev-only world seeds for the CUDA shake-out smoke sub-phase (NOT EVIDENCE)."""
    return namespace_seeds(SHAKEOUT_SMOKE_NAMESPACE, count)


def shakeout_probe_seeds(count: int = 2) -> list[int]:
    """Return dev-only world seeds for the CUDA shake-out probe harvest sub-phase (NOT EVIDENCE)."""
    return namespace_seeds(SHAKEOUT_PROBE_NAMESPACE, count)


def shakeout_train_seeds(count: int = 4) -> list[int]:
    """Return dev-only world seeds for the CUDA shake-out training sub-phase (NOT EVIDENCE)."""
    return namespace_seeds(SHAKEOUT_TRAIN_NAMESPACE, count)


def audit_shakeout_seeds(
    *,
    smoke_count: int = 16,
    probe_count: int = 8,
    train_count: int = 16,
    smoke_envs: int | None = None,
    probe_worlds: int | None = None,
    train_envs: int | None = None,
    index_horizon: int = AUDIT_INDEX_RANGE,
) -> dict[str, Any]:
    """Verify dev shake-out seeds are disjoint from each other and all registered namespaces."""
    n_smoke = int(smoke_envs) if smoke_envs is not None else int(smoke_count)
    n_probe = int(probe_worlds) if probe_worlds is not None else int(probe_count)
    n_train = int(train_envs) if train_envs is not None else int(train_count)
    s_smoke = set(shakeout_smoke_seeds(n_smoke))
    s_probe = set(shakeout_probe_seeds(n_probe))
    s_train = set(shakeout_train_seeds(n_train))
    dev_union = s_smoke | s_probe | s_train

    forbidden_registered: dict[str, set[int]] = {
        PREFLIGHT_NAMESPACE: set(namespace_seeds(PREFLIGHT_NAMESPACE, max(40, index_horizon))),
        T34_PREFLIGHT_NAMESPACE: set(
            namespace_seeds(T34_PREFLIGHT_NAMESPACE, max(4, index_horizon))
        ),
        TRAIN_NAMESPACE: set(
            namespace_seeds(TRAIN_NAMESPACE, max(TRAIN_WORLD_COUNT, index_horizon))
        ),
        PROBE32_NAMESPACE: set(
            namespace_seeds(PROBE32_NAMESPACE, max(PROBE32_WORLD_COUNT, index_horizon))
        ),
        SELECTIVITY_NAMESPACE: set(
            namespace_seeds(SELECTIVITY_NAMESPACE, max(SELECTIVITY_WORLD_COUNT, index_horizon))
        ),
        SCREEN_NAMESPACE: set(
            namespace_seeds(SCREEN_NAMESPACE, max(SCREEN_WORLD_COUNT, index_horizon))
        ),
    }
    known_by_source = collect_known_repository_seeds(index_horizon=index_horizon)
    overlaps: dict[str, list[int]] = {}
    if s_smoke & s_probe:
        overlaps["shakeout_smoke_vs_probe"] = sorted(s_smoke & s_probe)
    if s_smoke & s_train:
        overlaps["shakeout_smoke_vs_train"] = sorted(s_smoke & s_train)
    if s_probe & s_train:
        overlaps["shakeout_probe_vs_train"] = sorted(s_probe & s_train)
    for ns, r_seeds in forbidden_registered.items():
        hit = sorted(dev_union & r_seeds)
        if hit:
            overlaps[f"registered:{ns}"] = hit
    for src, k_seeds in known_by_source.items():
        hit = sorted(dev_union & k_seeds)
        if hit:
            overlaps[f"repo:{src}"] = hit
    ok = (
        len(s_smoke) == n_smoke
        and len(s_probe) == n_probe
        and len(s_train) == n_train
        and len(dev_union) == (n_smoke + n_probe + n_train)
        and not overlaps
    )
    return {
        "ok": bool(ok),
        "status": "PASS_ZERO_COLLISIONS" if ok else "FAIL_COLLISION_FOUND",
        "not_evidence": True,
        "shakeout_namespaces": {
            "smoke": SHAKEOUT_SMOKE_NAMESPACE,
            "probe": SHAKEOUT_PROBE_NAMESPACE,
            "train": SHAKEOUT_TRAIN_NAMESPACE,
        },
        "forbidden_registered_namespaces_untouched": [
            PREFLIGHT_NAMESPACE,
            T34_PREFLIGHT_NAMESPACE,
            SELECTIVITY_NAMESPACE,
            SCREEN_NAMESPACE,
            PROBE32_NAMESPACE,
            TRAIN_NAMESPACE,
        ],
        "overlaps": overlaps,
    }


def _assert_not_contaminated_path(path: Path | str) -> None:
    path_str = Path(path).as_posix()
    for bad_ns in FORBIDDEN_TRAINING_NAMESPACES:
        if bad_ns in path_str:
            raise RuntimeError(
                f"Refusing to open file under forbidden/contaminated namespace {bad_ns!r}: {path}"
            )


def _scan_code_roots_for_literal_seeds(
    scan_code_roots: Sequence[Path],
    tokens: dict[str, int],
) -> dict[str, list[int]]:
    literal_hits: dict[str, list[int]] = {}
    for root in scan_code_roots:
        root_p = Path(root)
        if root_p.is_dir():
            _assert_not_contaminated_path(root_p)
            resolved_root = root_p.resolve()
            hits = fixed_string_hits(resolved_root, tokens, word_regexp=True)
            for hit_path, matched_tokens in hits.items():
                _assert_not_contaminated_path(hit_path)
                try:
                    rel_str = Path(hit_path).resolve().relative_to(resolved_root).as_posix()
                except ValueError:
                    rel_str = Path(hit_path).as_posix()
                literal_hits[f"{root_p.name}/{rel_str}"] = sorted(tokens[t] for t in matched_tokens)
    return literal_hits


def _audit_runtime_modules_no_heldout_refs(
    gpu_sitl_dir: Path | None = None,
) -> dict[str, list[str]]:
    """Verify training/harvest/smoke modules never materialize or score held-out namespaces."""
    target_dir = (
        (REPO_ROOT / "research" / "gpu_sitl") if gpu_sitl_dir is None else Path(gpu_sitl_dir)
    )
    if not target_dir.is_dir():
        return {}
    forbidden_fn_names = {"selectivity_seeds", "screen_seeds"}
    forbidden_train_names = {
        "selectivity_seeds",
        "screen_seeds",
        "SELECTIVITY_NAMESPACE",
        "SCREEN_NAMESPACE",
        "CONTAMINATED_FORBIDDEN_DATA_NAMESPACES",
    }
    forbidden_train_strings = set(FORBIDDEN_TRAINING_NAMESPACES)
    forbidden_smoke_strings = set(CONTAMINATED_FORBIDDEN_DATA_NAMESPACES)
    violations: dict[str, list[str]] = {}
    for mod_name in ("open_stop_gpu.py", "probe_bank.py", "train_sitl.py", "smoke.py"):
        mod_path = target_dir / mod_name
        if not mod_path.is_file():
            continue
        tree = ast.parse(mod_path.read_text(encoding="utf-8"), filename=str(mod_path))
        bad_names = forbidden_fn_names if mod_name == "smoke.py" else forbidden_train_names
        bad_strings = forbidden_smoke_strings if mod_name == "smoke.py" else forbidden_train_strings
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in bad_names:
                found.add(f"name:{node.id}")
            elif isinstance(node, ast.Attribute) and node.attr in bad_names:
                found.add(f"attr:{node.attr}")
            elif isinstance(node, ast.alias) and node.name in bad_names:
                found.add(f"import:{node.name}")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in bad_strings:
                    found.add(f"literal:{node.value}")
        if found:
            violations[mod_name] = sorted(found)
    return violations


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

    by_source["known_fixture_seeds"] = {int(s) for s in KNOWN_FIXTURE_SEEDS} | {
        int(seed_for("t34-watch-replay-full-size-fixture-v1", 0))
    }
    by_source["t3_rollout_labels_dense"] = {int(s) for s in t3_fresh_seeds(512)}

    bank_seeds: set[int] = set()
    for rel_path in COMMITTED_BANK_FILES:
        _assert_not_contaminated_path(rel_path)
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
        _assert_not_contaminated_path(exclusions_path)
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

    internal_256 = [seed_for(PREFLIGHT_NAMESPACE, i) for i in range(max(256, int(index_horizon)))]
    internal_unique = len(internal_256) == len(set(internal_256))
    allocated_unique = len(all_allocated) == len(set(all_allocated))

    known_by_source = collect_known_repository_seeds(index_horizon=index_horizon)
    allocated_set = set(all_allocated)
    prefix_256_set = set(internal_256[:256])

    collisions_allocated: dict[str, list[int]] = {}
    collisions_prefix_256: dict[str, list[int]] = {}
    for source, seeds in known_by_source.items():
        if source == f"namespace:{PREFLIGHT_NAMESPACE}":
            continue
        hit_alloc = sorted(allocated_set & seeds)
        if hit_alloc:
            collisions_allocated[source] = hit_alloc
        hit_256 = sorted(prefix_256_set & seeds)
        if hit_256:
            collisions_prefix_256[source] = hit_256

    if scan_code_roots is None:
        scan_code_roots = list(DEFAULT_SCAN_CODE_ROOTS)
    tokens = {str(s): int(s) for s in all_allocated}
    literal_hits = _scan_code_roots_for_literal_seeds(scan_code_roots, tokens)

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

    contaminated_opened = 0
    for rel_path in COMMITTED_BANK_FILES:
        if any(bad_ns in str(rel_path) for bad_ns in CONTAMINATED_FORBIDDEN_DATA_NAMESPACES):
            contaminated_opened += 1

    # Confirm both contaminated namespaces are present in the algebraic seed registry
    for forbidden_ns in CONTAMINATED_FORBIDDEN_DATA_NAMESPACES:
        if f"namespace:{forbidden_ns}" not in known_by_source:
            raise AssertionError(
                f"Contaminated namespace {forbidden_ns!r} missing from known seed registry"
            )

    min_prefix = max(256, int(index_horizon))
    all_audited_specs: tuple[tuple[str, int, int], ...] = (
        (PREFLIGHT_NAMESPACE, 40, min_prefix),
        (T34_PREFLIGHT_NAMESPACE, 4, min_prefix),
        (TRAIN_NAMESPACE, TRAIN_WORLD_COUNT, max(512, int(index_horizon))),
        (PROBE32_NAMESPACE, PROBE32_WORLD_COUNT, min_prefix),
        (SELECTIVITY_NAMESPACE, SELECTIVITY_WORLD_COUNT, min_prefix),
        (SCREEN_NAMESPACE, SCREEN_WORLD_COUNT, min_prefix),
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
            if source == f"namespace:{ns}":
                continue
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
        scan_code_roots = list(DEFAULT_SCAN_CODE_ROOTS)
    tokens = {str(s): int(s) for s in set(all_allocated_union)}
    literal_hits = _scan_code_roots_for_literal_seeds(scan_code_roots, tokens)
    heldout_code_hits = _audit_runtime_modules_no_heldout_refs()

    all_ns_ok = all(info["ok"] for info in per_namespace.values())
    union_unique = len(all_allocated_union) == len(set(all_allocated_union))
    ok = (
        bool(preflight_report["ok"])
        and all_ns_ok
        and union_unique
        and not cross_collisions_allocated
        and not cross_collisions_prefix
        and not literal_hits
        and not heldout_code_hits
        and contaminated_opened == 0
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
        "contaminated_data_files_opened": int(contaminated_opened),
        "contaminated_namespaces_checked_algebraically": list(
            CONTAMINATED_FORBIDDEN_DATA_NAMESPACES
        ),
        "literal_code_hits": literal_hits,
        "heldout_code_reference_hits": heldout_code_hits,
    }


if __name__ == "__main__":
    print(json.dumps(audit_b1b_seeds(), indent=2, sort_keys=True))
