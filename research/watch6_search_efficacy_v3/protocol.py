"""Frozen V3 primary/reserve allocations and source-closure identity."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

NAMESPACE = "t34-watch6-exact-stop-efficacy-20261031-v3"
COUNT = 96
HORIZON = 14400
SHARDS = 24
WORKERS = 4
ALLOWED_WORKERS = (4,)
PAIRS_PER_SHARD = COUNT // SHARDS  # 4 pairs per shard
FIXTURE_SEEDS = [34890123, 34890124, 34890125]

# Pre-registered failure-handling caps (REGISTRATION_V3.md §3).
MAX_PRESTART_RETRIES_PER_SHARD = 6
MAX_PRESTART_RETRIES_STUDY = 72
MAX_COMPUTE_COMPLETIONS_PER_SHARD = 2
MAX_TOTAL_DISPATCHES_PER_SHARD = 10
MAX_RESERVE_SUBSTITUTIONS = 16

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "research/watch6_search_efficacy_v3/job.sh"
NON_SCIENCE_DOCS = {
    "research/watch6_search_efficacy_v3/JUDGE_QUESTION.md",
    "research/watch6_search_efficacy_v3/README.md",
}


def registration() -> Path:
    """Binding confirmation-lane pre-registration document for Watch6 V3."""
    return ROOT / "research/watch6_search_efficacy_v3/REGISTRATION_V3.md"


def primary_seed(index: int) -> int:
    """Deterministic uint32 primary world seed for slot index in 0..95."""
    if index not in range(COUNT):
        raise ValueError(f"slot index out of range: {index}")
    return int.from_bytes(
        hashlib.sha256(f"{NAMESPACE}/primary/{index}".encode()).digest()[:4], "big"
    )


def reserve_seed(index: int) -> int:
    """Deterministic 1-to-1 uint32 reserve world seed for slot index in 0..95."""
    if index not in range(COUNT):
        raise ValueError(f"slot index out of range: {index}")
    return int.from_bytes(
        hashlib.sha256(f"{NAMESPACE}/reserve/{index}".encode()).digest()[:4], "big"
    )


def allocation() -> list[dict]:
    """Frozen 96-slot primary allocation."""
    return [{"index": i, "seed": primary_seed(i), "horizon": HORIZON} for i in range(COUNT)]


def reserve_allocation() -> list[dict]:
    """Frozen 1-to-1 96-slot reserve allocation; used only under strict MCAR rule."""
    return [{"index": i, "seed": reserve_seed(i), "horizon": HORIZON} for i in range(COUNT)]


def shard_indices(shard: int, shards: int = SHARDS) -> list[int]:
    """Fixed strided slot partition across 24 shards (4 pairs per shard)."""
    if shards != SHARDS or shard not in range(shards):
        raise ValueError(f"invalid V3 shard partition: shard={shard} shards={shards}")
    return [i for i in range(COUNT) if i % shards == shard]


@lru_cache(maxsize=1)
def source_paths() -> list[str]:
    """Static transitive research imports plus shipped product source trees."""
    import ast

    pending = list((ROOT / "research/watch6_search_efficacy_v3").glob("*.py"))
    pending.append(ROOT / "research/apex_veto_serving_20261001/schema.py")
    pending += list((ROOT / "src").rglob("*.py"))
    pending += list((ROOT / "web/backend").rglob("*.py"))
    found = set()
    while pending:
        path = pending.pop()
        if path in found or not path.is_file():
            continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                base = node.module
                if node.level:
                    package = ".".join(path.relative_to(ROOT).parent.parts)
                    parts = package.split(".")
                    base = ".".join(parts[: len(parts) - node.level + 1]) + "." + node.module
                modules = [base, *[base + "." + a.name for a in node.names]]
            for module in modules:
                if module.startswith("research."):
                    target = ROOT / module.replace(".", "/")
                    for candidate in (target.with_suffix(".py"), target / "__init__.py"):
                        if candidate.is_file() and candidate not in found:
                            pending.append(candidate)
    names = {str(path.relative_to(ROOT)) for path in found}
    names |= {
        "research/watch6_search_efficacy_v3",
        "research/bl_served_search/requirements.txt",
        "tools/ops/package_spec.py",
        "src",
        "web/backend",
        "configs",
        "pyproject.toml",
        "docs/research/final_run_20261019/SUCCESS_CONTRACT.md",
        "docs/research/final_run_20261019/P7_STATISTICS.md",
    }
    return sorted(names)


@lru_cache(maxsize=1)
def identity() -> str:
    """Bind science code, world config, checkpoint pins, registration, and seed banks."""
    paths = []
    for name in source_paths():
        path = ROOT / name
        paths.extend(
            [
                item
                for item in path.rglob("*")
                if item.suffix in {".py", ".sh", ".md", ".yaml", ".json", ".txt"}
            ]
            if path.is_dir()
            else [path]
        )
    values: dict[str, object] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(paths))
        if path.is_file()
        and "__pycache__" not in str(path)
        and str(path.relative_to(ROOT)) not in NON_SCIENCE_DOCS
    }
    values["namespace"] = NAMESPACE
    values["primary_allocation"] = allocation()
    values["reserve_allocation"] = reserve_allocation()
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
