"""Frozen protocol, fresh seed allocation, 6-runtime sharding, and identity for Watch6-L1-V1."""

from __future__ import annotations

import ast
import hashlib
import json
from functools import lru_cache
from pathlib import Path

from tools.ops.package_spec import digest

ROOT = Path(__file__).resolve().parents[2]
NAMESPACE = "t34-watch6-l1-confirm-20261101-v1"
COUNT = 96
HORIZON = 14400
SEARCH_HORIZON = 300
TRIGGER_INTERVAL = 5
MAX_COMPUTE_COMPLETIONS_PER_SHARD = 2

# Disjoint operational fixture seeds for Mac mirror/preflight and Colab canaries only;
# never admitted into any scientific ledger or final look.
FIXTURE_SEEDS: list[int] = [34910101, 34910102, 34910103]

ARMS: tuple[str, str, str] = ("bare", "l2_prime", "l1")
ARM_SPECS: dict[str, dict[str, int | str | None]] = {
    "bare": {
        "label": "bare-b8-s3",
        "min_length": None,
        "interval": TRIGGER_INTERVAL,
        "search_horizon": 0,
    },
    "l2_prime": {
        "label": "exact-stop-l2-prime",
        "min_length": 600,
        "interval": TRIGGER_INTERVAL,
        "search_horizon": SEARCH_HORIZON,
    },
    "l1": {
        "label": "exact-stop-l1",
        "min_length": 1,
        "interval": TRIGGER_INTERVAL,
        "search_horizon": SEARCH_HORIZON,
    },
}

# Cyclic 6-permutation Latin design: across N = 96 = 16 * 6 paired worlds, each
# arm execution order runs exactly 16 times (12 times across A100 0..71, 4 times across L4 72..95).
ARM_ORDER_PERMUTATIONS: tuple[tuple[str, str, str], ...] = (
    ("bare", "l2_prime", "l1"),
    ("bare", "l1", "l2_prime"),
    ("l2_prime", "bare", "l1"),
    ("l2_prime", "l1", "bare"),
    ("l1", "bare", "l2_prime"),
    ("l1", "l2_prime", "bare"),
)

# Registered 6-runtime heterogeneous single-wave sharding plan (18 shards = 96 worlds):
# - Shards 00..11 (12 shards): A100 W=6, 6 worlds/shard = 72 worlds (4 waves across 3 A100s)
# - Shards 12..17 (6 shards):  L4   W=4, 4 worlds/shard = 24 worlds (2 waves across <=3 L4s)
A100_SHARDS = 12
A100_WORKERS = 6
A100_WORLDS_PER_SHARD = 6

L4_SHARDS = 6
L4_WORKERS = 4
L4_WORLDS_PER_SHARD = 4

SHARDS = A100_SHARDS + L4_SHARDS  # 18
TARGET_GPUS: tuple[str, str] = ("A100", "L4")
SCRIPT = "research/watch6_l1_confirm/job.sh"

SCIENCE_PACKAGE_MODULES: tuple[str, ...] = (
    "research/watch6_l1_confirm/__init__.py",
    "research/watch6_l1_confirm/protocol.py",
    "research/watch6_l1_confirm/engine.py",
    "research/watch6_l1_confirm/run.py",
    "research/watch6_l1_confirm/analyze.py",
    "research/watch6_l1_confirm/costs.py",
    "research/watch6_l1_confirm/mirror.py",
    "research/watch6_l1_confirm/resume.py",
)

EXCLUDED_WEB_NON_SCIENCE: frozenset[str] = frozenset(
    {
        "web/backend/app.py",
        "web/backend/watch_replay.py",
        "web/backend/serialize.py",
    }
)


def registration() -> Path:
    """Confirmation pre-registration document bound into science identity."""
    return ROOT / "research/watch6_l1_confirm/REGISTRATION_L1_V1.md"


def primary_seed(index: int) -> int:
    """Deterministic uint32 primary world seed for slot index 0..95."""
    if type(index) is not int or index not in range(COUNT):
        raise ValueError(f"slot index out of range 0..{COUNT - 1}: {index!r}")
    return int.from_bytes(
        hashlib.sha256(f"{NAMESPACE}/primary/{index}".encode("utf-8")).digest()[:4],
        "big",
    )


def arm_order_for_index(index: int) -> list[str]:
    """Deterministic balanced 3-arm execution order for world index 0..95 (or fixture >=0)."""
    if type(index) is not int or index < 0:
        raise ValueError(f"world index must be a nonnegative int, got {index!r}")
    return list(ARM_ORDER_PERMUTATIONS[index % len(ARM_ORDER_PERMUTATIONS)])


def allocation() -> list[dict]:
    """Frozen 96-world primary allocation (zero reserve seed substitution)."""
    return [
        {
            "index": i,
            "seed": primary_seed(i),
            "horizon": HORIZON,
            "arm_order": arm_order_for_index(i),
        }
        for i in range(COUNT)
    ]


def shard_spec(shard: int) -> dict:
    """Return target GPU, worker count, and disjoint slot indices for shard 0..17."""
    if type(shard) is not int or shard not in range(SHARDS):
        raise ValueError(f"invalid Watch6-L1-V1 shard index: {shard!r} (expected 0..{SHARDS - 1})")
    if shard < A100_SHARDS:
        start = shard * A100_WORLDS_PER_SHARD
        indices = list(range(start, start + A100_WORLDS_PER_SHARD))
        return {
            "shard": shard,
            "gpu": "A100",
            "workers": A100_WORKERS,
            "indices": indices,
            "world_count": len(indices),
        }
    l4_offset = shard - A100_SHARDS
    start = A100_SHARDS * A100_WORLDS_PER_SHARD + l4_offset * L4_WORLDS_PER_SHARD
    indices = list(range(start, start + L4_WORLDS_PER_SHARD))
    return {
        "shard": shard,
        "gpu": "L4",
        "workers": L4_WORKERS,
        "indices": indices,
        "world_count": len(indices),
    }


def shard_indices(shard: int, shards: int = SHARDS) -> list[int]:
    """Disjoint slot indices owned by shard 0..17."""
    if shards != SHARDS:
        raise ValueError(f"Watch6-L1-V1 freezes SHARDS={SHARDS}, got {shards!r}")
    return list(shard_spec(shard)["indices"])


def owner_shard_for_index(index: int) -> int:
    """Unique shard 0..17 that owns world slot index 0..95."""
    if type(index) is not int or index not in range(COUNT):
        raise ValueError(f"slot index out of range 0..{COUNT - 1}: {index!r}")
    a100_total = A100_SHARDS * A100_WORLDS_PER_SHARD
    if index < a100_total:
        return index // A100_WORLDS_PER_SHARD
    return A100_SHARDS + (index - a100_total) // L4_WORLDS_PER_SHARD


def _walk_research_imports(seed_files: list[Path], *, allow_colab_cli: bool) -> set[Path]:
    pending = list(seed_files)
    found: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in found or not path.is_file():
            continue
        found.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                base = node.module
                if node.level:
                    package = ".".join(path.relative_to(ROOT).parent.parts)
                    parts = package.split(".")
                    prefix = ".".join(parts[: len(parts) - node.level + 1])
                    base = f"{prefix}.{node.module}" if prefix else node.module
                modules = [base, *[f"{base}.{alias.name}" for alias in node.names]]
            for module in modules:
                if not module.startswith("research."):
                    continue
                if not allow_colab_cli and module.startswith("research.colab_cli"):
                    continue
                target = ROOT / module.replace(".", "/")
                for candidate in (target.with_suffix(".py"), target / "__init__.py"):
                    if candidate.is_file() and candidate not in found:
                        pending.append(candidate)
    return found


@lru_cache(maxsize=1)
def source_paths() -> list[str]:
    """Sparse-checkout closure required by colab_cli/source.py on Colab runtimes."""
    seed_files = [
        *sorted((ROOT / "research/watch6_l1_confirm").glob("*.py")),
        ROOT / "research/apex_veto_serving_20261001/schema.py",
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((ROOT / "web/backend").rglob("*.py")),
    ]
    found = _walk_research_imports(seed_files, allow_colab_cli=False)
    names = {str(path.relative_to(ROOT)) for path in found}
    names |= {
        "research/watch6_l1_confirm",
        "research/bl_served_search/requirements.txt",
        "research/t34_exit_b8/PUBLIC_INPUTS.json",
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
def science_paths() -> list[str]:
    """Exact scientific file set hashed into identity() (decoupled from ops/colab_cli/replay UI)."""
    seed_files = [ROOT / rel for rel in SCIENCE_PACKAGE_MODULES]
    seed_files.append(ROOT / "research/apex_veto_serving_20261001/schema.py")
    seed_files.extend(
        path for path in sorted((ROOT / "src").rglob("*.py")) if "__pycache__" not in path.parts
    )
    seed_files.extend(
        path
        for path in sorted((ROOT / "web/backend").rglob("*.py"))
        if "__pycache__" not in path.parts
        and str(path.relative_to(ROOT)) not in EXCLUDED_WEB_NON_SCIENCE
    )
    found = _walk_research_imports(seed_files, allow_colab_cli=False)
    rels = {
        str(path.relative_to(ROOT))
        for path in found
        if not str(path.relative_to(ROOT)).startswith(("tools/ops/", "research/colab_cli/"))
        and str(path.relative_to(ROOT)) not in EXCLUDED_WEB_NON_SCIENCE
    }
    # Exclude operational-only modules in research/watch6_l1_confirm from science identity.
    rels = {
        rel
        for rel in rels
        if not rel.startswith("research/watch6_l1_confirm/") or rel in SCIENCE_PACKAGE_MODULES
    }
    rels |= {
        SCRIPT,
        str(registration().relative_to(ROOT)),
        "configs/mechanics_v2.yaml",
        "web/backend/served_checkpoint_pins.json",
        "docs/research/final_run_20261019/SUCCESS_CONTRACT.md",
        "docs/research/final_run_20261019/P7_STATISTICS.md",
    }
    return sorted(rels)


def compute_identity() -> str:
    """Compute SHA-256 over science code, world config, pins, registration, and 96-world bank."""
    values: dict[str, object] = {
        rel: digest(ROOT / rel)
        for rel in science_paths()
        if (ROOT / rel).is_file() and not (ROOT / rel).is_symlink()
    }
    values["namespace"] = NAMESPACE
    values["arms"] = ARM_SPECS
    values["primary_allocation"] = allocation()
    values["shard_layout"] = [shard_spec(s) for s in range(SHARDS)]
    values["max_compute_completions_per_shard"] = MAX_COMPUTE_COMPLETIONS_PER_SHARD
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def identity() -> str:
    """Cached science identity for current process."""
    return compute_identity()
