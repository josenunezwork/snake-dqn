"""Frozen exploration allocation, arm specs, decision rule, and dependency closure."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from tools.ops.package_spec import unit_id

LANE = "exploration"
SCHEMA = "watch6-alllen-screen-contract/v1"
UNIT_SCHEMA = "watch6-alllen-screen-unit/v1"
UNIT_KIND = "paired-watch6-session"

NAMESPACE = "t34-explore-watch6-alllen-screen-20261031-r1"
CANARY_NAMESPACE = "t34-explore-watch6-alllen-canary-20261031-r1"
PREFLIGHT_NAMESPACE = "t34-explore-watch6-alllen-preflight-20261031-r1"
BRANCH_BENCHMARK_SEEDS = (91852001, 91852002, 91852003)

COUNT = 48
HORIZON = 6000
CANARY_COUNT = 1
CANARY_HORIZON = 120
PREFLIGHT_COUNT = 2
PREFLIGHT_HORIZON = 12

ADOPT_MASS_DELTA_MIN = 15.0
ADOPT_DEATH_DELTA_PER_1K_MAX = 0.0
DECISION_RULE = (
    "Adopt L>=1 exact-stop for new Watch replay banks iff paired mean incremental mass "
    "delta (exact_stop_alllen - l2_prime_600) >= +15.0 mass AND paired mean deaths/1k "
    "delta (exact_stop_alllen - l2_prime_600) <= 0.0 (no worse)."
)

ARM_SPECS: dict[str, dict[str, Any]] = {
    "bare": {
        "min_length": None,
        "interval": 5,
        "label": "B (bare B8 s3)",
    },
    "l2_prime_600": {
        "min_length": 600,
        "interval": 5,
        "label": "L2' exact-stop at L>=600 (served)",
    },
    "exact_stop_alllen": {
        "min_length": 1,
        "interval": 5,
        "label": "exact-stop at L>=1 (all lengths)",
    },
}
ARMS = tuple(ARM_SPECS.keys())

ARM_ORDER_PERMUTATIONS: tuple[tuple[str, str, str], ...] = (
    ("bare", "l2_prime_600", "exact_stop_alllen"),
    ("l2_prime_600", "exact_stop_alllen", "bare"),
    ("exact_stop_alllen", "bare", "l2_prime_600"),
    ("bare", "exact_stop_alllen", "l2_prime_600"),
    ("l2_prime_600", "bare", "exact_stop_alllen"),
    ("exact_stop_alllen", "l2_prime_600", "bare"),
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "research/watch6_alllen_screen/job.sh"
PACKAGE = "research/watch6_alllen_screen"


def canonical_bytes(payload: Any) -> bytes:
    """Deterministic UTF-8 JSON encoding for contract hashing."""
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def seed_for(namespace: str, index: int) -> int:
    """Deterministic 32-bit unsigned world seed from namespace and 0-based index."""
    if index < 0:
        raise ValueError("allocation index must be non-negative")
    return int.from_bytes(hashlib.sha256(f"{namespace}/{index}".encode()).digest()[:4], "big")


def arm_order_for_index(index: int) -> tuple[str, str, str]:
    """Balanced cyclic/transposed arm execution order across the 48 paired worlds."""
    return ARM_ORDER_PERMUTATIONS[index % len(ARM_ORDER_PERMUTATIONS)]


def allocation(
    namespace: str = NAMESPACE,
    count: int = COUNT,
    horizon: int = HORIZON,
    *,
    audit_live_mirror: bool = False,
) -> dict[str, Any]:
    """Build a frozen 3-arm Watch6-v2 exploration screen contract."""
    unit_id(namespace, UNIT_KIND, 0)
    if not 1 <= count <= 128:
        raise ValueError("invalid session count")
    if not 1 <= horizon <= 14400:
        raise ValueError("invalid horizon")
    return {
        "schema": SCHEMA,
        "lane": LANE,
        "world": "watch6-mechanics-v2-continuous-respawn",
        "namespace": namespace,
        "count": count,
        "horizon": horizon,
        "arms": list(ARMS),
        "audit_live_mirror": bool(audit_live_mirror),
        "rows": [
            {
                "index": i,
                "seed": seed_for(namespace, i),
            }
            for i in range(count)
        ],
    }


def contract_identity(contract: dict[str, Any]) -> dict[str, str]:
    """Bind every contract field (including commit) in the verified resume ledger."""
    return {"contract_sha256": hashlib.sha256(canonical_bytes(contract)).hexdigest()}


@lru_cache(maxsize=1)
def source_paths() -> list[str]:
    """Static transitive research imports plus the shipped product source trees."""
    import ast

    pending = list((ROOT / PACKAGE).glob("*.py"))
    pending.append(ROOT / "research/apex_veto_serving_20261001/schema.py")
    pending += list((ROOT / "src").rglob("*.py"))
    pending += list((ROOT / "web/backend").rglob("*.py"))
    found = set()
    while pending:
        path = pending.pop()
        if path in found or not path.is_file():
            continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                base = node.module
                if node.level:
                    package = ".".join(path.relative_to(ROOT).parent.parts)
                    parts = package.split(".")
                    base = ".".join(parts[: len(parts) - node.level + 1]) + "." + node.module
                modules = [base, *[base + "." + alias.name for alias in node.names]]
            for module in modules:
                if module.startswith("research."):
                    target = ROOT / module.replace(".", "/")
                    for candidate in (target.with_suffix(".py"), target / "__init__.py"):
                        if candidate.is_file() and candidate not in found:
                            pending.append(candidate)
    names = {str(path.relative_to(ROOT)) for path in found}
    names |= {
        PACKAGE,
        "research/watch_replay",
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
