"""Frozen allocation and identity; fixtures never enter the scientific bank."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

NAMESPACE = "t34-watch6-exact-stop-efficacy-20261030-v1"
COUNT = 24
HORIZON = 14400
FIXTURE_SEEDS = [34890123, 34890124, 34890125]
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "research/watch6_search_efficacy/job.sh"


def allocation() -> list[dict]:
    """Hash the namespace and frozen index, without outcome-dependent skipping."""
    return [
        {
            "index": i,
            "seed": int.from_bytes(hashlib.sha256(f"{NAMESPACE}/{i}".encode()).digest()[:4], "big"),
            "horizon": HORIZON,
        }
        for i in range(COUNT)
    ]


@lru_cache(maxsize=1)
def source_paths() -> list[str]:
    """Static transitive research imports plus the shipped product source trees."""
    import ast

    pending = list((ROOT / "research/watch6_search_efficacy").glob("*.py"))
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
                modules = [node.module, *[node.module + "." + a.name for a in node.names]]
            for module in modules:
                if module.startswith("research."):
                    target = ROOT / module.replace(".", "/")
                    for candidate in (target.with_suffix(".py"), target / "__init__.py"):
                        if candidate.is_file() and candidate not in found:
                            pending.append(candidate)
    names = {str(path.relative_to(ROOT)) for path in found}
    names |= {
        "research/watch6_search_efficacy",
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
    """Bind science code, world configuration, checkpoints pins and adopted rules."""
    paths = []
    for name in source_paths():
        path = ROOT / name
        paths.extend(
            [
                p
                for p in path.rglob("*")
                if p.suffix in {".py", ".sh", ".md", ".yaml", ".json", ".txt"}
            ]
            if path.is_dir()
            else [path]
        )
    values = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(paths))
        if path.is_file() and "__pycache__" not in str(path)
    }
    values["allocation"] = allocation()
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
