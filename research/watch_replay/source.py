"""Sparse static dependency closure, adopted from Watch6 efficacy packaging."""

from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def source_paths() -> list[str]:
    """Static transitive research imports plus the shipped product source trees."""
    import ast

    pending = list((ROOT / "research/watch_replay").glob("*.py"))
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
