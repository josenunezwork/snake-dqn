"""Fail-closed runtime adapter; no allocator and no Colab API calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def resolve_inputs(value: Any, inputs: Path) -> Any:
    """Resolve explicitly tagged input-relative filenames, refusing traversal."""
    if isinstance(value, dict):
        return {key: resolve_inputs(child, inputs) for key, child in value.items()}
    if isinstance(value, list):
        return [resolve_inputs(child, inputs) for child in value]
    if isinstance(value, str) and value.startswith("input:"):
        relative = Path(value[6:])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe input-relative path")
        path = (inputs / relative).resolve(strict=True)
        if not path.is_relative_to(inputs.resolve()):
            raise ValueError("input path escapes upload root")
        return str(path)
    if isinstance(value, str) and (value.startswith("/Users/") or "/drive/" in value):
        raise ValueError("runtime plan contains a Mac-only or Drive path")
    return value


def execute(plan_path: Path, out: Path) -> dict[str, Any]:
    """Refuse readiness placeholders before importing gameplay or touching outputs."""
    plan = json.loads(plan_path.read_text())
    if plan.get("ready") is not True:
        raise ValueError(f"NOT_DISPATCHABLE: {plan.get('blocking_reasons', ['inputs unbound'])}")
    experiment = plan.get("experiment")
    if experiment not in {"e6", "e7"}:
        raise ValueError("unknown experiment")
    plan = resolve_inputs(plan, plan_path.parent)
    if experiment == "e6":
        from research.bl_prefix.e6 import run
    else:
        from research.bl_prefix.e7 import run
    return run(plan, out, preflight=plan.get("preflight") is True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.plan, args.out), sort_keys=True))


if __name__ == "__main__":
    main()
