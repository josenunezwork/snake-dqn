"""Pre-first-probe round-64 audit, using the T3 inventory and manifest contracts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.redesign_m2_20261008 import ni_spec, ni_spec_m2b
from research.t2_screen_20261015.banks import seed_bank_audit as t2_audit
from research.t3_screen_20261015.banks import (
    _compact_inventory,
    _json_inventory,
    _manifest_binding,
    _seed_digest,
    _sha256,
)


def audit(repo: Path, artifacts: Path, output: Path) -> dict[str, Any]:
    """Check observed JSON worlds, registered banks, and all T3 label namespaces."""
    expected = set(ni_spec.distill_seeds(32, round_index=64))
    registered = t2_audit(repo)
    used = set(ni_spec.ni_seeds()) | set(ni_spec_m2b.ni_seeds())
    for values in registered["registered_rounds"].values():
        used.update(values)
    for item in registered["historical_distillation"].values():
        used.update(item["seeds"])
    reserved = set().union(*(set(ni_spec.distill_seeds(96, round_index=r)) for r in (62, 63)))
    roots = sorted(p for p in artifacts.glob("*probe*") if p.is_dir() and p != output)
    inventories = {str(p): _json_inventory(p, exclude=output) for p in roots}
    # Repo receipts include historical probes not retained in the external artifact tree.
    inventories[str(repo / "research")] = _json_inventory(
        repo / "research", exclude=repo / "research/t4_curve_20261015"
    )
    observed = set().union(*(set(item["seeds"]) for item in inventories.values()))
    namespaces = {}
    label_seeds: set[int] = set()
    paths = sorted((artifacts / "t3-rollout-labels-20261015").rglob("namespace.json"))
    for path in paths:
        namespace = json.loads(path.read_text())
        seeds = namespace.get("seeds", [])
        valid_namespace = (
            namespace.get("schema") == "t3-label-namespace/v1"
            and isinstance(seeds, list)
            and bool(seeds)
            and all(type(seed) is int and 0 <= seed < 2**32 for seed in seeds)
        )
        if not valid_namespace:
            raise ValueError(f"Invalid or empty T3 label namespace: {path}")
        binding = _manifest_binding(path.parent, path)
        namespaces[str(path)] = dict(
            sha256=_sha256(path),
            count=len(seeds),
            unique=len(set(seeds)) == len(seeds),
            seed_digest=_seed_digest(seeds),
            binding=binding,
        )
        label_seeds.update(seeds)
    checks = dict(
        unique=len(expected) == 32,
        registered_collisions=sorted(expected & used),
        reserved_62_63_collisions=sorted(expected & reserved),
        observed_collisions=sorted(expected & observed),
        label_collisions=sorted(expected & label_seeds),
    )
    passed = (
        registered["pass"]
        and checks["unique"]
        and not any(checks[k] for k in checks if k.endswith("collisions"))
        and bool(namespaces)
        and all(n["unique"] and n["binding"]["ok"] for n in namespaces.values())
    )
    return dict(
        schema="t4-curve-seed-audit/v1",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        round_index=64,
        worlds=32,
        seeds=sorted(expected),
        seed_digest=_seed_digest(expected),
        status="PASS" if passed else "FAIL",
        **{"pass": passed},
        checks=checks,
        registered_t2=registered,
        label_namespaces=namespaces,
        inventories={key: _compact_inventory(item) for key, item in inventories.items()},
        excluded_output=str(output),
        scope="Registered NI/distillation, reserved T3 62/63, observed JSON probes and "
        "repository research receipts, manifest-bound T3 label namespaces.",
        limitation="No claim about unrecorded experiments or arbitrary binary files.",
    )


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    artifacts = repo.parent / "snake-dqn-artifacts"
    output = artifacts / "redesign-t3-curve-probes-20261015"
    receipt = audit(repo, artifacts, output)
    if not receipt["pass"]:
        print(json.dumps(receipt, indent=2))
        return 1
    for path in (
        output / "seed_bank_audit.json",
        repo / "research/t4_curve_20261015/results/seed_bank_audit.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: receipt[key] for key in ("status", "checks", "seed_digest")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
