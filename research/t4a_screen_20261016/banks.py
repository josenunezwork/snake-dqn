"""Fresh-world audit for the T4a evaluation rounds.

The audit deliberately scans a bounded, recorded scope.  In particular, it
excludes the output directory being prepared so a saved receipt remains valid
after the watcher writes its own round-65/66 rows.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.t3_screen_20261015 import banks as prior_banks


def seed_bank_audit(
    repo: str | Path = ".",
    artifacts: str | Path | None = None,
    exclude_out: str | Path | None = None,
) -> dict[str, Any]:
    """Audit reserved rounds 65/66 against the registered and observed scope."""
    from research.redesign_m2_20261008 import ni_spec, ni_spec_m2b

    repo_path = Path(repo).resolve()
    artifact_root = (
        Path(artifacts).resolve()
        if artifacts is not None
        else repo_path.parent / "snake-dqn-artifacts"
    )
    output = Path(exclude_out).resolve() if exclude_out is not None else None
    t2 = prior_banks._registered_t2(repo_path)
    expected = {
        str(round_index): set(ni_spec.distill_seeds(96, round_index=round_index))
        for round_index in (65, 66)
    }
    registered = set(ni_spec.ni_seeds()) | set(ni_spec_m2b.ni_seeds())
    for values in t2["registered_rounds"].values():
        registered.update(values)
    for item in t2["historical_distillation"].values():
        registered.update(item["seeds"])
    for round_index in (62, 63, 64):
        registered.update(ni_spec.distill_seeds(96, round_index=round_index))

    roots = [repo_path / "research"] + sorted(
        path
        for path in artifact_root.iterdir()
        if path.is_dir()
        and any(token in path.name for token in ("probe", "eval", "strict", "gate"))
    )
    inventories = {str(root): prior_banks._json_inventory(root, exclude=output) for root in roots}
    observed = set().union(*(set(item["seeds"]) for item in inventories.values()))
    namespace_path = artifact_root / "t3-rollout-labels-20261015/bank/namespace.json"
    binding: dict[str, Any] = {"ok": False, "reason": "namespace.json missing"}
    namespace: dict[str, Any] = {"exists": False, "seeds": []}
    if namespace_path.is_file():
        raw = json.loads(namespace_path.read_text())
        seeds = [int(seed) for seed in raw.get("seeds", []) if type(seed) is int]
        binding = prior_banks._manifest_binding(namespace_path.parent, namespace_path)
        namespace = {
            "exists": True,
            "schema": raw.get("schema"),
            "count": len(seeds),
            "unique": len(seeds) == len(set(seeds)),
            "seed_digest": prior_banks._seed_digest(seeds),
            "manifest_binding": binding,
        }
        observed.update(seeds)
    checks = {}
    for round_index, seeds in expected.items():
        other = set().union(*(value for key, value in expected.items() if key != round_index))
        checks[round_index] = {
            "count": len(seeds),
            "unique": len(seeds) == 96,
            "seed_digest": prior_banks._seed_digest(seeds),
            "seeds": sorted(seeds),
            "registered_collisions": sorted(seeds & registered),
            "observed_json_or_label_collisions": sorted(seeds & observed),
            "other_reserved_round_collisions": sorted(seeds & other),
        }
    namespace_valid = (
        namespace.get("schema") == "t3-label-namespace/v1"
        and bool(namespace.get("count"))
        and namespace.get("unique") is True
    )
    passed = (
        t2["pass"]
        and binding["ok"]
        and namespace_valid
        and all(
            item["unique"]
            and not item["registered_collisions"]
            and not item["observed_json_or_label_collisions"]
            and not item["other_reserved_round_collisions"]
            for item in checks.values()
        )
    )
    return {
        "schema": "t4a-seed-bank-audit/v1",
        "status": "PASS" if passed else "FAIL",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "registered NI/distillation rounds, reserved rounds 62-64, repository research JSON, "
            "artifact probe/eval/strict/gate JSON, and immutable T3 label namespace"
        ),
        "excluded_output": str(output) if output is not None else None,
        "registered_t2_audit": {"pass": t2["pass"], "schema": t2["schema"]},
        "t3_rollout_training_namespace": namespace,
        "inventories": {
            name: prior_banks._compact_inventory(item) for name, item in inventories.items()
        },
        "checks": checks,
        "pass": passed,
        "limit": (
            "No claim is made about unrecorded experiments, arbitrary binary files, or external "
            "systems."
        ),
    }
