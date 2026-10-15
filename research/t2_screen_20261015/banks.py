"""Audit the registered T2 probe bank against historical distillation worlds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Sequence


def _manifest_seeds(root: Path, rounds: Sequence[int]) -> Dict[str, Any]:
    """Read generated shard manifests and return per-round seed inventories."""
    out: Dict[str, Any] = {}
    for round_index in rounds:
        paths = sorted(root.glob(f"round-{round_index}/chunk-*.json"))
        seeds = []
        for path in paths:
            data = json.loads(path.read_text())
            seeds.extend(int(seed) for seed in data.get("world_seeds", []))
        out[str(round_index)] = {
            "manifest_count": len(paths),
            "seed_count": len(seeds),
            "unique_seed_count": len(set(seeds)),
            "seeds": sorted(set(seeds)),
            "manifest_digest": hashlib.sha256(
                "\n".join(
                    f"{path.name}:{hashlib.sha256(path.read_bytes()).hexdigest()}" for path in paths
                ).encode()
            ).hexdigest(),
        }
    return out


def seed_bank_audit(repo: str | Path = ".") -> Dict[str, Any]:
    """Verify T2 rounds 60/61 are disjoint from all registered NI/distill worlds.

    Historical generated shards are the authoritative observed bank. Empty rounds are
    reported as empty rather than silently treated as complete.
    """
    from research.redesign_m2_20261008 import ni_spec, ni_spec_m2b

    root = Path(repo)
    m2 = root / "research/redesign_m2_20261008/results/data_manifests"
    m2b = root / "research/redesign_m2_20261008/results_m2b/data_manifests"
    historical = _manifest_seeds(m2, (0, 1, 2))
    historical.update({f"{r}": item for r, item in _manifest_seeds(m2b, (10, 11, 12)).items()})
    historical_seeds = set().union(*(set(v["seeds"]) for v in historical.values()))
    ni = set(ni_spec.ni_seeds()) | set(ni_spec_m2b.ni_seeds())
    rounds = {str(r): set(ni_spec.distill_seeds(96, round_index=r)) for r in (50, 51, 52, 60, 61)}
    checks = {
        str(r): {
            "count": len(seeds),
            "unique": len(seeds) == 96,
            "collisions": [],
        }
        for r, seeds in rounds.items()
    }
    # Compare explicitly to every other registered bank so the result is useful to
    # callers and does not report a round colliding with itself.
    for r, seeds in rounds.items():
        others = ni | historical_seeds | set().union(*(v for k, v in rounds.items() if k != r))
        checks[r]["collisions"] = sorted(seeds & others)
    return {
        "schema": "t2-seed-bank-audit/v1",
        "historical_distillation": historical,
        "registered_ni_count": len(ni),
        "registered_rounds": {r: sorted(v) for r, v in rounds.items()},
        "checks": checks,
        "historical_complete": all(
            item["manifest_count"] > 0 and item["seed_count"] > 0 for item in historical.values()
        ),
        "pass": all(c["unique"] and not c["collisions"] for c in checks.values())
        and all(
            item["manifest_count"] > 0 and item["seed_count"] > 0 for item in historical.values()
        ),
    }


def main() -> int:
    print(json.dumps(seed_bank_audit(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
