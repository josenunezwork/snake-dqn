"""Pre-probe audit for the registered T3 development-world banks.

The receipt intentionally excludes its own output directory from observed-artifact
scans.  That makes a saved, pre-first-probe receipt reproducible after probes add
round-62 rows, instead of failing by detecting those rows as historical use.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_digest(seeds: Iterable[int]) -> str:
    """Return a stable digest for a seed inventory retained outside the receipt."""
    return hashlib.sha256(",".join(str(seed) for seed in sorted(set(seeds))).encode()).hexdigest()


def _seed_values(value: Any) -> set[int]:
    """Collect only explicit world-seed fields from a decoded JSON value."""
    found: set[int] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"seed", "world_seed"} and isinstance(child, int) and child >= 0:
                found.add(int(child))
            elif key == "world_seeds" and isinstance(child, list):
                found.update(int(seed) for seed in child if isinstance(seed, int) and seed >= 0)
            elif key in {
                "mass",
                "deaths",
                "alive_frames",
                "death_frame",
                "length_at_death",
                "death_lengths",
                "causes",
            } and isinstance(child, Mapping):
                found.update(int(seed) for seed in child if str(seed).isdigit())
            else:
                found.update(_seed_values(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_seed_values(child))
    return found


def _json_inventory(root: Path, *, exclude: Path | None = None) -> dict[str, Any]:
    """Inventory explicit seed fields from JSON receipts without reading binary banks."""
    if not root.is_dir():
        return {"exists": False, "file_count": 0, "seed_count": 0, "seeds": [], "digest": None}
    paths = []
    for path in sorted((*root.rglob("*.json"), *root.rglob("*.jsonl"))):
        if exclude is not None and (path == exclude or exclude in path.parents):
            continue
        paths.append(path)
    seeds: set[int] = set()
    valid = 0
    digest = hashlib.sha256()
    for path in paths:
        raw = path.read_bytes()
        digest.update(str(path.relative_to(root)).encode() + b":" + hashlib.sha256(raw).digest())
        try:
            if path.suffix == ".jsonl":
                records = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
            else:
                records = [json.loads(raw)]
            for record in records:
                seeds.update(_seed_values(record))
            valid += 1
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return {
        "exists": True,
        "file_count": len(paths),
        "valid_json_count": valid,
        "seed_count": len(seeds),
        "seeds": sorted(seeds),
        "digest": digest.hexdigest(),
    }


def _compact_inventory(item: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the receipt small while retaining an inventory identity."""
    compact = {key: value for key, value in item.items() if key != "seeds"}
    compact["seed_digest"] = _seed_digest(item.get("seeds", []))
    return compact


def _manifest_binding(bank: Path, namespace_path: Path) -> dict[str, Any]:
    """Verify every manifest file digest, including the namespace declaration."""
    manifest_path = bank / "manifest.json"
    if not manifest_path.is_file():
        return {"ok": False, "reason": "manifest.json missing"}
    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("files")
    if manifest.get("schema") != "t3-label-bank/v1" or not isinstance(files, Mapping) or not files:
        return {"ok": False, "reason": "invalid manifest schema or files map"}
    namespace_key = str(namespace_path.relative_to(bank))
    if namespace_key not in files:
        return {"ok": False, "reason": "namespace absent from manifest files map"}
    mismatches = []
    for relative, expected in files.items():
        path = bank / str(relative)
        if not path.is_file() or _sha256(path) != expected:
            mismatches.append(str(relative))
    return {
        "ok": not mismatches,
        "manifest_sha256": _sha256(manifest_path),
        "file_count": len(files),
        "namespace_manifest_sha256": files[namespace_key],
        "mismatches": mismatches[:20],
    }


def _registered_t2(repo: Path) -> dict[str, Any]:
    from research.t2_screen_20261015 import banks as t2_banks

    return t2_banks.seed_bank_audit(repo)


def seed_bank_audit(
    repo: str | Path = ".",
    artifacts: str | Path | None = None,
    exclude_out: str | Path | None = None,
) -> dict[str, Any]:
    """Prove registered rounds 62/63 are unused in the declared pre-probe scope.

    Scope is bounded to the registered NI/distillation banks, observed JSON probe
    artifacts, and the manifest-bound T3 label-training namespace.  It does not
    claim a search of arbitrary binary files or unrecorded external experiments.
    """
    from research.redesign_m2_20261008 import ni_spec, ni_spec_m2b

    repo_path = Path(repo).resolve()
    artifact_root = (
        Path(artifacts).resolve()
        if artifacts is not None
        else repo_path.parent / "snake-dqn-artifacts"
    )
    output = Path(exclude_out).resolve() if exclude_out is not None else None
    t2 = _registered_t2(repo_path)
    expected = {
        str(round_index): set(ni_spec.distill_seeds(96, round_index=round_index))
        for round_index in (62, 63)
    }
    probe_roots = sorted(path for path in artifact_root.glob("*probe*") if path.is_dir())
    observed = {str(path): _json_inventory(path, exclude=output) for path in probe_roots}
    label_namespace = artifact_root / "t3-rollout-labels-20261015/bank/namespace.json"
    labels = (
        _json_inventory(label_namespace.parent)
        if label_namespace.is_file()
        else _json_inventory(label_namespace)
    )
    label_manifest_ok = False
    binding: dict[str, Any] = {"ok": False, "reason": "namespace.json missing"}
    if label_namespace.is_file():
        namespace = json.loads(label_namespace.read_text())
        declared = {int(seed) for seed in namespace.get("seeds", []) if isinstance(seed, int)}
        labels = {
            "exists": True,
            "path": str(label_namespace),
            "sha256": _sha256(label_namespace),
            "schema": namespace.get("schema"),
            "declared_seed_count": len(declared),
            "declared_seed_unique": len(declared) == len(namespace.get("seeds", [])),
            "seeds": sorted(declared),
        }
        binding = _manifest_binding(label_namespace.parent, label_namespace)
        label_manifest_ok = (
            namespace.get("schema") == "t3-label-namespace/v1" and bool(declared) and binding["ok"]
        )
    observed_seeds = set().union(
        *(set(item["seeds"]) for item in observed.values()), set(labels["seeds"])
    )
    registered_seeds: set[int] = set()
    ni = set(ni_spec.ni_seeds()) | set(ni_spec_m2b.ni_seeds())
    registered_seeds.update(ni)
    for values in t2["registered_rounds"].values():
        registered_seeds.update(values)
    for item in t2["historical_distillation"].values():
        registered_seeds.update(item["seeds"])
    checks = {}
    for round_index, seeds in expected.items():
        other_expected = set().union(
            *(other for key, other in expected.items() if key != round_index)
        )
        checks[round_index] = {
            "count": len(seeds),
            "unique": len(seeds) == 96,
            "registered_collisions": sorted(seeds & registered_seeds),
            "observed_probe_or_label_collisions": sorted(seeds & observed_seeds),
            "other_t3_round_collisions": sorted(seeds & other_expected),
        }
    passed = (
        t2["pass"]
        and label_manifest_ok
        and all(
            check["unique"]
            and not check["registered_collisions"]
            and not check["observed_probe_or_label_collisions"]
            and not check["other_t3_round_collisions"]
            for check in checks.values()
        )
    )
    return {
        "schema": "t3-seed-bank-audit/v1",
        "status": "PASS" if passed else "FAIL",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "registered NI/distillation, JSON probe artifacts, and manifest-bound "
            "T3 label namespace"
        ),
        "excluded_output": str(output) if output is not None else None,
        "t2_registered_audit": {"pass": t2["pass"], "schema": t2["schema"]},
        "registered_ni": {"count": len(ni), "seed_digest": _seed_digest(ni)},
        "expected_rounds": {
            round_index: {
                "count": len(seeds),
                "seed_digest": _seed_digest(seeds),
                "seeds": sorted(seeds),
            }
            for round_index, seeds in expected.items()
        },
        "registered_t2": {
            "rounds": {
                name: {"count": len(values), "seed_digest": _seed_digest(values)}
                for name, values in t2["registered_rounds"].items()
            },
            "historical": {
                name: _compact_inventory(item)
                for name, item in t2["historical_distillation"].items()
            },
        },
        "observed_probe_artifacts": {
            name: _compact_inventory(item) for name, item in observed.items()
        },
        "t3_rollout_training_namespace": {
            **_compact_inventory(labels),
            "manifest_binding": binding,
        },
        "checks": checks,
        "pass": passed,
        "limit": (
            "No claim is made about unrecorded experiments or arbitrary binary files outside "
            "this scope."
        ),
    }


def main() -> int:
    print(json.dumps(seed_bank_audit(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
