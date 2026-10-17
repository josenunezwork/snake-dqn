"""Fresh, purpose-labelled seed allocation for the T15 mixed Watch qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from research.redesign_t7_phase_r_20261016 import seed_audit as t7
from research.t4e_return_20261016 import seed_audit as scanner
from research.t12_b8_serving_20261017 import seed_audit as t12

NAMESPACE = "t15-mixed-roster-20261017/v1"
SCHEMA = "t15-mixed-roster-seed-audit/v1"
PURPOSE_COUNTS = {"watch": 25, "parity": 3, "latency": 3, "ui": 1, "controls": 1}
SMOKE_NAMESPACE = "t15-mixed-roster-20261017/smoke-v1"


def sha256(path: Path) -> str:
    """Return the digest of one regular file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_digest(values: Iterable[int]) -> str:
    return hashlib.sha256(",".join(map(str, sorted(set(values)))).encode()).hexdigest()


def seed_for(purpose: str, index: int, namespace: str = NAMESPACE) -> int:
    if purpose not in PURPOSE_COUNTS or not 0 <= index < PURPOSE_COUNTS[purpose]:
        raise ValueError(f"invalid T15 seed coordinate: {purpose}/{index}")
    return int.from_bytes(
        hashlib.sha256(f"{namespace}|{purpose}|{index}".encode()).digest()[:4], "big"
    )


def generate_banks(namespace: str = NAMESPACE) -> dict[str, Any]:
    banks = {
        key: [seed_for(key, index, namespace) for index in range(count)]
        for key, count in PURPOSE_COUNTS.items()
    }
    all_values = [value for values in banks.values() for value in values]
    if len(set(all_values)) != len(all_values):
        raise RuntimeError("T15 generated seed collision")
    return {
        "namespace": namespace,
        "recipe": "uint32_be(sha256(f'{namespace}|{purpose}|{i}')[:4])",
        "banks": {
            key: {"count": len(values), "seeds": values, "seed_digest": seed_digest(values)}
            for key, values in banks.items()
        },
        "count": len(all_values),
        "seed_digest": seed_digest(all_values),
    }


def _prior_receipt(path: Path) -> tuple[dict[str, Any], Path]:
    value = json.loads(path.read_text())
    if value.get("pass") is not True:
        raise ValueError(f"prior seed receipt is not passing: {path}")
    return value, path


def audit(repo: str | Path = ".", artifacts: str | Path | None = None) -> dict[str, Any]:
    """Audit all registered and visible JSON/JSONL seed identities without gameplay."""
    repo_path = Path(repo).resolve()
    artifact_root = (
        Path(artifacts).resolve() if artifacts else repo_path.parent / "snake-dqn-artifacts"
    )
    source = Path(__file__).resolve()
    source_hash = sha256(source)
    generated = generate_banks()
    candidate = {v for row in generated["banks"].values() for v in row["seeds"]}
    t7_receipt, t7_binding = t12._t7_receipt(repo_path)
    allocations, allocated = t12._t7_allocations(repo_path)
    registered = t7._registered(repo_path) | allocated
    mandatory = t7_receipt.get("mandatory_evidence", {})
    namespace_path = Path(mandatory["t3_namespace"]["path"])
    registered.update(json.loads(namespace_path.read_text())["seeds"])
    for name in ("m3_phase_r", "m4_phase_r"):
        plan = json.loads(Path(mandatory[name]["path"]).read_text())
        for values in plan["banks"].values():
            registered.update(values)
    bindings: dict[str, str] = {}
    # T12 remains the mandatory registered boundary; T12B and both T13 receipts add later banks.
    prior_paths = (
        repo_path / "research/t12_b8_serving_20261017/SEED_AUDIT_T12.json",
        repo_path / "research/t12_b8_serving_20261017/SEED_AUDIT_T12_V2.json",
        artifact_root / "t12b-check4-20261017/seed-audit-v1.json",
        repo_path / "research/t13_product_roster_20261017/SEED_AUDIT.json",
        repo_path / "research/t13b_six_snake_20261017/SEED_AUDIT.json",
    )
    for path in prior_paths:
        value, path = _prior_receipt(path)
        bindings[str(path)] = sha256(path)
        for row in value.get("banks", {}).values():
            registered.update(row.get("seeds", ()))
        for row in value.get("allocation", {}).get("banks", {}).values():
            registered.update(row.get("seeds", ()))
        for key in ("world_seeds", "smoke_seeds"):
            registered.update(value.get(key, ()))
        for key in ("registered_collision_seeds", "observed_json_jsonl_collision_seeds"):
            registered.update(value.get(key, ()))
    exclusions = sorted(artifact_root.glob("frp-*/namespaces*/training_exclusions.json"))
    if len(exclusions) < 7:
        raise FileNotFoundError("historical FRP exclusion inventories incomplete")
    exclusion_bindings = []
    for path in exclusions:
        values = json.loads(path.read_text())
        if not isinstance(values, list) or not all(type(v) is int for v in values):
            raise ValueError(f"invalid historical exclusions: {path}")
        registered.update(values)
        exclusion_bindings.append({"path": str(path), "sha256": sha256(path), "count": len(values)})
    args = [
        "--hidden",
        "--no-ignore",
        "--fixed-strings",
        "--word-regexp",
        "--glob",
        "*.json",
        "--glob",
        "*.jsonl",
        "--only-matching",
    ]
    for value in sorted(candidate):
        args.extend(["-e", str(value)])
    own = (Path(__file__).resolve().parent, artifact_root / "t15-mixed-roster-20261017")
    scans = {
        str(path): scanner._partitioned_rg(
            path, args, own, "t15-visible-json-jsonl-seed-identities"
        )
        for path in (repo_path / "research", artifact_root)
    }
    observed = {int(hit) for hits, _ in scans.values() for hit in hits if hit.isdigit()}
    collisions = sorted(candidate & registered)
    passed = not collisions and not observed
    if sha256(source) != source_hash:
        raise RuntimeError("seed audit source changed during scan")
    return {
        "schema": SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "namespace": NAMESPACE,
        "allocation": generated,
        "audit_source_sha256": source_hash,
        "registered_count": len(registered),
        "registered_collision_seeds": collisions,
        "observed_json_jsonl_collision_seeds": sorted(observed),
        "prior_bindings": bindings,
        "t7_receipt": t7_binding,
        "t7_allocations": allocations,
        "historical_frp_training_exclusions": exclusion_bindings,
        "text_scan": {
            "roots": list(scans),
            "hits": {name: sorted(hits) for name, (hits, _) in scans.items()},
            "progress": {name: progress for name, (_, progress) in scans.items()},
            "excluded_self_roots": [str(path) for path in own],
        },
        "limitations": [
            "Visible JSON/JSONL and registered manifests only; unrecorded or external runs are "
            "outside this evidence boundary."
        ],
        "t7_receipt_generated_utc": t7_receipt.get("generated_utc"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite: {args.out}")
    result = audit(artifacts=args.artifacts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=1)
        stream.write("\n")
    print(json.dumps({"pass": result["pass"], "status": result["status"]}))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
