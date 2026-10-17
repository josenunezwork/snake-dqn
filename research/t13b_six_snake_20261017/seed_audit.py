"""T13b fresh, paired world allocation against the full T7/T12/T13 audit boundary."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.redesign_t7_phase_r_20261016 import seed_audit as t7
from research.t12_b8_serving_20261017 import seed_audit as t12
from research.t4e_return_20261016 import seed_audit as scanner

NAMESPACE = "t13b-six-snake-20261017/v1"
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / "SEED_AUDIT.json"


def seeds(purpose: str = "worlds", count: int = 25) -> list[int]:
    return [
        int.from_bytes(hashlib.sha256(f"{NAMESPACE}|{purpose}|{i}".encode()).digest()[:4], "big")
        for i in range(count)
    ]


def audit() -> dict[str, Any]:
    artifacts = ROOT.parent / "snake-dqn-artifacts"
    receipt, binding = t12._t7_receipt(ROOT)
    allocations, allocated = t12._t7_allocations(ROOT)
    registered = t7._registered(ROOT) | allocated
    sources = {}
    for name in ("SEED_AUDIT_T12.json", "SEED_AUDIT_T12_V2.json"):
        path = ROOT / "research/t12_b8_serving_20261017" / name
        old = json.loads(path.read_text())
        if not old["pass"]:
            raise ValueError(f"nonpassing prior audit: {path}")
        for bank in old["banks"].values():
            registered.update(bank["seeds"])
        sources[str(path.relative_to(ROOT))] = t12.sha256(path)
    prior_path = ROOT / "research/t13_product_roster_20261017/SEED_AUDIT.json"
    prior = json.loads(prior_path.read_text())
    if not prior["pass"]:
        raise ValueError("T13 prior audit failed")
    registered.update(prior["world_seeds"])
    registered.update(prior["smoke_seeds"])
    sources[str(prior_path.relative_to(ROOT))] = t12.sha256(prior_path)
    exclusions = sorted(artifacts.glob("frp-*/namespaces*/training_exclusions.json"))
    if len(exclusions) < 7:
        raise ValueError("missing historical exclusion inventories")
    for path in exclusions:
        values = json.loads(path.read_text())
        if not isinstance(values, list) or not all(type(v) is int for v in values):
            raise ValueError(f"invalid exclusions: {path}")
        registered.update(values)
        sources[str(path)] = t12.sha256(path)
    # T7 additionally reads source namespace seeds and Phase R plans explicitly.
    mandatory = receipt["mandatory_evidence"]
    namespace = json.loads(Path(mandatory["t3_namespace"]["path"]).read_text())
    registered.update(namespace["seeds"])
    for name in ("m3_phase_r", "m4_phase_r"):
        plan = json.loads(Path(mandatory[name]["path"]).read_text())
        for bank in plan["banks"].values():
            registered.update(bank)
    worlds, smoke = seeds(), seeds("smoke", 1)
    candidate = set(worlds + smoke)
    if len(candidate) != 26:
        raise ValueError("T13b internal seed collision")
    scan_args = [
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
        scan_args.extend(["-e", str(value)])
    own = (Path(__file__).resolve().parent, artifacts / "t13b-six-snake-20261017")
    scans = {}
    observed = set()
    for root in (ROOT / "research", artifacts):
        hits, progress = scanner._partitioned_rg(root, scan_args, own, "t13b-seed-identities")
        observed.update(int(hit) for hit in hits if hit.isdigit())
        scans[str(root)] = {"hits": sorted(hits), "progress": progress}
    collisions = sorted(candidate & registered)
    return {
        "schema": "t13b-seed-audit/v1",
        "pass": not collisions and not observed,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "namespace": NAMESPACE,
        "recipe": "uint32_be(sha256(f'{namespace}|{purpose}|{i}')[:4])",
        "world_seeds": worlds,
        "smoke_seeds": smoke,
        "registered_count": len(registered),
        "registered_collisions": collisions,
        "observed_collisions": sorted(observed),
        "source_bindings": sources,
        "t7_receipt": binding,
        "t7_allocations": allocations,
        "scans": scans,
        "excluded_self_roots": [str(p) for p in own],
        "audit_source_sha256": t12.sha256(Path(__file__)),
        "limitations": [
            "Visible JSON/JSONL and mandatory binary-bank manifests only; no claim about unrecorded experiments, opaque binaries or external systems."
        ],
    }


if __name__ == "__main__":
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result = audit()
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "pass",
                    "registered_count",
                    "registered_collisions",
                    "observed_collisions",
                )
            }
        )
    )
    raise SystemExit(0 if result["pass"] else 1)
