"""Collision-audited supplemental worlds for the Q0 B8 initializer fidelity fill.

These worlds exist solely to obtain a missing fidelity-coverage witness after
the already-complete Q0 mass packet.  They are not efficacy worlds and use a
separate, fixed namespace so they cannot silently reuse a mass or Q1 row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from research.apex_safety_20260926 import dev_screen
from research.t23_r0_screen_20261017 import bank as t23_bank

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS = ROOT.parent / "snake-dqn-artifacts"
SCHEMA = "q0-b8-init-fidelity-fill-bank/v1"
NAMESPACE = "q0-b8-init-fidelity-fill/20261018/a2/v1"
PROFILE_ID = "q0-terminal-h5000/v1"
WORLD_COUNT = 12
MIXES = ("scripted", "frozen", "mixed")
RECEIPT_ENV = "Q0Q1_INIT_FILL_BANK_RECEIPT"
RECEIPT_SHA256_ENV = "Q0Q1_INIT_FILL_BANK_RECEIPT_SHA256"


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _slot(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "slot": int(row["slot"]),
        "kind": "member",
        "member_sha256": str(row["member_sha256"]),
        "eligible_member_sha256s": [str(value) for value in row["eligible_member_sha256s"]],
    }


def build_world_bank() -> dict[str, Any]:
    """Materialize exactly twelve round-robin mix worlds in the A2 namespace."""
    worlds: list[dict[str, Any]] = []
    for number in range(WORLD_COUNT):
        mix = MIXES[number % len(MIXES)]
        seed = dev_screen.screen_seeds(1, NAMESPACE, f"world-{number}/{mix}")[0]
        row = next(item for item in dev_screen._design_rows([seed]) if item["mix"] == mix)
        slots = [_slot(item) for item in row["slots"]]
        semantic = {
            "namespace": NAMESPACE,
            "ordinal": number,
            "mix": mix,
            "world_seed": int(row["world_seed"]),
            "profile_id": PROFILE_ID,
            "horizon": 5000,
            "terminal_hero_slot": 0,
            "opponent_slots": slots,
        }
        worlds.append(
            {
                **semantic,
                "opponents": [["checkpoint", item["member_sha256"]] for item in slots],
                "roster_semantic_sha256": canonical_sha({"mix": mix, "slots": slots}),
                "world_semantic_sha256": canonical_sha(semantic),
            }
        )
    if len({item["world_seed"] for item in worlds}) != WORLD_COUNT:
        raise RuntimeError("supplemental fill namespace reused a world seed")
    return {
        "schema": SCHEMA,
        "namespace": NAMESPACE,
        "profile_id": PROFILE_ID,
        "world_count": WORLD_COUNT,
        "mixes": list(MIXES),
        "worlds": worlds,
        "allocation_semantic_sha256": canonical_sha(worlds),
    }


def source_manifest(repo: Path = ROOT) -> dict[str, str]:
    """Bind the evaluator closure plus Q0 probe, partial, and trainer helpers."""
    repo = Path(repo)
    from research.t23_r0_screen_20261017 import evaluate as t23_evaluate

    relative = set(t23_evaluate.source_manifest())
    relative.update(
        str(path.relative_to(ROOT))
        for folder in ("research/q0q1_20261017", "research/redesign_m3c_20261008")
        for path in sorted((ROOT / folder).glob("*.py"))
    )
    relative.add("research/q0q1_20261017/PREREGISTRATION.md")
    return {name: sha256_file(repo / name) for name in sorted(relative)}


def audit_world_bank(repo: Path = ROOT, artifacts: Path = DEFAULT_ARTIFACTS) -> dict[str, Any]:
    """Run the full Mac historical collision audit for the A2 fill namespace."""
    repo, artifacts = Path(repo).resolve(), Path(artifacts).resolve()
    bank = build_world_bank()
    candidate = {int(row["world_seed"]) for row in bank["worlds"]}
    bindings, historical = t23_bank._prior_strict_seed_inventory(repo, artifacts, candidate)
    # Include the existing Q0 and Q1 registered allocations explicitly, as well
    # as the strict inventory's scan of research/artifact receipts.
    from research.q0q1_20261017 import audit_q0_bank, q1_bank

    q0 = audit_q0_bank.build_world_bank()
    q1 = q1_bank.build_world_bank()
    mass = {row["world_seed"] for rows in q0["worlds"].values() for row in rows}
    q1_worlds = {row["world_seed"] for rows in q1["worlds"].values() for row in rows} | {
        row["world_seed"] for row in q1["parity_worlds"]
    }
    collisions = sorted(candidate & (historical | mass | q1_worlds))
    return {
        "schema": "q0-b8-init-fidelity-fill-seed-audit/v1",
        "namespace": NAMESPACE,
        "world_bank_semantic_sha256": bank["allocation_semantic_sha256"],
        "candidate_count": len(candidate),
        "registered_strict_count": len(historical),
        "registered_collision_seeds": collisions,
        "mass_bank_semantic_sha256": q0["allocation_semantic_sha256"],
        "q1_bank_semantic_sha256": q1["allocation_semantic_sha256"],
        "internal_unique": len(candidate) == WORLD_COUNT,
        "mandatory_evidence": bindings,
        "pass": len(candidate) == WORLD_COUNT and not collisions,
    }


def create_receipt(repo: Path = ROOT, artifacts: Path = DEFAULT_ARTIFACTS) -> dict[str, Any]:
    repo, artifacts = Path(repo).resolve(), Path(artifacts).resolve()
    bank = build_world_bank()
    audit = audit_world_bank(repo, artifacts)
    if not audit["pass"]:
        raise ValueError(
            f"supplemental fill collision audit failed: {audit['registered_collision_seeds']}"
        )
    return {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "producer": "Mac full historical strict-inventory audit",
        "source_sha256": source_manifest(repo),
        "bank": bank,
        "audit": audit,
    }


def load_receipt(*, env: Mapping[str, str] | None = None, repo: Path = ROOT) -> dict[str, Any]:
    environment = os.environ if env is None else env
    location, expected = environment.get(RECEIPT_ENV), environment.get(RECEIPT_SHA256_ENV)
    if (
        not location
        or not expected
        or len(expected) != 64
        or set(expected) - set("0123456789abcdef")
    ):
        raise ValueError("missing hash-bound Mac supplemental fill bank receipt environment")
    path = Path(location)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError("supplemental fill bank receipt is missing or SHA-256 mismatched") from exc
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("supplemental fill bank receipt is missing or SHA-256 mismatched")
    try:
        receipt = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("supplemental fill bank receipt is not valid JSON") from exc
    bank = build_world_bank()
    audit = receipt.get("audit") if isinstance(receipt, dict) else None
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("source_sha256") != source_manifest(Path(repo))
        or receipt.get("bank") != bank
        or not isinstance(audit, dict)
        or audit.get("pass") is not True
        or audit.get("registered_collision_seeds") != []
        or audit.get("world_bank_semantic_sha256") != bank["allocation_semantic_sha256"]
    ):
        raise ValueError(
            "supplemental fill bank receipt is stale, failing, or not bound to this source"
        )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    args = parser.parse_args(argv)
    receipt = create_receipt(args.repo, args.artifacts)
    sidecar = args.out.with_suffix(args.out.suffix + ".sha256")
    if args.out.exists() or sidecar.exists():
        raise FileExistsError("supplemental fill receipt or SHA-256 sidecar already exists")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with args.out.open("xb") as stream:
        stream.write(raw)
    with sidecar.open("x") as stream:
        stream.write(hashlib.sha256(raw).hexdigest() + "  " + args.out.name + "\n")
    print(f"{SCHEMA} {hashlib.sha256(raw).hexdigest()} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
