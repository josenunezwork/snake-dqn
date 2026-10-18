"""Hash-bound Mac collision-inventory receipts for the Q0/Q1 Colab jobs.

The historical FRP exclusion inventories deliberately remain Mac-resident.  A
Colab job must therefore consume a receipt made by the full Mac audit, never a
partial reimplementation over its staged inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from research.q0q1_20261017 import audit_q0_bank, q1_bank

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS = ROOT.parent / "snake-dqn-artifacts"
SCHEMA = "q0q1-bank-inventory-receipt/v1"
RECEIPT_ENV = "Q0Q1_BANK_INVENTORY_RECEIPT"
RECEIPT_SHA256_ENV = "Q0Q1_BANK_INVENTORY_RECEIPT_SHA256"
KIND_Q0 = "q0_development"
KIND_Q1 = "q1_evaluation"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one file without interpreting its bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_bindings(repo: Path) -> dict[str, str]:
    paths = (
        "research/q0q1_20261017/audit_q0_bank.py",
        "research/q0q1_20261017/q1_bank.py",
        "research/apex_safety_20260926/dev_screen.py",
        "research/t23_r0_screen_20261017/bank.py",
        "research/redesign_t7_phase_r_20261016/seed_audit.py",
        "research/t3_screen_20261015/banks.py",
        "research/t2_screen_20261015/banks.py",
        "research/t4e_return_20261016/seed_audit.py",
        "research/t4f_return_20261016/seed_audit.py",
        "research/redesign_m2_20261008/ni_spec.py",
        "research/redesign_m2_20261008/ni_spec_m2b.py",
        "research/q0q1_20261017/inventory_receipt.py",
    )
    return {relative: sha256_file(repo / relative) for relative in paths}


def source_manifest(repo: Path = ROOT) -> dict[str, str]:
    """Expose the consumer/producer source binding for queue-test fixtures."""
    return _source_bindings(Path(repo))


def create_receipt(repo: Path = ROOT, artifacts: Path = DEFAULT_ARTIFACTS) -> dict[str, Any]:
    """Run both full inventory-dependent audits and make their portable receipt."""
    repo, artifacts = Path(repo).resolve(), Path(artifacts).resolve()
    q0_bank = audit_q0_bank.build_world_bank()
    q1_world_bank = q1_bank.build_world_bank()
    q0_audit = audit_q0_bank.audit_world_bank(repo, artifacts=artifacts, allow_receipt=False)
    q1_audit = q1_bank.audit_world_bank(repo, artifacts)
    if not q0_audit.get("pass") or not q1_audit.get("pass"):
        raise ValueError("Mac historical collision audit did not pass")
    return {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "producer": "Mac full historical strict-inventory audit",
        "source_sha256": _source_bindings(repo),
        "audits": {
            KIND_Q0: {
                "world_bank_semantic_sha256": q0_bank["allocation_semantic_sha256"],
                "audit": q0_audit,
            },
            KIND_Q1: {
                "world_bank_semantic_sha256": q1_world_bank["allocation_semantic_sha256"],
                "audit": q1_audit,
            },
        },
    }


# A public semantic name for callers that build a receipt in an explicitly
# supplied Mac inventory fixture.
build_receipt = create_receipt


def load_receipt_audit(
    bank_kind: str,
    bank: Mapping[str, Any],
    *,
    env: Mapping[str, str] | None = None,
    repo: Path = ROOT,
) -> dict[str, Any]:
    """Load a job-bound receipt and reject stale, substituted, or failed audits."""
    if bank_kind not in (KIND_Q0, KIND_Q1):
        raise ValueError(f"unknown inventory receipt bank kind: {bank_kind}")
    environment = os.environ if env is None else env
    location, expected = environment.get(RECEIPT_ENV), environment.get(RECEIPT_SHA256_ENV)
    if not location or not expected:
        raise ValueError("missing hash-bound Mac bank inventory receipt environment")
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("bank inventory receipt SHA-256 is malformed")
    path = Path(location)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError("bank inventory receipt is missing or SHA-256 mismatched") from exc
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("bank inventory receipt is missing or SHA-256 mismatched")
    try:
        receipt = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("bank inventory receipt is not valid JSON") from exc
    if receipt.get("schema") != SCHEMA:
        raise ValueError("bank inventory receipt schema is unrecognized")
    if receipt.get("source_sha256") != _source_bindings(Path(repo)):
        raise ValueError("bank inventory receipt source binding is stale")
    entry = receipt.get("audits", {}).get(bank_kind)
    if not isinstance(entry, dict) or entry.get("world_bank_semantic_sha256") != bank.get(
        "allocation_semantic_sha256"
    ):
        raise ValueError("bank inventory receipt does not bind this world bank")
    audit = entry.get("audit")
    if (
        not isinstance(audit, dict)
        or audit.get("pass") is not True
        or audit.get("world_bank_semantic_sha256") != bank.get("allocation_semantic_sha256")
        or audit.get("registered_collision_seeds") != []
    ):
        raise ValueError("bank inventory receipt records a failing or incomplete collision audit")
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    args = parser.parse_args(argv)
    receipt = create_receipt(args.repo, args.artifacts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sidecar = args.out.with_suffix(args.out.suffix + ".sha256")
    if args.out.exists() or sidecar.exists():
        raise FileExistsError("inventory receipt or SHA-256 sidecar already exists")
    encoded = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    with args.out.open("xb") as stream:
        stream.write(encoded)
    with sidecar.open("x") as stream:
        stream.write(digest + "  " + args.out.name + "\n")
    print(f"{SCHEMA} {digest} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
