"""Freeze operator ratification before dispatch; never ratify on import."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from research.t34_round0.bank import validate_bank
from research.t34_round0.common import (
    PACKAGE,
    ROOT,
    head,
    read_json,
    sha256,
    source_hashes,
    write_once,
)

PREREG = PACKAGE / "PREREGISTRATION_ROUND0.md"


def check_audit(bank_path: Path, audit_path: Path, *, preflight: bool = False) -> dict[str, Any]:
    """Validate the saved Mac audit instead of re-scanning partial Colab history."""
    bank, audit = read_json(bank_path), read_json(audit_path)
    validate_bank(bank, preflight=preflight)
    if not preflight:
        tables = read_json(PACKAGE / "TABULAR_SEED_AUDIT.json")
        if (
            tables.get("status") != "CLEAR"
            or tables.get("collisions") != []
            or tables.get("bank_sha256") != sha256(bank_path)
            or tables.get("candidate_count") != 242
            or not tables.get("files")
        ):
            raise ValueError("CSV/TSV historical-bank audit does not bind this allocation")
    seeds = {r["world_seed"] for r in bank["rows"]}
    if (
        audit.get("schema") != "t34-round0-seed-audit/v1"
        or audit.get("status") != "CLEAR"
        or audit.get("collisions") != []
        or audit.get("bank_sha256") != sha256(bank_path)
        or any(audit.get("concrete_hits", {}).values())
        or not seeds.issubset(set(audit.get("seeds", [])))
        or len(audit.get("historical_frp_exclusions", [])) < 7
        or not audit.get("retention_bank", {}).get("sha256")
    ):
        raise ValueError("missing, incomplete, or mismatched collision audit")
    return bank


def require_registration(
    registration: Mapping[str, Any], bank_path: Path, audit_path: Path, *, repo: Path = ROOT
) -> dict[str, Any]:
    """Reject drafts, changed rules, checkpoints, source files, or bank identities."""
    bank = check_audit(bank_path, audit_path)
    required = {
        "schema": "t34-round0-ratification/v1",
        "status": "RATIFIED",
        "registration_sha256": sha256(repo / "research/t34_round0/PREREGISTRATION_ROUND0.md"),
        "bank_sha256": sha256(bank_path),
        "audit_sha256": sha256(audit_path),
    }
    if any(registration.get(k) != v for k, v in required.items()):
        raise ValueError("Round 0 is not ratified for these exact rules and bank")
    if not all(
        isinstance(registration.get(k), str) and registration[k].strip()
        for k in ("operator", "approved_at", "commit")
    ):
        raise ValueError("operator identity, date and commit are required")
    commit = registration["commit"]
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("ratification needs a full source SHA")
    if head(repo) != commit:
        raise ValueError("checkout differs from ratified source commit")
    hashes = registration.get("source_hashes")
    if not isinstance(hashes, dict) or len(hashes) < 10:
        raise ValueError("ratification lacks the executable source closure")
    if hashes != source_hashes(repo):
        raise ValueError("source inventory differs from ratified executable closure")
    preflight = registration.get("preflight", {})
    if (
        not isinstance(preflight, dict)
        or preflight.get("pairs") != 2
        or preflight.get("horizon") != 500
        or preflight.get("source_hashes") != hashes
        or not isinstance(preflight.get("sha256"), str)
        or len(preflight["sha256"]) != 64
    ):
        raise ValueError("ratification lacks the source-bound real A6 preflight")
    for name, digest in hashes.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or sha256(repo / path) != digest:
            raise ValueError(f"ratified source bytes differ: {name}")
    a5 = registration.get("a5", {})
    if not isinstance(a5.get("sha256"), str) or len(a5["sha256"]) != 64:
        raise ValueError("ratification lacks the A5 evidence binding")
    return bank


def ratify(
    *,
    operator: str,
    a5_path: Path,
    preflight_record: Path,
    out: Path,
    bank_path: Path = PACKAGE / "BANK_ROUND0.json",
    audit_path: Path = PACKAGE / "SEED_AUDIT_ROUND0.json",
    repo: Path = ROOT,
) -> dict[str, Any]:
    """Create the operator's explicit approval after committed preflight and A5."""
    if not operator.strip():
        raise ValueError("operator name is required")
    check_audit(bank_path, audit_path)
    from research.t34_round0.make_manifest import validate_a5

    a5 = read_json(a5_path)
    validate_a5(a5)
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise ValueError("ratification requires committed tracked source")
    hashes = source_hashes(repo)
    verification = read_json(preflight_record)
    if (
        verification.get("schema") != "t34-round0-preparation-validation/v1"
        or verification.get("status") != "PASS"
        or verification.get("source_hashes") != hashes
        or verification.get("preflight", {}).get("pairs") != 2
        or verification.get("preflight", {}).get("horizon") != 500
    ):
        raise ValueError("ratification requires a source-bound successful real A6 preflight")
    commit = head(repo)
    # Bind every executable file to the same committed tree, not just a clean status.
    for name, digest in hashes.items():
        payload = subprocess.run(
            ["git", "show", f"{commit}:{name}"], cwd=repo, check=True, capture_output=True
        ).stdout
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(f"uncommitted source: {name}")
    payload = {
        "schema": "t34-round0-ratification/v1",
        "status": "RATIFIED",
        "operator": operator,
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "registration_sha256": sha256(PREREG),
        "bank_sha256": sha256(bank_path),
        "audit_sha256": sha256(audit_path),
        "a5": {"path": str(a5_path.resolve()), "sha256": sha256(a5_path)},
        "preflight": {
            "path": str(preflight_record.resolve()),
            "sha256": sha256(preflight_record),
            "source_hashes": hashes,
            "pairs": 2,
            "horizon": 500,
        },
        "source_hashes": hashes,
        "release_authorized": False,
        "scope": "Round 0 development only; reviewed manifest plan caps still required",
    }
    require_registration(payload, bank_path, audit_path, repo=repo)
    write_once(out, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--a5", type=Path, required=True)
    parser.add_argument("--preflight-record", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ratify(
        operator=args.operator,
        a5_path=args.a5,
        preflight_record=args.preflight_record,
        out=args.out,
    )


if __name__ == "__main__":
    main()
