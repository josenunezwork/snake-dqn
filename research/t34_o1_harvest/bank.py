"""Allocate O1 once using the existing Round 0 rosters and safe seed auditor."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from research.t34_round0.bank import _rows
from research.t34_round0.common import read_json, sha256, write_once
from research.t34_round0d.bank import audit as existing_audit
from research.t34_round0d.bank import scan_history

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "research/t34_o1_harvest"
NAMESPACE = "t34-o1-harvest-20261021/v1"


def materialize() -> dict[str, Any]:
    """Derive all study and excluded identities before the collision scan."""
    return {
        "schema": "t34-o1-bank/v1",
        "namespace": NAMESPACE,
        "mechanics_version": 2,
        "rows": _rows(
            NAMESPACE + "/study", dict.fromkeys(("scripted", "frozen", "mixed"), 16), 5000
        ),
        "excluded_timing_rows": _rows(NAMESPACE + "/timing", {"frozen": 1, "mixed": 1}, 5000),
        "excluded_preflight_rows": _rows(
            NAMESPACE + "/mac-preflight", {"scripted": 1, "frozen": 1}, 40
        ),
        "excluded_parity_rows": _rows(
            NAMESPACE + "/parity", dict.fromkeys(("scripted", "frozen", "mixed"), 1), 5000
        ),
    }


def allocate() -> None:
    """Reject the whole allocation on a collision; never replace a seed."""
    bank_path, audit_path = PACKAGE / "BANK_O1.json", PACKAGE / "SEED_AUDIT_O1.json"
    if bank_path.exists() or audit_path.exists():
        raise FileExistsError("O1 allocation is create-only")
    bank = materialize()
    # Existing audit checks registered banks, T3 census, literal history incl.
    # CSV/TSV and >=7 FRP training exclusions, with positive scanner controls.
    # scan_history excludes the sealed Round 0 directory BEFORE rg traverses it.
    payload = {
        "rows": bank["rows"],
        "excluded_preflight_rows": bank["excluded_preflight_rows"],
        "excluded_a5d_rows": bank["excluded_timing_rows"] + bank["excluded_parity_rows"],
    }
    receipt = existing_audit(payload, scan=scan_history)
    receipt.update(
        schema="t34-o1-seed-audit/v1",
        namespace=NAMESPACE,
        excluded_timing_count=2,
        excluded_parity_count=3,
    )
    write_once(bank_path, bank)
    receipt["bank_sha256"] = sha256(bank_path)
    write_once(audit_path, receipt)


def validate_bank(bank: dict[str, Any]) -> None:
    """Refuse edited seeds, rosters, counts, namespace or horizon."""
    if bank != materialize():
        raise ValueError("O1 bank differs from the deterministic pre-data allocation")


def supplement() -> None:
    """Check main's published audit inventories, including hash-only namespaces."""
    from research.t34_claim_a.seed_audit import _bank_seeds, namespace_seeds, seed_digest

    own = read_json(PACKAGE / "BANK_O1.json")
    candidates = _bank_seeds(own)
    inventories = []
    for path in sorted((ROOT / "research").rglob("*SEED_AUDIT*.json")):
        if path.parent == PACKAGE:
            continue
        value = read_json(path)
        concrete = _bank_seeds(value)
        derived = set()
        if value.get("domain") == "t34-exit-b8-20261019/v1":
            for namespace, entry in value.get("namespaces", {}).items():
                seeds = namespace_seeds(namespace, int(entry["count"]))
                if seed_digest(seeds) != entry["sha256"]:
                    raise ValueError(f"published namespace digest differs: {path}")
                derived.update(seeds)
        if candidates.intersection(concrete | derived):
            raise ValueError(f"whole O1 allocation collides with published inventory: {path}")
        inventories.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "concrete_seed_values": len(concrete),
                "reconstructed_hash_only_seeds": len(derived),
                "collisions": 0,
            }
        )
    write_once(
        PACKAGE / "SEED_AUDIT_MAIN_UPDATE.json",
        {
            "schema": "t34-o1-main-update-audit/v1",
            "status": "CLEAR",
            "bank_sha256": sha256(PACKAGE / "BANK_O1.json"),
            "note": "supplements original full-history audit after merging concurrent main",
            "inventories": inventories,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocate", action="store_true")
    parser.add_argument("--supplement", action="store_true")
    args = parser.parse_args()
    if args.supplement:
        supplement()
    elif args.allocate:
        allocate()
    else:
        validate_bank(read_json(PACKAGE / "BANK_O1.json"))


if __name__ == "__main__":
    main()
