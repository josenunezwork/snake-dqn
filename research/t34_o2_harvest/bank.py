"""Create-only O2 paired worlds and fail-closed freshness receipt; no gameplay."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from research.t34_round0.bank import _rows
from research.t34_round0.common import read_json, sha256, write_once
from research.t34_round0d.bank import audit as historical_audit
from research.t34_round0d.bank import scan_history

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "research/t34_o2_harvest"
NAMESPACE = "t34-o2-harvest-20261021/v1"
COUNTS = dict.fromkeys(("scripted", "frozen", "mixed"), 32)


def materialize() -> dict[str, Any]:
    """All selection and excluded fixture seeds are fixed before data."""
    return {
        "schema": "t34-o2-bank/v1",
        "namespace": NAMESPACE,
        "mechanics_version": 2,
        "counts": COUNTS,
        "rows": _rows(NAMESPACE + "/study", COUNTS, 5000),
        "excluded_preflight_rows": _rows(
            NAMESPACE + "/mac-preflight", dict.fromkeys(COUNTS, 1), 40
        ),
        "excluded_timing_rows": _rows(NAMESPACE + "/timing", dict.fromkeys(COUNTS, 1), 5000),
        "excluded_parity_rows": _rows(NAMESPACE + "/parity", dict.fromkeys(COUNTS, 1), 500),
    }


def validate_bank(bank: dict[str, Any]) -> None:
    """Refuse any edit to counts, seeds, mix, roster, horizon or namespaces."""
    if bank != materialize():
        raise ValueError("O2 bank differs from deterministic pre-data allocation")


def published_inventories(candidates: set[int]) -> list[dict[str, Any]]:
    """Check concrete banks and reconstruct Claim A's hash-only inventories.

    Only repository paths are enumerated here. Artifact history is scanned by the
    existing safe scanner, whose rg exclusion prunes the sealed Round 0 root
    before traversal. No output directory beneath that root is opened.
    """
    from research.t34_claim_a.seed_audit import _bank_seeds, namespace_seeds, seed_digest

    entries = []
    required = (
        "research/t34_o1_harvest/BANK_O1.json",
        "research/bl_served_search/BANK_L2.json",
        "research/bl_prefix/SEED_AUDIT.json",
    )
    paths = set((ROOT / "research").rglob("*BANK*.json"))
    paths.update((ROOT / "research").rglob("*SEED_AUDIT*.json"))
    for name in required:
        if not (ROOT / name).is_file():
            raise ValueError(f"required prior-bank inventory missing: {name}")
        paths.add(ROOT / name)
    for path in sorted(paths):
        if path.parent == PACKAGE:
            continue
        value = read_json(path)
        concrete, derived = _bank_seeds(value), set()
        if isinstance(value, dict) and value.get("domain") == "t34-exit-b8-20261019/v1":
            for namespace, entry in value.get("namespaces", {}).items():
                seeds = namespace_seeds(namespace, int(entry["count"]))
                if seed_digest(seeds) != entry["sha256"]:
                    raise ValueError(f"published namespace digest differs: {path}")
                derived.update(seeds)
        if candidates.intersection(concrete | derived):
            raise ValueError(f"whole O2 allocation collides with prior inventory: {path}")
        entries.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "concrete_seed_values": len(concrete),
                "reconstructed_hash_only_seeds": len(derived),
                "collisions": 0,
            }
        )
    return entries


def allocate() -> None:
    """Collision rejects whole allocation; never reroll individual worlds."""
    bank_path, audit_path = PACKAGE / "BANK_O2.json", PACKAGE / "SEED_AUDIT_O2.json"
    if bank_path.exists() or audit_path.exists():
        raise FileExistsError("O2 allocation is create-only")
    bank = materialize()
    allocation = {
        "rows": bank["rows"],
        "excluded_preflight_rows": bank["excluded_preflight_rows"],
        "excluded_a5d_rows": bank["excluded_timing_rows"] + bank["excluded_parity_rows"],
    }
    receipt = historical_audit(allocation, scan=scan_history)
    receipt.update(
        schema="t34-o2-seed-audit/v1",
        namespace=NAMESPACE,
        excluded_timing_count=3,
        excluded_parity_count=3,
        published_inventories=published_inventories(set(receipt["seeds"])),
    )
    write_once(bank_path, bank)
    receipt["bank_sha256"] = sha256(bank_path)
    write_once(audit_path, receipt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocate", action="store_true")
    args = parser.parse_args()
    if args.allocate:
        allocate()
    else:
        validate_bank(read_json(PACKAGE / "BANK_O2.json"))


if __name__ == "__main__":
    main()
