"""Extend the unused O2 bank prospectively, retaining all original worlds."""

from __future__ import annotations

import argparse
from typing import Any

from research.t34_o2_harvest.bank import PACKAGE, published_inventories
from research.t34_o2_harvest.plan_v2 import MIXES, optimal_capacity
from research.t34_round0.bank import _rows
from research.t34_round0.common import read_json, sha256, write_once
from research.t34_round0d.bank import audit as historical_audit
from research.t34_round0d.bank import scan_history

ROOT = PACKAGE.parents[1]
NAMESPACE = "t34-o2-harvest-20261021/v2/round-5"


def materialize() -> dict[str, Any]:
    """Retain the v1 96 and add deterministic fresh worlds and fixtures."""
    prior = read_json(PACKAGE / "BANK_O2.json")
    counts = optimal_capacity()["mix_counts"]
    added = _rows(NAMESPACE + "/added-study", {m: counts[m] - 32 for m in MIXES}, 5000)
    combined = prior["rows"] + added
    # Equal interleaving keeps every chunk close to balanced, fixed before data.
    groups = {m: [r for r in combined if r["mix"] == m] for m in MIXES}
    rows = [groups[m][i] for i in range(counts[MIXES[0]]) for m in MIXES]
    return {
        "schema": "t34-o2-bank/v2",
        "namespace": NAMESPACE,
        "mechanics_version": 2,
        "counts": counts,
        "rows": rows,
        "inherited_v1_worlds": 96,
        "inherited_bank_sha256": sha256(PACKAGE / "BANK_O2.json"),
        "excluded_preflight_rows": _rows(NAMESPACE + "/mac-preflight", dict.fromkeys(MIXES, 1), 40),
        "excluded_timing_rows": _rows(NAMESPACE + "/timing", dict.fromkeys(MIXES, 1), 5000),
        "excluded_parity_rows": _rows(NAMESPACE + "/parity", dict.fromkeys(MIXES, 1), 500),
    }


def validate_bank(bank: dict[str, Any]) -> None:
    """Reject any size, seed, roster or fixture drift."""
    if bank != materialize():
        raise ValueError("O2 V2 bank differs from deterministic pre-data allocation")


def allocate() -> None:
    """Audit added identities before saving; inherited worlds keep v1 audit."""
    bank_path, audit_path = PACKAGE / "BANK_O2_V2.json", PACKAGE / "SEED_AUDIT_O2_V2.json"
    if bank_path.exists() or audit_path.exists():
        raise FileExistsError("V2 allocation is create-only")
    bank = materialize()
    all_rows = [
        bank_row
        for key in (
            "rows",
            "excluded_preflight_rows",
            "excluded_timing_rows",
            "excluded_parity_rows",
        )
        for bank_row in bank[key]
    ]
    if len({r["world_seed"] for r in all_rows}) != len(all_rows):
        raise ValueError("whole V2 allocation rejected: internal or inherited seed collision")
    added = [r for r in bank["rows"] if r["namespace"].startswith(NAMESPACE + "/added-study/")]
    allocation = {
        "rows": added,
        "excluded_preflight_rows": bank["excluded_preflight_rows"],
        "excluded_a5d_rows": bank["excluded_timing_rows"] + bank["excluded_parity_rows"],
    }
    receipt = historical_audit(allocation, scan=scan_history)
    receipt.update(
        schema="t34-o2-seed-audit/v2",
        namespace=NAMESPACE,
        published_inventories=published_inventories(set(receipt["seeds"])),
        inherited_v1_worlds=96,
        inherited_v1_audit_sha256=sha256(PACKAGE / "SEED_AUDIT_O2.json"),
    )
    # The shared scanner skips this package; explicitly reject v1 selection/fixture reuse.
    v1 = read_json(PACKAGE / "BANK_O2.json")
    old = {
        r["world_seed"]
        for key in (
            "rows",
            "excluded_preflight_rows",
            "excluded_timing_rows",
            "excluded_parity_rows",
        )
        for r in v1[key]
    }
    if old.intersection(receipt["seeds"]):
        raise ValueError("added V2 allocation collides with v1")
    write_once(bank_path, bank)
    receipt["bank_sha256"] = sha256(bank_path)
    write_once(audit_path, receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocate", action="store_true")
    args = parser.parse_args()
    if args.allocate:
        allocate()
    else:
        validate_bank(read_json(PACKAGE / "BANK_O2_V2.json"))
