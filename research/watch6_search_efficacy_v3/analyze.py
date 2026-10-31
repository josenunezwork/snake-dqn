"""Mac-only one-look reduction of the 96-pair Watch6 V3 union reusing V2 analyze.decide."""

from __future__ import annotations

import argparse
import copy
import json
import platform
from pathlib import Path
from unittest.mock import patch

from research.watch6_search_efficacy import analyze as v2_analyze
from research.watch6_search_efficacy_v3 import protocol as p
from research.watch6_search_efficacy_v3.resume import build as build_verified_ledger
from tools.ops.package_spec import publish_json, unit_id, verify_immutable


def decide_v3(rows: list[dict], *, reserve_verified_indices: set[int] | None = None) -> dict:
    """Validate V3 primary/reserve rows and delegate paired-t + Fieller df=95 math to V2."""
    if len(rows) != p.COUNT or {r.get("allocation", {}).get("index") for r in rows} != set(
        range(p.COUNT)
    ):
        raise ValueError("INCOMPLETE or duplicate union; no scientific look")

    reserve_allowed = set(reserve_verified_indices or set())
    if len(reserve_allowed) > p.MAX_RESERVE_SUBSTITUTIONS:
        raise ValueError(
            f"reserve substitutions {len(reserve_allowed)} exceed cap {p.MAX_RESERVE_SUBSTITUTIONS}"
        )

    ordered = sorted(rows, key=lambda r: r["allocation"]["index"])
    primary_alloc = p.allocation()
    reserve_alloc = p.reserve_allocation()
    effective_alloc: list[dict] = []
    reserve_used: list[int] = []

    for index, row in enumerate(ordered):
        seed_kind = row.get("seed_kind")
        if (
            row.get("schema") != "watch6-search-pair/v3"
            or row.get("namespace") != p.NAMESPACE
            or row.get("science") != p.identity()
            or row.get("fixture") is not False
            or row.get("unit_id") != unit_id(p.NAMESPACE, "watch6", index)
            or seed_kind not in ("primary", "reserve")
        ):
            raise ValueError(f"foreign V3 schema, namespace, fixture, or identity at slot {index}")
        if seed_kind == "reserve":
            if index not in reserve_allowed:
                raise ValueError(
                    f"slot {index} claims reserve seed without verified reserve lineage"
                )
            expected = reserve_alloc[index]
            reserve_used.append(index)
        else:
            if index in reserve_allowed:
                raise ValueError(
                    f"slot {index} mismatch between primary payload and reserve lineage"
                )
            expected = primary_alloc[index]
        if row.get("allocation") != expected:
            raise ValueError(
                f"slot {index} seed/horizon mismatch against frozen {seed_kind} allocation"
            )
        effective_alloc.append(expected)

    if len(reserve_used) > p.MAX_RESERVE_SUBSTITUTIONS:
        raise ValueError("reserve substitutions exceed pre-registered study-wide cap")

    normalized = [copy.deepcopy(r) for r in ordered]
    with patch.multiple(
        v2_analyze.p,
        NAMESPACE=p.NAMESPACE,
        COUNT=p.COUNT,
        HORIZON=p.HORIZON,
        identity=p.identity,
        allocation=lambda: effective_alloc,
    ):
        decision = v2_analyze.decide(normalized)

    fieller = decision["mass_relative_ci95_fieller"]
    return {
        **decision,
        "death_pass": bool(decision["death_pass"]),
        "mass_pass": bool(decision["mass_pass"]),
        "mass_relative_ci95_fieller": (
            [float(x) for x in fieller] if fieller is not None else None
        ),
        "schema": "watch6-v3-efficacy-decision/v1",
        "namespace": p.NAMESPACE,
        "science": p.identity(),
        "primary_pairs_count": p.COUNT - len(reserve_used),
        "reserve_substitutions_count": len(reserve_used),
        "reserve_substituted_indices": reserve_used,
    }


def analyze_verified_union(unit_paths: list[Path], receipt_paths: list[Path]) -> dict:
    """Verify receipts, sidecars, .meta.json, and reserve lineage BEFORE opening outcomes."""
    ledger = build_verified_ledger(unit_paths, receipt_paths)
    if len(ledger["units"]) != p.COUNT:
        found_n = len(ledger["units"])
        raise ValueError(
            f"INVALID_OR_INCOMPLETE: expected {p.COUNT} verified pairs, found {found_n}"
        )

    reserve_indices: set[int] = set()
    rows = []
    for path in sorted(unit_paths, key=lambda item: item.name):
        verify_immutable(path)
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        if meta.get("seed_kind") == "reserve":
            reserve_indices.add(int(meta["allocation"]["index"]))
        rows.append(json.loads(path.read_text()))

    decision = decide_v3(rows, reserve_verified_indices=reserve_indices)
    decision["verified_receipt_count"] = len(receipt_paths)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", nargs="+", type=Path, required=True)
    parser.add_argument("--receipts", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("registered look is Mac-only")
    decision = analyze_verified_union(args.units, args.receipts)
    publish_json(args.out, decision)
    print(
        json.dumps(
            {
                "verdict": decision["verdict"],
                "n": decision["n"],
                "reserve_substitutions_count": decision["reserve_substitutions_count"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
